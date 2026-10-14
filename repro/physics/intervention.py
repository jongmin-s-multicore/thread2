"""Frozen PointMaze action intervention: prepare CPU oracles, then GPU inference.

Run phases separately. Never instantiate an optimizer or modify source results.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def save(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)


def read(p):return json.loads(Path(p).read_text())


def csvwrite(p,rows):
    if not rows:return
    with Path(p).open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def grid():
    rows=[dict(name='zero',radius=0.,direction=-1,control=[0.,0.],stress=False)]
    for radius in [.05,.1,.2,.4,1.]:
        for direction in range(8):
            angle=direction*np.pi/4
            rows.append(dict(name=f'r{radius:g}_d{direction}',radius=radius,direction=direction,
                             control=(radius*np.array([np.cos(angle),np.sin(angle)])).tolist(),stress=radius==1.))
    return rows


def clearance(q,params):
    distances=[]
    for w in params['walls']:
        d=np.abs(np.asarray(q)-w['center'])-w['halfsize']
        distances.append(np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(np.max(d,axis=-1),0)-params['particle_radius'])
    return np.min(distances,axis=0)


def analytic(st,actions,params):
    a,b=np.array(params['Ah']),np.array(params['Bh']);q=np.array(st[:2],float);v=np.array(st[2:],float)
    path=[np.r_[q,v].tolist()];clips=0
    for u in actions:
        clips+=int(np.any(np.abs(v)>5));v=np.clip(v,-5,5)
        for _ in range(5):
            v=a@v+b@np.clip(u,-1,1);q=q+params['dt']*v;path.append(np.r_[q,v].tolist())
    return np.array(path),clips


def branch_actions(history,future,mean,std):
    """Three action blocks: two past blocks, then the manipulated future block."""
    raw=np.concatenate([history,future],axis=0)
    assert raw.shape==(15,2)
    normalized=(raw-mean)/std
    assert np.max(np.abs(normalized*std+mean-raw))<1e-6
    return normalized.reshape(3,10)


def sources(root,repo):
    base=root/'work/physics';src=base/'pointmaze-velocity-diagnosis-dev20-20261009'
    return dict(base=base,src=src,audit=base/'pointmaze-dynamics-audit-20261009',
                b=base/'pointmaze-dynamics-b-20261010',previous=base/'pointmaze-proprio-transition-dev20-20261009',
                data=root/'work/data/point_maze',cfg=root/'work/checkpoints/outputs/point_maze/hydra.yaml',repo=repo)


def prepare(root,repo,out,reuse=None):
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);paths=sources(root,repo)
    assert read(paths['b']/'checks.json')['pass_all']
    assert read(paths['b']/'completion.json')['status']=='DYNAMICS_B_COMPLETE_DEVELOPMENT'
    params=read(paths['audit']/'parameters.json');save(out/'parameters.json',params)
    data=paths['data'];checkpoint=paths['cfg'].parent/'checkpoints/model_latest.pth'
    files=[data/x for x in ['states.pth','actions.pth','seq_lengths.pth']]
    files += [checkpoint,paths['cfg'],paths['src']/'feature_cache.joblib',paths['src']/'split_manifest.json',
              paths['b']/'new_oracle_traces.json',paths['previous']/'viewer_data.json',paths['audit']/'parameters.json',
              repo/'models/visual_world_model.py',repo/'models/dino.py',repo/'env/pointmaze/maze_model.py']
    hashes={str(f):sha(f) for f in files}
    prior=read(paths['b']/'manifest.json')['sources']
    for f in [data/'states.pth',data/'actions.pth',checkpoint,paths['cfg']]:assert hashes[str(f)]==prior[str(f)]
    for relative in ['models/visual_world_model.py','models/dino.py','env/pointmaze/maze_model.py']:
        assert sha(repo/relative)==sha(root/'repo'/relative),'Development upstream code differs from validated version'
    save(out/'manifest.json',dict(sources=hashes,script_sha256=sha(__file__),
         protocol_sha256=sha(Path(__file__).with_name('intervention_protocol.md')),
         code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),
         previous_status=read(root/'work/STATUS.json'),new_training=False,new_model_inference=True,
         bootstrap_seed=0,bootstrap_count=10000,scope='C one-step pilot, not D or unseen WM evaluation'))
    split=read(paths['src']/'split_manifest.json');rng=np.random.default_rng(20261009)
    groups=dict(calibration_fit=rng.choice(split['inner_fit'],16,replace=False).tolist(),
                calibration_tune=rng.choice(split['inner_tune'],4,replace=False).tolist(),development=split['fresh_development'])
    ids=sum(groups.values(),[]);assert len(set(ids))==40 and not set(ids)&set(split['sealed_wm_test']+split['old_development'])
    save(out/'split.json',groups);conditions=grid();save(out/'conditions.json',conditions)
    states=torch.load(data/'states.pth',weights_only=False,map_location='cpu').float().numpy()
    # Preserve simulator's original raw precision; the WM historically casts to
    # float32 before normalization. Never label a float32 key as the old raw key.
    actions=torch.load(data/'actions.pth',weights_only=False,map_location='cpu').numpy()
    lengths=torch.as_tensor(torch.load(data/'seq_lengths.pth',weights_only=False)).numpy()
    assert len(set(hashlib.sha256(states[i].tobytes()+actions[i].tobytes()).hexdigest() for i in ids))==40
    # Select before model inference. Free analytic paths are conservative candidates,
    # and every selected branch is subsequently checked in the simulator.
    selected=[];selection=[];margin=.004
    for splitname,episodes in groups.items():
        for ep in episodes:
            found=False
            for macro in range(2,15):
                k=macro*5
                if k+25>=int(lengths[ep]):continue
                st=states[ep,k];minimum=float('inf');clip=False
                for c in conditions[:33]:
                    tr,cl=analytic(st,np.tile(c['control'],(5,1)),params)
                    minimum=min(minimum,float(np.min(clearance(tr[:,:2],params))));clip|=cl>0
                valid=minimum>margin and not clip
                selection.append(dict(split=splitname,episode=ep,raw_anchor=k,min_clearance=minimum,any_velocity_clip=clip,selected=valid))
                if valid:
                    selected.append(dict(split=splitname,episode=ep,raw_anchor=k,start=st.tolist(),
                                         context_indices=[k-10,k-5,k],history_actions=actions[ep,k-10:k].tolist(),
                                         recorded_actions=actions[ep,k:k+5].tolist()));found=True;break
            if not found:selection.append(dict(split=splitname,episode=ep,raw_anchor=-1,min_clearance=None,any_velocity_clip=None,selected=False))
    save(out/'selected.json',selected);save(out/'selection_audit.json',selection)
    assert selected,'No feasible context; do not relax the predeclared rule'
    # Reuse B's completed oracle keys and previous GT traces.
    def key(st,ac,k):return(np.array(st,dtype=float).tobytes(),np.array(ac,dtype=float).tobytes(),k)
    cached={}
    for r in read(paths['b']/'new_oracle_traces.json'):
        cached[key(r['trace'][0],r['requested_actions'],r['raw_start'])]=r
    for item in read(paths['previous']/'viewer_data.json')['rows']:
        k=(2+item['from_horizon'])*5;ac=actions[item['episode'],k:(2+item['to_horizon'])*5]
        if item['readout']=='proprio':
            v=item['variants']['true_both'];tr=v['trace']
            cached[key(tr[0],ac,k)]=dict(trace=tr,contacts=v['wall_contact_substeps'],
                clips=int(sum(np.any(np.abs(np.array(tr)[i,2:])>5) for i in range(0,len(tr)-1,5))))
    if reuse is not None:
        prior_manifest=read(reuse.parent/'manifest.json')['sources']
        assert all(prior_manifest[str(f)]==hashes[str(f)] for f in [data/'states.pth',data/'actions.pth',checkpoint,paths['cfg']])
        prior_rows=read(reuse)
        for r in prior_rows:
            cached[key(r['trace'][0],r['requested_actions'],r['raw_anchor'])]=dict(
                trace=r['trace'],contacts=r['wall_contact_substeps'],clips=r['clipped_boundaries'])
        save(out/'reuse_partial.json',dict(source=str(reuse),sha256=sha(reuse),rows=len(prior_rows),reason='Preserve successful oracle computations from the precision-gate diagnostic attempt'))
    sys.path.insert(0,str(repo));import gym,env,mujoco_py
    cfg=OmegaConf.load(paths['cfg']);e=gym.make(cfg.env.name,*cfg.env.args,**dict(cfg.env.kwargs)).unwrapped;m,s=e.model,e.sim
    assert m.nq==m.nv==2 and m.na==m.nmocap==0 and e.frame_skip==5 and s.nsubsteps==1
    assert m.opt.integrator==0 and m.opt.disableflags==0 and abs(m.opt.timestep-.01)<1e-12
    particle=m.geom_name2id('particle_geom');walls={i for i in range(m.ngeom) if (m.geom_id2name(i) or '').startswith('wall_')}
    assert abs(float(m.geom_margin[particle]+max(m.geom_margin[i] for i in walls))-margin)<1e-12
    results=[];reused=0;new=0;recorded_gate=[]
    for context in selected:
        ep,k=context['episode'],context['raw_anchor'];st=np.array(context['start'])
        for c in conditions+[dict(name='recorded',radius=None,direction=None,control=None,stress=False)]:
            ac=np.array(context['recorded_actions']) if c['name']=='recorded' else np.tile(np.array(c['control'],dtype=np.float32),(5,1)).astype(float)
            kk=key(st,ac,k)
            if kk in cached:
                result=cached[kk];reused+=1;origin='reused_B_or_previous'
            else:
                new+=1;origin='new_action_branch';s.reset();s.set_state(mujoco_py.MjSimState(k*.05,st[:2].copy(),st[2:].copy(),None,{}));s.forward();s.data.qacc_warmstart[:]=0
                tr=[st.tolist()];contacts=0;clips=0
                for u in ac:
                    clips+=int(np.any(np.abs(s.data.qvel)>5));e.clip_velocity();s.data.ctrl[:]=np.clip(u,-1,1)
                    for _ in range(5):
                        s.step();tr.append(np.r_[s.data.qpos.copy(),s.data.qvel.copy()].tolist())
                        contacts+=int(any((int(cc.geom1)==particle and int(cc.geom2) in walls) or (int(cc.geom2)==particle and int(cc.geom1) in walls) for cc in s.data.contact[:s.data.ncon]))
                assert abs(s.data.time-(k+5)*.05)<1e-11
                result=dict(trace=tr,contacts=contacts,clips=clips)
            tr=np.array(result['trace']);clear=float(np.min(clearance(tr[:,:2],params)))
            free=result['contacts']==0 and result['clips']==0 and clear>margin
            theory,_=analytic(st,ac,params)
            eq=float(np.max(np.abs(tr-theory))) if free else None
            if free and eq>=1e-10:
                save(out/'oracle_gate_diagnostic.json',dict(context=context,condition=c,actions=ac.tolist(),
                     trace=tr.tolist(),theory=theory.tolist(),max_abs=eq,provenance=origin,
                     constraint=s.data.qfrc_constraint.tolist(),passive=s.data.qfrc_passive.tolist(),
                     actuator=s.data.qfrc_actuator.tolist(),mass_matrix_packed=s.data.qM.tolist()))
                save(out/'partial_oracles.json',results)
                raise AssertionError(('free_oracle_equation',ep,k,c['name'],eq,origin))
            if c['name']=='recorded':recorded_gate.append(float(np.max(np.abs(tr[-1]-states[ep,k+5]))))
            results.append(dict(split=context['split'],episode=ep,raw_anchor=k,condition=c['name'],
                requested_actions=ac.tolist(),effective_controls=np.clip(ac,-1,1).tolist(),trace=tr.tolist(),endpoint=tr[-1].tolist(),
                wall_contact_substeps=result['contacts'],clipped_boundaries=result['clips'],min_clearance=clear,free=free,
                analytic_max_abs=eq,provenance=origin))
    assert max(recorded_gate)<=1e-4
    e.close();save(out/'oracles.json',results)
    # Training action distribution reference: observations only, not a new fit.
    train=np.concatenate([actions[i,:int(lengths[i])] for i in split['train']])
    changes=np.concatenate([np.linalg.norm(np.diff(actions[i,:int(lengths[i])],axis=0),axis=1) for i in split['train']])
    blocks=np.array([actions[i,k:k+5] for i in split['train'] for k in range(0,int(lengths[i])-5,5)])
    norms=np.linalg.norm(train,axis=1);hist=np.histogram(np.arctan2(train[:,1],train[:,0]),bins=8,range=(-np.pi,np.pi))[0]
    save(out/'action_distribution.json',dict(norm_quantiles=np.quantile(norms,[0,.01,.1,.5,.9,.99,1]).tolist(),
        consecutive_change_quantiles=np.quantile(changes,[0,.01,.1,.5,.9,.99,1]).tolist(),direction_histogram=hist.tolist(),
        constant_5action_fraction=float(np.mean(np.max(np.abs(blocks-blocks[:,:1]),axis=(1,2))<1e-8)),
        intervention_norm_percentiles={str(r):float(np.mean(norms<=r)) for r in [.05,.1,.2,.4,1.]}))
    assert all(sha(f)==v for f,v in hashes.items())
    save(out/'prepare_complete.json',dict(selected_contexts=len(selected),selected_by_split={k:sum(r['split']==k for r in selected) for k in groups},
        new_oracle_branches=new,reused_oracle_branches=reused,recorded_gt_max_abs=max(recorded_gate),
        main_nonfree=sum(not r['free'] for r in results if r['condition'] not in ['recorded'] and not r['condition'].startswith('r1_')),
        source_hashes_unchanged=True))
    print(json.dumps(read(out/'prepare_complete.json')),flush=True)


def infer(root,repo,out):
    import torch
    import hydra
    import joblib
    from omegaconf import OmegaConf
    assert (out/'prepare_complete.json').exists() and not (out/'infer_complete.json').exists()
    paths=sources(root,repo);manifest=read(out/'manifest.json')
    assert all(sha(p)==v for p,v in manifest['sources'].items())
    sys.path.insert(0,str(repo));os.environ['TORCH_HOME']=str(root/'work/torch_home');os.environ['DATASET_DIR']=str(root/'work/data')
    from datasets.img_transforms import default_transform
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.60,0)
    cfg=OmegaConf.load(paths['cfg']);assert cfg.num_hist==3 and cfg.num_pred==1 and cfg.frameskip==5 and cfg.concat_dim==1
    # Use installed pinned hub source and weights only. Network downloads are disabled.
    original_load=torch.hub.load
    hub=root/'work/torch_home/hub/facebookresearch_dinov2_85a24602099d397264d5b30461ad7f3bfd726ca1'
    assert (hub/'hubconf.py').exists()
    def local_hub(repo_or_dir,model,*args,**kwargs):
        if str(repo_or_dir).startswith('facebookresearch/dinov2:'):
            return original_load(str(hub),model,*args,source='local',**kwargs)
        raise RuntimeError('Unexpected hub source; no downloads allowed')
    torch.hub.load=local_hub
    def no_download(*args,**kwargs):raise RuntimeError('A required weight is missing; download disabled')
    torch.hub.download_url_to_file=no_download
    payload=torch.load(paths['cfg'].parent/'checkpoints/model_latest.pth',weights_only=False,map_location='cuda:0')
    encoder=payload.get('encoder')
    if encoder is None:encoder=hydra.utils.instantiate(cfg.encoder)
    wm=hydra.utils.instantiate(cfg.model,encoder=encoder,predictor=payload['predictor'],decoder=None,
         proprio_encoder=payload['proprio_encoder'],action_encoder=payload['action_encoder'],
         proprio_dim=cfg.proprio_emb_dim,action_dim=cfg.action_emb_dim,concat_dim=cfg.concat_dim,
         num_action_repeat=cfg.num_action_repeat,num_proprio_repeat=cfg.num_proprio_repeat).to('cuda:0')
    wm.eval();wm.requires_grad_(False)
    for module in wm.modules():
        if isinstance(getattr(module,'bias',None),torch.Tensor) and 'bias' not in module._parameters and 'bias' not in module._buffers:module.bias=module.bias.to('cuda:0')
    def weights_hash():
        digest=hashlib.sha256()
        for k,v in sorted(wm.state_dict().items()):digest.update(k.encode());digest.update(v.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
    before=weights_hash()
    data=paths['data'];states=torch.load(data/'states.pth',weights_only=False,map_location='cpu').float()
    actions=torch.load(data/'actions.pth',weights_only=False,map_location='cpu').float()
    lengths=torch.as_tensor(torch.load(data/'seq_lengths.pth',weights_only=False))
    allstate=torch.cat([states[i,:int(n)] for i,n in enumerate(lengths)]);allaction=torch.cat([actions[i,:int(n)] for i,n in enumerate(lengths)])
    sm,ss=allstate.mean(0),allstate.std(0);am,ast=allaction.mean(0),allaction.std(0)
    assert cfg.normalize_action
    cache=joblib.load(paths['src']/'feature_cache.joblib');lookup={(r['episode'],r['horizon'],r['mode']):r for r in cache['endpoints']}
    np.testing.assert_array_equal(sm.numpy(),cache['sm']);np.testing.assert_array_equal(ss.numpy(),cache['ss'])
    w=wm.proprio_encoder.patch_embed.weight[:,:,0].detach().cpu().double().numpy();bias=wm.proprio_encoder.patch_embed.bias.detach().cpu().double().numpy()
    assert np.linalg.matrix_rank(w)==4;inv=np.linalg.pinv(w)
    def decode(z):
        z=np.array(z,dtype=float);norm=(z-bias)@inv.T
        return norm*ss.numpy()+sm.numpy(),float(np.linalg.norm(z-(norm@w.T+bias)))
    save(out/'normalization.json',dict(state_mean=sm.tolist(),state_std=ss.tolist(),action_mean=am.tolist(),action_std=ast.tolist(),
         scope='Original full dataset statistics preserved; no new normalization fit',weight=w.tolist(),bias=bias.tolist()))
    contexts=read(out/'selected.json');conditions=read(out/'conditions.json');groups=read(out/'split.json');baseline_checks=[];rows=[];contextmeta=[]
    tokens=out/'token_cache';tokens.mkdir(exist_ok=False)
    transform=default_transform(int(cfg.img_size));source_hashes={}
    def encode_context(ep,k):
        path=data/'obses'/f'episode_{ep:03d}.pth';source_hashes[str(path)]=sha(path)
        images=torch.load(path,weights_only=False,map_location='cpu')[[k-10,k-5,k]]
        images=transform(images.float().permute(0,3,1,2)/255.)
        obs=dict(visual=images[None].to('cuda:0'),proprio=((states[ep,[k-10,k-5,k]]-sm)/ss)[None].to('cuda:0'))
        act=((actions[ep,k-10:k+5]-am)/ast).reshape(1,3,10).to('cuda:0')
        z=wm.encode(obs,act)
        assert not z.requires_grad
        return z,act
    encoded={}
    with torch.inference_mode():
        # Only two original H1 re-inferences for index/normalization reproduction.
        for ep in groups['calibration_fit'][:2]:
            z,act=encode_context(ep,10);encoded[ep,10]=(z,act)
            y=wm.predict(z)[:,-1:];prop=wm.separate_emb(y)[0]['proprio'][0,0].cpu().numpy();state,residual=decode(prop)
            expected=lookup[ep,1,'teacher_forced'];err=float(np.max(np.abs(state-expected['proprio'])))
            assert err<1e-4,('Baseline reproduction failed',ep,err)
            assert np.max(np.abs(expected['proprio']-lookup[ep,1,'open_loop']['proprio']))<1e-10
            baseline_checks.append(dict(episode=ep,raw_anchor=10,max_abs=err,off_manifold=residual))
        for index,context in enumerate(contexts):
            ep,k=context['episode'],context['raw_anchor']
            if (ep,k) in encoded:z,act=encoded.pop((ep,k))
            else:z,act=encode_context(ep,k)
            fixed=z[...,:-wm.action_dim].clone();past=z[:,:2].clone()
            context_hash=hashlib.sha256(fixed.cpu().numpy().tobytes()+past.cpu().numpy().tobytes()).hexdigest()
            torch.save(dict(z=z.cpu(),actions=act.cpu(),context=context),tokens/f'context_{ep}_{k}.pth')
            predicted_tokens={};max_roundtrip=0.;reused_baseline=0
            for c in conditions+[dict(name='recorded',radius=None,direction=None,control=None,stress=False)]:
                raw=np.array(context['recorded_actions'],dtype=np.float32) if c['name']=='recorded' else np.tile(np.array(c['control'],dtype=np.float32),(5,1))
                na=branch_actions(np.array(context['history_actions'],dtype=np.float32),raw,am.numpy(),ast.numpy()).astype(np.float32)
                max_roundtrip=max(max_roundtrip,float(np.max(np.abs(na.reshape(15,2)*ast.numpy()+am.numpy()-np.r_[context['history_actions'],raw]))))
                raw_all=torch.from_numpy(np.concatenate([np.array(context['history_actions'],dtype=np.float32),raw]))
                native_normalized=((raw_all-am)/ast).reshape(1,3,10)
                assert np.max(np.abs(native_normalized.numpy()[0]-na))<1e-6
                na=native_normalized.numpy()[0]
                branch_act=native_normalized.to('cuda:0')
                zz=wm.replace_actions_from_z(z.clone(),branch_act)
                assert torch.equal(zz[...,:-wm.action_dim],fixed) and torch.equal(zz[:,:2],past),'Prebranch context was changed'
                if c['name']=='recorded':assert torch.equal(zz,z),'Recorded action input mismatch'
                target=k//5-1
                prior=lookup.get((ep,target,'teacher_forced')) if c['name']=='recorded' else None
                if prior is not None:
                    state=np.array(prior['proprio']);residual=prior['off_manifold'];spread=None;origin='reused_cached_teacher_forced';reused_baseline+=1
                else:
                    y=wm.predict(zz)[:,-1:];prop=wm.separate_emb(y)[0]['proprio'][0,0].cpu().numpy()
                    state,residual=decode(prop)
                    pp=y[0,0,:, -(wm.proprio_dim+wm.action_dim):-wm.action_dim][:,:len(bias)]
                    spread=float(torch.max(torch.abs(pp-pp[:1])).cpu())
                    predicted_tokens[c['name']]=y.cpu();origin='new_frozen_inference'
                assert np.isfinite(state).all()
                rows.append(dict(split=context['split'],episode=ep,raw_anchor=k,condition=c['name'],
                     prediction=state.tolist(),off_manifold=float(residual),patch_proprio_max_difference=spread,
                     normalized_actions=na.tolist(),requested_actions=context['recorded_actions'] if c['name']=='recorded' else raw.tolist(),
                     model_raw_float32=raw.tolist(),context_sha256=context_hash,
                     provenance=origin))
            torch.save(predicted_tokens,tokens/f'predictions_{ep}_{k}.pth')
            contextmeta.append(dict(episode=ep,raw_anchor=k,context_sha256=context_hash,action_roundtrip_max=max_roundtrip,reused_recorded_baseline=reused_baseline))
            print('INFER_CONTEXT',index+1,len(contexts),ep,k,flush=True)
    assert before==weights_hash() and all(not p.requires_grad for p in wm.parameters())
    assert all(sha(p)==v for p,v in manifest['sources'].items()) and all(sha(p)==v for p,v in source_hashes.items())
    save(out/'predictions.json',rows);save(out/'contexts.json',contextmeta);save(out/'image_sources.json',source_hashes)
    save(out/'infer_complete.json',dict(weights_unchanged=True,weight_tensor_sha256=before,checkpoint_unchanged=True,
         baseline_checks=baseline_checks,context_unchanged=True,modified_action_block=2,past_action_blocks_unchanged=[0,1],
         action_roundtrip_max=max(r['action_roundtrip_max'] for r in contextmeta),future_observations_used=False,
         new_training=False,rows=len(rows),cached_recorded_outputs=sum(r['provenance'].startswith('reused') for r in rows),
         torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','infer'],required=True)
    p.add_argument('--reuse-oracles',type=Path)
    a=p.parse_args();start=time.monotonic()
    try:
        if a.stage=='prepare':prepare(a.root.resolve(),a.repo.resolve(),a.out.resolve(),a.reuse_oracles)
        else:infer(a.root.resolve(),a.repo.resolve(),a.out.resolve())
        save(a.out/f'{a.stage}_timing.json',dict(elapsed_seconds=time.monotonic()-start))
    except BaseException as ex:
        if a.out.exists():save(a.out/f'{a.stage}_failure_{time.time_ns()}.json',dict(reason=repr(ex)))
        traceback.print_exc();raise


if __name__=='__main__':main()
