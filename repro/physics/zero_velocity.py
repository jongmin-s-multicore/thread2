"""Coherent zero-input velocity/history intervention, frozen WM only."""
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
from intervention import sha, save, read, csvwrite, sources, clearance

C_RUN='pointmaze-dynamics-c-20261010-v3'

def conditions():
    vs=[('zero',[0.,0.],0.,-1,0)]
    for r in [.1,.25,.5]:
        for j in range(2):
            for sign in [-1,1]:
                v=[0.,0.];v[j]=sign*r
                vs.append((f'r{r:g}_{j}_{sign}',v,r,j,sign))
    rows=[]
    for history,u in [('coast',0.),('push_positive',.05),('push_negative',-.05)]:
        for name,v,r,j,sign in vs:
            if history!='coast' and r not in [0.,.25]:continue
            rows.append(dict(name=history+'_'+name,history=history,velocity=v,radius=r,axis=j,sign=sign,past_control=[u,0.]))
    return rows


def coherent_trace(q,v,past,params):
    """Indices 0..76 correspond -0.51..0.25s, anchor at51; no clipping."""
    A=np.array(params['Ah']);B=np.array(params['Bh']);dt=params['dt'];st=np.r_[q,v].astype(float)
    back=[st.copy()]
    for _ in range(51):
        st=np.r_[st[:2]-dt*st[2:],np.linalg.solve(A,st[2:]-B@past)];back.append(st)
    tr=list(reversed(back));st=tr[-1]
    for _ in range(25):
        vn=A@st[2:];st=np.r_[st[:2]+dt*vn,vn];tr.append(st)
    return np.array(tr)


def decompose(f0,fp,fm,r,alpha):
    J=np.column_stack([(fp[j]-fm[j])/(2*r) for j in range(2)])
    return J,J-alpha*np.eye(2)


def prepare(root,repo,out):
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False)
    paths=sources(root,repo);c=paths['base']/C_RUN
    assert read(c/'checks.json')['pass_all'] and read(c/'completion.json')['status']=='DYNAMICS_C_COMPLETE_DEVELOPMENT'
    cm=read(c/'manifest.json');assert all(sha(f)==h for f,h in cm['sources'].items())
    previous=read(root/'work/STATUS.json');assert previous['state']=='DYNAMICS_C_COMPLETE_DEVELOPMENT'
    files=[c/n for n in ['manifest.json','checks.json','completion.json','selected.json','parameters.json','predictions.json','oracles.json','normalization.json','infer_complete.json','split.json']]
    contexts=read(c/'selected.json');assert len({x['episode'] for x in contexts})==len(contexts)
    for x in contexts:files.append(c/'token_cache'/f"context_{x['episode']}_{x['raw_anchor']}.pth")
    hashes={str(f):sha(f) for f in files};hashes.update(cm['sources'])
    save(out/'manifest.json',dict(sources=hashes,previous_status=previous,code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),script_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).with_name('zero_velocity_protocol.md')),new_training=False,new_physics_loss=False,parent_run=C_RUN))
    params=read(c/'parameters.json');save(out/'parameters.json',params);save(out/'normalization.json',read(c/'normalization.json'))
    cs=conditions();save(out/'conditions.json',cs);selected=[];audit=[]
    states=torch.load(paths['data']/'states.pth',weights_only=False,map_location='cpu').float().numpy()
    candidate_groups=read(c/'split.json')
    candidates=[dict(episode=ep,split=g,raw_anchor=k,start=states[ep,k].tolist()) for g,eps in candidate_groups.items() for ep in eps for k in range(10,71,5)]
    found=set()
    for x in candidates:
        if x['episode'] in found:continue
        ts=[coherent_trace(x['start'][:2],y['velocity'],np.array(y['past_control']),params) for y in cs]
        margin=min(float(np.min(clearance(t[:,:2],params))) for t in ts)
        vmax=max(float(np.max(np.abs(t[:,2:]))) for t in ts)
        ok=margin>.004 and vmax<5
        audit.append(dict(episode=x['episode'],raw_anchor=x['raw_anchor'],split=x['split'],min_margin=margin,max_abs_velocity=vmax,selected=ok))
        if ok:selected.append(x);found.add(x['episode'])
    save(out/'selection_audit.json',audit);save(out/'selected.json',selected)
    assert selected and any(x['split']=='development' for x in selected),'No common development sites'
    sys.path.insert(0,str(repo));import gym,env,mujoco_py
    cfg=OmegaConf.load(paths['cfg']);e=gym.make(cfg.env.name,*cfg.env.args,**dict(cfg.env.kwargs)).unwrapped
    renderer=gym.make(cfg.env.name,*cfg.env.args,**dict(cfg.env.kwargs)).unwrapped
    assert not e.with_target and not renderer.with_target
    renderer.prepare_for_render()
    def render(st):
        renderer.sim.reset();renderer.sim.set_state(mujoco_py.MjSimState(0.,np.array(st[:2]),np.array(st[2:]),None,{}));renderer.sim.forward();renderer.set_marker()
        return renderer._render_frame().copy()
    # Match the original lagged image timing; checks are limited to calibration.
    states=torch.load(paths['data']/'states.pth',weights_only=False).numpy();actions=torch.load(paths['data']/'actions.pth',weights_only=False).numpy()
    render_checks=[]
    for x in [z for z in contexts if z['split']=='calibration_fit'][:2]:
        ep,k=x['episode'],x['raw_anchor'];st=states[ep,k].astype(float)
        imst=np.r_[st[:2]-params['dt']*st[2:],np.linalg.solve(np.array(params['Ah']),st[2:]-np.array(params['Bh'])@actions[ep,k-1])]
        f=paths['data']/'obses'/f'episode_{ep:03d}.pth';old=torch.load(f,weights_only=False)[k].numpy();new=render(imst)
        mad=float(np.mean(np.abs(old.astype(float)-new.astype(float))))
        render_checks.append(dict(episode=ep,raw_anchor=k,mae_rgb_255=mad,source_sha256=sha(f)))
        from PIL import Image
        Image.fromarray(np.concatenate([old,new],axis=1)).save(out/f'render_check_{ep}.png')
    save(out/'render_checks.json',render_checks)
    assert max(r['mae_rgb_255'] for r in render_checks)<=3,'Rendering mismatch'
    (out/'images').mkdir();rows=[];particle=e.model.geom_name2id('particle_geom');walls={i for i in range(e.model.ngeom) if (e.model.geom_id2name(i) or '').startswith('wall_')}
    assert e.model.nq==e.model.nv==2 and e.model.na==e.model.nmocap==0 and e.frame_skip==5 and e.sim.nsubsteps==1
    assert e.model.opt.integrator==0 and e.model.opt.disableflags==0 and abs(e.model.opt.timestep-params['dt'])<1e-12
    M=np.array(params['M']);D=np.array(params['D']);energymax=0.
    for i,x in enumerate(selected):
        for cc in cs:
            tr=coherent_trace(x['start'][:2],cc['velocity'],np.array(cc['past_control']),params)
            s=e.sim;s.reset();s.set_state(mujoco_py.MjSimState(-.01,tr[0,:2].copy(),tr[0,2:].copy(),None,{}));s.forward();s.data.qacc_warmstart[:]=0.
            actual=[tr[0].tolist()];contacts=0;clips=0
            for n in range(76):
                if n>=1 and (n-1)%5==0:
                    clips+=int(np.any(np.abs(s.data.qvel)>5));e.clip_velocity()
                s.data.ctrl[:]=cc['past_control'] if n<51 else 0.
                s.step();actual.append(np.r_[s.data.qpos.copy(),s.data.qvel.copy()].tolist())
                contacts+=sum((int(k.geom1)==particle and int(k.geom2) in walls) or (int(k.geom2)==particle and int(k.geom1) in walls) for k in s.data.contact[:s.data.ncon])
            actual=np.array(actual);err=float(np.max(np.abs(actual-tr)));assert err<1e-10 and contacts==clips==0
            assert abs(s.data.time-.75)<1e-12
            for v0,v1 in zip(actual[51:-1,2:],actual[52:,2:]):
                residual=.5*(v1@M@v1-v0@M@v0)+params['dt']*(v1@D@v1)+.5*((v1-v0)@M@(v1-v0));energymax=max(energymax,abs(float(residual)))
            images=np.stack([render(actual[n]) for n in [0,25,50]])
            filename=f"{x['episode']}_{cc['name']}.npz";np.savez_compressed(out/'images'/filename,images=images)
            rows.append(dict(episode=x['episode'],split=x['split'],raw_anchor=x['raw_anchor'],**cc,
                start=actual[51].tolist(),endpoint=actual[-1].tolist(),context_states=actual[[1,26,51]].tolist(),image_states=actual[[0,25,50]].tolist(),trace=actual.tolist(),
                min_clearance=float(np.min(clearance(actual[:,:2],params))),wall_contact_substeps=int(contacts),clips=int(clips),analytic_max_abs=err,image_file='images/'+filename,image_sha256=sha(out/'images'/filename)))
        print('PREPARE_SITE',i+1,len(selected),x['episode'],flush=True)
    e.close();renderer.close();assert energymax<1e-10
    save(out/'branches.json',rows)
    # Existing original-history baselines are preserved for reference, not rerun.
    oldp={(r['episode'],r['raw_anchor']):r for r in read(c/'predictions.json') if r['condition']=='zero'}
    oldo={(r['episode'],r['raw_anchor']):r for r in read(c/'oracles.json') if r['condition']=='zero'}
    save(out/'original_c_zero.json',[dict(context=x,prediction=oldp[x['episode'],x['raw_anchor']],oracle=oldo[x['episode'],x['raw_anchor']]) for x in contexts])
    assert all(sha(f)==h for f,h in hashes.items())
    save(out/'prepare_complete.json',dict(sites=len(selected),by_split={g:sum(x['split']==g for x in selected) for g in ['calibration_fit','calibration_tune','development']},branches=len(rows),energy_max_abs=energymax,oracle_max_abs=max(r['analytic_max_abs'] for r in rows),reused_c_zero=len(contexts),source_hashes_unchanged=True))


def infer(root,repo,out):
    import torch,hydra
    from omegaconf import OmegaConf
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert (out/'prepare_complete.json').exists();manifest=read(out/'manifest.json');assert all(sha(f)==h for f,h in manifest['sources'].items())
    paths=sources(root,repo);c=paths['base']/C_RUN
    sys.path.insert(0,str(repo));os.environ['TORCH_HOME']=str(root/'work/torch_home')
    from datasets.img_transforms import default_transform
    assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.6,0)
    original_load=torch.hub.load;hub=root/'work/torch_home/hub/facebookresearch_dinov2_85a24602099d397264d5b30461ad7f3bfd726ca1'
    def local_hub(repo_or_dir,model,*args,**kwargs):
        assert str(repo_or_dir).startswith('facebookresearch/dinov2:')
        return original_load(str(hub),model,*args,source='local',**kwargs)
    torch.hub.load=local_hub
    def no_download(*args,**kwargs):raise RuntimeError('No downloads')
    torch.hub.download_url_to_file=no_download
    cfg=OmegaConf.load(paths['cfg']);payload=torch.load(paths['cfg'].parent/'checkpoints/model_latest.pth',weights_only=False,map_location='cuda:0')
    encoder=payload.get('encoder')
    if encoder is None:encoder=hydra.utils.instantiate(cfg.encoder)
    wm=hydra.utils.instantiate(cfg.model,encoder=encoder,predictor=payload['predictor'],decoder=None,proprio_encoder=payload['proprio_encoder'],action_encoder=payload['action_encoder'],proprio_dim=cfg.proprio_emb_dim,action_dim=cfg.action_emb_dim,concat_dim=cfg.concat_dim,num_action_repeat=cfg.num_action_repeat,num_proprio_repeat=cfg.num_proprio_repeat).to('cuda:0')
    wm.eval();wm.requires_grad_(False)
    for module in wm.modules():
        if isinstance(getattr(module,'bias',None),torch.Tensor) and 'bias' not in module._parameters and 'bias' not in module._buffers:module.bias=module.bias.to('cuda:0')
    def whash():
        h=hashlib.sha256()
        for k,v in sorted(wm.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=whash();assert before==read(c/'infer_complete.json')['weight_tensor_sha256']
    norm=read(out/'normalization.json');sm=torch.tensor(norm['state_mean']);ss=torch.tensor(norm['state_std']);am=torch.tensor(norm['action_mean']);ast=torch.tensor(norm['action_std'])
    w=np.array(norm['weight']);bias=np.array(norm['bias']);inv=np.linalg.pinv(w)
    def decode(z):
        z=np.asarray(z,dtype=float);n=(z-bias)@inv.T
        return n*ss.numpy()+sm.numpy(),float(np.linalg.norm(z-(n@w.T+bias)))
    checks=[];rows=[];roundtrip=0.;tokens=out/'token_cache';tokens.mkdir(exist_ok=False)
    transform=default_transform(int(cfg.img_size))
    with torch.inference_mode():
        if manifest.get('baseline_reuse_source'):
            checks=read(Path(manifest['baseline_reuse_source']))['baseline_checks']
        for x in ([] if manifest.get('baseline_reuse_source') else [z for z in read(c/'selected.json') if z['split']=='calibration_fit'][:2]):
            cache=torch.load(c/'token_cache'/f"context_{x['episode']}_{x['raw_anchor']}.pth",weights_only=False)
            z=cache['z'].to('cuda:0');act=cache['actions'].to('cuda:0');act[:,2]=((-am)/ast).repeat(5).to('cuda:0')
            y=wm.predict(wm.replace_actions_from_z(z,act))[:,-1:];pred,_=decode(wm.separate_emb(y)[0]['proprio'][0,0].cpu().numpy())
            old=next(r['prediction']['prediction'] for r in read(out/'original_c_zero.json') if r['context']['episode']==x['episode'])
            err=float(np.max(np.abs(pred-old)));assert err<=1e-4;checks.append(dict(episode=x['episode'],max_abs=err))
        for i,r in enumerate(read(out/'branches.json')):
            assert sha(out/r['image_file'])==r['image_sha256'];images=torch.from_numpy(np.load(out/r['image_file'])['images'])
            prop=torch.tensor(r['context_states'],dtype=torch.float32);normalized=(prop-sm)/ss
            roundtrip=max(roundtrip,float(torch.max(torch.abs(normalized*ss+sm-prop))))
            raw=torch.tensor(r['raw_actions'] if 'raw_actions' in r else [r['past_control']]*10+[[0.,0.]]*5,dtype=torch.float32);acts=((raw-am)/ast).reshape(1,3,10).to('cuda:0')
            obs=dict(visual=transform(images.float().permute(0,3,1,2)/255.)[None].to('cuda:0'),proprio=normalized[None].to('cuda:0'))
            z=wm.encode(obs,acts);encoded=wm.separate_emb(z)[0]['proprio'][0].cpu().numpy()
            for j in range(3):
                decoded,_=decode(encoded[j]);roundtrip=max(roundtrip,float(np.max(np.abs(decoded-prop[j].numpy()))))
            y=wm.predict(z)[:,-1:];p,residual=decode(wm.separate_emb(y)[0]['proprio'][0,0].cpu().numpy());assert np.isfinite(p).all()
            filename=f"{r['episode']}_{r['name']}.pth";torch.save(dict(context=z.cpu(),prediction=y.cpu()),tokens/filename)
            rows.append(dict(episode=r['episode'],split=r['split'],name=r['name'],prediction=p.tolist(),off_manifold=residual,normalized_actions=acts.cpu().numpy().tolist(),context_sha256=hashlib.sha256(z.cpu().numpy().tobytes()).hexdigest()))
            if i%23==22:print('INFER_SITE',(i+1)//23,flush=True)
    assert roundtrip<1e-6 and before==whash() and all(not p.requires_grad for p in wm.parameters())
    assert all(sha(f)==h for f,h in manifest['sources'].items())
    save(out/'predictions.json',rows);save(out/'infer_complete.json',dict(baseline_checks=checks,weights_unchanged=True,weight_tensor_sha256=before,proprio_roundtrip_max=roundtrip,new_training=False,rows=len(rows),future_observations_used=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','infer','analyze'],required=True);p.add_argument('--render-audit',action='store_true');a=p.parse_args();t=time.monotonic()
    try:
        if a.render_audit and a.stage!='infer':
            from zero_render_audit import prepare_audit,analyze_audit
            (prepare_audit if a.stage=='prepare' else analyze_audit)(a.root.resolve(),a.repo.resolve(),a.out.resolve())
        elif a.stage=='analyze':
            from analyze_zero_velocity import analyze
            analyze(a.root.resolve(),a.repo.resolve(),a.out.resolve())
        else:globals()[a.stage](a.root.resolve(),a.repo.resolve(),a.out.resolve())
        save(a.out/f'{a.stage}_timing.json',dict(elapsed_seconds=time.monotonic()-t))
    except BaseException as ex:
        if a.out.exists():save(a.out/f'{a.stage}_failure_{time.time_ns()}.json',dict(reason=repr(ex)))
        traceback.print_exc();raise
if __name__=='__main__':main()
