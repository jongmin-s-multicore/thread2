"""훈련 whitelist에 한정한 v2.0 작은 검증. 비교 학습 entrypoint와 분리."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
import traceback
# [repro] 서버·실제 argv 관문은 모델 import보다 먼저 실행한다.
if __name__!='__main__':
    raise RuntimeError('unsupported import: approved CLI required before model/data access')
if __name__=='__main__':
    from natural_v2_guard import production_entry_guard
    import sys
    production_entry_guard(sys.argv)
import numpy as np
from intervention import read,save,sha,sources
from natural_v2_core import require,anchors,transition,parts,flatgrad,objective,CELLS,schedule,evaluation_mode
from natural_v2_control import verify_lock,git
# [repro] metadata 감사 범위와 update 사건을 수치 검사와 분리한다.
from natural_v2_guard import (read_document, verify_documents, audit_manifests, recorded_update,
    update_counts, verify_job_authorization)
from natural_v2_data import WhitelistData,natural_window,alignment_pass,replay,numpy_transition


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False,default=serial)


def check(rows,name,value,limit=None,ok=None):
    passed=bool(value<limit) if ok is None else bool(ok)
    rows.append(dict(name=name,actual=value,tolerance=limit,pass_test=passed))
    require(passed,name)


def selection_tests():
    from natural_v2_selection import G,REST,STAGE1,select_stage1,select_base,select_aux
    a=dict(complete=True,verified=True,cohort={'fixture':[1,2]},metrics={k:1. for k in set(G+REST)})
    cases={c:copy.deepcopy(a) for c in CELLS}
    require(select_stage1(a,cases)['cell']=='I-N','stage1 tie')
    rows=[dict(copy.deepcopy(a),mixture=m,step=s) for m in ('R06','R12') for s in (120,240,480)]
    b=select_base(a,rows);require(b['mixture']=='R06' and b['step']==120,'base tie')
    arms=[dict(copy.deepcopy(b),strength=c) for c in (.5,1.)]
    for r in arms:
        r['metrics'][REST[0]]=.8;r['metrics'][REST[1]]=.8
    pp=select_aux(a,b,arms,'P');require(pp['incremental_gate'] and pp['strength']==.5,'P gate/tie')
    bad=copy.deepcopy(rows)
    for r in bad:r['metrics'][G[0]]=1.051
    require(select_base(a,bad)['status']=='NO_PRESERVING_BASELINE','B failure')
    fallback=copy.deepcopy(arms)
    for r in fallback:r['metrics'][G[0]]=1.2
    require(not select_aux(a,b,fallback,'S')['preserving'],'S fallback')
    negatives=[]
    for kind in ('empty','nonfinite','technical','checkpoint','missing_cell'):
        try:
            if kind=='checkpoint':
                cc=copy.deepcopy(arms);cc[0]['step']=240;select_aux(a,b,cc,'P')
            else:
                cc=copy.deepcopy(cases)
                if kind=='empty':cc['K-F']['cohort']={}
                if kind=='nonfinite':cc['K-F']['metrics'][STAGE1[0]]=float('nan')
                if kind=='technical':cc['K-F']['complete']=False
                if kind=='missing_cell':del cc['K-F']
                select_stage1(a,cc)
        except RuntimeError:negatives.append(kind)
    require(len(negatives)==5,'fail-closed selection')
    return dict(pass_all=True,fixtures=['stage1_tie','B_tie','B_gate','P_gate','S_fallback']+negatives,real_metrics_used=False)


def metadata(root,repo,out):
    import torch
    verify_documents(out)
    split=read_document(out,'physics_natural_split_v2_0.json');paths=sources(root,repo)
    original=paths['src']/'split_manifest.json'
    require(sha(original)==split['source_split_manifest_sha256'],'source split hash')
    old=read(original)
    ids=split['stage2_natural_train_episodes']+split['stage2_tune_episodes']+split['stage2_fresh_evaluation_episodes']
    require(len(ids)==len(set(ids))==64,'split disjoint sizes')
    perm=torch.randperm(2000,generator=torch.Generator().manual_seed(42)).tolist()
    require(set(perm[1800:])==set(old['sealed_wm_test']),'WM membership reconstruction')
    require(set(ids)<=set(perm[:1800]),'WM train membership')
    targets=set(split['stage2_tune_episodes']+split['stage2_fresh_evaluation_episodes'])
    manifest_paths=[p for p in sorted(paths['base'].rglob('*.json'))
        if not any(part.startswith('pointmaze-physics-natural-v2-') for part in p.parts)
        and not p.name.startswith('._')
        and any(k in p.name for k in ('manifest','selection','selected','split'))]
    audit=audit_manifests(manifest_paths,targets,excluded=[
        '실제 관측/cases/branches/tensor는 이 metadata 감사 대상 아님',
        '명시적 episode metadata가 없는 cache 및 서버 외부 이력은 미확인'])
    write(out/'historical_usage_audit.json',audit)
    scanned={r['path']:r['sha256'] for r in audit['scanned']};hits=audit['overlap']
    require(not hits,'prior research overlap: '+json.dumps(hits))
    lengths=torch.load(paths['data']/'seq_lengths.pth',map_location='cpu',weights_only=False)
    planned={str(ep):anchors(int(lengths[ep])) for ep in split['stage2_natural_train_episodes']}
    require(all((paths['data']/'obses'/f'episode_{ep:03d}.pth').is_file() for ep in ids),'image files absent')
    prior=root/'work/physics/pointmaze-physics-loss-pilot-20261011'
    attribution=root/'work/physics/pointmaze-physics-attribution-20261011'
    hashes={};reuse_checks={}
    for parent,arm in ((prior,'D'),(attribution,'DM')):
        complete=read(parent/arm/'complete.json');cp=parent/arm/'checkpoint.pth'
        require(sha(cp)==complete['checkpoint_sha256'],'reuse checkpoint hash')
        require(read(parent/arm/'exposure.json')==read(prior/'B/exposure.json'),'reuse exposure')
        for p in (cp,parent/arm/'exposure.json',parent/f'evaluation_{arm}.json',parent/'manifest.json'):
            hashes[str(p)]=sha(p)
        reuse_checks[arm]=dict(checkpoint=str(cp),sha256=sha(cp),steps=complete['steps'])
    for rel,h in read(prior/'encoded_and_checkpoint_hashes.json').items():
        require(sha(prior/rel)==h,'cache changed '+rel);hashes[str(prior/rel)]=h
    evidence=read_document(out,'physics_natural_evidence_v2_0.json')
    for rel,h in evidence['sha256_relative_to_mac_root'].items():
        if rel.startswith('repo/'):
            path=repo/rel[5:];require(sha(path)==h,'baseline source hash '+rel);hashes[str(path)]=h
    for name in ('states.pth','actions.pth','seq_lengths.pth'):
        p=paths['data']/name;hashes[str(p)]=sha(p)
        require(hashes[str(p)]==read(prior/'manifest.json')['sources'][str(p)],'raw source changed')
    checkpoint=paths['cfg'].parent/'checkpoints/model_latest.pth'
    require(sha(checkpoint)=='78c654baede0e9289b751e9ad251839b8045a6074cc1ec23ab54caaf11a25700','A checkpoint changed')
    hashes[str(checkpoint)]=sha(checkpoint)
    data=dict(split=split,wm_membership_verified=True,provenance_files=scanned,overlap=hits,
              provenance_scope=audit['scope'],provenance_unknown=audit['unknown'],provenance_excluded=audit['excluded'],
              train_anchors=planned,evaluation_content_opened=False,tune_content_opened=False,
              source_hashes=hashes)
    write(out/'data_manifest.json',data)
    write(out/'reuse_audit.json',dict(checkpoint_and_cache_hashes_pass=True,exposure_identical=True,cells={'K-F':reuse_checks['D'],'K-N':reuse_checks['DM']},loss_gradient_equivalence='PENDING_GPU'))
    return split,prior


def cpu(root,repo,out,job):
    import torch
    from physics_depth import make_env
    torch.set_num_threads(2);rows=[]
    write(out/'selection_tests.json',selection_tests())
    split,prior=metadata(root,repo,out)
    params=read(prior/'parameters.json')
    loader=WhitelistData(sources(root,repo)['data'],split['stage2_natural_train_episodes'])
    try:loader.row('states',split['stage2_fresh_evaluation_episodes'][0])
    except RuntimeError:check(rows,'whitelist_reject_eval',True,ok=True)
    else:raise RuntimeError('whitelist bypass')
    check(rows,'anchor_length100',anchors(100),ok=anchors(100)==[10,31,52,74])
    # 합성 fixture로 전체 480 step의 고정 schedule을 검증한다.
    fixture=[dict(id=f'{f}{i}',schedule_family=f) for f in 'NILS' for i in range(3)]
    for m in ('R06','R12'):
        s=schedule(fixture,m);check(rows,'schedule_'+m,len(s),ok=len(s)==480 and s==schedule(fixture,m))
    state=np.array([1.5,1.5,.1,-.15]);u=np.array([[.1,-.2],[-.3,.4],[.2,.15],[-.1,-.25],[.4,.05]])
    ts=torch.tensor(state,dtype=torch.float64,requires_grad=True);tu=torch.tensor(u,dtype=torch.float64,requires_grad=True)
    actual=transition(ts,tu,params);expect=numpy_transition(state,u,params)[-1]
    check(rows,'variable_action_numpy_double',float(np.max(np.abs(actual.detach().numpy()-expect))),1e-10)
    grad=torch.autograd.grad(actual.square().sum(),(ts,tu));maxerr=0.
    for which,array in enumerate((state,u)):
        for index in np.ndindex(array.shape):
            aa=state.copy();bb=u.copy();cc=state.copy();dd=u.copy();plus=(aa,bb)[which];minus=(cc,dd)[which]
            plus[index]+=1e-6;minus[index]-=1e-6
            fd=(np.square(numpy_transition(aa,bb,params)[-1]).sum()-np.square(numpy_transition(cc,dd,params)[-1]).sum())/2e-6
            err=abs(fd-float(grad[which][index]));maxerr=max(maxerr,err)
            require(err<=1e-5+1e-3*abs(fd),'finite difference gradient')
    check(rows,'finite_difference_max_abs',maxerr,1e-5)
    import physics_loss_pilot as pilot
    cu=torch.tensor([.1,-.2],dtype=torch.float64)
    check(rows,'constant_action_regression',float((transition(ts,cu.expand(5,2),params)-pilot.transition(ts,cu,params)).abs().max()),1e-12)
    env=make_env(root,repo);cases=[];alignment=[]
    try:
        for ep in split['stage2_natural_train_episodes'][:2]:
            k=anchors(int(loader.row('seq_lengths',ep)))[0]
            c=natural_window(loader,ep,k,env,params,images=True);cases.append(c)
            alignment.append({key:value for key,value in c.items() if key not in ('images',)})
        # 기존 synthetic train family마다 한 branch의 5 구간을 다시 확인한다.
        pc=read(prior/'cases.json');oracle=read(prior/'oracles.json');synthetic=[]
        for family in ('static','low_speed','recorded'):
            c=next(c for c in pc if c['split']=='calibration_fit' and c['family']==family)
            for h,old in enumerate(oracle[c['id']]):
                seg=replay(env,old['trace'][0],np.tile(c['u'],(5,1)),params)
                err=float(np.max(np.abs(np.asarray(seg['trace'])-old['trace'])))
                synthetic.append(dict(id=c['id'],family=family,horizon=h+1,max_abs=err,free_same=seg['free']==old['free']))
                check(rows,f'synthetic_{family}_h{h+1}',err,1e-10)
                require(seg['free']==old['free'],'synthetic mask changed')
        var=replay(env,cases[0]['start'],u,params)
        check(rows,'nonconstant_mujoco_finite',bool(np.isfinite(var['trace']).all()),ok=bool(np.isfinite(var['trace']).all()))
    finally:env.close()
    write(out/'alignment_checks.json',dict(natural=alignment,synthetic=synthetic,nonconstant=var,access_log=loader.access,
        global_load_note=loader.global_load_note,tune_evaluation_content_opened=False,
        pass_all=all(alignment_pass(c) for c in cases)))
    # 정렬 실패는 결과를 기록한 뒤 종료한다. 표본 교체/제외는 하지 않는다.
    require(all(alignment_pass(c) for c in cases),'natural replay/target gate failed; report to design, do not replace samples')
    for c in cases:
        with (out/(c['id']+'.npz')).open('xb') as f:
            np.savez_compressed(f,**{k:c[k] for k in ('images','context_states','truth','start','controls','context_actions','future_actions','mask')})
    write(out/'preflight_natural_cases.json',[{k:c[k] for k in ('id','episode','family','schedule_family','condition','raw_anchor','frames')} for c in cases])
    write(out/'cpu_tests.json',dict(pass_all=True,checks=rows,optimizer_updates=0))


def unit(root,repo,out,job):
    from test_natural_v2 import run
    write(out/'unit_tests.json',run())
    from natural_v2_control import approval
    try:approval(repo,out,None,'stage1')
    except RuntimeError:write(out/'approval_gate_test.json',dict(pass_all=True,comparison_rejected_without_approval=True))
    else:raise RuntimeError('comparison approval bypass')
    split=read_document(out,'physics_natural_split_v2_0.json')
    prior=root/'work/physics/pointmaze-physics-loss-pilot-20261011'
    plans=[dict(id=f'N_{ep}_{k}',schedule_family='N') for ep in split['stage2_natural_train_episodes'] for k in anchors(100)]
    plans += [dict(id=c['id'],schedule_family={'static':'S','low_speed':'L','recorded':'I'}[c['family']]) for c in read(prior/'cases.json') if c['split']=='calibration_fit']
    write(out/'schedule_manifest.json',dict(cases=plans,exposure={m:schedule(plans,m) for m in ('R06','R12')},
        effective_batch=2,seed=0,steps=480,same_B_P_S_exposure=True))


def gpu(root,repo,out,job):
    import torch
    import hashlib
    import physics_loss_pilot as pilot
    import physics_attribution as old
    from natural_v2_data import encode_case
    from frozen_runtime import weight_hash
    require(read(out/'cpu_tests.json')['pass_all'], 'CPU gate not passed')
    prior=root/'work/physics/pointmaze-physics-loss-pilot-20261011'
    params=read(prior/'parameters.json');rows=[];details={};times=[]
    wm,cfg,n=pilot.runtime(root,repo,prior)
    original=pilot.initial(wm);before=pilot.frozen_hash(wm)
    def extra_frozen_hash():
        h=hashlib.sha256()
        for k,v in sorted(n.items()):h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
        for name,m in wm.named_modules():
            v=getattr(m,'bias',None)
            if isinstance(v,torch.Tensor) and 'bias' not in m._parameters and 'bias' not in m._buffers:
                h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    extra=extra_frozen_hash();initial_hash=weight_hash(wm)
    check(rows,'predictor_parameters',sum(p.numel() for p in wm.predictor.parameters()),ok=sum(p.numel() for p in wm.predictor.parameters())==20122600)
    check(rows,'only_predictor_trainable',True,ok=all(name.startswith('predictor.') for name,p in wm.named_parameters() if p.requires_grad))
    cases=[c for c in read(prior/'cases.json') if c['split']=='calibration_fit']
    batches=pilot.schedule(cases,120)
    check(rows,'legacy_exposure',True,ok=[[c['id'] for c in cs] for cs in batches]==read(prior/'B/exposure.json'))
    b=pilot.batch(prior,batches[0]);wm.predictor.train()
    def reset():wm.predictor.load_state_dict(original);torch.manual_seed(0);wm.zero_grad(set_to_none=True)
    for cell,(stop,mode) in CELLS.items():
        reset();ll,st,paths=parts(wm,b,n,params,stop_input=stop,paths=True)
        loss,scale=objective(ll,wm,b['mask'],'P',mode)
        gradient=flatgrad(loss,wm)
        # 입력 경로와 target 경로의 VJP를 별도로 검사한다.
        gi=torch.autograd.grad(paths['s2'].sum(),paths['y1'],retain_graph=True,allow_unused=True)[0]
        gt=torch.autograd.grad(paths['t2'].square().sum(),paths['y1'],retain_graph=True)[0]
        details[cell]=dict(loss=float(loss.detach()),physics=float(ll['physics'].detach()),base=float(ll['base'].detach()),
            input_vjp_norm=0. if gi is None else float(gi.norm()),target_vjp_norm=float(gt.norm()),scale=scale)
        if not stop:
            ref_target=gt.detach().clone();ref_value=float(loss.detach())
        else:
            check(rows,cell+'_input_vjp_zero',details[cell]['input_vjp_norm'],ok=details[cell]['input_vjp_norm']==0)
            check(rows,cell+'_target_vjp_nonzero',details[cell]['target_vjp_norm'],ok=details[cell]['target_vjp_norm']>0)
            check(rows,cell+'_target_vjp_unchanged',float((gt-ref_target).abs().max()),1e-7)
        if mode=='norm':check(rows,cell+'_ratio_error',abs(scale['ratio']-.1),1e-5)
        del gi,gt,paths
        if not stop:
            saved_loss=float(loss.detach());saved_grad=gradient.detach();del ll,st,loss,gradient
            reset()
            if mode=='fixed':
                legacy,_=pilot.loss_parts(wm,b,n,params);legacy_loss=legacy['base']+.14603252771223255*legacy['physics']
            else:
                legacy,_=old.parts(wm,b,n,params);gb=old.flatgrad(legacy['base'],wm);ga=old.flatgrad(legacy['D'],wm)
                legacy_loss=legacy['base']+float(.1*gb.norm()/ga.norm().clamp(min=1e-12))*legacy['D'];del gb,ga
            gg=flatgrad(legacy_loss,wm)
            check(rows,cell+'_legacy_loss_abs',abs(saved_loss-float(legacy_loss.detach())),1e-7)
            check(rows,cell+'_legacy_gradient_relative',float((gg-saved_grad).norm()/gg.norm().clamp(min=1e-12)),1e-5)
            del legacy,legacy_loss,gg,saved_grad
        else:del ll,st,loss,gradient
    for mode in ('F','N'):
        # norm coefficient는 gradient 경로 변경에 따라 달라질 수 있으므로 raw loss만 비교.
        check(rows,'keep_stop_physics_forward_'+mode,abs(details['K-'+mode]['physics']-details['I-'+mode]['physics']),1e-7)
    reset();ll,_=parts(wm,b,n,params);g=flatgrad(ll['base'],wm)
    zero,info=objective(ll,wm,b['mask'],'P','norm',0.)
    check(rows,'zero_coefficient_gradient',float((flatgrad(zero,wm)-g).abs().max()),1e-7)
    del ll,g,zero
    bb=dict(b,mask=torch.zeros_like(b['mask']));reset();ll,_=parts(wm,bb,n,params)
    zero,info=objective(ll,wm,bb['mask'],'P','norm')
    check(rows,'zero_mask_skipped',info,ok=info['lambda']==info['aux_norm']==0)
    del ll,zero
    reset();ll,_=parts(wm,b,n,params)
    for name,factor in [('degenerate',0.),('excessive_lambda',1e-10)]:
        scaled=dict(ll,physics=ll['physics']*factor)
        try:objective(scaled,wm,b['mask'],'P','norm')
        except RuntimeError:check(rows,'reject_'+name,True,ok=True)
        else:raise RuntimeError('invalid norm not rejected')
    del ll
    reset();wm.eval();encoded=[];encoding_times=[]
    for c in read(out/'preflight_natural_cases.json'):
        data=dict(c,**dict(np.load(out/(c['id']+'.npz'))))
        torch.cuda.synchronize();t=time.monotonic();encoded.append(encode_case(wm,cfg,n,data));torch.cuda.synchronize();encoding_times.append(time.monotonic()-t)
    bn={k:torch.cat([x[k] for x in encoded],0) for k in encoded[0]}
    # H5 padding action은 visual/proprio target에 영향을 주지 않아야 한다.
    with torch.no_grad():
        zz=encoded[0]['future'][:,-1:];replaced=wm.replace_actions_from_z(zz.clone(),torch.ones((1,1,10),device='cuda'))
        check(rows,'H5_padding_observation_invariance',float((zz[...,:-wm.action_dim]-replaced[...,:-wm.action_dim]).abs().max()),1e-7)
    wm.predictor.train()
    # 여섯 disposable update. 각 설정은 원본에서 시작하며 checkpoint 저장 안 함.
    configs=[(name,b,stop,mode) for name,(stop,mode) in CELLS.items()]+[('natural_fixed',bn,True,'fixed'),('natural_norm',bn,True,'norm')]
    for i,(name,bb,stop,mode) in enumerate(configs):
        reset();opt=pilot.optimize(wm);wm.predictor.train();torch.cuda.synchronize();t=time.monotonic()
        ll,st=parts(wm,bb,n,params,stop_input=stop);loss,scale=objective(ll,wm,bb['mask'],'P',mode)
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(wm.predictor.parameters(),1.,error_if_nonfinite=True)
        used=read(job/'budget_at_launch.json')['ledger']['optimizer_updates']+update_counts(job)['attempted']
        require(used<12,'optimizer budget')
        recorded_update(job,name,i+1,opt.step); torch.cuda.synchronize();elapsed=time.monotonic()-t
        delta=max(float((v.detach().cpu()-original[k]).abs().max()) for k,v in wm.predictor.state_dict().items())
        check(rows,name+'_updated',delta,ok=delta>0)
        check(rows,name+'_frozen',True,ok=pilot.frozen_hash(wm)==before and extra_frozen_hash()==extra)
        times.append(dict(name=name,seconds=elapsed,gradient_norm=float(gn),clipped=bool(gn>1),scale=scale))
        del ll,st,loss,opt
    # 원 checkpoint에서 predictor를 다시 로딩한 뒤 초기 tensor hash를 확인한다.
    cp=sources(root,repo)['cfg'].parent/'checkpoints/model_latest.pth'
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    wm.predictor.load_state_dict(payload['predictor'].state_dict());del payload
    check(rows,'reloaded_original_tensor_hash',True,ok=weight_hash(wm)==initial_hash)
    from natural_v2_metrics import rollout
    cpu_rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone();mode=wm.predictor.training
    torch.cuda.synchronize();t=time.monotonic();rollout(wm,encoded[0],n);torch.cuda.synchronize();rollout_time=time.monotonic()-t
    check(rows,'evaluation_rng_and_mode_restored',True,ok=torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(cuda_rng,torch.cuda.get_rng_state()) and mode==wm.predictor.training)
    write(out/'gradient_paths.json',dict(pass_all=True,cells=details,checks=rows,benchmark=times,optimizer_updates=6,checkpoints_saved=0))
    write(out/'gpu_tests.json',dict(pass_all=True,checks=rows))
    write(out/'implementation_manifest.json',dict(code_commit=git(repo,'rev-parse','HEAD'),branch=git(repo,'branch','--show-current'),dirty_diff='clean',
        protocol_sha256=sha(out/'physics_natural_protocol_v2_0.md'),split_sha256=sha(out/'physics_natural_split_v2_0.json'),
        sources=read(out/'source_hashes.json'),python=sys.version,python_executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
        imports={name:str(Path(sys.modules[name].__file__).resolve()) for name in ('models.visual_world_model','physics_loss_pilot','natural_v2_core','natural_v2_data')},
        A_checkpoint_sha256=sha(cp),A_tensor_sha256=initial_hash,normalization_sha256=sha(prior/'normalization.json'),frozen_hash=before,
        previous_runs=[str(prior),str(root/'work/physics/pointmaze-physics-attribution-20261011')]))
    train_est=240*max(t['seconds'] for t in times)+2880*max(t['seconds'] for t in times)
    # family별 최대 14 synthetic + 4 natural 및 모든 tune checkpoint를 포함한다.
    encode_windows=(32*4)+(12+20)*18
    evaluation_windows=2*56 + 6*4*12*18 + 4*20*18
    estimate=train_est+encode_windows*max(encoding_times)+evaluation_windows*rollout_time
    write(out/'resource_forecast.json',dict(measured_step=times,measured_encoding_seconds_per_window=encoding_times,
        measured_h5_rollout_seconds_per_window=rollout_time,remaining_training_updates=3120,
        encoding_windows_upper_bound=encode_windows,evaluation_windows_upper_bound=evaluation_windows,
        raw_estimate_seconds=estimate,conservative_remaining_seconds=2*estimate+120,
        formula='prior + all preflight GPU jobs + 2*(3120*max_step + windows_encode*max_encode + windows_evaluate*measured_H5) + 120',
        estimate_not_authorization=True,assumptions='single train window timing; synthetic counts use maximum; CPU preparation excluded from GPU time',
        comparison_training_started=False))


def extra(root,repo,out,job):
    """추가 검증만: nonconstant float·실제 encode padding·mask 불변. update 없음."""
    import torch
    import physics_loss_pilot as pilot
    from natural_v2_data import encode_case
    from natural_v2_metrics import rollout
    prior=root/'work/physics/pointmaze-physics-loss-pilot-20261011'
    wm,cfg,n=pilot.runtime(root,repo,prior);params=read(prior/'parameters.json');checks=[]
    cpu=read(out/'alignment_checks.json');free=[]
    for case in cpu['natural']:
        for h,seg in enumerate(case['segments']):
            if not seg['free']:continue
            controls=np.asarray(case['controls'][h]);st=np.asarray(seg['trace'][0])
            varying=not np.all(controls==controls[0])
            actual=transition(torch.tensor(st,dtype=torch.float32,device='cuda'),torch.tensor(controls,dtype=torch.float32,device='cuda'),params)
            error=float(np.max(np.abs(actual.cpu().numpy()-numpy_transition(st,controls,params)[-1])))
            check(checks,f'float_nonconstant_{case["episode"]}_h{h+1}',error,1e-5)
            if varying:free.append(dict(episode=case['episode'],horizon=h+1,analytic_fine_error=seg['analytic_max_abs'],float_error=error))
    require(free,'no verified nonconstant free segment')
    c=read(out/'preflight_natural_cases.json')[0];raw=dict(np.load(out/(c['id']+'.npz')));case=dict(c,**raw)
    wm.eval();first=encode_case(wm,cfg,n,case);altered=copy.deepcopy(case);altered['future_actions'][-1]=.37
    second=encode_case(wm,cfg,n,altered)
    check(checks,'actual_encode_padding_invariance',float((first['future'][...,:-wm.action_dim]-second['future'][...,:-wm.action_dim]).abs().max()),1e-7)
    original_mask=first['mask'].clone();rollout(wm,first,n)
    check(checks,'mask_unchanged_by_prediction',True,ok=torch.equal(original_mask,first['mask']))
    # 실제 forward가 기록 action의 다음 block만 받는지 호출을 계측한다.
    seen=[];replace=wm.replace_actions_from_z
    def record(z,actions):seen.append(actions.detach().cpu().clone());return replace(z,actions)
    wm.replace_actions_from_z=record
    rollout(wm,first,n)
    wm.replace_actions_from_z=replace
    check(checks,'horizon_action_sequence',True,ok=len(seen)==4 and all(torch.equal(v,first['actions'][:,h+1:h+2].cpu()) for h,v in enumerate(seen)))
    write(out/'extra_gpu_tests.json',dict(pass_all=True,checks=checks,nonconstant_free_segments=free,optimizer_updates=0,
        code_commit=git(repo,'rev-parse','HEAD'),source_hashes=read(out/'source_hashes.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['cpu','gpu','unit','extra'],required=True);p.add_argument('--job',type=Path,required=True);p.add_argument('--preflight-only',action='store_true',required=True);a=p.parse_args()
    verify_lock(a.repo,a.out)
    try:globals()[a.stage](a.root,a.repo,a.out,a.job)
    except BaseException as ex:
        write(a.job/'failure.json',dict(error=repr(ex),traceback=traceback.format_exc()))
        raise
if __name__=='__main__':main()
