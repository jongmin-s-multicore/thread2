"""검토 승인 후에만 사용하는 v2.0 비교 실행. preflight에서 호출하지 않는다."""
import argparse
import json
from pathlib import Path
import time
# [repro] 서버·실제 argv 관문은 모델 import보다 먼저 실행한다.
if __name__!='__main__':
    raise RuntimeError('unsupported import: approved CLI required before model/data access')
if __name__=='__main__':
    from natural_v2_guard import production_entry_guard
    import sys
    production_entry_guard(sys.argv)
import numpy as np
from intervention import read,save,sha,sources
import physics_loss_pilot as pilot
from natural_v2_core import require,CELLS,parts,objective,schedule,upgrade_batch
from natural_v2_control import approval,verify_lock,ledger
# [repro] 실제 입력/갱신/실행 승인의 공통 운영 관문.
from natural_v2_guard import (read_document, verify_documents, recorded_update, update_counts,
    reconcile_curve, verify_job_authorization, budget_for_run)
from natural_v2_data import WhitelistData,anchors,natural_window,alignment_pass,synthetic_sites,encode_case
from natural_v2_metrics import rollout,case_metrics,action_metrics,aggregate,selection_row,paired
from natural_v2_selection import select_stage1,select_base,select_aux,final_success


def locked_save(path,value):
    save(path,value);save(path.with_suffix('.sha256.json'),{'sha256':sha(path)})


def locked_read(path):
    require(sha(path)==read(path.with_suffix('.sha256.json'))['sha256'],'selection lock changed')
    return read(path)


def old_cases(root,split):
    prior=root/'work/physics/pointmaze-physics-loss-pilot-20261011'
    oracles=read(prior/'oracles.json');result=[]
    for c in read(prior/'cases.json'):
        if c['split']!=split:continue
        c=dict(c);c['legacy_family']=c['family'];c['family']='intervention' if c['family']=='recorded' else c['family']
        c['segments']=oracles[c['id']];c['truth']=[s['trace'][-1] for s in c['segments']]
        c['controls']=np.tile(c['u'],(5,5,1)).tolist();c['cache']=str(prior/'encoded'/f'{c["id"]}.pth')
        c['cache_sha256']=sha(c['cache']);c['schedule_family']={'static':'S','low_speed':'L','intervention':'I'}[c['family']]
        result.append(c)
    return result


def load_batch(cases):
    import torch
    rows=[]
    for c in cases:
        require(sha(c['cache'])==c['cache_sha256'],'encoded cache changed')
        rows.append(upgrade_batch(torch.load(c['cache'],map_location='cpu',weights_only=False)))
    keys=('context','future','actions','controls','start','truth','mask','static')
    result={k:torch.cat([r[k][:,:2] if k=='mask' else r[k] for r in rows]).cuda() for k in keys}
    return result


def evaluate(wm,cases,n,params,folder,reused=None):
    folder.mkdir(parents=True,exist_ok=False)
    predictions={}
    for c in cases:
        predictions[c['id']]=reused[c['id']] if reused is not None else rollout(wm,load_batch([c]),n)
    rows=[]
    for c in cases:rows.extend(case_metrics(c,predictions[c['id']],params,n['state_std'].cpu().numpy()))
    rows.extend(action_metrics(cases,predictions))
    summary=aggregate(rows)
    save(folder/'predictions.json',predictions);save(folder/'per_case_metrics.json',rows);save(folder/'summary.json',summary)
    row=selection_row(summary,cases);save(folder/'selection_row.json',row)
    return row


def train_one(root,repo,out,arm,exposure,case_index,tune,stop,mode,strength,steps,params,eval_steps,job):
    import torch
    prior=root/'work/physics/pointmaze-physics-loss-pilot-20261011'
    folder=out/arm;folder.mkdir(exist_ok=False)
    wm,cfg,n=pilot.runtime(root,repo,prior);before=pilot.frozen_hash(wm);wm.predictor.train();torch.manual_seed(0);np.random.seed(0)
    opt=pilot.optimize(wm);curve=[];evaluations=[]
    family='S' if arm.startswith('S-') else 'B' if arm.startswith('B-') else 'P'
    require(len(exposure)==steps,'wrong step count')
    save(folder/'exposure.json',exposure)
    save(folder/'input_hashes.json',{cid:case_index[cid]['cache_sha256'] for batch in exposure for cid in batch})
    with (folder/'curve.jsonl').open('x') as log:
        for i,ids in enumerate(exposure):
            batch=load_batch([case_index[c] for c in ids]);opt.zero_grad(set_to_none=True)
            ll,st=parts(wm,batch,n,params,stop_input=stop)
            loss,scale=objective(ll,wm,batch['mask'],family,mode,strength)
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(wm.predictor.parameters(),1.,error_if_nonfinite=True)
            event=recorded_update(job,arm,i+1,opt.step)
            item=dict(step=i+1,update_event=event,update_accounting=update_counts(job,arm),loss=float(loss.detach()),components={k:float(v.detach()) for k,v in ll.items()},scale=scale,gradient_norm=float(gn),clipped=bool(gn>1),
                family_mask_counts=[dict(family=case_index[c]['schedule_family'],mask=batch['mask'][j].cpu().tolist()) for j,c in enumerate(ids)])
            log.write(json.dumps(item,allow_nan=False)+'\n');log.flush();curve.append(item)
            if i+1 in eval_steps:
                cp=folder/f'predictor_at{i+1}.pth';torch.save(dict(predictor=pilot.initial(wm),step=i+1),cp)
                result=evaluate(wm,tune,n,params,folder/f'tune_at{i+1}')
                result.update(step=i+1,checkpoint=str(cp),checkpoint_sha256=sha(cp));evaluations.append(result)
    require(pilot.frozen_hash(wm)==before,'frozen parameters changed')
    save(folder/'curve.json',curve);save(folder/'complete.json',dict(steps=steps,update_accounting=reconcile_curve(job,arm,curve),frozen_hash=before,code_commit=read(out/'run_lock.json')['code_commit']))
    return evaluations


def stage1(root,repo,out,job):
    verify_documents(out)  # [repro] 문서 사용 직전 재검사.
    import torch
    prior=root/'work/physics/pointmaze-physics-loss-pilot-20261011';attr=root/'work/physics/pointmaze-physics-attribution-20261011'
    train=old_cases(root,'calibration_fit');tune=old_cases(root,'calibration_tune');index={c['id']:c for c in train}
    require(len(train)==140 and len(tune)==56,'stage1 counts')
    exposure=read(prior/'B/exposure.json');params=read(prior/'parameters.json')
    wm,cfg,n=pilot.runtime(root,repo,prior);cells={}
    a=evaluate(wm,tune,n,params,out/'A-stage1',reused={r['id']:r for r in read(prior/'evaluation_A.json')})
    for cell,parent,old in (('K-F',prior,'D'),('K-N',attr,'DM')):
        complete=read(parent/old/'complete.json');require(sha(parent/old/'checkpoint.pth')==complete['checkpoint_sha256'],'reused checkpoint changed')
        cells[cell]=evaluate(wm,tune,n,params,out/cell,reused={r['id']:r for r in read(parent/f'evaluation_{old}.json')})
        cells[cell].update(checkpoint=str(parent/old/'checkpoint.pth'),checkpoint_sha256=complete['checkpoint_sha256'],step=120)
    del wm;torch.cuda.empty_cache()
    for cell in ('I-F','I-N'):
        stop,mode=CELLS[cell]
        cells[cell]=train_one(root,repo,out,cell,exposure,index,tune,stop,mode,1.,120,params,(120,),job)[0]
    choice=select_stage1(a,cells)
    locked_save(out/'stage1_choice.json',dict(choice=choice,cells=cells,baseline=a))


def save_case(out,c):
    folder=out/'raw';folder.mkdir(exist_ok=True)
    arrays={k:c[k] for k in ('images','context_states','truth','start','controls','context_actions','future_actions','mask')}
    f=folder/(c['id']+'.npz')
    with f.open('xb') as stream:np.savez_compressed(stream,**arrays)
    metadata={k:v for k,v in c.items() if k not in arrays}
    metadata.update(truth=np.asarray(c['truth']).tolist(),start=np.asarray(c['start']).tolist(),controls=np.asarray(c['controls']).tolist(),raw_path=str(f),raw_sha256=sha(f))
    return metadata


def prepare_data(root,repo,out,final=False):
    verify_documents(out)  # [repro] 문서 사용 직전 재검사.
    from physics_depth import make_env
    if final:locked_read(out/'final_eval_lock.json')
    else:locked_read(out/'stage1_choice.json')
    verify_documents(out)
    split=read_document(out,'physics_natural_split_v2_0.json');params=read(root/'work/physics/pointmaze-physics-loss-pilot-20261011/parameters.json')
    groups={'evaluation':split['stage2_fresh_evaluation_episodes']} if final else {
        'train':split['stage2_natural_train_episodes'],'tune':split['stage2_tune_episodes']}
    loader=WhitelistData(sources(root,repo)['data'],sum(groups.values(),[]))
    env=make_env(root,repo);renderer=make_env(root,repo);renderer.prepare_for_render();cases={};audit=[]
    try:
        for group,ids in groups.items():
            rows=[]
            for ep in ids:
                for k in anchors(int(loader.row('seq_lengths',ep))):
                    c=natural_window(loader,ep,k,env,params)
                    require(alignment_pass(c),'natural data gate; no sample exclusion')
                    rows.append(save_case(out,c))
                if group!='train':
                    synthetic,checks=synthetic_sites(loader,ep,env,renderer,params)
                    audit.extend(checks);rows.extend(save_case(out,c) for c in synthetic)
            if group=='train':rows.extend(old_cases(root,'calibration_fit'))
            else:
                minimum=12 if final else 8
                for family in ('static','low_speed','intervention'):
                    require(len({c['episode'] for c in rows if c['family']==family})>=minimum,'synthetic cohort data gate')
            cases[group]=rows
    finally:env.close();renderer.close()
    locked_save(out/('eval_data.json' if final else 'train_tune_data.json'),dict(cases=cases,selection_audit=audit,access_log=loader.access))


def encode_data(root,repo,out,final=False):
    verify_documents(out)  # [repro] 문서 사용 직전 재검사.
    import torch
    manifest=locked_read(out/('eval_data.json' if final else 'train_tune_data.json'))
    if final:locked_read(out/'final_eval_lock.json')
    wm,cfg,n=pilot.runtime(root,repo,root/'work/physics/pointmaze-physics-loss-pilot-20261011');wm.eval()
    folder=out/'encoded';folder.mkdir(exist_ok=True)
    for group,rows in manifest['cases'].items():
        for c in rows:
            if 'cache' in c:continue
            require(sha(c['raw_path'])==c['raw_sha256'],'raw arrays changed')
            arrays=dict(np.load(c['raw_path']));batch=encode_case(wm,cfg,n,dict(c,**arrays))
            f=folder/(c['id']+'.pth');require(not f.exists(),'encoded output exists')
            torch.save({k:v.cpu() for k,v in batch.items()},f);c.update(cache=str(f),cache_sha256=sha(f))
    locked_save(out/('eval_encoded.json' if final else 'train_tune_encoded.json'),manifest)


def stage2_base(root,repo,out,job):
    verify_documents(out)  # [repro] 문서 사용 직전 재검사.
    import torch
    locked_read(out/'stage1_choice.json');data=locked_read(out/'train_tune_encoded.json')['cases']
    params=read(root/'work/physics/pointmaze-physics-loss-pilot-20261011/parameters.json')
    train,tune=data['train'],data['tune'];index={c['id']:c for c in train}
    wm,cfg,n=pilot.runtime(root,repo,root/'work/physics/pointmaze-physics-loss-pilot-20261011')
    a=evaluate(wm,tune,n,params,out/'A-tune');save(out/'baseline_tune.json',a)
    del wm;torch.cuda.empty_cache();candidates=[]
    stop,mode=CELLS[locked_read(out/'stage1_choice.json')['choice']['cell']]
    for mixture in ('R06','R12'):
        exposure=schedule(train,mixture)
        evaluated=train_one(root,repo,out,'B-'+mixture,exposure,index,tune,stop,mode,1.,480,params,(120,240,480),job)
        candidates.extend(dict(r,mixture=mixture) for r in evaluated)
        save(out/('B-'+mixture)/'tune_at0_reference.json',dict(path=str(out/'A-tune'),sha256=sha(out/'A-tune/summary.json'),step=0))
    chosen=select_base(a,candidates);locked_save(out/'base_choice.json',chosen)


def stage2_aux(root,repo,out,job):
    verify_documents(out)  # [repro] 문서 사용 직전 재검사.
    base=locked_read(out/'base_choice.json');require(base['status']=='BASE_CHOSEN','NO_PRESERVING_BASELINE')
    stop,mode=CELLS[locked_read(out/'stage1_choice.json')['choice']['cell']]
    data=locked_read(out/'train_tune_encoded.json')['cases'];train,tune=data['train'],data['tune'];index={c['id']:c for c in train}
    params=read(root/'work/physics/pointmaze-physics-loss-pilot-20261011/parameters.json')
    a=read(out/'baseline_tune.json');choices={}
    for family in ('P','S'):
        candidates=[]
        for strength in (.5,1.):
            arm=f'{family}-{base["mixture"]}-c{strength}'
            evaluated=train_one(root,repo,out,arm,schedule(train,base['mixture']),index,tune,stop,mode,strength,480,params,(120,240,480),job)
            candidates.extend(dict(r,mixture=base['mixture'],strength=strength) for r in evaluated if r['step']==base['step'])
            save(out/arm/'tune_at0_reference.json',dict(path=str(out/'A-tune'),sha256=sha(out/'A-tune/summary.json'),step=0))
        choices[family]=select_aux(a,base,candidates,family)
    locked_save(out/'aux_choice.json',choices)
    if not choices['P']['incremental_gate']:
        locked_save(out/'tune_stop.json',dict(status='NO_TUNE_INCREMENTAL_BENEFIT'));return
    checkpoint=sources(root,repo)['cfg'].parent/'checkpoints/model_latest.pth'
    locked_save(out/'final_eval_lock.json',dict(A=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)),B=base,**choices,
        stage1_choice_sha256=sha(out/'stage1_choice.json'),base_choice_sha256=sha(out/'base_choice.json'),aux_choice_sha256=sha(out/'aux_choice.json'),
        tune_data_sha256=sha(out/'train_tune_encoded.json'),evaluation_ids=read_document(out,'physics_natural_split_v2_0.json')['stage2_fresh_evaluation_episodes']))


def final_eval(root,repo,out,job):
    verify_documents(out)  # [repro] 문서 사용 직전 재검사.
    import torch
    choices=locked_read(out/'final_eval_lock.json');cases=locked_read(out/'eval_encoded.json')['cases']['evaluation']
    params=read(root/'work/physics/pointmaze-physics-loss-pilot-20261011/parameters.json')
    outputs={};summaries={}
    # 전체 final eval을 한 번만 시작할 수 있다. 실패시 성능에 따른 재실행은 불가.
    save(out/'final_eval_started.json',dict(lock_sha256=sha(out/'final_eval_lock.json')))
    for arm in ('A','B','P','S'):
        c=choices[arm];require(sha(c['checkpoint'])==c['checkpoint_sha256'],'selected checkpoint changed')
        wm,cfg,n=pilot.runtime(root,repo,root/'work/physics/pointmaze-physics-loss-pilot-20261011')
        if arm!='A':wm.predictor.load_state_dict(torch.load(c['checkpoint'],map_location='cpu',weights_only=False)['predictor'])
        outputs[arm]=evaluate(wm,cases,n,params,out/f'final-{arm}')
        summaries[arm]=read(out/f'final-{arm}/summary.json');del wm;torch.cuda.empty_cache()
    pairs={x+'-'+y:paired(summaries[x],summaries[y]) for x,y in (('P','B'),('P','S'),('B','A'),('P','A'))}
    save(out/'final_paired.json',pairs)
    success=final_success(outputs['A'],outputs['B'],outputs['P'],pairs['P-B']['static.h5.drift.all']['ci95'])
    save(out/'final_outcome.json',dict(success=success,single_seed=True,wm_train_evaluation=True,sealed_opened=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',required=True);p.add_argument('--job',type=Path,required=True);p.add_argument('--approval',type=Path,required=True);a=p.parse_args()
    approved=approval(a.repo,a.out,a.approval,a.stage)
    budget=read(a.job/'budget_at_launch.json')
    require(budget['stage']==a.stage,'job stage mismatch')
    fresh=budget_for_run(a.out,approved,budget['ledger']['jobs'],budget['ledger']['known_prior_gpu_seconds'],a.stage)
    require(fresh==budget['remaining_forecast'],'launch budget changed')
    stages={'stage1':stage1,'stage2-base':stage2_base,'stage2-aux':stage2_aux,'final-eval':final_eval}
    if a.stage in stages:stages[a.stage](a.root,a.repo,a.out,a.job)
    elif a.stage in ('prepare-tune','prepare-eval'):prepare_data(a.root,a.repo,a.out,a.stage=='prepare-eval')
    elif a.stage in ('encode-tune','encode-eval'):encode_data(a.root,a.repo,a.out,a.stage=='encode-eval')
    else:raise RuntimeError('unregistered stage')
if __name__=='__main__':main()
