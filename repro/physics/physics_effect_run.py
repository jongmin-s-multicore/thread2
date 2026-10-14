"""[repro] One model load per fixed v2 job; scientific functions reused unchanged."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from physics_effect_control import (read,save,sha,require,locked_save,locked_read,admit,
    updated,exposure_check,bootstrap_ci,checked_paired,gate,row_gate,import_paths,verify_imports,forecast,resource_state,preserve_budget_lock)

# Scientific imports occur only after manifest admission in main.
def scientific_imports():
    global np,pilot,CELLS,parts,objective,schedule,upgrade_batch,sources
    global WhitelistData,anchors,natural_window,alignment_pass,synthetic_sites,encode_case
    global rollout,case_metrics,action_metrics,aggregate,selection_row
    global select_stage1,select_base,select_aux,final_success,evaluation_mode
    import numpy as np
    import physics_loss_pilot as pilot
    from intervention import sources
    from natural_v2_core import CELLS,parts,objective,schedule,upgrade_batch,evaluation_mode
    from natural_v2_data import WhitelistData,anchors,natural_window,alignment_pass,synthetic_sites,encode_case
    from natural_v2_metrics import rollout,case_metrics,action_metrics,aggregate,selection_row
    from natural_v2_selection import select_stage1,select_base,select_aux,final_success


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
    evaluation_start=time.monotonic();predictions={}
    for c in cases:
        predictions[c['id']]=reused[c['id']] if reused is not None else rollout(wm,load_batch([c]),n)
    rows=[]
    for c in cases:rows.extend(case_metrics(c,predictions[c['id']],params,(n['state_std'].cpu().numpy() if hasattr(n['state_std'],'cpu') else np.asarray(n['state_std']))))
    rows.extend(action_metrics(cases,predictions))
    summary=aggregate(rows)
    save(folder/'report_ci.json',bootstrap_ci(summary));save(folder/'predictions.json',predictions);save(folder/'per_case_metrics.json',rows);save(folder/'summary.json',summary)
    row=selection_row(summary,cases);save(folder/'selection_row.json',row)
    if reused is None:save(Path(JOBDIR)/('measurement_'+folder.parent.name+'_'+folder.name+'.json'),dict(kind='roll',seconds=time.monotonic()-evaluation_start,count=len(cases)))
    return row


def train_one(root,repo,out,arm,exposure,case_index,tune,stop,mode,strength,steps,params,eval_steps,job):
    import torch
    prior=root/'work/physics/pointmaze-physics-loss-pilot-20261011'
    folder=out/arm;folder.mkdir(exist_ok=False)
    wm,cfg,n=pilot.runtime(root,repo,prior);before=pilot.frozen_hash(wm);wm.eval();wm.predictor.train();torch.manual_seed(0);np.random.seed(0)
    require(sum(p.numel() for p in wm.predictor.parameters())==20122600,'predictor count');opt=pilot.optimize(wm);curve=[];evaluations=[];timings=[]
    family='S' if arm.startswith('S-') else 'B' if arm.startswith('B-') else 'P'
    exposure_check(exposure,steps,strength=strength)
    signature={'exposure':exposure,'cache_sha256':{cid:case_index[cid]['cache_sha256'] for batch in exposure for cid in batch},'seed':0}
    if steps==480:
        key=arm.split('-')[1];lock=out/('exposure_'+key+'.json')
        if lock.exists():require(locked_read(lock)==signature,'B/P/S input exposure mismatch')
        else:require(arm.startswith('B-'),'base exposure required');locked_save(lock,signature)
    save(folder/'exposure.json',exposure)
    save(folder/'input_hashes.json',{cid:case_index[cid]['cache_sha256'] for batch in exposure for cid in batch})
    with (folder/'curve.jsonl').open('x') as log:
        for i,ids in enumerate(exposure):
            step_start=time.monotonic();batch=load_batch([case_index[c] for c in ids]);opt.zero_grad(set_to_none=True)
            ll,st=parts(wm,batch,n,params,stop_input=stop)
            loss,scale=objective(ll,wm,batch['mask'],family,mode,strength)
            loss.backward();gn=torch.nn.utils.clip_grad_norm_(wm.predictor.parameters(),1.,error_if_nonfinite=True)
            def verify_update():
                require(all(bool(torch.isfinite(x).all()) for x in wm.predictor.state_dict().values()),'nonfinite predictor')
                for state in opt.state.values():
                    for value in state.values():
                        if torch.is_tensor(value):require(bool(torch.isfinite(value).all()),'nonfinite optimizer')
                require(pilot.frozen_hash(wm)==before,'frozen state changed after update')
            updated(job,i+1,steps,opt.step,torch.cuda.synchronize,verify_update)
            timings.append(time.monotonic()-step_start)
            if arm=='I-F' and i==7:
                ledger,fc,_,_=resource_state(M,observations=[{'step':max(timings)}],running={'job':Path(job).name,'elapsed':max(time.time()-(Path(job)/'attempt.json').stat().st_mtime,sum(timings)),'completed_steps':i+1})
                preserve_budget_lock(M['out'],fc)
                save(Path(job)/'first8_forecast.json',fc)
            event={'attempted':i+1,'completed':i+1}
            item=dict(step=i+1,update_event=event,update_accounting={'attempted':i+1,'completed':i+1},loss=float(loss.detach()),components={k:float(v.detach()) for k,v in ll.items()},scale=scale,gradient_norm=float(gn),clipped=bool(gn>1),
                family_mask_counts=[dict(family=case_index[c]['schedule_family'],mask=batch['mask'][j].cpu().tolist()) for j,c in enumerate(ids)])
            log.write(json.dumps(item,allow_nan=False)+'\n');log.flush();curve.append(item)
            if i+1 in eval_steps:
                cp=folder/f'predictor_at{i+1}.pth';torch.save(dict(predictor=pilot.initial(wm),step=i+1),cp)
                result=evaluate(wm,tune,n,params,folder/f'tune_at{i+1}')
                result.update(step=i+1,checkpoint=str(cp),checkpoint_sha256=sha(cp));evaluations.append(result)
    require(pilot.frozen_hash(wm)==before,'frozen parameters changed')
    save(folder/'curve.json',curve);save(folder/'complete.json',dict(steps=steps,update_accounting={'attempted':len(curve),'completed':len(curve)},frozen_hash=before,code_commit=M['host']['commit']))
    save(Path(job)/'timing.json',dict(max_step_seconds=max(timings),first8=timings[:8]))
    return evaluations


def save_case(out,c):
    folder=out/'raw';folder.mkdir(exist_ok=True)
    arrays={k:c[k] for k in ('images','context_states','truth','start','controls','context_actions','future_actions','mask')}
    f=folder/(c['id']+'.npz')
    with f.open('xb') as stream:np.savez_compressed(stream,**arrays)
    metadata={k:v for k,v in c.items() if k not in arrays}
    metadata.update(truth=np.asarray(c['truth']).tolist(),start=np.asarray(c['start']).tolist(),controls=np.asarray(c['controls']).tolist(),raw_path=str(f),raw_sha256=sha(f))
    return metadata


def prepare_data(root,repo,out,final=False):
    gate(out,JOB)
    from physics_depth import make_env
    if final:locked_read(out/'final_eval_lock.json')
    else:locked_read(out/'stage1_choice.json')
    admit(M,JOB,DIGEST,claimed=True)
    split=read(M['split_path']);params=read(root/'work/physics/pointmaze-physics-loss-pilot-20261011/parameters.json')
    groups={'evaluation':split['stage2_fresh_evaluation_episodes']} if final else {
        'train':split['stage2_natural_train_episodes'],'tune':split['stage2_tune_episodes']}
    loader=WhitelistData(sources(root,repo)['data'],sum(groups.values(),[]))
    env=make_env(root,repo);renderer=make_env(root,repo);renderer.prepare_for_render();cases={};audit=[]
    try:
        for group,ids in groups.items():
            rows=[]
            for ep in ids:row_gate(group,set(ids),ep,final and (out/'final_eval_lock.json').exists())
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
    finally:
        save(Path(JOBDIR)/'data_access.json',dict(global_deserialize=loader.global_load_note,rows=loader.access));env.close();renderer.close()
    locked_save(out/('eval_data.json' if final else 'train_tune_data.json'),dict(cases=cases,selection_audit=audit,access_log=loader.access))


def encode_data(root,repo,out,final=False):
    gate(out,JOB)
    import torch
    gate(out,JOB)
    manifest=locked_read(out/('eval_data.json' if final else 'train_tune_data.json'))
    if final:locked_read(out/'final_eval_lock.json')
    wm,cfg,n=pilot.runtime(root,repo,root/'work/physics/pointmaze-physics-loss-pilot-20261011');wm.eval()
    folder=out/'encoded';folder.mkdir(exist_ok=True);encoding_start=time.monotonic();encoded_count=0
    for group,rows in manifest['cases'].items():
        for c in rows:
            if 'cache' in c:continue
            require(sha(c['raw_path'])==c['raw_sha256'],'raw arrays changed')
            arrays=dict(np.load(c['raw_path']));batch=encode_case(wm,cfg,n,dict(c,**arrays))
            f=folder/(c['id']+'.pth');require(not f.exists(),'encoded output exists')
            torch.save({k:v.cpu() for k,v in batch.items()},f);encoded_count+=1;c.update(cache=str(f),cache_sha256=sha(f))
    save(Path(JOBDIR)/'measurement_encoding.json',dict(kind='enc',seconds=time.monotonic()-encoding_start,count=encoded_count))
    locked_save(out/('eval_encoded.json' if final else 'train_tune_encoded.json'),manifest)



def dispatch(root,repo,out,job):
    prior=root/'work/physics/pointmaze-physics-loss-pilot-20261011';params=read(prior/'parameters.json')
    if job.startswith('stage1-'):
        train=old_cases(root,'calibration_fit');tune=old_cases(root,'calibration_tune')
        require(len(train)==140 and len(tune)==56,'stage1 fixed cohorts')
        if job!='stage1-select':
            cell=job.removeprefix('stage1-');stop,mode=CELLS[cell]
            values=train_one(root,repo,out,cell,read(prior/'B/exposure.json'),{c['id']:c for c in train},tune,stop,mode,1.,120,params,(120,),JOBDIR)
            locked_save(out/(job+'_row.json'),values[0]);return
        # Historical predictions are aggregated on exactly tune56, without loading any model.
        n={'state_std':read(prior/'normalization.json')['state_std']}
        a=evaluate(None,tune,n,params,out/'A-stage1',reused={r['id']:r for r in read(prior/'evaluation_A.json')})
        cells={}
        for cell,parent,arm in [('K-F',prior,'D'),('K-N',root/'work/physics/pointmaze-physics-attribution-20261011','DM')]:
            complete=read(parent/arm/'complete.json');cp=parent/arm/'checkpoint.pth';require(sha(cp)==complete['checkpoint_sha256'],'reused checkpoint')
            rows={r['id']:r for r in read(parent/f'evaluation_{arm}.json')}
            require(all(c['id'] in rows for c in tune),'historical tune cohort missing')
            cells[cell]=evaluate(None,tune,n,params,out/cell,reused=rows);cells[cell].update(checkpoint=str(cp),checkpoint_sha256=sha(cp),step=120)
        for cell in ('I-F','I-N'):cells[cell]=locked_read(out/('stage1-'+cell+'_row.json'))
        locked_save(out/'stage1_choice.json',{'baseline':a,'cells':cells,'choice':select_stage1(a,cells)});return
    if job.startswith('prepare-'):prepare_data(root,repo,out,job=='prepare-eval');return
    if job.startswith('encode-'):encode_data(root,repo,out,job=='encode-eval');return
    if job=='A-tune':
        wm,cfg,n=pilot.runtime(root,repo,prior);cases=locked_read(out/'train_tune_encoded.json')['cases']['tune']
        locked_save(out/'baseline_tune.json',evaluate(wm,cases,n,params,out/'A-tune'));return
    if job.startswith(('B-','P-','S-')):
        import torch
        data=locked_read(out/'train_tune_encoded.json')['cases'];train,tune=data['train'],data['tune']
        stop,mode=CELLS[locked_read(out/'stage1_choice.json')['choice']['cell']]
        if job.startswith('B-'):mixture=job[2:];strength=1.
        else:mixture=locked_read(out/'base_choice.json')['mixture'];strength=float(job.split('c')[1])
        arm=job if job.startswith('B-') else job[0]+'-'+mixture+'-c'+str(strength)
        rows=train_one(root,repo,out,arm,schedule(train,mixture),{c['id']:c for c in train},tune,stop,mode,strength,480,params,(120,240,480),JOBDIR)
        rows=[dict(x,mixture=mixture,strength=strength) for x in rows]
        locked_save(out/(job+'_rows.json'),rows)
        save(out/arm/'tune_at0_reference.json',{'path':str(out/'A-tune'),'sha256':sha(out/'A-tune/summary.json'),'step':0});return
    if job=='base-select':
        candidates=sum([locked_read(out/(x+'_rows.json')) for x in ('B-R06','B-R12')],[])
        locked_save(out/'base_choice.json',select_base(locked_read(out/'baseline_tune.json'),candidates));return
    if job=='aux-select':
        a=locked_read(out/'baseline_tune.json');b=locked_read(out/'base_choice.json');choices={}
        for family in ('P','S'):
            candidates=[x for strength in ('0.5','1.0') for x in locked_read(out/(family+'-c'+strength+'_rows.json')) if x['step']==b['step']]
            choices[family]=select_aux(a,b,candidates,family)
        locked_save(out/'aux_choice.json',choices)
        if not choices['P']['incremental_gate']:locked_save(out/'tune_stop.json',{'status':'NO_TUNE_INCREMENTAL_BENEFIT'});return
        cp=sources(root,repo)['cfg'].parent/'checkpoints/model_latest.pth'
        locked_save(out/'final_eval_lock.json',dict(A={'checkpoint':str(cp),'checkpoint_sha256':sha(cp)},B=b,**choices,
          stage1_choice_sha256=sha(out/'stage1_choice.json'),base_choice_sha256=sha(out/'base_choice.json'),aux_choice_sha256=sha(out/'aux_choice.json'),
          tune_data_sha256=sha(out/'train_tune_encoded.json'),evaluation_ids=read(M['split_path'])['stage2_fresh_evaluation_episodes']));return
    if job.startswith('final-') and job!='final-report':
        import torch
        arm=job[6:];choice=locked_read(out/'final_eval_lock.json')[arm];require(sha(choice['checkpoint'])==choice['checkpoint_sha256'],'selected checkpoint')
        wm,cfg,n=pilot.runtime(root,repo,prior)
        if arm!='A':wm.predictor.load_state_dict(torch.load(choice['checkpoint'],map_location='cpu',weights_only=False)['predictor'])
        row=evaluate(wm,locked_read(out/'eval_encoded.json')['cases']['evaluation'],n,params,out/job)
        locked_save(out/(job+'_row.json'),row);return
    if job=='final-report':
        summaries={x:read(out/('final-'+x)/'summary.json') for x in ('A','B','P','S')}
        pairs={x+'-'+y:checked_paired(summaries[x],summaries[y]) for x,y in [('P','B'),('P','S'),('B','A'),('P','A')]}
        save(out/'final_paired.json',pairs)
        rows={x:locked_read(out/('final-'+x+'_row.json')) for x in ('A','B','P','S')}
        save(out/'final_outcome.json',{'success':final_success(rows['A'],rows['B'],rows['P'],pairs['P-B']['static.h5.drift.all']['ci95']),'single_seed':True,'wm_train_evaluation':True,'sealed_opened':False});return
    raise RuntimeError('unknown job')


def install_load_guard():
    import torch
    original=torch.load
    allowed=dict(M['model_input_hashes'])
    if JOB.startswith('stage1-'):
        for case in old_cases(Path(M['root']),'calibration_fit')+old_cases(Path(M['root']),'calibration_tune'):allowed[case['cache']]=case['cache_sha256']
    for row in M['file_pins']['hub_weights']:allowed[row['path']]=row['sha256']
    out=Path(M['out'])
    for name in ('train_tune_encoded.json','eval_encoded.json'):
        if (out/name).exists():
            if name=='eval_encoded.json':gate(out,JOB)
            for cases in locked_read(out/name)['cases'].values():
                for case in cases:allowed[case['cache']]=case['cache_sha256']
    if JOB.startswith('final-'):
        for row in locked_read(out/'final_eval_lock.json').values():
            if isinstance(row,dict) and 'checkpoint' in row:allowed[row['checkpoint']]=row['checkpoint_sha256']
    # Raw whitelist files are pinned in the preparation job's scoped file map, never fitted.
    if JOB.startswith('prepare-'):
        key='evaluation' if JOB=='prepare-eval' else 'train_tune'
        if key=='evaluation':gate(out,JOB)
        require(sha(M['raw_input_manifest_path'])==M['raw_input_manifest_sha256'],'raw source manifest changed')
        raw=read(M['raw_input_manifest_path'])
        if key=='evaluation':
            split=read(M['split_path']);base=Path(M['root'])/'work/data/point_maze'
            files=[base/(n+'.pth') for n in ('states','actions','seq_lengths')]+[base/'obses'/f'episode_{ep:03d}.pth' for ep in split['stage2_fresh_evaluation_episodes']]
            scoped={str(p):sha(p) for p in files};locked_save(Path(M['out'])/'eval_raw_input_pins.json',scoped);allowed.update(scoped)
        else:allowed.update(raw[key])
    def load(path,*args,**kwargs):
        resolved=str(Path(path).resolve());require(resolved in allowed and sha(resolved)==allowed[resolved],'checkpoint/cache/raw file not pinned')
        with (JOBDIR/'load_access.jsonl').open('a') as f:f.write(json.dumps({'path':resolved,'sha256':allowed[resolved]})+'\n')
        return original(path,*args,**kwargs)
    torch.load=load


def main():
    global M,JOB,JOBDIR,DIGEST
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--job',required=True);a=p.parse_args()
    M=read(a.manifest);JOB=a.job;DIGEST=a.manifest_sha;JOBDIR=Path(M['out'])/'jobs'/JOB
    admit(M,JOB,DIGEST,claimed=True)
    require(sys.argv[1:]==['--manifest',a.manifest,'--manifest-sha',DIGEST,'--job',JOB],'actual argv')
    save(JOBDIR/'child_started.json',{'job':JOB,'time':time.monotonic()})
    try:
        if JOB in ('cpu-regression','cpu-correction-h01-h04'):
            import unittest
            from test_physics_effect import ChangedPathTests, CorrectionTests
            case=CorrectionTests if JOB=='cpu-correction-h01-h04' else ChangedPathTests
            expected=6 if case is CorrectionTests else 7
            class NamedResult(unittest.TextTestResult):
                def addSuccess(self,test):super().addSuccess(test);named.append({'name':test.id(),'status':'pass'})
                def addFailure(self,test,err):super().addFailure(test,err);named.append({'name':test.id(),'status':'fail'})
                def addError(self,test,err):super().addError(test,err);named.append({'name':test.id(),'status':'error'})
                def addSkip(self,test,reason):super().addSkip(test,reason);named.append({'name':test.id(),'status':'skip'})
            named=[]
            suite=unittest.defaultTestLoader.loadTestsFromTestCase(case)
            if JOB=='cpu-correction-h01-h04':
                require(sorted(t.id() for t in suite)==sorted(M['expected_cpu_tests']),'exact focused case set')
            result=unittest.TextTestRunner(verbosity=2,resultclass=NamedResult).run(suite)
            save(JOBDIR/'cpu_checks.json',{'run':result.testsRun,'named':named,'failures':[(str(t),x) for t,x in result.failures],'errors':[(str(t),x) for t,x in result.errors],'skipped':[(str(t),x) for t,x in result.skipped]})
            require(result.testsRun==expected and result.wasSuccessful() and not result.skipped,'changed-path suite failed')
        else:
            if JOB.startswith('prepare-'):sys.path.append(M['model_site_packages'])
            scientific_imports();install_load_guard();dispatch(Path(M['root']),Path(M['repo']),Path(M['out']),JOB)
        imports=import_paths();save(JOBDIR/'actual_imports.json',imports);verify_imports(M,imports)
        save(JOBDIR/'result.json',{'status':'COMPLETE','job':JOB})
    except BaseException:
        if not (JOBDIR/'actual_imports.json').exists():save(JOBDIR/'actual_imports.json',import_paths())
        save(JOBDIR/'failure.json',{'traceback':traceback.format_exc(),'job':JOB});raise


if __name__=='__main__':main()
