"""[repro] Fixed v2 comparison jobs; no model imports in admission/CPU helpers."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from simple_server_control import (require,read,sha,observe,bind,validate_sources,
    verify_imports,import_paths,supervise,device)

TRAIN = {'stage1-I-F':120,'stage1-I-N':120,'B-R06':480,'B-R12':480,
         'P-c0.5':480,'P-c1.0':480,'S-c0.5':480,'S-c1.0':480}
GPU = set(TRAIN)|{'encode-train-tune','A-tune','encode-eval','final-A','final-B','final-P','final-S'}
CPU_CHECKS = {'cpu-regression','cpu-correction-h01-h04'}
STAGE1 = {'stage1-I-F','stage1-I-N','stage1-select'}
JOBS = GPU|CPU_CHECKS|{'stage1-select','prepare-train-tune','base-select','aux-select','prepare-eval','final-report'}


def save(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,allow_nan=False,indent=2);f.flush();os.fsync(f.fileno())


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def locked_save(path,value):
    save(path,value);save(Path(path).with_suffix('.sha256.json'),{'sha256':sha(path)})


def locked_read(path):
    require(sha(path)==read(Path(path).with_suffix('.sha256.json'))['sha256'],'choice hash mismatch')
    return read(path)


def updated(job,step,ceiling,perform,sync,verify):
    require(type(step) is int and 1<=step<=ceiling,'update ceiling')
    if step>1:require((Path(job)/('update_%04d_complete.json'%(step-1))).exists(),'previous update incomplete')
    save(Path(job)/('update_%04d_attempt.json'%step),{'step':step,'time':time.monotonic()})
    perform();sync();verify()
    save(Path(job)/('update_%04d_complete.json'%step),{'step':step,'time':time.monotonic()})


def exposure_check(exposure,steps,seed=0,strength=1.,reference=None):
    require(seed==0 and strength in (.5,1.) and steps in (120,480),'unapproved seed/grid/steps')
    require(len(exposure)==steps and all(len(b)==2 for b in exposure),'batch count')
    require(sum(TRAIN.values())==3120,'total update contract')
    h=digest(exposure)
    if reference is not None:require(h==reference,'B/P/S exposure mismatch')
    return h


def merge_cost(rows):
    seen={};lower=0.;charged=0.;unknown=[]
    for row in rows:
        key=row['identity']
        if key in seen:
            require(seen[key]==row,'conflicting same receipt');continue
        seen[key]=row
        if not row['gpu']:continue
        value=row.get('seconds');upper=row.get('upper_seconds')
        require(value is None or math.isfinite(value) and value>=0,'invalid cost')
        if value is not None:lower+=value
        if row.get('complete') and value is not None:charged+=value
        elif upper is not None:
            require(math.isfinite(upper) and upper>= (value or 0),'invalid upper charge');charged+=upper;unknown.append(key)
        else:unknown.append(key);charged=None;break
    return {'known_lower_bound':lower,'charged_gpu_seconds':charged,'unknown':unknown,'receipts':list(seen.values())}


def forecast(ledger,remaining_raw):
    used=ledger['charged_gpu_seconds'];require(used is not None,'unbounded historical GPU interval')
    require(all(math.isfinite(x) and x>=0 for x in remaining_raw.values()),'forecast invalid')
    need=2*sum(remaining_raw.values())+120
    require(used+need<=3600,'forecast exceeds 3600 including future stages')
    return {'used':used,'remaining_raw_by_job':remaining_raw,'remaining_raw':sum(remaining_raw.values()),'reserve':120,'forecast':need,'headroom':3600-used-need}


def disk_bytes(roots):
    seen=set();total=0
    for root in roots:
        for p in Path(root).rglob('*'):
            if not p.is_file():continue
            st=p.stat();key=(st.st_dev,st.st_ino)
            if key not in seen:seen.add(key);total+=st.st_size
    return total


def gate(out,job):
    out=Path(out)
    if job in CPU_CHECKS or job.startswith('stage1-'):return {}
    require(not (out/'stage2_budget_lock.json').exists(),'stage2 budget locked')
    choice=locked_read(out/'stage1_choice.json');parents={'stage1':sha(out/'stage1_choice.json')}
    if job not in ('prepare-train-tune',):
        if job!='encode-train-tune':locked_read(out/'train_tune_encoded.json')
    if job.startswith(('P-','S-')) or job in ('aux-select','prepare-eval','encode-eval','final-report') or job.startswith('final-'):
        base=locked_read(out/'base_choice.json');require(base['status']=='BASE_CHOSEN','NO_PRESERVING_BASELINE');parents['base']=sha(out/'base_choice.json')
    if job in ('prepare-eval','encode-eval','final-report') or job.startswith('final-'):
        lock=locked_read(out/'final_eval_lock.json');require(lock['P']['incremental_gate'] is True,'final gate refused');parents['final']=sha(out/'final_eval_lock.json')
    return parents


def row_gate(split,allowed,episode,final_locked):
    require(split in ('train','tune','evaluation'),'sealed/unknown split')
    require(split!='evaluation' or final_locked,'eval row/cache/encoding locked')
    require(episode in allowed,'outside row whitelist')


def admit(m,job,manifest_sha,claimed=False):
    require(job in JOBS and job in m['jobs'],'unknown job')
    require(sha(m['manifest_path'])==manifest_sha,'manifest mismatch')
    bind(m['host'],observe(m['repo']));validate_sources(m)
    for p,h in m['documents'].items():require(sha(p)==h,'document mismatch')
    require(m['seed']==0 and m['max_updates']==3120 and m['max_gpu_jobs']==15 and m['max_train_jobs']==8,'scientific job caps')
    if job not in CPU_CHECKS:
        review=read(m['scientific_review_path'])
        scientific_binding(m,job,manifest_sha,review,read(m['registry_path']),sha(m['registry_path']),read(m['registry_activation_path']))
        for p,h in m['input_hashes'].items():require(sha(p)==h,'fixed input hash')
        for row in m['file_pins']['hub_sources']+m['file_pins']['hub_weights']:require(sha(row['path'])==row['sha256'],'local DINO source/weight')
        if not job.startswith('stage1-'):
            audit=read(m['provenance_clearance_path']);require(audit['cleared'] is True and audit['manifest_sha256']==manifest_sha,'stage2 provenance unresolved')
    parents=gate(m['out'],job)
    marker=Path(m['out'])/'jobs'/job/'attempt.json'
    if claimed:
        rec=read(marker);require(rec['manifest_sha256']==manifest_sha and rec['candidate']==m['host']['commit'] and rec['parents']==parents,'job/parent/candidate mismatch')
    else:require(not marker.exists(),'duplicate job')
    return parents


def component_forecast(work,base_rates,observations=()):
    # [repro] Both first-eight and next-launch use this exact component sum.
    rates=dict(base_rates)
    for observed in observations:
        for kind,value in observed.items():
            require(kind in rates and math.isfinite(value) and value>=0,'invalid measured rate')
            rates[kind]=max(rates[kind],value)
    return {job:sum(rates[kind]*count for kind,count in counts.items()) for job,counts in work.items()}


def forecast_scope(ledger,remaining,stage2_locked=False):
    used=ledger['charged_gpu_seconds'];require(used is not None,'unbounded historical GPU interval')
    full=used+2*sum(remaining.values())+120<=3600
    if full and not stage2_locked:
        return {**forecast(ledger,remaining),'scope':'FULL','stage2_locked':False}
    stage1={k:v for k,v in remaining.items() if k in STAGE1}
    return {**forecast(ledger,stage1),'scope':'STAGE1_ONLY','stage2_locked':True,
            'full_remaining_raw_by_job':remaining,'full_forecast_feasible':full}


def scientific_binding(m,job,manifest_sha,review,registry,actual_sha,activation):
    expected=m['registry_sha256'];commit=m['host']['commit']
    require(actual_sha==expected and m['documents'][m['registry_path']]==expected,'registry bytes mismatch')
    code=registry['code_review'];scope=registry['execution_scope']
    require(registry['revision']==3 and registry['phase']=='IMPLEMENTATION_REVIEWED','registry not reviewed')
    require(code['implementation_review_status']=='APPROVED_STAGE1' and code['reviewed_implementation_commit']==commit,'registry candidate not reviewed')
    require(set(code['approved_comparison_stages'])==STAGE1 and job in STAGE1,'registry job scope')
    require(scope['comparison_training_allowed'] is True and scope['comparison_evaluation_allowed'] is True,'registry execution scope')
    require(review['approved'] is True and review['candidate_code_commit']==commit and review['manifest_sha256']==manifest_sha,'exact candidate scientific review required')
    require(review['registry_sha256']==expected and review['protocol_sha256']==registry['protocol']['sha256'] and set(review['allowed_jobs'])==STAGE1,'review registry/scope mismatch')
    require(registry['protocol']['sha256']==m['protocol_sha256'],'protocol mismatch')
    require(activation['candidate_code_commit']==commit and activation['old_sha256']==m['previous_registry_sha256'] and
            activation['mac_registry_sha256']==expected and activation['server_registry_sha256']==expected and
            activation['mac_path']==registry['registry_paths']['mac'] and activation['server_path']==m['registry_path'],
            'Mac/server registry activation proof required')


def terminal_checks(rec,job,read_result,post_admit):
    require(rec is not None and rec['returncode']==0 and rec['reaped'] and rec['reason'] is None,'supervisor/child failed')
    done=read_result();require(done['status']=='COMPLETE' and done['job']==job,'required COMPLETE result missing')
    post_admit()


def terminal_receipt(identity,gpu,seconds,ceiling,rec,success,error):
    # Cost completeness describes the measured interval, never scientific success.
    measured_complete=rec is not None and rec['reaped'] and math.isfinite(seconds)
    return {'identity':identity,'gpu':gpu,'seconds':seconds,'upper_seconds':max(ceiling,seconds),
            'complete':measured_complete,'success':bool(success),'failure':error,
            'returncode':rec['returncode'] if rec else None,'child_reaped':rec['reaped'] if rec else False,
            'supervisor_reason':rec['reason'] if rec else 'supervisor receipt unavailable'}


def cpu_remaining(prior,rows):
    charged=prior;seen={}
    for row in rows:
        if row['gpu']:continue
        if row['identity'] in seen:
            require(seen[row['identity']]==row,'conflicting CPU receipt');continue
        seen[row['identity']]=row
        seconds=row.get('seconds');upper=row.get('upper_seconds')
        require(seconds is None or math.isfinite(seconds) and seconds>=0,'invalid CPU seconds')
        if row.get('complete') and seconds is not None:charged+=seconds
        else:
            require(upper is not None and math.isfinite(upper) and upper>=(seconds or 0),'unknown CPU cost without bound')
            charged+=upper
    return max(0.,900-charged)


def job_wall(m,job,fc,cpu_left,setup_seconds):
    wall=m['jobs'][job]['seconds']-setup_seconds-2
    if job in GPU:
        require(job in fc['remaining_raw_by_job'],'stage2 budget locked')
        wall=min(wall,3600-fc['used'],2*fc['remaining_raw_by_job'][job]+120/15)
    else:wall=min(wall,cpu_left-setup_seconds-2)
    require(wall>0,'job setup/CPU budget exhausted wall')
    return wall


def resource_state(m,observations=(),running=None):
    # [repro] running={'job','elapsed','completed_steps'}: the job that is asking now. Its pending attempt is charged at measured
    # elapsed seconds (not the whole ceiling) and only its unfinished steps stay in the forecast, so it is counted once.
    if running is not None:require(math.isfinite(running['elapsed']) and running['elapsed']>=0,'invalid running elapsed')
    rows=list(m['historical_receipts']);root=Path(m['out'])/'jobs';blocked=[];finished=set()
    for attempt in root.glob('*/attempt.json'):
        job=attempt.parent.name;finish=attempt.parent/'receipt.json';a=read(attempt)
        if finish.exists():
            receipt=read(finish);rows.append(receipt)
            if receipt.get('success') is True:finished.add(job)
            else:blocked.append(job+':failed_or_legacy_without_explicit_success')
        elif running is not None and job==running['job']:
            blocked.append(job+':pending')
            rows.append({'identity':str(attempt),'gpu':job in GPU,'seconds':running['elapsed'],'upper_seconds':a['whole_ceiling_seconds'],'complete':True})
        else:
            blocked.append(job+':pending')
            rows.append({'identity':str(attempt),'gpu':job in GPU,'seconds':None,'upper_seconds':a['whole_ceiling_seconds'],'complete':False})
    ledger=merge_cost(rows);work={k:v for k,v in m['forecast_work'].items() if k not in finished}
    if running is not None and running['job'] in work:
        left=dict(work[running['job']]);left['step']=max(0,left['step']-running['completed_steps']);work[running['job']]=left
    observed=list(observations)
    for p in root.glob('*/timing.json'):observed.append({'step':read(p)['max_step_seconds']})
    for p in root.glob('*/measurement_*.json'):
        v=read(p)
        if v['count']>0:observed.append({v['kind']:v['seconds']/v['count']})
    remaining=component_forecast(work,m['forecast_rates'],observed)
    fc=forecast_scope(ledger,remaining,(Path(m['out'])/'stage2_budget_lock.json').exists())
    return ledger,fc,blocked,cpu_remaining(m['prior_cpu_seconds'],rows)


def require_ready(blocked):
    require(not blocked,'prior failed/pending job; no retry')


def preserve_budget_lock(out,fc):
    p=Path(out)/'stage2_budget_lock.json'
    if fc['stage2_locked'] and not p.exists():locked_save(p,fc)


def bootstrap_ci(summary):
    import numpy as np
    out={}
    for key,row in summary.items():
        ids=sorted(row['episodes']);v=np.asarray([row['episodes'][x] for x in ids],dtype=float)
        require(np.isfinite(v).all(),'nonfinite episode metric')
        rng=np.random.default_rng(0)
        ci=np.quantile(v[rng.integers(0,len(v),(10000,len(v)))].mean(1),[.025,.975]).tolist() if len(v) else None
        out[key]={**row,'ci95':ci,'ci_scope':'report_only_episode_bootstrap_seed0_10000'}
    return out


def checked_paired(first,second):
    from natural_v2_metrics import paired
    for summary in (first,second):
        for row in summary.values():require(all(math.isfinite(x) for x in row['episodes'].values()),'nonfinite paired input')
    return paired(first,second)


def main():
    import signal
    started=time.monotonic()
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--manifest-sha',required=True);p.add_argument('--job',required=True);a=p.parse_args();m=read(a.manifest)
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('whole job timeout')))
    signal.setitimer(signal.ITIMER_REAL,m['jobs'][a.job]['seconds'])
    require(str(Path(a.manifest).resolve())==m['manifest_path'],'actual manifest path')
    out=Path(m['out']);(out/'jobs').mkdir(exist_ok=True)
    with (out/'active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger,fc,pending,cpu_left=resource_state(m)
        early_remaining=min(m['jobs'][a.job]['seconds'],cpu_left if a.job not in GPU else m['jobs'][a.job]['seconds'])-(time.monotonic()-started)
        require(early_remaining>0,'whole CPU budget exhausted before admission')
        signal.setitimer(signal.ITIMER_REAL,early_remaining)
        require_ready(pending)
        parents=admit(m,a.job,a.manifest_sha);job=out/'jobs'/a.job;job.mkdir()
        save(job/'resource_admission.json',{'ledger':ledger,'forecast':fc,'blocked':pending,'cpu_remaining':cpu_left})
        require_ready(pending)
        preserve_budget_lock(out,fc)
        require(disk_bytes(m['disk_roots'])<10*1024**3,'cumulative disk cap')
        ceiling=m['jobs'][a.job]['seconds'];wall=job_wall(m,a.job,fc,cpu_left,time.monotonic()-started)
        # CPU bound includes setup, supervisor/finalization and cleanup, not only child.
        whole_remaining=min(ceiling,cpu_left if a.job not in GPU else ceiling)-(time.monotonic()-started)
        require(whole_remaining>0,'whole CPU budget exhausted')
        signal.setitimer(signal.ITIMER_REAL,whole_remaining)
        save(job/'attempt.json',{'job':a.job,'candidate':m['host']['commit'],'manifest_sha256':a.manifest_sha,'parents':parents,'wall_seconds':wall,'whole_ceiling_seconds':min(ceiling,cpu_left) if a.job not in GPU else ceiling,'gpu':a.job in GPU,'forecast':fc})
        python=m['oracle_python'] if a.job.startswith('prepare-') else m['python']
        argv=[python,'-B',m['repo']+'/repro/physics/physics_effect_run.py','--manifest',a.manifest,'--manifest-sha',a.manifest_sha,'--job',a.job]
        start=time.monotonic();rec=None;success=False;error=None
        try:
            os.sched_setaffinity(0,m['affinity'])
            gpu=(lambda:device(m)) if a.job in GPU else None
            if gpu:require(gpu()<m['gpu_device_mib'],'no GPU headroom')
            # Existing tested supervisor; per-file limit enlarged for authorized predictor checkpoints.
            rec=effect_supervise(argv,job,wall,m,gpu)
            terminal_checks(rec,a.job,lambda:read(job/'result.json'),lambda:admit(m,a.job,a.manifest_sha,claimed=True))
            success=True
        except BaseException as exc:
            error=repr(exc);raise
        finally:
            save(job/'receipt.json',terminal_receipt(str(job/'attempt.json'),a.job in GPU,time.monotonic()-started,
                 min(ceiling,cpu_left) if a.job not in GPU else ceiling,rec,success,error))


def apply_file_limit(resource):
    # [repro] A child may tighten inherited limits, never raise a hard ceiling.
    _,hard=resource.getrlimit(resource.RLIMIT_FSIZE)
    ceiling=512*1024**2 if hard==resource.RLIM_INFINITY else min(512*1024**2,hard)
    require(ceiling>0,'inherited file limit exhausted')
    resource.setrlimit(resource.RLIMIT_FSIZE,(ceiling,ceiling))
    return ceiling


def effect_supervise(argv,job,wall,m,gpu):
    # Same own-session monitor as validated simple runner, with checkpoint-sized file bound.
    import selectors,signal,resource
    from simple_server_control import tree_rss,terminate
    start=time.monotonic();samples=[];proc=None;reason=None;sel=selectors.DefaultSelector();count=0
    def setup():os.sched_setaffinity(0,m['affinity']);apply_file_limit(resource)
    try:
        env={**os.environ,**m.get('oracle_env',{})} if 'prepare-' in argv[-1] else dict(os.environ)
        env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=setup,env=env)
        os.set_blocking(proc.stdout.fileno(),False);sel.register(proc.stdout,selectors.EVENT_READ)
        with (job/'stdout.log').open('xb') as log:
            while True:
                require(time.monotonic()-start<wall,'timeout')
                rss,pids=tree_rss(proc.pid);g=gpu() if gpu else 0
                b=disk_bytes(m['disk_roots']);samples.append({'seconds':time.monotonic()-start,'rss':rss,'gpu_mib':g,'bytes':b})
                require(rss<=12*1024**3 and g<=11000 and b<=10*1024**3,'monitored resource cap')
                for key,_ in sel.select(.1):
                    data=os.read(key.fd,65536)
                    if data:count+=len(data);require(count<8*1024**2,'log cap');log.write(data);log.flush()
                    else:sel.unregister(key.fileobj)
                if proc.poll() is not None and not sel.get_map():break
    except BaseException as e:reason=repr(e)
    finally:
        if proc is not None:terminate(proc)
        sel.close();rec={'returncode':proc.returncode if proc else None,'reaped':proc is not None and proc.poll() is not None,'reason':reason,'elapsed_seconds':time.monotonic()-start,'samples':samples,'argv':argv}
        save(job/'supervisor.json',rec)
    return rec


if __name__=='__main__':main()
