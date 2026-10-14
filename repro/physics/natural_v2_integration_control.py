"""[repro] Proposed integration-only lifecycle; requires separate, concrete execution approval.

This module has no model imports. Missing delegated cgroup or current budget evidence
is a refusal, never a request to downgrade enforcement or reset historical costs.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from natural_v2_guard import (require, read, save, capture_context, bind_actual,
                              verify_execution_authorization, update_counts, CONTEXT_KEYS)

REQUIRED = ('I01_admission', 'I02_input_scope', 'I03_model', 'I04_natural_encode',
            'I05_zero_mask', 'I06_active_aux', 'I07_update', 'I08_restore', 'I09_completion')
CAPS = dict(wall_seconds=180, metadata_seconds=90, gpu_seconds=60,
            finalization_seconds=30, cpu_cores=2, ram_bytes=12*1024**3,
            gpu_device_mib=11000, disk_bytes=64*1024**2, updates=2,
            max_executions=1, max_retries=0, new_checkpoints=0, new_fixture_invocations=0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def pinned(path, expected):
    path = Path(path)
    require(path.is_absolute() and str(path.resolve()) == str(path) and not path.is_symlink(),
            'noncanonical input')
    require(digest(path) == expected, 'input hash mismatch: '+str(path))
    return path


# [repro] 단일 manifest 계약. draft는 같은 키를 가지지만 실행 검증을 통과하지 못한다.
SCOPE = 'natural-v2-integration-once-v1'
DOCUMENTS = ('registry','protocol','split','evidence','policy','review','source_plan')
DEPENDENCIES = tuple('repro/physics/'+name for name in (
    'natural_v2_integration_check.py','natural_v2_integration_control.py',
    'natural_v2_integration_bootstrap.sh','natural_v2_integration_watchdog.py',
    'test_natural_v2_integration.py','natural_v2_guard.py','natural_v2_core.py',
    'natural_v2_data.py','natural_v2_metrics.py','natural_v2_selection.py',
    'physics_loss_pilot.py','physics_attribution.py','physics_depth.py','frozen_runtime.py',
    'intervention.py','zero_velocity.py'))
FIELDS = tuple(('schema_version status scope scope_id candidate_code_commit uuid root repo out job '
    'python python_version caps host environment documents source_pins file_pins input_allowlist '
    'ledger bootstrap controller_argv child_argv required_checks data_scope').split())
COST_KEYS = ('research_gpu_seconds','preflight_gpu_seconds','preflight_updates',
             'preflight_cpu_seconds','preflight_bytes','research_bytes')
RESERVATION = dict(zip(COST_KEYS,(60,60,2,180,64*1024**2,64*1024**2)))
CEILINGS = dict(zip(COST_KEYS,(3600,120,12,900,1024**3,10*1024**3)))


def strict_read(path):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key');result[key]=value
        return result
    def bad(value):raise ValueError('nonfinite JSON: '+value)
    require(Path(path).stat().st_size<=4*1024**2,'oversize JSON')
    return json.loads(Path(path).read_text(),object_pairs_hook=pairs,parse_constant=bad)


def finite_tree(value, tensor_check=None):
    if isinstance(value,dict):
        for child in value.values():finite_tree(child,tensor_check)
    elif isinstance(value,(list,tuple)):
        for child in value:finite_tree(child,tensor_check)
    elif type(value) in (int,float):require(math.isfinite(value),'nonfinite number')
    elif tensor_check is not None and tensor_check(value):pass
    elif value is None or isinstance(value,(str,bool)):pass
    else:raise RuntimeError('unsupported finite value')


def canonical(path):
    require(isinstance(path,str) and Path(path).is_absolute() and str(Path(path).resolve())==path,
            'noncanonical path')
    return path


def pin_record(row):
    require(set(row)=={'path','sha256'},'pin schema')
    canonical(row['path'])
    require(isinstance(row['sha256'],str) and len(row['sha256'])==64 and
            all(x in '0123456789abcdef' for x in row['sha256']),'pin digest')


def draft_manifest():
    draft=dict.fromkeys(FIELDS)
    draft.update(schema_version='integration_execution_v2',status='PROPOSED',scope='integration_only',
        scope_id=SCOPE,caps=dict(CAPS),required_checks=list(REQUIRED),data_scope='train-cache-only:78,88,311,529')
    return draft


def make_argv(plan,role,plan_path,plan_hash='<PLAN_SHA256>'):
    if role=='bootstrap':
        return ['/bin/bash',plan['repo']+'/repro/physics/natural_v2_integration_bootstrap.sh',
            plan['bootstrap']['trust'],plan_path,plan_hash,plan['bootstrap']['approval_path'],
            '<APPROVAL_SHA256>',plan['bootstrap']['unit']]
    if role=='watchdog':
        return [plan['python'],'-B',plan['repo']+'/repro/physics/natural_v2_integration_watchdog.py',
            '--trust',plan['bootstrap']['trust'],'--execution-plan',plan_path,'--plan-sha256',plan_hash,
            '--resource-approval',plan['bootstrap']['approval_path'],'--approval-sha256','<APPROVAL_SHA256>',
            '--unit',plan['bootstrap']['unit']]
    require(role in ('controller','child'),'argv role')
    script=plan['repo']+'/repro/physics/natural_v2_integration_'+('check' if role=='child' else 'control')+'.py'
    args=[plan['python'],'-B',script]
    for key in ('root','repo','out','job','uuid'):args+=['--'+key,plan[key]]
    args+=['--stage','integration','--execution-plan',plan_path,'--plan-sha256',plan_hash]
    if role=='controller':
        args+=['--resource-approval',plan['bootstrap']['approval_path'],
               '--approval-sha256','<APPROVAL_SHA256>']
    return args


# [repro] 입력 확장 정보는 metadata에만 둔다. 실행 필드는 추가/누락을 허용하지 않는다.
ALLOW_FIELDS = {'episode_ids','natural_order','synthetic_order','hashes','tensor_hashes',
                'A_path','A_tensor_sha256','normalization_path','parameters_path',
                'natural','synthetic','hub_load_counts'}
NATURAL_FIELDS = {'id','episode','path','raw_anchor','frames','image_shape','split'}
SYNTHETIC_FIELDS = {'id','episode','path','split'}


def exact_metadata_keys(value,required):
    require(type(value) is dict and required<=set(value)<=required|{'metadata'},'nested input keys')
    if 'metadata' in value:
        meta=value['metadata']
        require(type(meta) is dict and len(meta)<=32 and all(type(k) is str and
                type(v) in (str,int,float,bool) for k,v in meta.items()),'metadata extension type')
        finite_tree(meta)


def digest_value(value):
    require(type(value) is str and len(value)==64 and all(x in '0123456789abcdef' for x in value),'digest format')


def validate_input_allowlist(allow):
    exact_metadata_keys(allow,ALLOW_FIELDS)
    digest_value(allow['A_tensor_sha256'])
    for key,extension in (('A_path','.pth'),('normalization_path','.json'),('parameters_path','.json')):
        require(Path(canonical(allow[key])).suffix==extension,'input extension')
    for key in ('hashes','tensor_hashes'):
        require(type(allow[key]) is dict and bool(allow[key]),'digest mapping')
        for path,h in allow[key].items():canonical(path);digest_value(h)
    require(type(allow['episode_ids']) is list and all(type(x) is int for x in allow['episode_ids']) and
            allow['episode_ids']==[78,88,311,529],'training episode types/order')
    require(allow['natural_order']==['N_78_10','N_88_10'] and type(allow['natural_order']) is list and
            allow['synthetic_order']==['Z_529','C_311_r0.2_d0'] and type(allow['synthetic_order']) is list,'case order')
    for category,fields,episodes,extension in (('natural',NATURAL_FIELDS,(78,88),'.npz'),
                                               ('synthetic',SYNTHETIC_FIELDS,(529,311),'.pth')):
        require(type(allow[category]) is dict and set(allow[category])==set(allow[category+'_order']),'case inventory')
        for key,ep in zip(allow[category+'_order'],episodes):
            row=allow[category][key];exact_metadata_keys(row,fields)
            require(type(row['id']) is str and row['id']==key and type(row['episode']) is int and row['episode']==ep,'case identity')
            require(Path(canonical(row['path'])).suffix==extension and Path(row['path']).stem==key,'cache filename')
            require(row['path'] in allow['hashes'],'cache digest missing')
            if category=='natural':
                require(type(row['raw_anchor']) is int and row['raw_anchor']==10 and row['split']=='stage2_natural_train','natural metadata')
                require(type(row['frames']) is list and all(type(x) is int for x in row['frames']) and
                        row['frames']==[0,5,10,15,20,25,30,35],'frames')
                shape=row['image_shape']
                require(type(shape) is list and len(shape)==4 and all(type(x) is int and x>0 for x in shape) and
                        shape[0]==8 and shape[-1]==3,'image shape unknown')
            else:require(row['split']=='calibration_fit','synthetic split')
    require(type(allow['hub_load_counts']) is dict and bool(allow['hub_load_counts']),'hub counts unknown')
    for path,count in allow['hub_load_counts'].items():
        canonical(path);require(type(count) is int and 0<=count<=2,'hub load count')
    return allow


def validate_manifest(plan,plan_path):
    require(type(plan) is dict and set(plan)==set(FIELDS),'manifest keys')
    require(plan['schema_version']=='integration_execution_v2' and
            plan['status']=='AUTHORIZED_INTEGRATION_ONLY','draft/unknown schema')
    require(plan['scope']=='integration_only' and plan['scope_id']==SCOPE,'scope')
    require(plan['caps']==CAPS and all(type(plan['caps'][k]) is int for k in CAPS),'caps')
    import uuid
    require(str(uuid.UUID(plan['uuid']))==plan['uuid'],'UUID')
    require(len(plan['candidate_code_commit'])==40 and
            all(x in '0123456789abcdef' for x in plan['candidate_code_commit']),'commit')
    for key in ('root','repo','out','job','python'):canonical(plan[key])
    require(plan['job']==plan['out']+'/jobs/'+plan['uuid'] and
            not Path(plan['out']).is_relative_to(plan['repo']),'job/output')
    require(type(plan['python_version']) is str and bool(plan['python_version']),'Python version')
    require(plan['required_checks']==list(REQUIRED),'required check set')
    require(set(plan['documents'])==set(DOCUMENTS),'document inventory')
    for row in plan['documents'].values():pin_record(row)
    require(set(plan['source_pins'])>=set(DEPENDENCIES),'mandatory source closure')
    require(any(x.startswith('models/') for x in plan['source_pins']) and
            any('preprocess' in x or x.startswith('utils') for x in plan['source_pins']),
            'model/preprocessing closure missing')
    for relative,h in plan['source_pins'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'source escape')
        pin_record(dict(path=plan['repo']+'/'+relative,sha256=h))
    require(set(plan['file_pins'])=={'A','config','normalization','parameters','caches','hub_sources','hub_weights','environment'},'input roles')
    for key in ('A','config','normalization','parameters','environment'):pin_record(plan['file_pins'][key])
    for key in ('caches','hub_sources','hub_weights'):
        require(type(plan['file_pins'][key]) is list and bool(plan['file_pins'][key]),'empty input role')
        for row in plan['file_pins'][key]:pin_record(row)
    require(len(plan['file_pins']['caches'])==4,'four cache pins')
    require(set(plan['host'])=={'actual_context','gpu_uuid','gpu_index','cpu_affinity','nvidia_smi','import_inventory'},'host schema')
    require(type(plan['host']['gpu_index']) is int and plan['host']['gpu_index']>=0,'device')
    require(len(set(plan['host']['cpu_affinity']))==2 and all(type(x) is int and x>=0 for x in plan['host']['cpu_affinity']),'affinity')
    context=plan['host']['actual_context']
    require(set(context)==set(CONTEXT_KEYS) and context['system']=='Linux' and
            context['code_commit']==plan['candidate_code_commit'],'actual context schema')
    require(context['dirty_sha256']==hashlib.sha256(b'').hexdigest(),'dirty context')
    for key in ('root','repo','out','job'):require(context[key]==plan[key],'context path')
    require(context['stage']=='integration' and context['argv']==plan['child_argv'][2:],'context argv')
    for key,role in (('registry_sha256','registry'),('protocol_sha256','protocol'),('input_sha256','evidence'),('policy_sha256','policy')):
        require(context[key]==plan['documents'][role]['sha256'],'document context binding')
    require(plan['host']['import_inventory'],'import inventory absent')
    for row in plan['host']['import_inventory'].values():pin_record(row)
    require(set(plan['environment'])=={'CUDA_VISIBLE_DEVICES','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','MKL_NUM_THREADS'},'environment keys')
    require(plan['environment']==dict(CUDA_VISIBLE_DEVICES=str(plan['host']['gpu_index']),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'),'environment binding')
    require(set(plan['ledger'])=={'path','sha256','historical_floor','fixture_history'},'ledger schema')
    pin_record({k:plan['ledger'][k] for k in ('path','sha256')})
    require(set(plan['ledger']['historical_floor'])==set(COST_KEYS),'floor keys')
    require(set(plan['bootstrap'])=={'trust','unit','scope_anchor','approval_path','plan_path','exec_start','control_group','boot_id','start_clock','quota_path'},'bootstrap schema')
    for key in ('trust','scope_anchor','approval_path','plan_path','quota_path'):canonical(plan['bootstrap'][key])
    require(plan['bootstrap']['plan_path']==plan_path,'plan location')
    for key in ('unit','control_group','boot_id'):
        require(type(plan['bootstrap'][key]) is str and bool(plan['bootstrap'][key]),'unknown service field')
    require(plan['bootstrap']['start_clock']=='ExecMainStartTimestampMonotonic','untrusted start clock')
    require(plan['bootstrap']['exec_start']==make_argv(plan,'bootstrap',plan_path),'external argv binding')
    require(plan['controller_argv']==make_argv(plan,'controller',plan_path) and
            plan['child_argv']==make_argv(plan,'child',plan_path),'argv schema drift')
    require(plan['data_scope']=='train-cache-only:78,88,311,529','data scope')
    allow=validate_input_allowlist(plan['input_allowlist']);pins=plan['file_pins']
    require(allow['A_path']==pins['A']['path'] and allow['normalization_path']==pins['normalization']['path'] and
            allow['parameters_path']==pins['parameters']['path'],'input role alias')
    expected_hashes={row['path']:row['sha256'] for key in ('A','config','normalization','parameters') for row in [pins[key]]}
    expected_hashes.update({row['path']:row['sha256'] for row in pins['caches']})
    require(allow['hashes']==expected_hashes,'allowlist/pin mismatch')
    data_paths=[x['path'] for x in allow['natural'].values()]+[x['path'] for x in allow['synthetic'].values()]
    require(len(set(data_paths))==4 and set(data_paths)=={r['path'] for r in pins['caches']},'cache scope mismatch')
    require(set(allow['hub_load_counts'])=={r['path'] for r in pins['hub_weights']} and
            all(type(x) is int and 0<=x<=2 for x in allow['hub_load_counts'].values()),'hub load counts unknown')
    expected_tensors={pins['A']['path']:pins['A']['sha256']}
    expected_tensors.update({row['path']:allow['hashes'][row['path']] for row in allow['synthetic'].values()})
    require(allow['tensor_hashes']==expected_tensors,'tensor scope')
    require(plan['documents']['registry']['path']==plan['root']+'/CURRENT_PROTOCOL.json','registry location')
    for role,name in (('protocol','physics_natural_protocol_v2_0.md'),('split','physics_natural_split_v2_0.json'),('evidence','physics_natural_evidence_v2_0.json')):
        require(plan['documents'][role]['path']==plan['repo']+'/repro/physics/'+name,'document role location')
    require(plan['documents']['policy']['path']==plan['root']+'/orchestration/SERVER_EXECUTION_POLICY.txt','policy location')
    finite_tree(plan)
    return plan


def all_pins(plan):
    rows=list(plan['documents'].values())
    rows += [dict(path=plan['repo']+'/'+p,sha256=h) for p,h in plan['source_pins'].items()]
    for value in plan['file_pins'].values():rows.extend(value if isinstance(value,list) else [value])
    rows+=list(plan['host']['import_inventory'].values())
    return rows


def verify_pins(plan):
    for row in all_pins(plan):pinned(row['path'],row['sha256'])


def validate_cases(allow,split):
    require(allow['natural_order']==['N_78_10','N_88_10'] and
            allow['synthetic_order']==['Z_529','C_311_r0.2_d0'] and
            allow['episode_ids']==[78,88,311,529],'case selection')
    require(set(allow['natural'])==set(allow['natural_order']) and
            set(allow['synthetic'])==set(allow['synthetic_order']),'case set')
    paths=[]
    for key,ep in zip(allow['natural_order'],(78,88)):
        row=allow['natural'][key]
        require(row['id']==key and row['episode']==ep and row['raw_anchor']==10 and
                row['frames']==[0,5,10,15,20,25,30,35] and row['split']=='stage2_natural_train','natural identity')
        require(ep in split['stage2_natural_train_episodes'],'natural train membership')
        shape=row['image_shape'];require(type(shape) is list and len(shape)==4 and
            all(type(x) is int and x>0 for x in shape) and shape[0]==8 and shape[-1]==3,'image shape unknown')
        paths.append(canonical(row['path']))
    for key,ep in zip(allow['synthetic_order'],(529,311)):
        row=allow['synthetic'][key]
        require(row['id']==key and row['episode']==ep and row['split']=='calibration_fit','synthetic identity')
        require(ep in split['stage1_train_episodes'] and ep in split['stage2_synthetic_train_episodes'],'synthetic train membership')
        paths.append(canonical(row['path']))
    forbidden=set(split['stage1_tune_episodes']+split['stage2_tune_episodes']+split['stage2_fresh_evaluation_episodes'])
    require(not forbidden.intersection(allow['episode_ids']) and len(set(paths))==4,'forbidden/alias input')
    return paths


def check_access(access,allow):
    expected=[('natural',allow['natural'][k]['path']) for k in allow['natural_order']]
    expected += [('synthetic',allow['synthetic'][k]['path']) for k in allow['synthetic_order']]
    observed=[(x['kind'],x['path']) for x in access if x['kind'] in ('natural','synthetic')]
    require(sorted(observed)==sorted(expected),'data access multiset')
    for row in access:require(row['sha256']==allow['hashes'].get(row['path'],row.get('approved_sha256')),'access digest')
    require(all(x['kind'] in ('natural','synthetic','model','hub') for x in access),'unknown access role')


def validate_update_values(steps,delta):
    finite_tree([steps,delta])
    require(bool(steps) and all(x==1 for x in steps) and delta>0,'optimizer state/delta mismatch')


def update_once(job,arm,step,perform,synchronize,verify):
    from natural_v2_guard import recorded_update
    require(step in (1,2) and update_counts(job)==dict(attempted=step-1,completed=step-1,uncertain=0,legacy_reserved=0),'update cap/uncertain')
    def callback():
        perform();synchronize();verify()
    recorded_update(job,arm,step,callback)


def validate_checks(checks,expected=REQUIRED):
    require([x['id'] for x in checks]==list(expected),'missing/duplicate checks')
    for row in checks:
        require(set(row)=={'id','expected','actual','tolerance','evidence','passed'} and
                row['passed'] is True and bool(row['evidence']) and row['actual'] is not None,'incomplete check')
        def known(value):
            if isinstance(value,dict):return all(known(x) for x in value.values())
            if isinstance(value,(list,tuple)):return all(known(x) for x in value)
            return value is not None and not (isinstance(value,str) and value.upper() in ('UNKNOWN','NOT_RUN','UNVERIFIED','FAILED','PENDING'))
        require(known(row['actual']),'unknown check result')
        finite_tree(row)


def pointer(doc,path):
    require(path.startswith('/'),'JSON pointer')
    for key in path[1:].split('/'):
        key=key.replace('~1','/').replace('~0','~');doc=doc[int(key)] if isinstance(doc,list) else doc[key]
    return doc


def validate_exit_proof(job,docs):
    proof=job['exit_proof']
    require(set(proof)=={'kind','path','job_pointer','host_pointer','invocation_pointer','observation_pointer'},'exit proof schema')
    require(proof['path'] in job['evidence'],'exit receipt ownership')
    doc=docs[proof['path']]
    require(type(job['host_id']) is str and bool(job['host_id']) and
            type(job['invocation_id']) is str and bool(job['invocation_id']),'job host/invocation identity')
    for key,field in (('job_pointer','id'),('host_pointer','host_id'),('invocation_pointer','invocation_id')):
        require(pointer(doc,proof[key])==job[field],'exit identity mismatch')
    observed=pointer(doc,proof['observation_pointer'])
    if proof['kind']=='reaped':require(observed is True,'process not reaped')
    elif proof['kind']=='service_exit':
        require(type(observed) is dict and set(observed)=={'active_state','main_pid','cgroup_populated','exit_status'},'service exit observation')
        require(observed['active_state'] in ('inactive','failed') and type(observed['main_pid']) is int and
                observed['main_pid']==0 and observed['cgroup_populated'] is False and
                type(observed['exit_status']) is int,'service/tree not exited')
    else:raise RuntimeError('unsupported exit proof; logical FAILED is not process exit')


def reconcile_ledger(plan):
    spec=plan['ledger'];ledger=strict_read(pinned(spec['path'],spec['sha256']))
    require(ledger['unresolved_jobs']==[] and ledger['integration_attempts_used']==0,'running/consumed scope')
    discovered=set();count=0
    for scan in ledger['inventory']:
        root=Path(canonical(scan['root']));require(root.is_dir(),'missing receipt root')
        for directory,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(directory)/d).is_symlink()]
            import fnmatch
            for name in files:
                count+=1;require(count<=10000,'inventory bound')
                relative=str((Path(directory)/name).relative_to(root))
                if any(fnmatch.fnmatch(relative,pat) for pat in scan['patterns']):
                    path=Path(directory)/name;require(not path.is_symlink(),'receipt symlink')
                    discovered.add(str(path))
    require(discovered==set(ledger['evidence']),'receipt inventory mismatch')
    docs={};total_bytes=0
    for path,row in ledger['evidence'].items():
        pinned(path,row['sha256']);total_bytes+=Path(path).stat().st_size
        require(total_bytes<=8*1024**2,'receipt byte bound');docs[path]=strict_read(path)
    totals=dict.fromkeys(COST_KEYS,0);seen=set();owned=set()
    for job in ledger['jobs']:
        require(job['id'] not in seen and job['state'] in ('COMPLETE','FAILED','UNKNOWN','CANCELLED'),'job identity/state')
        seen.add(job['id'])
        require(job['evidence'] and not owned.intersection(job['evidence']),'duplicate receipt ownership')
        owned.update(job['evidence'])
        for path in job['evidence']:
            row=ledger['evidence'][path]
            require(row['owner']==job['id'] and pointer(docs[path],row['identity_pointer'])==row['identity_value'],'receipt identity mismatch')
        terminal=job['terminal']
        require(terminal['path'] in job['evidence'] and pointer(docs[terminal['path']],terminal['pointer'])==terminal['value'] and terminal['value']==job['state'],'logical state mismatch')
        validate_exit_proof(job,docs)
        for key in COST_KEYS:
            cost=job['costs'][key];charged=cost['charged'];finite_tree(cost)
            require(type(charged) in (int,float) and charged>=0,'invalid charge')
            values=[];increments=set();cumulative_coverage=[];increment_coverage=set()
            for proof in cost['proofs']:
                require(proof['path'] in job['evidence'] and proof['unit']==('bytes' if key.endswith('bytes') else 'updates' if key.endswith('updates') else 'seconds'),'cost ownership/unit')
                value=pointer(docs[proof['path']],proof['pointer'])
                require(type(value) in (int,float) and math.isfinite(value) and value>=0,'invalid measured cost')
                require(proof['semantics'] in ('cumulative','increment'),'cost semantics')
                coverage=proof['coverage']
                require(type(coverage) is list and coverage and len(set(coverage))==len(coverage) and
                        all(type(x) is str and bool(x) for x in coverage),'cost coverage missing')
                if proof['semantics']=='cumulative':cumulative_coverage.append(set(coverage))
                if proof['semantics']=='increment':
                    require(coverage==[proof['event_id']] and not increment_coverage.intersection(coverage),'increment coverage overlap')
                    increment_coverage.update(coverage)
                    require(proof['event_id'] not in increments,'duplicate increment');increments.add(proof['event_id'])
                values.append((proof['semantics'],value))
            require(all(a<=b or b<=a for a in cumulative_coverage for b in cumulative_coverage),'non-nested cumulative coverage')
            require(not any(a.intersection(increment_coverage) for a in cumulative_coverage),'cumulative/increment double count')
            lower=max([v for mode,v in values if mode=='cumulative'] or [0])+sum(v for mode,v in values if mode=='increment')
            require(charged>=lower and charged>=cost['floor']>=0,'undercharge')
            # N/A is intentionally unsupported: explicit measured zero or positive reservation only.
            if not values or job['state'] in ('FAILED','UNKNOWN'):
                require(cost['reservation']>0 and charged>=cost['reservation'],'unknown cost not reserved')
            totals[key]+=charged
    require(owned==set(docs),'unmapped evidence')
    for key in COST_KEYS:
        require(totals[key]>=spec['historical_floor'][key] and totals[key]+RESERVATION[key]<=CEILINGS[key],'history/budget')
    historical=strict_read(pinned(plan['documents']['source_plan']['path'],plan['documents']['source_plan']['sha256']))['proposal']['historical_budget_ledger']
    minimum=dict(zip(COST_KEYS,(historical['known_total_gpu_seconds'],historical['completed_v2_preflight_gpu_seconds'],
        historical['completed_preflight_updates'],900-historical['v2_cpu_conservative_cross_scope_remaining_upper_seconds'],
        historical['historical_preflight_bytes_snapshot'],historical['historical_research_bytes_snapshot'])))
    require(all(spec['historical_floor'][key]>=minimum[key] for key in COST_KEYS),'historical floor understated')
    history=ledger['fixture_history']
    require(history==spec['fixture_history'] and history['invocations_used']==4 and
            history['invocation_cap']==4 and history['historical_reserved_windows_seconds']==6000 and
            history==historical['fixture_separate_ledger'] and history['old_deadlines_expired'] is True,'fixture history lost')
    return dict(totals=totals,reserved=RESERVATION,ledger_sha256=spec['sha256'])


class OutputBudget:
    """[repro] Reserve bytes before writing; final receipts have independent headroom."""
    def __init__(self, out):
        self.out = Path(out)
    def used(self):
        return sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
    def write(self, name, value, final=False):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        require(len(data)<=1024**2,'record size')
        path=self.out/name
        require(self.out.resolve() in path.resolve().parents,'output escape')
        with (self.out/'budget.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            require(self.used()+len(data)<=56*1024**2 and len(list(self.out.rglob('*')))<512,'worker output quota')
            with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            fd=os.open(path.parent,os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)


def gpu_usage(plan):
    # [repro] Queries only the approved device; never stop unrelated workloads.
    text = subprocess.check_output([plan['host']['nvidia_smi'], '--query-gpu=uuid,memory.used',
                                   '--format=csv,noheader,nounits', '-i', str(plan['host']['gpu_index'])],
                                  text=True, timeout=2).strip()
    uuid, used = [x.strip() for x in text.split(',')]
    require(uuid == plan['host']['gpu_uuid'], 'GPU UUID mismatch')
    return int(used)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('root','repo','out','job','uuid','execution-plan','resource-approval',
                'approval-sha256','plan-sha256'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--stage', required=True, choices=['integration'])
    return p


def main():
    args=parser().parse_args()
    plan=validate_manifest(strict_read(pinned(args.execution_plan,args.plan_sha256)),args.execution_plan)
    expected=make_argv(plan,'controller',args.execution_plan,args.plan_sha256)
    expected=[args.approval_sha256 if x=='<APPROVAL_SHA256>' else x for x in expected]
    require([sys.executable,'-B']+sys.argv==expected,'controller argv')
    require(str(Path(sys.executable).resolve())==plan['python'] and sys.version.split()[0]==plan['python_version'],'Python binding')
    out=Path(plan['out']);job=Path(plan['job']);budget=OutputBudget(out)
    supervisor=strict_read(out/'supervisor.json')
    require(supervisor['uuid']==plan['uuid'] and supervisor['plan_sha256']==args.plan_sha256,'supervisor binding')
    worker=Path(supervisor['worker_cgroup'])
    require(str(os.getpid()) in (worker/'cgroup.procs').read_text().split(),'direct controller denied')
    deadline=supervisor['start_monotonic']+90
    def alarm(*_):raise TimeoutError('fixed controller deadline')
    signal.signal(signal.SIGALRM,alarm)
    require(time.monotonic()<deadline,'metadata deadline consumed')
    signal.setitimer(signal.ITIMER_REAL,deadline-time.monotonic())
    child=None
    try:
        verify_pins(plan)
        approval=strict_read(pinned(args.resource_approval,args.approval_sha256))
        require(approval['scope']==plan['scope'] and approval['fixture'] is False and
                approval['uuid']==plan['uuid'],'production approval required')
        allow=plan['input_allowlist'];split=strict_read(plan['documents']['split']['path'])
        validate_cases(allow,split)
        child_argv=make_argv(plan,'child',args.execution_plan,args.plan_sha256)
        actual=capture_context(plan['root'],plan['repo'],out,'integration',job,child_argv[2:])
        context=dict(plan['host']['actual_context']);context['argv']=child_argv[2:]
        bind_actual(context,actual)
        require(actual['code_commit']==plan['candidate_code_commit'] and approval['actual_context']==actual,'candidate/context')
        request=dict(plan_sha256=args.plan_sha256,code_commit=actual['code_commit'],argv=child_argv,
                     data_scope=plan['data_scope'],limits={k:CAPS[k] for k in ('gpu_seconds','cpu_cores','ram_bytes','wall_seconds','disk_bytes','gpu_device_mib')},
                     max_executions=1,max_retries=0,actual_context=actual)
        receipt=verify_execution_authorization(approval,request)
        budget.write('run_lock.json',dict(code_commit=actual['code_commit'],plan_sha256=args.plan_sha256,
            approval_sha256=args.approval_sha256,actual_context=actual,documents=plan['documents'],
            source_pins=plan['source_pins'],file_pins=plan['file_pins'],ledger=plan['ledger'],
            scope_marker=supervisor['scope_marker'],required_checks=list(REQUIRED),caps=CAPS,
            start_monotonic=supervisor['start_monotonic']))
        budget.write('jobs/'+plan['uuid']+'/execution_authorization.json',dict(receipt,
            plan_path=args.execution_plan,approval_path=args.resource_approval,approval_sha256=args.approval_sha256))
        gpu_peak=gpu_usage(plan)
        require(gpu_peak+6500<=CAPS['gpu_device_mib'],'insufficient GPU headroom')
        budget.write('admitted.json',dict(uuid=plan['uuid'],plan_sha256=args.plan_sha256,context=actual,cgroup=str(worker)))
        started=time.monotonic();require(started<deadline,'metadata deadline')
        model_deadline=min(started+60,supervisor['start_monotonic']+150)
        budget.write('model_start.json',dict(monotonic=started,deadline=model_deadline))
        signal.setitimer(signal.ITIMER_REAL,max(.001,model_deadline-time.monotonic()))
        env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','LANG','LC_ALL')}
        env.update(plan['environment'])
        child=subprocess.Popen(child_argv,cwd=plan['repo'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        selector=selectors.DefaultSelector();log=bytearray()
        for stream in (child.stdout,child.stderr):
            os.set_blocking(stream.fileno(),False);selector.register(stream,selectors.EVENT_READ)
        while child.poll() is None or selector.get_map():
            require(time.monotonic()<model_deadline,'GPU deadline')
            gpu_peak=max(gpu_peak,gpu_usage(plan))
            require(gpu_peak<=CAPS['gpu_device_mib'],'GPU memory limit')
            require(digest(plan['ledger']['path'])==plan['ledger']['sha256'],'ledger changed')
            require(update_counts(job)['attempted']<=2,'update cap')
            for key,_ in selector.select(.02):
                block=os.read(key.fd,4096)
                if not block:selector.unregister(key.fileobj)
                log.extend(block);require(len(log)<=65536,'log quota')
        require(child.returncode==0,'model child failed')
        finished=time.monotonic()
        budget.write('model_end.json',dict(monotonic=finished))
        signal.setitimer(signal.ITIMER_REAL,max(.001,min(finished+25,supervisor['start_monotonic']+175)-time.monotonic()))
        checks=strict_read(out/'model_checks.json')['checks'];validate_checks(checks,REQUIRED[:-1])
        require(update_counts(job)==dict(attempted=2,completed=2,uncertain=0,legacy_reserved=0),'update accounting')
        verify_pins(plan);bind_actual(actual,capture_context(plan['root'],plan['repo'],out,'integration',job,child_argv[2:]))
        budget.write('bounded_log.json',dict(text=log.decode(errors='replace')))
        budget.write('controller_result.json',dict(status='READY_FOR_FINALIZATION',checks=checks,uuid=plan['uuid'],
            resources=dict(gpu_device_peak_observed_mib=gpu_peak,model_wall_seconds=finished-started,
                           metadata_wall_seconds=started-supervisor['start_monotonic'])))
        return 0
    except BaseException as ex:
        budget.write('controller_failure.json',dict(status='UNKNOWN',type=type(ex).__name__,message=str(ex)[:2000]))
        return 1
    # watchdog owns bounded worker kill/reap and completion, including errors in this handler.


if __name__=='__main__':
    sys.exit(main())
