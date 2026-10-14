"""[repro] F1~F6의 모델 없는 callback/sentinel 검사와 제한된 실행기."""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback
from natural_v2_guard import (
    save, read, sha, require, document_manifest, guarded_input, stage_costs,
    remaining_budget, update_event, recorded_update, update_counts, reconcile_curve,
    verify_execution_authorization, authorized_call, verify_constants, audit_manifests,
)


def run_fixtures(folder, test):
    def rejected(fn):
        try:fn()
        except (RuntimeError,KeyError,FileNotFoundError,FileExistsError,ValueError):return True
        return False
    docs=folder/'documents';docs.mkdir()
    names=['physics_natural_split_v2_0.json','physics_natural_evidence_v2_0.json','review.md','handoff.md']
    for name in names:save(docs/name,dict(fixture=True))
    registry=dict(documents=[dict(server_path='/fixture/'+n,sha256=sha(docs/n)) for n in names])
    save(docs/'registry_snapshot.json',registry)
    lock=dict(registry_sha256=sha(docs/'registry_snapshot.json'),document_sha256=document_manifest(registry),input_manifest_sha256=sha(docs/names[1]))
    save(docs/'run_lock.json',lock);calls=[]
    test('F1','matching_documents_callback',lambda:guarded_input(docs,lambda:'read_sentinel'),'read_sentinel')
    for name in names:
        original=(docs/name).read_bytes()
        if name.endswith('.md'):(docs/name).unlink()
        else:(docs/name).write_text('{"fixture":"changed"}')
        test('F1','reject_'+name,lambda:rejected(lambda:guarded_input(docs,lambda:calls.append('forbidden'))),True)
        (docs/name).write_bytes(original)
    wrong=dict(lock,document_sha256={});(docs/'run_lock.json').write_text(json.dumps(wrong))
    test('F1','registry_manifest_mismatch',lambda:rejected(lambda:guarded_input(docs,lambda:calls.append('forbidden'))),True)
    (docs/'run_lock.json').write_text(json.dumps(dict(lock,input_manifest_sha256='0'*64)))
    test('F1','input_manifest_mismatch',lambda:rejected(lambda:guarded_input(docs,lambda:calls.append('forbidden'))),True)
    (docs/'run_lock.json').write_text(json.dumps(lock))
    test('F1','invalid_documents_never_read_data',lambda:calls,[])
    forecast=dict(measured_step=[dict(seconds=.1)],measured_encoding_seconds_per_window=[.01],measured_h5_rollout_seconds_per_window=.02,raw_estimate_seconds=312+7.04+134.72)
    costs=stage_costs(forecast)
    jobs=[]
    test('F2','initial_full_cost',lambda:round(remaining_budget(costs,jobs,100,'stage1')['projected_gpu_seconds'],2),1127.52)
    complete=dict(job_id='fixture-job1',stage='stage1',comparison_stage=True,state='COMPLETE',gpu=True,elapsed_seconds=30,returncode=0,reason=None)
    test('F2','completed_cost_removed_actual_retained',lambda:round(remaining_budget(costs,[complete],100,'prepare-tune')['projected_gpu_seconds'],2),1105.04)
    failed=dict(complete,job_id='fixture-failed',state='FAILED',returncode=1,elapsed_seconds=5)
    test('F2','failed_time_kept_remaining_unchanged',lambda:round(remaining_budget(costs,[failed],100,'stage1')['actual_gpu_seconds'],2),105)
    cancellation=dict(failed,job_id='fixture-cancel',state='CANCELLED',elapsed_seconds=7)
    test('F2','cancelled_and_validation_time_included',lambda:remaining_budget(costs,[failed,cancellation,dict(failed,job_id='validation',comparison_stage=False,elapsed_seconds=2)],100,'stage1')['actual_gpu_seconds'],114)
    prior_boundary=3600-(2*sum(x['raw_seconds'] for x in costs['stages'].values())+120)
    test('F2','exact_cap',lambda:remaining_budget(costs,[],prior_boundary,'stage1')['projected_gpu_seconds'],3600.)
    test('F2','over_cap',lambda:rejected(lambda:remaining_budget(costs,[],prior_boundary+1,'stage1')),True)
    for state in ['RUNNING','UNKNOWN']:
        test('F2','reject_'+state,lambda state=state:rejected(lambda:remaining_budget(costs,[dict(complete,state=state)],0,'stage1')),True)
    test('F2','completed_stage_duplicate',lambda:rejected(lambda:remaining_budget(costs,[complete],0,'stage1')),True)
    test('F2','ambiguous_complete',lambda:rejected(lambda:remaining_budget(costs,[dict(complete,returncode=1)],0,'prepare-tune')),True)
    job=folder/'job';job.mkdir()
    test('F3','zero',lambda:update_counts(job),dict(attempted=0,completed=0,uncertain=0,legacy_reserved=0))
    calls=[]
    recorded_update(job,'arm',1,lambda:calls.append(1));recorded_update(job,'arm',2,lambda:calls.append(2))
    test('F3','two_completed',lambda:reconcile_curve(job,'arm',[dict(step=1),dict(step=2)]),dict(attempted=2,completed=2,uncertain=0,legacy_reserved=0))
    test('F3','duplicate_attempt_no_callback',lambda:rejected(lambda:recorded_update(job,'arm',2,lambda:calls.append(3))),True)
    test('F3','callback_count',lambda:calls,[1,2])
    test('F3','curve_mismatch',lambda:rejected(lambda:reconcile_curve(job,'arm',[dict(step=1)])),True)
    failure=folder/'failure_job';failure.mkdir();recorded_update(failure,'arm',1,lambda:None)
    def explode():raise RuntimeError('fixture callback exception')
    rejected(lambda:recorded_update(failure,'arm',2,explode))
    test('F3','exception_keeps_uncertain',lambda:update_counts(failure),dict(attempted=2,completed=1,uncertain=1,legacy_reserved=0))
    update_event(job,'other',1,'attempt')
    test('F3','after_callback_before_complete_interruption',lambda:update_counts(job),dict(attempted=3,completed=2,uncertain=1,legacy_reserved=0))
    test('F3','multiple_jobs_arms',lambda:sum(update_counts(j)['attempted'] for j in [job,failure]),5)
    request=dict(plan_sha256='a'*64,code_commit='b'*40,argv=['fixture_only'],data_scope=dict(episode_ids=[9001]),limits=dict(gpu_seconds=0,cpu_cores=1,ram_bytes=1024,wall_seconds=1,disk_bytes=1024,gpu_device_mib=0),max_executions=1,max_retries=0)
    auth=dict(copy.deepcopy(request),status='APPROVED',issuer='user',fixture=True);launch=[]
    test('F4','matching_fixture_sentinel',lambda:authorized_call(auth,request,lambda:'sentinel',True)[0],'sentinel')
    test('F4','fixture_not_real_authority',lambda:rejected(lambda:authorized_call(auth,request,lambda:launch.append(1))),True)
    test('F4','missing_user_approval',lambda:rejected(lambda:authorized_call(None,request,lambda:launch.append(1))),True)
    test('F4','research_approval_only',lambda:rejected(lambda:authorized_call(dict(status='APPROVED',code_commit='b'*40),request,lambda:launch.append(1))),True)
    for field in request:
        bad=copy.deepcopy(auth);del bad[field]
        test('F4','missing_'+field,lambda bad=bad:rejected(lambda:authorized_call(bad,request,lambda:launch.append(1),True)),True)
    for field in ['plan_sha256','code_commit','argv','data_scope']:
        bad=copy.deepcopy(request);bad[field]='changed'
        test('F4','changed_'+field,lambda bad=bad:rejected(lambda:authorized_call(auth,bad,lambda:launch.append(1),True)),True)
    for field in request['limits']:
        bad=copy.deepcopy(request);del bad['limits'][field]
        test('F4','missing_limit_'+field,lambda bad=bad:rejected(lambda:authorized_call(auth,bad,lambda:launch.append(1),True)),True)
        bad=copy.deepcopy(request);bad['limits'][field]+=1
        test('F4','increased_limit_'+field,lambda bad=bad:rejected(lambda:authorized_call(auth,bad,lambda:launch.append(1),True)),True)
    for field in ['max_executions','max_retries']:
        bad=copy.deepcopy(request);bad[field]+=1
        test('F4','increased_'+field,lambda bad=bad:rejected(lambda:authorized_call(auth,bad,lambda:launch.append(1),True)),True)
    small=copy.deepcopy(request);small['limits']['ram_bytes']=512
    test('F4','reduced_numeric_limit',lambda:verify_execution_authorization(auth,small,True)['verified'],True)
    small=copy.deepcopy(request);small['data_scope']={'episode_ids':[]}
    test('F4','changed_data_subset_requires_new_approval',lambda:rejected(lambda:verify_execution_authorization(auth,small,True)),True)
    test('F4','rejected_requests_never_launch',lambda:launch,[])
    test('F5','same_constant',lambda:verify_constants('X=[1,2]','X=[1,2]',('X',))['equal'],True)
    test('F5','same_value_expression',lambda:verify_constants("X=list('NS')","X=['N','S']",('X',))['equal'],True)
    test('F5','schedule_value_changed',lambda:rejected(lambda:verify_constants("MIXTURES={'R':['N']}","MIXTURES={'R':['S']}",('MIXTURES',))),True)
    test('F5','unresolved_constant_fail_closed',lambda:rejected(lambda:verify_constants('X=dynamic()','X=[]',('X',))),True)
    test('F5','nested_mutation_fail_closed',lambda:rejected(lambda:verify_constants("X={'R':[1]}\nX['R'][0]=2","X={'R':[1]}",('X',))),True)
    manifests=folder/'manifests';manifests.mkdir();save(manifests/'nested.json',dict(fixture=True,nested=[dict(episode_id=9001)]))
    (manifests/'invalid.json').write_text('{invalid')
    test('F6','nested_overlap_stop',lambda:audit_manifests([manifests/'nested.json'],[9001])['status'],'OVERLAP_STOP')
    test('F6','missing_and_invalid_unknown',lambda:len(audit_manifests([manifests/'invalid.json',manifests/'absent.json'],[])['unknown']),2)
    test('F6','scoped_no_overlap',lambda:audit_manifests([manifests/'nested.json'],[9002])['status'],'NO_OVERLAP_IN_SCANNED_METADATA')
    test('F6','excluded_scope_incomplete',lambda:audit_manifests([manifests/'nested.json'],[9002],['server unavailable'])['status'],'INCOMPLETE')
    # [repro] 연결 검사는 연구 모듈을 import하지 않고 AST로 호출 위치만 확인한다.
    here=Path(__file__).parent
    trees={n:ast.parse((here/n).read_text()) for n in ['natural_v2_control.py','natural_v2_comparison.py','natural_v2_preflight.py']}
    def calls_in(file,function):
        node=next(n for n in trees[file].body if isinstance(n,ast.FunctionDef) and n.name==function)
        return [(n.func.id if isinstance(n.func,ast.Name) else ast.unparse(n.func),n.lineno) for n in ast.walk(node) if isinstance(n,ast.Call)]
    control=calls_in('natural_v2_control.py','supervise')
    test('F4','authorization_before_external_calls',lambda:max(line for name,line in control if name=='verify_execution_authorization')<min(line for name,line in control if name in ['subprocess.Popen','subprocess.check_output']),True)
    test('F1','verify_lock_checks_all_documents',lambda:any(n=='verify_documents' for n,_ in calls_in('natural_v2_control.py','verify_lock')),True)
    test('F3','comparison_uses_recorded_update',lambda:any(n=='recorded_update' for n,_ in calls_in('natural_v2_comparison.py','train_one')),True)
    test('F5','reuse_compares_constants',lambda:any(n=='verify_constants' for n,_ in calls_in('natural_v2_control.py','reuse_cpu')),True)
    test('F6','preflight_records_audit',lambda:any(n=='audit_manifests' for n,_ in calls_in('natural_v2_preflight.py','metadata')),True)


if __name__=='__main__':
    raise SystemExit("server_fixture_entry.py만 허용; 직접 실행 금지")


def run_followup(out,m,actual,store,test,forbidden):
    # [repro] Invocation3 required tests; previous128 results are immutable historical evidence.
    import signal,time,importlib.util
    import natural_v2_guard as g
    base=out/'subcases';base.mkdir()
    def rejects(fn):
        try:fn()
        except (RuntimeError,ValueError,KeyError,FileExistsError,FileNotFoundError):return True
        return False
    def lifecycle_case(name,validation=lambda:None,body=lambda:None,final=lambda x:None,seconds=.05):
        folder=base/name
        result=g.protected_lifecycle(folder,validation,body,final,seconds,store)
        return result['status']=='failed' and (folder/'failed.json').exists() and not (folder/'complete.json').exists()
    def invalid_reader():
        g.require((base/'invalid/reserved.json').exists(),'reservation absent')
        raise ValueError('injected invalid manifest')
    test('lifecycle','reservation_before_manifest_read',lambda:lifecycle_case('invalid',validation=invalid_reader))
    test('lifecycle','invalid_manifest_consumes_attempt',lambda:(base/'invalid/reserved.json').exists() and g.read(base/'invalid/failed.json')['error']=='ValueError')
    test('lifecycle','initial_delay_deadline',lambda:lifecycle_case('initial_delay',validation=lambda:time.sleep(.25)))
    test('lifecycle','final_write_delay_deadline',lambda:lifecycle_case('final_delay',final=lambda _:time.sleep(.25)))
    test('lifecycle','memory_error_keeps_failure',lambda:lifecycle_case('memory',body=lambda:(_ for _ in ()).throw(MemoryError('controlled'))))
    def controlled_signal():
        previous=signal.getsignal(signal.SIGUSR1)
        def handler(*a):raise RuntimeError('controlled SIGUSR1')
        signal.signal(signal.SIGUSR1,handler)
        try:return lifecycle_case('signal',body=lambda:os.kill(os.getpid(),signal.SIGUSR1))
        finally:signal.signal(signal.SIGUSR1,previous)
    test('lifecycle','handled_signal_keeps_failure',controlled_signal)
    def small_writer():
        small=g.BoundedStore(base,256,128)
        blocked=rejects(lambda:small.write(base/'large.json',{'text':'x'*150}))
        small.write(base/'small_failure.json',{'status':'failed'},terminal=True)
        return blocked and (base/'small_failure.json').exists()
    test('lifecycle','write_budget_reserve',small_writer)
    test('lifecycle','incomplete_result_not_pass',lambda:all(not g.coverage({'g':['a','b']},r)['passed'] for r in [[],[dict(id='g',name='a',passed=True)],[dict(id='g',name='a',passed=True)]*2]))
    test('lifecycle','lost_transport_no_relaunch',lambda:g.reconcile_receipt(m['identity'],None)==dict(status='UNKNOWN',attempt_consumed=True,relaunch=False))
    cases={'multi_target_alias':'X=Y=[1]\nY.append(2)','reversed_multi_target_alias':'Y=X=[1]\nY.append(2)','nested_alias':"X={'a':[1]}\nY=X['a']\nY.append(2)",'function_global_mutation':'X=[1]\ndef f():\n global X\n X.append(2)\nf()','function_alias_mutation':'X=[1]\ndef f():\n y=X\n y.append(2)\nf()','unknown_top_level_call':'X=[1]\nf()'}
    for name,source in cases.items():test('alias',name,lambda source=source:rejects(lambda:g.verify_constants(source,'X=[1]',('X',))))
    test('alias','literal_equivalence',lambda:g.verify_constants("X=list('NS')","X=['N','S']",('X',))['equal'])
    test('alias','actual_constant_change',lambda:rejects(lambda:g.verify_constants('X=[1]','X=[2]',('X',))))
    def admission(name,mutate=None,production=False,replay=False,copied=False):
        folder=base/name;folder.mkdir();job=folder/'job';job.mkdir()
        argv=['fixture','--root',actual['root'],'--repo',actual['repo'],'--out',str(folder),'--stage','fixture','--job',str(job)]
        context=dict(actual,out=str(folder),job=str(job),stage='fixture',argv=argv)
        plan=folder/'plan.json';g.save(plan,{'fixture':True});approval_path=folder/'approval.json'
        request=dict(plan_sha256=g.sha(plan),code_commit=m['candidate'],argv=argv,data_scope={'fixture':True},limits={k:0 for k in g.LIMIT_KEYS},max_executions=1,max_retries=0,actual_context=context)
        approval=dict(request,status='APPROVED',issuer='user',fixture=True);g.save(approval_path,approval)
        g.save(folder/'run_lock.json',dict(code_commit=m['candidate']))
        receipt=dict(fixture=True,request=request,plan_path=str(plan),approval_path=str(approval_path),approval_sha256=g.sha(approval_path));g.save(job/'execution_authorization.json',receipt)
        observed=dict(context);calls=[]
        if mutate:mutate(folder,job,observed)
        target=argv
        if copied:
            other=folder/'other';other.mkdir();(other/'execution_authorization.json').write_bytes((job/'execution_authorization.json').read_bytes());target=list(argv);target[-1]=str(other);observed['job']=str(other);observed['argv']=target
        fn=lambda:g.admission_pipeline(target,lambda v,a:observed,lambda:calls.append('inert'),allow_fixture=not production)
        if name=='valid':fn();return calls==['inert']
        if replay:fn();calls.clear()
        return rejects(fn) and calls==[]
    test('admission','valid_fixture_full_pipeline_once',lambda:admission('valid'))
    test('admission','fixture_cannot_authorize_production',lambda:admission('production',production=True))
    test('admission','missing_approval_blocks_callbacks',lambda:admission('missingapproval',lambda f,j,a:(f/'approval.json').unlink()))
    test('admission','changed_plan_hash_blocks_callbacks',lambda:admission('planchange',lambda f,j,a:(f/'plan.json').write_text('{}')))
    test('admission','changed_approval_hash_blocks_callbacks',lambda:admission('approvalchange',lambda f,j,a:(f/'approval.json').write_text('{}')))
    test('admission','copied_receipt_blocks_callbacks',lambda:admission('copyreceipt',copied=True))
    test('admission','replayed_receipt_blocks_callbacks',lambda:admission('replayreceipt',replay=True))
    for key in g.CONTEXT_KEYS:
        test('admission','changed_'+key,lambda key=key:admission('changed_'+key,lambda f,j,a:a.__setitem__(key,'changed')))
        test('admission','missing_'+key,lambda key=key:admission('missing_'+key,lambda f,j,a:a.pop(key)))
    for name,module in [('comparison','natural_v2_comparison.py'),('preflight','natural_v2_preflight.py')]:
        def import_rejected(module=module):
            before=len(forbidden);spec=importlib.util.spec_from_file_location('unsupported_scientific_import',Path(m['repo'])/'repro/physics'/module);mod=importlib.util.module_from_spec(spec)
            try:spec.loader.exec_module(mod)
            except RuntimeError as ex:return 'unsupported import' in str(ex) and len(forbidden)==before and not hasattr(mod,'main')
            return False
        test('admission',name+'_import_rejected_before_model',import_rejected)
    identity=m['identity'];receipt=dict(identity,status='PREPARED',duration_monotonic_seconds=.1,epoch=-999)
    test('receipts','same_uuid_chain',lambda:g.reconcile_receipt(identity,receipt)['status']=='PREPARED')
    for name,key in [('different_uuid_rejected','uuid'),('different_candidate_rejected','candidate'),('different_manifest_rejected','manifest_sha256')]:
        test('receipts',name,lambda key=key:rejects(lambda:g.reconcile_receipt(identity,dict(receipt,**{key:'changed'}))))
    data=(Path(m['repo'])/'repro/physics/natural_v2_guard.py').read_bytes();h=hashlib.sha256(data).hexdigest()
    test('receipts','source_hash_transfer_match',lambda:g.verify_copy(data,m['tracked_sha256']['repro/physics/natural_v2_guard.py']))
    test('receipts','changed_copy_rejected',lambda:rejects(lambda:g.verify_copy(data+b'changed',h)))
    test('receipts','host_epochs_not_subtracted',lambda:g.reconcile_receipt(dict(identity,epoch=9999999999),receipt)['status']=='PREPARED')
    test('receipts','prior_records_unchanged',lambda:all(g.sha(out/n)==h for n,h in m['prior_run_hashes'].items()))


def run_constant_import(out,m,actual,store,test,forbidden):
    # [repro] Only frozen import/binding cases and affected alias regressions; no source exec.
    import natural_v2_guard as g
    plan=g.read(m['plan_path']);observations=[]
    g.require(g.sha(m['plan_path'])==m['plan_sha256'],'focused plan changed')
    g.require(m['required']==plan['required_checks'],'focused required names changed')
    def inspect(case):
        source=case['source'];names=tuple(case['protected_names']);expected=case['expected']
        try:
            values=g.static_constants(source,names)
            reference='\n'.join(name+'='+repr(value) for name,value in case.get('expected_values',{}).items())
            equal=g.verify_constants(source,reference,names)['equal'] if expected=='resolved' else False
            observed=dict(status='resolved',values=values,equal=equal)
            passed=expected=='resolved' and values==case['expected_values'] and equal
        except RuntimeError as ex:
            status='manual_review_required' if 'manual_review_required' in str(ex) or 'manual review required' in str(ex) else 'unexpected_error'
            observed=dict(status=status,error=str(ex));passed=expected=='manual_review_required' and status==expected
        observations.append(dict(name=case['name'],expected=expected,expected_values=case.get('expected_values'),actual=observed,passed=passed))
        return passed
    for case in plan['focused_cases']:test('import_binding',case['name'],lambda case=case:inspect(case))
    regressions={'multi_target_alias':'X=Y=[1]\nY.append(2)','reversed_multi_target_alias':'Y=X=[1]\nY.append(2)','nested_alias':"X={'a':[1]}\nY=X['a']\nY.append(2)",'function_global_mutation':'X=[1]\ndef f():\n global X\n X.append(2)\nf()','function_alias_mutation':'X=[1]\ndef f():\n y=X\n y.append(2)\nf()','unknown_top_level_call':'X=[1]\nf()'}
    for name,source in regressions.items():
        case=dict(name=name,source=source,protected_names=['X'],expected='manual_review_required')
        test('alias',name,lambda case=case:inspect(case))
    test('alias','literal_equivalence',lambda:g.verify_constants("X=list('NS')","X=['N','S']",('X',))['equal'])
    def changed():
        try:g.verify_constants('X=[1]','X=[2]',('X',))
        except RuntimeError as ex:return str(ex)=='referenced constants changed'
        return False
    test('alias','actual_constant_change',changed)
    test('ledger','prior_three_records_byte_identical',lambda:len(m['prior_run_hashes'])==3 and all(g.sha(out/n)==h and g.sha(out/'history'/n)==h for n,h in m['prior_run_hashes'].items()))
    def identity_cost():
        reservation=g.read(out/'server/invocation4.reservation.json');history=g.read(m['history_path'])
        return reservation['invocation']==m['invocation']==4 and reservation['uuid']==m['uuid'] and history['prior_fixture_invocations']==3 and m['prior_fixture_charged_seconds']==history['prior_fixture_charged_wall_seconds'] and m['prior_fixture_charged_seconds']+120<=300
    test('ledger','invocation4_identity_and_cost_preserved',identity_cost)
    def coverage_check():
        req={'sample':['a','b']};one=dict(id='sample',name='a',passed=True)
        try:g.coverage({},[])
        except RuntimeError:empty=True
        else:empty=False
        return empty and all(not g.coverage(req,rows)['passed'] for rows in [[],[one],[one,one]]) and g.coverage(req,[one,dict(id='sample',name='b',passed=True)])['passed']
    test('ledger','required_coverage_nonempty_exact',coverage_check)
    store.write(out/'server/binding_case_results.json',dict(m['identity'],cases=observations))
