"""[repro] S01–S07 회귀검사 소스. 이번 제출에서는 실행하지 않았다.

독립 승인된 CPU fixture에서만 사용한다. 실제 모델·torch·서버 호출이 없다.
OS 강제 종료/동시 프로세스의 입증을 대신하지 않는다.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import natural_v2_integration_control as c
import natural_v2_integration_watchdog as w
from natural_v2_guard import update_counts, admission_pipeline


class LifecycleRuntime(w.Runtime):
    """가짜 환경/프로세스만 주입하고 production main과 alarm/write wrapper는 그대로 사용."""
    def __init__(self,plan,mode):
        super().__init__();self.plan=plan;self.mode=mode;self.clock=11.;self.alarms=[]
        self.launched=0;self.killed=0;self.verifications=0;self.ticks=0
        self.anchor=Path(plan['bootstrap']['scope_anchor']);(self.anchor/c.SCOPE).mkdir(parents=True)
        self.marker=w.reserve_scope(self.anchor)
        self.worker=self.anchor/'worker';self.worker.mkdir()
        for name,value in {'memory.events':'oom 0\noom_kill 0\n','cpu.stat':'usage_usec 1','memory.peak':'1'}.items():
            (self.worker/name).write_text(value)
    def now(self):return self.clock
    def prelude(self,a):return [str(self.anchor),a.unit,'fixture','fixture','fixture'],self.marker,10.
    def pin(self,path,sha):return Path(path)  # inert files only; pin tests use production separately
    def read(self,path):
        if str(path)==self.plan['bootstrap']['plan_path']:
            if self.mode=='parse_failure':raise ValueError('initial JSON parse')
            return self.plan
        if str(path)==self.plan['bootstrap']['approval_path']:
            return dict(status='APPROVED',issuer='user',fixture=False,plan_sha256='a'*64,uuid=self.plan['uuid'])
        return super().read(path)
    def envelope(self,plan):return dict(start_monotonic=10.,worker_cgroup=str(self.worker),parent_cgroup=str(self.anchor))
    def arm(self,phase,deadline):
        with patch.object(w.signal,'signal'),patch.object(w.signal,'setitimer',side_effect=lambda _,seconds:self.alarms.append((phase,self.clock+seconds))):
            super().arm(phase,deadline)
    def verify(self,plan):
        self.verifications+=1
        if self.mode=='metadata_delay' and self.verifications==1:self.clock=101.
        if self.mode=='final_rehash_delay' and self.verifications==2:self.clock+=31
    def ledger(self,plan):return dict(reserved=c.RESERVATION)
    def launch(self,args,repo,worker):
        from types import SimpleNamespace
        self.launched+=1;self.clock=13.
        out=Path(self.plan['out']);(out/'model_start.json').write_text(json.dumps(dict(monotonic=12.)))
        if self.mode!='manager_lost':
            for step in (1,2):c.update_once(self.plan['job'],'inert',step,lambda:None,lambda:None,lambda:None)
            checks=[dict(id=k,expected='fixture',actual=1,tolerance=0,evidence=['fixture'],passed=True) for k in c.REQUIRED[:-1]]
            (out/'controller_result.json').write_text(json.dumps(dict(status='READY_FOR_FINALIZATION',uuid=self.plan['uuid'],checks=checks,resources={})))
        return SimpleNamespace(returncode=-9 if self.mode=='manager_lost' else 0)
    def running(self,proc):self.ticks+=1;return self.ticks<=2
    def drain(self):
        if self.mode!='manager_lost':
            (Path(self.plan['out'])/'model_end.json').write_text(json.dumps(dict(monotonic=14.)))
            self.clock=15.
        return []
    def kill(self,*args):self.killed+=1;return True
    def write_io(self,path,value,limit):
        if Path(path).name=='completion.json':
            if self.mode=='final_write_failure':raise OSError('fsync unavailable')
            if self.mode=='final_write_delay':
                self.clock+=31;return None  # 지연된 I/O 복귀는 production write의 후검사에서 거부
        return super().write_io(path,value,limit)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.job=self.root/'job';self.job.mkdir()

    def cases(self):
        natural={k:dict(id=k,episode=ep,raw_anchor=10,frames=[0,5,10,15,20,25,30,35],
            image_shape=[8,224,224,3],split='stage2_natural_train',path=str(self.root/(k+'.npz')))
            for k,ep in [('N_78_10',78),('N_88_10',88)]}
        synthetic={k:dict(id=k,episode=ep,split='calibration_fit',path=str(self.root/(k+'.pth')))
            for k,ep in [('Z_529',529),('C_311_r0.2_d0',311)]}
        a=dict(natural_order=list(natural),synthetic_order=list(synthetic),episode_ids=[78,88,311,529],
               natural=natural,synthetic=synthetic,hashes={})
        split=dict(stage2_natural_train_episodes=[78,88],stage1_train_episodes=[529,311],
            stage2_synthetic_train_episodes=[529,311],stage1_tune_episodes=[9],
            stage2_tune_episodes=[10],stage2_fresh_evaluation_episodes=[11])
        return a,split

    def manifest(self):
        p={k:None for k in c.FIELDS};repo=str(self.root/'repo');out=str(self.root/'out')
        pin=lambda name:dict(path=str(self.root/name),sha256='a'*64)
        uid='00000000-0000-4000-8000-000000000001'
        p.update(schema_version='integration_execution_v2',status='AUTHORIZED_INTEGRATION_ONLY',scope='integration_only',
            scope_id=c.SCOPE,candidate_code_commit='b'*40,uuid=uid,root=str(self.root),repo=repo,out=out,
            job=out+'/jobs/'+uid,python=str(self.root/'python'),python_version='3.10.21',caps=dict(c.CAPS),
            documents={k:pin(k) for k in c.DOCUMENTS},source_pins={x:'a'*64 for x in c.DEPENDENCIES},
            required_checks=list(c.REQUIRED),data_scope='train-cache-only:78,88,311,529')
        p['documents']['registry']=pin('CURRENT_PROTOCOL.json')
        for role,name in (('protocol','physics_natural_protocol_v2_0.md'),('split','physics_natural_split_v2_0.json'),('evidence','physics_natural_evidence_v2_0.json')):
            p['documents'][role]=dict(path=repo+'/repro/physics/'+name,sha256='a'*64)
        p['documents']['policy']=pin('orchestration/SERVER_EXECUTION_POLICY.txt')
        p['source_pins'].update({'models/example.py':'a'*64,'utils.py':'a'*64})
        a,_=self.cases();p['input_allowlist']=a
        p['file_pins']={k:pin(k+('.pth' if k=='A' else '.json')) for k in ('A','config','normalization','parameters','environment')}
        p['file_pins'].update(caches=[dict(path=r['path'],sha256='a'*64) for r in list(a['natural'].values())+list(a['synthetic'].values())],hub_sources=[pin('hub.py')],hub_weights=[pin('hub.pth')])
        a['hub_load_counts']={p['file_pins']['hub_weights'][0]['path']:1}
        a.update(A_tensor_sha256='c'*64,A_path=p['file_pins']['A']['path'],normalization_path=p['file_pins']['normalization']['path'],parameters_path=p['file_pins']['parameters']['path'])
        a['hashes']={r['path']:r['sha256'] for r in p['file_pins']['caches']+[p['file_pins'][k] for k in ('A','config','normalization','parameters')]}
        a['tensor_hashes']={a['A_path']:'a'*64,**{r['path']:'a'*64 for r in a['synthetic'].values()}}
        p['host']=dict(actual_context={},gpu_uuid='fixture-device',gpu_index=3,cpu_affinity=[2,3],nvidia_smi='/bin/false',import_inventory={'fake':pin('fake.py')})
        p['environment']=dict(CUDA_VISIBLE_DEVICES='3',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        p['ledger']=dict(pin('ledger.json'),historical_floor=dict.fromkeys(c.COST_KEYS,0),fixture_history={})
        p['bootstrap']=dict(trust=str(self.root/'trust'),unit='fixture.service',scope_anchor=str(self.root/'anchor'),approval_path=str(self.root/'approval.json'),plan_path=str(self.root/'plan.json'),exec_start='fixture',control_group='/fixture',boot_id='fixture',start_clock='ExecMainStartTimestampMonotonic',quota_path=str(self.root))
        p['bootstrap']['exec_start']=c.make_argv(p,'bootstrap',p['bootstrap']['plan_path'])
        p['controller_argv']=c.make_argv(p,'controller',p['bootstrap']['plan_path']);p['child_argv']=c.make_argv(p,'child',p['bootstrap']['plan_path'])
        ctx=dict.fromkeys(c.CONTEXT_KEYS,'fixture')
        ctx.update(system='Linux',code_commit=p['candidate_code_commit'],dirty_sha256=c.hashlib.sha256(b'').hexdigest(),
            stage='integration',argv=p['child_argv'][2:])
        for key in ('root','repo','out','job'):ctx[key]=p[key]
        for key in ('registry_sha256','protocol_sha256','input_sha256','policy_sha256'):ctx[key]='a'*64
        p['host']['actual_context']=ctx
        return p

    def valid(self,p):return c.validate_manifest(p,p['bootstrap']['plan_path'])

    def test_T01_schema_valid(self):
        p=self.manifest();self.assertIs(self.valid(p),p)
        self.assertEqual(c.make_argv(p,'child',p['bootstrap']['plan_path'],'x')[-1],'x')

    def test_T02_schema_refuse(self):
        p=self.manifest()
        for key,value in [('status','PROPOSED'),('caps',{}),('extra',1),('uuid',None)]:
            q=copy.deepcopy(p);q[key]=value
            with self.assertRaises((RuntimeError,ValueError,TypeError,AttributeError)):self.valid(q)
        for field in c.ALLOW_FIELDS:
            q=copy.deepcopy(p);del q['input_allowlist'][field]
            with self.assertRaises(RuntimeError):self.valid(q)
        for value in (None,'short','Z'*64):
            q=copy.deepcopy(p);q['input_allowlist']['A_tensor_sha256']=value
            with self.assertRaises(RuntimeError):self.valid(q)
        for category,key,field in (('natural','N_78_10','image_shape'),('synthetic','Z_529','episode')):
            q=copy.deepcopy(p);del q['input_allowlist'][category][key][field]
            with self.assertRaises(RuntimeError):self.valid(q)
        q=copy.deepcopy(p);q['input_allowlist']['natural']['N_78_10']['extra']=1
        with self.assertRaises(RuntimeError):self.valid(q)
        q=copy.deepcopy(p);q['input_allowlist']['metadata']={'note':'non-execution annotation'};self.valid(q)
        for text in ('{"a":1,"a":2}','{"a":NaN}','{"a":Infinity}'):
            path=self.root/'bad.json';path.write_text(text)
            with self.assertRaises((RuntimeError,ValueError)):c.strict_read(path)

    def test_T03_binding_mismatch(self):
        expected=self.manifest()['host']['actual_context']
        c.bind_actual(expected,expected)
        for key in c.CONTEXT_KEYS:
            actual=copy.deepcopy(expected);actual[key]=['wrong'] if key=='argv' else 'wrong'
            with self.assertRaises(RuntimeError):c.bind_actual(expected,actual)
        p=self.manifest();p['child_argv'][-1]='other'
        with self.assertRaises(RuntimeError):self.valid(p)

    def test_T04_envelope_absent(self):
        with self.assertRaises((RuntimeError,KeyError)):
            w.inspect_envelope(self.manifest(),lambda *_:'KillMode=process\nSendSIGKILL=no\nRestart=always')

    def lifecycle(self,mode):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        prior=self.root;self.root=Path(temp.name).resolve();plan=self.manifest();self.root=prior
        runtime=LifecycleRuntime(plan,mode)
        argv=['--trust',plan['bootstrap']['trust'],'--unit',plan['bootstrap']['unit'],
              '--execution-plan',plan['bootstrap']['plan_path'],'--plan-sha256','a'*64,
              '--resource-approval',plan['bootstrap']['approval_path'],'--approval-sha256','d'*64]
        return w.main(argv,runtime),runtime

    def test_T05_deadline_not_reset(self):
        code,r=self.lifecycle('normal');self.assertEqual(code,0)
        self.assertIn(('metadata',100.),r.alarms);self.assertIn(('gpu',72.),r.alarms)
        self.assertTrue(all(deadline==44 for phase,deadline in r.alarms if phase=='final'))
        for mode in ('metadata_delay','final_rehash_delay','final_write_delay'):
            code,r=self.lifecycle(mode);self.assertEqual(code,1)
            self.assertTrue(r.marker.exists())
            if mode=='metadata_delay':self.assertEqual(r.launched,0)
            else:self.assertFalse((Path(r.plan['out'])/'completion.json').exists())

    def scope(self):
        (self.root/c.SCOPE).mkdir();return w.reserve_scope(self.root)

    def test_T06_attempt_replay(self):
        marker=self.scope();self.assertTrue(marker.exists())
        # Empty marker, no UUID/output/approval knowledge: still permanently consumed.
        with self.assertRaises(FileExistsError):w.reserve_scope(self.root)

    def test_T07_attempt_race(self):
        # 두 contender가 같은 사전 ABSENT 관측을 한 뒤 가능한 mkdir 순서를 모두 모델링.
        # OS 병렬 실행 시험이 아니며 양 순서 모두 실제 reserve_scope를 호출한다.
        for order in (('A','B'),('B','A')):
            anchor=self.root/(''.join(order));(anchor/c.SCOPE).mkdir(parents=True)
            observed={who:not (anchor/c.SCOPE/'attempt-1').exists() for who in order}
            outcomes={}
            for who in order:
                self.assertTrue(observed[who])
                try:w.reserve_scope(anchor);outcomes[who]='won'
                except FileExistsError:outcomes[who]='lost'
            self.assertEqual(list(outcomes.values()),['won','lost'])

    def test_T08_early_error(self):
        code,r=self.lifecycle('parse_failure');self.assertEqual(code,1);self.assertEqual(r.launched,0)
        receipt=c.strict_read(r.marker/'completion.json')
        self.assertEqual(receipt['status'],'FAILED_OR_UNKNOWN')
        with self.assertRaises(FileExistsError):w.reserve_scope(r.anchor)

    def test_T09_manager_lost(self):
        code,r=self.lifecycle('manager_lost');self.assertEqual(code,1);self.assertEqual(r.killed,1)
        receipt=c.strict_read(Path(r.plan['out'])/'completion.json')
        self.assertEqual(receipt['status'],'FAILED_OR_UNKNOWN');self.assertEqual(receipt['charged'],c.RESERVATION)

    def test_T10_finalize_failure(self):
        code,r=self.lifecycle('final_write_failure');self.assertEqual(code,1)
        self.assertTrue((r.marker/'reservation.json').exists())
        self.assertFalse((Path(r.plan['out'])/'completion.json').exists())
        with self.assertRaises(FileExistsError):w.reserve_scope(r.anchor)

    def test_T11_output_race(self):
        out=self.root/'out';out.mkdir();budget=c.OutputBudget(out)
        with patch.object(budget,'used',return_value=56*1024**2):
            with self.assertRaises(RuntimeError):budget.write('x.json',{'a':1})
        budget.write('x.json',{'a':1})
        with self.assertRaises(FileExistsError):budget.write('x.json',{'a':2})

    def test_T12_update_order(self):
        order=[]
        def phase(name):
            counts=update_counts(self.job)
            self.assertEqual((counts['attempted'],counts['completed']),(1,0));order.append(name)
        c.update_once(self.job,'natural',1,lambda:phase('step'),lambda:phase('sync'),lambda:phase('verify'))
        self.assertEqual(order,['step','sync','verify'])
        self.assertEqual(update_counts(self.job)['completed'],1)

    def test_T13_update_sync_failure(self):
        def bad():raise RuntimeError('async failure')
        with self.assertRaises(RuntimeError):c.update_once(self.job,'natural',1,lambda:None,bad,lambda:None)
        counts=update_counts(self.job);self.assertEqual((counts['attempted'],counts['completed'],counts['uncertain']),(1,0,1))
        with self.assertRaises(RuntimeError):c.update_once(self.job,'synthetic',2,lambda:self.fail('next update'),lambda:None,lambda:None)

    def test_T14_update_cap(self):
        with self.assertRaises(RuntimeError):c.validate_update_values([2],1.)
        with self.assertRaises(RuntimeError):c.validate_update_values([1],float('inf'))
        with self.assertRaises(RuntimeError):c.update_once(self.job,'x',3,lambda:self.fail('third'),lambda:None,lambda:None)
        with patch.object(c,'update_counts',return_value=dict(attempted=1,completed=2,uncertain=0,legacy_reserved=0)):
            with self.assertRaises(RuntimeError):c.update_once(self.job,'x',2,lambda:None,lambda:None,lambda:None)

    def test_T15_finite_contract(self):
        for name in ('loss','state','gradient','predictor','optimizer','delta'):
            for value in (float('nan'),float('inf')):
                with self.assertRaises(RuntimeError):c.finite_tree({name:[value]})
        c.finite_tree({'loss':[0.,1.], 'optimizer':{'step':1}})

    def test_T16_fixed_case_membership(self):
        a,s=self.cases();self.assertEqual(len(c.validate_cases(a,s)),4)
        s['stage2_tune_episodes']=[78]
        with self.assertRaises(RuntimeError):c.validate_cases(a,s)
        a,s=self.cases();a['natural']['N_78_10']['raw_anchor']=15
        with self.assertRaises(RuntimeError):c.validate_cases(a,s)

    def test_T17_access_inventory(self):
        a,_=self.cases();rows=[]
        for kind in ('natural','synthetic'):
            for row in a[kind].values():a['hashes'][row['path']]='h';rows.append(dict(kind=kind,path=row['path'],sha256='h'))
        c.check_access(rows,a)
        for bad in (rows[:-1],rows+rows[:1],rows+[dict(kind='global',path='x',sha256='h')]):
            with self.assertRaises(RuntimeError):c.check_access(bad,a)

    def test_T18_mandatory_pins(self):
        for key in c.DOCUMENTS:
            p=self.manifest();del p['documents'][key]
            with self.assertRaises(RuntimeError):self.valid(p)
        p=self.manifest();del p['source_pins']['repro/physics/natural_v2_guard.py']
        with self.assertRaises(RuntimeError):self.valid(p)

    def ledger(self):
        # Positive end-to-end receipt mapping with synthetic (not historical research) documents.
        receipt=self.root/'receipt.json';receipt.write_text(json.dumps(dict(id='job1',state='COMPLETE',cost=10,host='host1',invocation='inv1',reaped=True)))
        history=dict(invocations_used=4,invocation_cap=4,historical_reserved_windows_seconds=6000,old_deadlines_expired=True,charged_seconds=123)
        source=self.root/'source.json';source.write_text(json.dumps({'proposal':{'historical_budget_ledger':dict(known_total_gpu_seconds=0,completed_v2_preflight_gpu_seconds=0,completed_preflight_updates=0,v2_cpu_conservative_cross_scope_remaining_upper_seconds=900,historical_preflight_bytes_snapshot=0,historical_research_bytes_snapshot=0,fixture_separate_ledger=history)}}))
        costs={}
        for key in c.COST_KEYS:
            unit='bytes' if key.endswith('bytes') else 'updates' if key.endswith('updates') else 'seconds'
            costs[key]=dict(charged=10,floor=0,reservation=10,proofs=[dict(path=str(receipt),pointer='/cost',unit=unit,semantics='cumulative',coverage=['cost-event-1'])])
        costs['preflight_updates']['charged']=0;costs['preflight_updates']['proofs']=[];costs['preflight_updates']['reservation']=1;costs['preflight_updates']['charged']=1
        ledger=dict(unresolved_jobs=[],integration_attempts_used=0,inventory=[dict(root=str(self.root),patterns=['receipt.json'])],
            evidence={str(receipt):dict(sha256=c.digest(receipt),owner='job1',identity_pointer='/id',identity_value='job1')},
            jobs=[dict(id='job1',state='COMPLETE',host_id='host1',invocation_id='inv1',evidence=[str(receipt)],terminal=dict(path=str(receipt),pointer='/state',value='COMPLETE'),
                exit_proof=dict(kind='reaped',path=str(receipt),job_pointer='/id',host_pointer='/host',invocation_pointer='/invocation',observation_pointer='/reaped'),costs=costs)],fixture_history=history)
        path=self.root/'ledger.json'
        plan=dict(ledger=dict(path=str(path),sha256='',historical_floor=dict.fromkeys(c.COST_KEYS,0),fixture_history=history),documents={'source_plan':dict(path=str(source),sha256=c.digest(source))})
        return plan,ledger

    def reconcile(self,p,l):
        path=Path(p['ledger']['path']);path.write_text(json.dumps(l));p['ledger']['sha256']=c.digest(path)
        return c.reconcile_ledger(p)

    def test_T19_ledger_good(self):
        p,l=self.ledger();first=next(iter(l['evidence']));second=self.root/'timing.json'
        second.write_bytes(Path(first).read_bytes());l['inventory'][0]['patterns'].append('timing.json')
        l['evidence'][str(second)]=dict(l['evidence'][first],sha256=c.digest(second))
        l['jobs'][0]['evidence'].append(str(second))
        proof=dict(l['jobs'][0]['costs']['research_gpu_seconds']['proofs'][0],path=str(second))
        l['jobs'][0]['costs']['research_gpu_seconds']['proofs'].append(proof)
        self.assertEqual(self.reconcile(p,l)['totals']['research_gpu_seconds'],10)

    def test_T20_ledger_missing(self):
        for field in ('exit_proof','host_id','invocation_id'):
            p,l=self.ledger();del l['jobs'][0][field]
            with self.assertRaises((RuntimeError,KeyError)):self.reconcile(p,l)
        for pointer in ('/state','/cost'):
            p,l=self.ledger();l['jobs'][0]['exit_proof']['observation_pointer']=pointer
            with self.assertRaises(RuntimeError):self.reconcile(p,l)
        p,l=self.ledger();l['jobs'][0]['host_id']='different-host'
        with self.assertRaises(RuntimeError):self.reconcile(p,l)
        p,l=self.ledger();l['jobs'][0]['evidence']=[]
        with self.assertRaises(RuntimeError):self.reconcile(p,l)
        p,l=self.ledger();l['inventory'][0]['patterns']=[]
        with self.assertRaises(RuntimeError):self.reconcile(p,l)

    def test_T21_ledger_undercharge(self):
        p,l=self.ledger();cost=l['jobs'][0]['costs']['research_gpu_seconds']
        cost['proofs'].append(dict(cost['proofs'][0],semantics='increment',event_id='cost-event-1'))
        with self.assertRaises(RuntimeError):self.reconcile(p,l)
        for value in (0,-1,float('nan')):
            p,l=self.ledger();l['jobs'][0]['costs']['research_gpu_seconds']['charged']=value
            with self.assertRaises((RuntimeError,ValueError)):self.reconcile(p,l)

    def test_T22_ledger_budget(self):
        p,l=self.ledger();l['jobs'][0]['costs']['research_gpu_seconds']['charged']=3600
        with self.assertRaises(RuntimeError):self.reconcile(p,l)
        p,l=self.ledger();l['unresolved_jobs']=['running']
        with self.assertRaises(RuntimeError):self.reconcile(p,l)

    def test_T23_required_nine(self):
        rows=[dict(id=k,expected='finite',actual=1,tolerance=0,evidence=['fixture'],passed=True) for k in c.REQUIRED]
        c.validate_checks(rows)
        for bad in ([],rows[:-1],rows+rows[:1]):
            with self.assertRaises(RuntimeError):c.validate_checks(bad)
        unknown=copy.deepcopy(rows);unknown[0]['actual']='UNKNOWN'
        with self.assertRaises(RuntimeError):c.validate_checks(unknown)
        rows[0]['actual']=float('inf')
        with self.assertRaises(RuntimeError):c.validate_checks(rows)

    def test_T24_entrypoint_boundary(self):
        from natural_v2_guard import verify_execution_authorization
        called=[]
        with self.assertRaises(RuntimeError):admission_pipeline([],lambda *_:None,lambda:called.append(1))
        out=self.root/'auth';job=out/'jobs'/'inv';job.mkdir(parents=True)
        p=self.manifest();ctx=copy.deepcopy(p['host']['actual_context'])
        argv=['script','--root',p['root'],'--repo',p['repo'],'--out',str(out),'--stage','integration','--job',str(job)]
        ctx.update(out=str(out),job=str(job),argv=argv)
        planfile=self.root/'inert-plan.json';planfile.write_text('{}')
        request=dict(plan_sha256=c.digest(planfile),code_commit=ctx['code_commit'],argv=['python','-B']+argv,
            data_scope='inert fixture only',limits={k:0 for k in ('gpu_seconds','cpu_cores','ram_bytes','wall_seconds','disk_bytes','gpu_device_mib')},
            max_executions=1,max_retries=0,actual_context=ctx)
        # fixture=True cannot grant production authority; only supplied observer is inert.
        approval=dict(request,status='APPROVED',issuer='user',fixture=True)
        approvalfile=self.root/'inert-approval.json';approvalfile.write_text(json.dumps(approval))
        receipt=verify_execution_authorization(approval,request,allow_fixture=True)
        receipt.update(plan_path=str(planfile),approval_path=str(approvalfile),approval_sha256=c.digest(approvalfile))
        (job/'execution_authorization.json').write_text(json.dumps(receipt))
        (out/'run_lock.json').write_text(json.dumps(dict(code_commit=ctx['code_commit'])))
        observe=lambda v,a:dict(ctx,**v,argv=a)
        with self.assertRaises(RuntimeError):admission_pipeline(argv,observe,lambda:called.append(1))
        self.assertEqual(called,[])
        value,_=admission_pipeline(argv,observe,lambda:'inert',allow_fixture=True)
        self.assertEqual(value,'inert')
        with self.assertRaises(FileExistsError):admission_pipeline(argv,observe,lambda:called.append(1),allow_fixture=True)
        self.assertEqual(called,[])


if __name__=='__main__':
    unittest.main()
