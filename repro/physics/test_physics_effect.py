"""[repro] Only changed orchestration paths; inert callbacks, no model/data load."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import physics_effect_control as c


class ChangedPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
    def test_01_update_sync_finite_frozen(self):
        order=[]
        c.updated(self.out,1,120,lambda:order.append('step'),lambda:order.append('sync'),lambda:order.extend(['finite','frozen']))
        self.assertEqual(order,['step','sync','finite','frozen']);self.assertTrue((self.out/'update_0001_complete.json').exists())
        for failing in ('step','sync','verify'):
            d=self.out/failing;d.mkdir()
            def bad():raise RuntimeError('injected')
            with self.assertRaises(RuntimeError):c.updated(d,1,120,bad if failing=='step' else lambda:None,bad if failing=='sync' else lambda:None,bad if failing=='verify' else lambda:None)
            self.assertFalse((d/'update_0001_complete.json').exists())
    def test_02_admission_mismatch_duplicate(self):
        f=self.out/'manifest.json';c.save(f,{})
        host={'system':'Linux','repo':'repo','hostname_sha256':'host','branch':'own','commit':'commit','dirty':False}
        m={'manifest_path':str(f),'host':host,'repo':'repo','documents':{},'seed':0,'max_updates':3120,'max_gpu_jobs':15,'max_train_jobs':8,'jobs':{'cpu-regression':{}},'out':str(self.out)}
        with patch.object(c,'observe',return_value=host),patch.object(c,'validate_sources'):
            c.admit(m,'cpu-regression',c.sha(f))
            for job,h in [('invalid',c.sha(f)),('cpu-regression','wrong')]:
                with self.assertRaises(RuntimeError):c.admit(m,job,h)
            wrong=copy.deepcopy(m);wrong['host']['commit']='wrong'
            with self.assertRaises(RuntimeError):c.admit(wrong,'cpu-regression',c.sha(f))
            d=self.out/'jobs/cpu-regression';d.mkdir(parents=True);c.save(d/'attempt.json',{'candidate':'commit','manifest_sha256':c.sha(f),'parents':{'wrong':'choice'}})
            with self.assertRaises(RuntimeError):c.admit(m,'cpu-regression',c.sha(f))
            with self.assertRaises(RuntimeError):c.admit(m,'cpu-regression',c.sha(f),claimed=True)
    def test_03_eval_lock(self):
        for access in ('row','cache','encoding'):
            with self.assertRaises(RuntimeError):c.row_gate('evaluation',{4},4,False)
            with self.assertRaises(FileNotFoundError):c.gate(self.out,'encode-eval')
        c.row_gate('evaluation',{4},4,True)
        with self.assertRaises(RuntimeError):c.row_gate('sealed',{4},4,True)
    def test_04_exposure_grid_cap(self):
        exposure=[['a','b']]*480;h=c.exposure_check(exposure,480)
        for arm in ('B','P','S'):self.assertEqual(c.exposure_check(exposure,480,reference=h),h)
        with self.assertRaises(RuntimeError):c.exposure_check([['b','a']]*480,480,reference=h)
        for change in ({'seed':1},{'strength':2.}):
            with self.assertRaises(RuntimeError):c.exposure_check(exposure,480,**change)
        self.assertEqual(sum(c.TRAIN.values()),3120)
        with self.assertRaises(RuntimeError):c.updated(self.out,481,480,lambda:None,lambda:None,lambda:None)
    def test_05_cost_unknown_dedup_forecast(self):
        a={'identity':'a','gpu':True,'seconds':10.,'complete':True};b={'identity':'failed','gpu':True,'seconds':None,'upper_seconds':20.,'complete':False}
        v=c.merge_cost([a,a,b]);self.assertEqual(v['charged_gpu_seconds'],30);self.assertEqual(v['unknown'],['failed'])
        bad=c.merge_cost([{'identity':'missing','gpu':True,'seconds':None,'complete':False}]);self.assertIsNone(bad['charged_gpu_seconds'])
        with self.assertRaises(RuntimeError):c.forecast(bad,{'future-eval':1})
        with self.assertRaises(RuntimeError):c.forecast(v,{'future-train':1000,'future-encoding':1000,'future-eval':1})
        self.assertEqual(c.forecast(v,{'future-eval':10})['forecast'],140)
    def test_06_masks_keep_batch_count_reused_math(self):
        from natural_v2_core import schedule
        cases=[{'id':f'{f}{i}','schedule_family':f} for f in ('N','I','L','S') for i in range(2)]
        for mixture in ('R06','R12'):
            exposure=schedule(cases,mixture)
            for mask in ('zero','active'):self.assertEqual(len(exposure),480);self.assertEqual(sum(len(b) for b in exposure),960)
    def test_07_bootstrap_pairing(self):
        x={'metric':{'episodes':{'1':1.,'2':2.},'n_episodes':2,'n_samples':4,'mean':1.5}}
        paired=c.checked_paired(x,x);self.assertEqual(paired['metric']['mean_delta'],0);self.assertEqual(paired['metric']['ci95'],[0.,0.])
        self.assertEqual(c.bootstrap_ci(x)['metric']['ci_scope'],'report_only_episode_bootstrap_seed0_10000')
        empty={'metric':{'episodes':{},'n_episodes':0,'n_samples':0,'mean':None}}
        self.assertIsNone(c.bootstrap_ci(empty)['metric']['ci95']);self.assertIsNone(c.checked_paired(empty,empty)['metric']['ci95'])
        bad=copy.deepcopy(x);bad['metric']['episodes']['1']=float('nan')
        with self.assertRaises(RuntimeError):c.bootstrap_ci(bad)
        with self.assertRaises(RuntimeError):c.checked_paired(x,bad)


class CorrectionTests(unittest.TestCase):
    """[repro] Only H01-H04; do not rerun the earlier scientific or seven tests."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.out=Path(self.tmp.name)
        self.m={'out':str(self.out),'historical_receipts':[], 'prior_cpu_seconds':0,
                'forecast_work':{'stage1-I-F':{'step':120,'roll':56},'stage1-I-N':{'step':120,'roll':56},'B-R06':{'step':480,'roll':828}},
                'forecast_rates':{'step':.2552428860217333,'roll':.06564707984216511,'enc':.02736684400588274},
                'jobs':{'stage1-select':{'seconds':120},'stage1-I-F':{'seconds':180}}}
    def test_H01_component_sum_first8_nextlaunch_stage1_only(self):
        forecast=c.component_forecast(self.m['forecast_work'],self.m['forecast_rates'],[{'step':.5}])
        self.assertAlmostEqual(forecast['B-R06'],294.3557821093127)
        self.assertNotEqual(forecast['B-R06'],240)
        first=c.resource_state(self.m,observations=[{'step':.5}])[1]
        job=self.out/'jobs/stage1-I-F';job.mkdir(parents=True)
        c.save(job/'timing.json',{'max_step_seconds':.5})
        following=c.resource_state(self.m)[1]
        self.assertEqual(first,following)
        self.m['historical_receipts']=[{'identity':'past','gpu':True,'seconds':3000.,'complete':True}]
        _,fc,_,_=c.resource_state(self.m)
        self.assertEqual(fc['scope'],'STAGE1_ONLY');self.assertTrue(fc['stage2_locked'])
        self.assertEqual(set(fc['remaining_raw_by_job']),{'stage1-I-F','stage1-I-N'})
        c.preserve_budget_lock(self.out,fc)
        c.gate(self.out,'stage1-I-N');c.gate(self.out,'stage1-select')
        with self.assertRaisesRegex(RuntimeError,'stage2 budget locked'):c.gate(self.out,'B-R06')
        # Lower subsequent rates cannot unlock a previously closed stage2.
        self.m['historical_receipts']=[]
        self.assertTrue(c.resource_state(self.m)[1]['stage2_locked'])
    def test_H02_child_exit_is_not_job_success_and_cost_preserved(self):
        for case in ('missing','incomplete','post_error','supervisor_reason','complete'):
            with self.subTest(case=case):
                root=self.out/case;job=root/'jobs/stage1-select';job.mkdir(parents=True)
                m={**self.m,'out':str(root)}
                c.save(job/'attempt.json',{'whole_ceiling_seconds':120})
                if case!='missing':c.save(job/'result.json',{'job':'stage1-select','status':'INCOMPLETE' if case=='incomplete' else 'COMPLETE'})
                rec={'returncode':0,'reaped':True,'reason':'timeout' if case=='supervisor_reason' else None}
                def post():
                    if case=='post_error':raise RuntimeError('post-admission mismatch')
                success=False;failure=None
                try:
                    c.terminal_checks(rec,'stage1-select',lambda:c.read(job/'result.json'),post);success=True
                except (RuntimeError,FileNotFoundError) as exc:failure=str(exc)
                receipt=c.terminal_receipt(str(job/'attempt.json'),False,3.,120,rec,success,failure)
                c.save(job/'receipt.json',receipt)
                _,_,blocked,left=c.resource_state(m)
                self.assertEqual(receipt['seconds'],3.);self.assertEqual(left,897.)
                if case=='complete':c.require_ready(blocked);self.assertTrue(receipt['success'])
                else:
                    with self.assertRaises(RuntimeError):c.require_ready(blocked)
                    self.assertFalse(receipt['success']);self.assertTrue(receipt['complete'])
        unknown=c.terminal_receipt('lost',True,3.,120,None,False,'lost')
        self.assertEqual(c.merge_cost([unknown])['charged_gpu_seconds'],120)
        self.assertEqual(c.merge_cost([unknown])['known_lower_bound'],3)
    def test_H03_actual_registry_manifest_review_binding(self):
        registry=self.out/'registry.json';activation=self.out/'activation.json';review=self.out/'review.json';manifest=self.out/'manifest.json'
        reg={'revision':3,'phase':'IMPLEMENTATION_REVIEWED','protocol':{'sha256':'protocol'},'registry_paths':{'mac':'mac','server':str(registry)},
             'code_review':{'implementation_review_status':'APPROVED_STAGE1','reviewed_implementation_commit':'commit','approved_comparison_stages':sorted(c.STAGE1)},
             'execution_scope':{'comparison_training_allowed':True,'comparison_evaluation_allowed':True}}
        c.save(registry,reg);h=c.sha(registry)
        host={'system':'Linux','repo':'repo','hostname_sha256':'host','branch':'own','commit':'commit','dirty':False}
        m={**self.m,'manifest_path':str(manifest),'host':host,'repo':'repo','seed':0,'max_updates':3120,'max_gpu_jobs':15,'max_train_jobs':8,
           'documents':{str(registry):h},'registry_path':str(registry),'registry_sha256':h,'previous_registry_sha256':'old',
           'protocol_sha256':'protocol','registry_activation_path':str(activation),'scientific_review_path':str(review),'input_hashes':{},'file_pins':{'hub_sources':[],'hub_weights':[]}}
        c.save(manifest,m);mh=c.sha(manifest)
        rev={'approved':True,'candidate_code_commit':'commit','manifest_sha256':mh,'registry_sha256':h,'protocol_sha256':'protocol','allowed_jobs':sorted(c.STAGE1)}
        act={'candidate_code_commit':'commit','old_sha256':'old','mac_registry_sha256':h,'server_registry_sha256':h,'mac_path':'mac','server_path':str(registry)}
        c.save(review,rev);c.save(activation,act)
        with patch.object(c,'observe',return_value=host),patch.object(c,'validate_sources'):
            c.admit(m,'stage1-I-F',mh)
        for field,value in [('revision',2),('phase','PROTOCOL_FINALIZED')]:
            bad=copy.deepcopy(reg);bad[field]=value
            with self.assertRaises(RuntimeError):c.scientific_binding(m,'stage1-I-F',mh,rev,bad,h,act)
        bad=copy.deepcopy(reg);bad['code_review']['reviewed_implementation_commit']='other'
        with self.assertRaises(RuntimeError):c.scientific_binding(m,'stage1-I-F',mh,rev,bad,h,act)
        for job,actual in [('B-R06',h),('stage1-I-F','changed_bytes')]:
            with self.assertRaises(RuntimeError):c.scientific_binding(m,job,mh,rev,reg,actual,act)
        for key,value in [('candidate_code_commit','other'),('manifest_sha256','other'),('registry_sha256','other'),('allowed_jobs',['stage1-I-F'])]:
            bad={**rev,key:value}
            with self.assertRaises(RuntimeError):c.scientific_binding(m,'stage1-I-F',mh,bad,reg,h,act)
        with self.assertRaises(RuntimeError):c.scientific_binding(m,'stage1-I-F',mh,rev,reg,h,{**act,'mac_registry_sha256':'other'})
    def test_H04_cpu_remaining_cap_and_pending_attempt(self):
        self.m['prior_cpu_seconds']=890
        _,fc,blocked,left=c.resource_state(self.m)
        self.assertEqual(left,10);c.require_ready(blocked)
        self.assertEqual(c.job_wall(self.m,'stage1-select',fc,left,0),8)
        self.m['prior_cpu_seconds']=900
        self.assertEqual(c.resource_state(self.m)[3],0)
        with self.assertRaises(RuntimeError):c.job_wall(self.m,'stage1-select',fc,0,0)
        self.m['prior_cpu_seconds']=700
        job=self.out/'jobs/stage1-select';job.mkdir(parents=True)
        c.save(job/'attempt.json',{'whole_ceiling_seconds':120})
        _,_,blocked,left=c.resource_state(self.m)
        self.assertEqual(left,80)
        with self.assertRaises(RuntimeError):c.require_ready(blocked)
        self.assertEqual(c.cpu_remaining(700,[{'identity':'pending','gpu':False,'seconds':None,'upper_seconds':120,'complete':False}]),80)

    def test_H05_inherited_file_limit_never_raised(self):
        class Limits:
            RLIMIT_FSIZE=1;RLIM_INFINITY=-1
            def __init__(self,hard):self.hard=hard;self.applied=None
            def getrlimit(self,key):return (self.hard,self.hard)
            def setrlimit(self,key,value):
                if self.hard!=-1 and value[1]>self.hard:raise ValueError('raising inherited hard limit')
                self.applied=value
        for hard,expected in [(8*1024**2,8*1024**2),(-1,512*1024**2),(1024**3,512*1024**2)]:
            fake=Limits(hard)
            self.assertEqual(c.apply_file_limit(fake),expected)
            self.assertEqual(fake.applied,(expected,expected))
        with self.assertRaises(RuntimeError):c.apply_file_limit(Limits(0))
    def test_D1_first8_counts_running_job_once(self):
        # [repro] stage1-I-F is running (attempt.json, no receipt) at its first-8 check; it must not be charged at its ceiling AND for all its work.
        m={**self.m,'historical_receipts':[{'identity':'past','gpu':True,'seconds':2800.,'complete':True}]}
        job=self.out/'jobs/stage1-I-F';job.mkdir(parents=True);c.save(job/'attempt.json',{'whole_ceiling_seconds':180})
        observed=[{'step':.34}]
        legacy=c.resource_state(m,observations=observed)[1]
        self.assertEqual(legacy['scope'],'STAGE1_ONLY')
        first=c.resource_state(m,observations=observed,running={'job':'stage1-I-F','elapsed':20.,'completed_steps':8})[1]
        self.assertEqual(first['scope'],'FULL');self.assertFalse(first['stage2_locked']);self.assertEqual(first['used'],2820.)
        self.assertLessEqual(first['used']+first['forecast'],3600)
        c.preserve_budget_lock(self.out,first);self.assertFalse((self.out/'stage2_budget_lock.json').exists())
        c.save(job/'timing.json',{'max_step_seconds':.34})
        c.save(job/'receipt.json',{'identity':str(job/'attempt.json'),'gpu':True,'seconds':60.,'upper_seconds':180,'complete':True,'success':True})
        following=c.resource_state(m)[1]
        self.assertEqual(following['scope'],'FULL');self.assertFalse(following['stage2_locked'])
        self.assertLessEqual(following['used']+following['forecast'],3600)
