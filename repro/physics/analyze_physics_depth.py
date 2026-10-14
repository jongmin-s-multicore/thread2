"""Separate global rollout error, local physical consistency, and rendering effects."""
import numpy as np
from intervention import read,save,sha,csvwrite,clearance
from analyze_intervention import aggregate,energy_terms,norm
from physics_depth import make_env,simulate,diagnose

def analyze(root,repo,out):
 assert read(out/'infer_complete.json')['weights_unchanged']
 cases={x['id']:x for x in read(out/'cases.json')};oracles={x['id']:x for x in read(out/'oracles.json')};rolls=read(out/'rollouts.json');params=read(out/'parameters.json');M,D,G=[np.array(params[k]) for k in ['M','D','G']]
 e=make_env(root,repo);metrics=[];selforacles=[];fine_max=0.
 for rollout in rolls:
  case=cases[rollout['id']];start=np.array(case['start']);prev=start;u=np.array(case['u']);prefix=True
  fixed_free=all(s['free'] for s in oracles[case['id']]['segments'])
  for prediction,oracle in zip(rollout['predictions'],oracles[case['id']]['segments']):
   h=prediction['horizon'];p=np.array(prediction['state']);gt=np.array(oracle['trace'][-1]);prefix&=oracle['free']
   if h==1:local=dict(trace=oracle['trace'],contacts=oracle['contacts'],clips=oracle['clips'])
   else:local=simulate(e,prev,u,(h-1)*.25)
   diag=diagnose(local['trace'],u,params,local['contacts'],local['clips']);localend=np.array(local['trace'][-1]);terms=energy_terms(prev,p,G@u,M,D)
   if diag['free']:fine_max=max(fine_max,abs(diag['fine_energy_residual']))
   r=dict(id=case['id'],family=case['family'],episode=case['episode'],split=case['split'],condition=case['condition'],variant=rollout['variant'],horizon=h,oracle_step_free=oracle['free'],oracle_prefix_free=prefix,oracle_fixed_h5_free=fixed_free,local_free=diag['free'],oracle_contacts=oracle['contacts'],oracle_clips=oracle['clips'],local_contacts=diag['contacts'],local_clips=diag['clips'],
    position_error=norm(p[:2]-gt[:2]),velocity_error=norm(p[2:]-gt[2:]),signed_vx_error=float(p[2]-gt[2]),signed_vy_error=float(p[3]-gt[3]),local_position_error=norm(p[:2]-localend[:2]),local_velocity_error=norm(p[2:]-localend[2:]),
    constant_velocity_position_error=norm(start[:2]+h*.25*start[2:]-gt[:2]),hold_velocity_error=norm(start[2:]-gt[2:]),endpoint_penetration=max(0.,-float(clearance(p[:2],params))),endpoint_overlap=int(clearance(p[:2],params)<0),predicted_speed=norm(p[2:]),
    kinetic_energy=float(.5*p[2:]@M@p[2:]),energy_excess=terms['necessary_excess'],energy_increase=int(terms['delta_k']>1e-8),energy_necessary_failure=int(terms['necessary_excess']>1e-8) if diag['free'] else None,kinematic_error=terms['kinematic_error'],state=p.tolist(),oracle_state=gt.tolist(),start=prev.tolist(),u=u.tolist())
   metrics.append(r);selforacles.append(dict(id=case['id'],variant=rollout['variant'],horizon=h,trace=local['trace'],**diag));prev=p
 e.close()
 lookup={(r['id'],r['variant'],r['horizon']):r for r in metrics};responses=[]
 for r in metrics:
  if r['family']!='recorded' or r['condition']=='zero':continue
  ep,h=r['episode'],r['horizon'];z=lookup[f'C_{ep}_zero','original',h];dv=np.array(r['state'][2:])-z['state'][2:];de=np.array(r['oracle_state'][2:])-z['oracle_state'][2:];direction=int(r['condition'].split('_d')[1]);flip=lookup[f'C_{ep}_r0.2_d{(direction+4)%8}','original',h];wrong=lookup[f'C_{ep}_r0.2_d{(direction+2)%8}','original',h]
  err=norm(dv-de);angle=float(np.degrees(np.arccos(np.clip(dv@de/(norm(dv)*norm(de)),-1,1)))) if norm(dv)>1e-6 and norm(de)>1e-6 else None
  responses.append(dict(episode=ep,split=r['split'],horizon=h,condition=r['condition'],pair_prefix_free=r['oracle_prefix_free'] and z['oracle_prefix_free'],response_error=err,angle_degrees=angle,fixed_zero_error=norm(de),sign_flip_error=norm(np.array(flip['state'][2:])-z['state'][2:]-de),wrong_branch_error=norm(np.array(wrong['state'][2:])-z['state'][2:]-de)))
 render=[];meta={r['episode']:r for r in read(out/'render_contexts.json')}
 for r in read(out/'render_predictions.json'):
  ep=r['episode'];original=np.array(r['original']);gt=np.array(r['oracle']);rr=dict(episode=ep,split=r['split'],original_velocity_error=norm(original[2:]-gt[2:]),original_vy_bias=float(original[3]-gt[3]))
  for variant in ['raw','matched']:
   v=np.array(r[variant]);rr.update({f'{variant}_velocity_difference':norm(v[2:]-original[2:]),f'{variant}_position_difference':norm(v[:2]-original[:2]),f'{variant}_velocity_error':norm(v[2:]-gt[2:]),f'{variant}_vy_bias':float(v[3]-gt[3]),f'{variant}_pixel_mae':meta[ep][f'{variant}_pixel_mae']})
  rr['paired_matching_reduction']=rr['raw_velocity_difference']-rr['matched_velocity_difference'];render.append(rr)
 keys=['position_error','velocity_error','signed_vx_error','signed_vy_error','local_position_error','local_velocity_error','hold_velocity_error','constant_velocity_position_error','endpoint_overlap','predicted_speed','kinetic_energy','energy_necessary_failure','kinematic_error']
 summaries=[]
 for split in sorted({r['split'] for r in metrics}):
  for family,variant in [('recorded','original'),('static','raw'),('static','matched')]:
   for h in range(1,6):
    for subset,mask in [('all',lambda r:True),('oracle_prefix_free',lambda r:r['oracle_prefix_free']),('fixed_h5_free',lambda r:r['oracle_fixed_h5_free']),('local_free',lambda r:r['local_free'])]:
     rows=[r for r in metrics if (r['split'],r['family'],r['variant'],r['horizon'])==(split,family,variant,h) and mask(r)]
     summaries.append(dict(split=split,family=family,variant=variant,horizon=h,subset=subset,episodes=len({r['episode'] for r in rows}),rows=len(rows),metrics={k:aggregate(rows,k) for k in keys}))
 resp=[]
 for split in ['calibration_fit','development']:
  for h in range(1,6):
   for subset in ['all','pair_prefix_free']:
    rows=[r for r in responses if r['split']==split and r['horizon']==h and (subset=='all' or r['pair_prefix_free'])]
    resp.append(dict(split=split,horizon=h,subset=subset,metrics={k:aggregate(rows,k) for k in ['response_error','angle_degrees','fixed_zero_error','sign_flip_error','wrong_branch_error']}))
 paired=[]
 for r in metrics:
  if r['family']=='static' and r['variant']=='raw':
   other=lookup[r['id'],'matched',r['horizon']];paired.append(dict(episode=r['episode'],split=r['split'],horizon=r['horizon'],speed_reduction=r['predicted_speed']-other['predicted_speed'],position_error_reduction=r['position_error']-other['position_error'],velocity_difference=norm(np.array(r['state'][2:])-other['state'][2:])))
 pairs=[dict(split=g,horizon=h,metrics={k:aggregate([r for r in paired if r['split']==g and r['horizon']==h],k) for k in ['speed_reduction','position_error_reduction','velocity_difference']}) for g in ['calibration_fit','development'] for h in range(1,6)]
 changes=[]
 for family,variant in [('recorded','original'),('static','raw'),('static','matched')]:
  for g in ['calibration_fit','development']:
   for subset in ['all','fixed_h5_free']:
    rows=[]
    for r in metrics:
     if (r['family'],r['variant'],r['split'],r['horizon'])!=(family,variant,g,5) or (subset=='fixed_h5_free' and not r['oracle_fixed_h5_free']):continue
     one=lookup[r['id'],variant,1];rows.append(dict(episode=r['episode'],velocity_error_growth=r['velocity_error']-one['velocity_error'],position_error_growth=r['position_error']-one['position_error']))
    changes.append(dict(family=family,variant=variant,split=g,subset=subset,metrics={k:aggregate(rows,k) for k in ['velocity_error_growth','position_error_growth']}))
 assert all(sha(f)==h for f,h in read(out/'manifest.json')['sources'].items())
 checks=dict(pass_all=True,source_hashes_unchanged=True,weights_unchanged=True,h1_cache_max_abs=read(out/'infer_complete.json')['h1_cache_max_abs'],public_rollout_max_abs=max(x['max_abs'] for x in read(out/'infer_complete.json')['public_rollout_checks']),free_local_energy_max_abs=fine_max,rows=len(metrics),no_training=True,sealed_test_opened=False)
 save(out/'metrics.json',metrics);save(out/'self_oracles.json',selforacles);save(out/'responses.json',responses);save(out/'render_metrics.json',render);save(out/'summary.json',dict(rollout=summaries,response=resp,render={g:{k:aggregate([r for r in render if r['split']==g],k) for k in render[0] if k not in ['episode','split']} for g in sorted({r['split'] for r in render})},static_paired=pairs,horizon_paired=changes));save(out/'checks.json',checks)
 csvwrite(out/'metrics.csv',metrics)
 save(out/'analysis_manifest.json',dict(script_sha256=sha(__file__),sources={n:sha(out/n) for n in ['cases.json','oracles.json','rollouts.json','render_predictions.json']},bootstrap_count=10000,bootstrap_seed=0,unit='episode',new_training=False))
 save(out/'completion.json',dict(status='PHYSICS_DEPTH_COMPLETE_DEVELOPMENT',phase_d='MULTISTEP_DEVELOPMENT_COMPLETE',heldout_evaluation='NOT_RUN',new_training=False,new_physics_loss=False));print(checks,flush=True)
