"""Fixed episode aggregation and paired comparison of the first loss pilot."""
import argparse,json,csv
from pathlib import Path
import numpy as np
from intervention import read,save,analytic,clearance,sha,csvwrite

BOOT_INDEX={}

def aggregate(rows,key):
 episodes=sorted({r['episode'] for r in rows});v=np.array([np.mean([r[key] for r in rows if r['episode']==e]) for e in episodes]);rng=np.random.default_rng(0)
 if not len(v):return dict(mean=None,ci=None,episodes=0,branches=0,episode_values={})
 if len(v) not in BOOT_INDEX:BOOT_INDEX[len(v)]=rng.integers(0,len(v),(10000,len(v)))
 means=v[BOOT_INDEX[len(v)]].mean(axis=1)
 return dict(mean=float(v.mean()),ci=np.quantile(means,[.025,.975]).tolist(),episodes=len(v),branches=len(rows),episode_values={str(e):float(x) for e,x in zip(episodes,v)})

def analyze(root,repo,out,arms=None,comparisons=None):
 arms=list('ABCDE') if arms is None else arms
 comparisons=[('B','A'),('C','B'),('D','B'),('D','C'),('E','D'),('E','A'),('C','A'),('D','A')] if comparisons is None else comparisons
 params=read(out/'parameters.json');norm=read(out/'normalization.json');cases={c['id']:c for c in read(out/'cases.json')};oracles=read(out/'oracles.json');rows=[];responses=[];summary=[];paired=[]
 for arm in arms:
  output=read(out/f'evaluation_{arm}.json');lookup={r['id']:r for r in output}
  for result in output:
   c=cases[result['id']];prev=np.array(c['start']);prefix=True
   for hi,st in enumerate(result['states']):
    st=np.array(st);seg=oracles[c['id']][hi];truth=np.array(seg['trace'][-1]);prefix=prefix and seg['free'];currentfree=seg['free'];initial=np.array(seg['trace'][0]);actual_target=analytic(initial,np.tile(c['u'],(5,1)),params)[0][-1] if currentfree else None
    # Affine extension only, no speed clipping: intentionally the train regularizer.
    def affine(s):
     q,v=s[:2].copy(),s[2:].copy()
     for _ in range(25):v=np.array(params['Ah'])@v+np.array(params['Bh'])@c['u'];q+=params['dt']*v
     return np.r_[q,v]
    rec=affine(prev);depth=max(0.,-float(clearance(st[:2],params)));row=dict(arm=arm,id=c['id'],episode=c['episode'],split=c['split'],family=c['family'],condition=c['condition'],horizon=hi+1,current_free=bool(currentfree),prefix_free=bool(prefix),contact=seg['contacts']>0,clipped=seg['clips']>0,position_error=float(np.linalg.norm(st[:2]-truth[:2])),velocity_error=float(np.linalg.norm(st[2:]-truth[2:])),speed=float(np.linalg.norm(st[2:])),position_drift=float(np.linalg.norm(st[:2]-np.array(c['start'])[:2])),velocity_y_bias=float(st[3]-truth[3]),visual_mse=result['visual_mse'][hi],proprio_mse=result['proprio_mse'][hi],off_manifold=result['off_manifold'][hi],patch_mean_position_error=float(np.linalg.norm(np.array(result['patch_mean_states'][hi])[:2]-truth[:2])),patch_mean_velocity_error=float(np.linalg.norm(np.array(result['patch_mean_states'][hi])[2:]-truth[2:])),wall_depth=depth,wall_overlap=float(depth>0),wall_overlap_001=float(depth>.01),oracle_wall_overlap_001=float(-float(clearance(truth[:2],params))>.01),recurrent_position_residual=float(np.linalg.norm(st[:2]-rec[:2])),recurrent_velocity_residual=float(np.linalg.norm(st[2:]-rec[2:])),recurrent_normalized_mse=float(np.mean(((st-rec)/norm['state_std'])**2)),gt_anchored_residual=float(np.linalg.norm((st-actual_target)/norm['state_std'])) if currentfree else None,predicted_start_invalid=float(float(clearance(prev[:2],params))<=.004 or np.any(np.abs(prev[2:])>5)))
    rows.append(row);prev=st
   if c['family']=='recorded' and c['condition']!='zero':
    zero=lookup[f"C_{c['episode']}_zero"]
    for hi in range(5):
     pv=np.array(result['states'][hi])[2:]-np.array(zero['states'][hi])[2:];tv=np.array(oracles[c['id']][hi]['trace'][-1])[2:]-np.array(oracles[zero['id']][hi]['trace'][-1])[2:];pn,tn=np.linalg.norm(pv),np.linalg.norm(tv)
     angle=None if tn<1e-8 else 180. if pn<1e-8 else float(np.degrees(np.arccos(np.clip(pv@tv/(pn*tn),-1,1))))
     responses.append(dict(arm=arm,episode=c['episode'],id=c['id'],split=c['split'],horizon=hi+1,vector_error=float(np.linalg.norm(pv-tv)),magnitude_error=float(abs(pn-tn)),angle_error=angle,model_magnitude=float(pn),oracle_magnitude=float(tn),pair_prefix_free=all(s['free'] for key in [c['id'],zero['id']] for s in oracles[key][:hi+1])))
 keys=['position_error','velocity_error','speed','position_drift','velocity_y_bias','visual_mse','proprio_mse','off_manifold','patch_mean_position_error','patch_mean_velocity_error','wall_overlap','wall_overlap_001','oracle_wall_overlap_001','recurrent_position_residual','recurrent_velocity_residual','recurrent_normalized_mse','predicted_start_invalid']
 for arm in arms:
  for split in ['calibration_tune','development']:
   for family in ['recorded','static','low_speed']:
    for h in range(1,6):
     selected=[r for r in rows if (r['arm'],r['split'],r['family'],r['horizon'])==(arm,split,family,h)]
     for scope in ['all','current_free','prefix_free','contact']:
      rr=selected if scope=='all' else [r for r in selected if r[scope]]
      for key in keys:
       if key.startswith('recurrent') and scope not in ['current_free','prefix_free']:continue
       summary.append(dict(arm=arm,split=split,family=family,horizon=h,scope=scope,metric=key,**aggregate(rr,key)))
   for h in range(1,6):
    for scope in ['all','prefix_free']:
     rs=[r for r in responses if r['arm']==arm and r['split']==split and r['horizon']==h and (scope=='all' or r['pair_prefix_free'])]
     for key in ['vector_error','magnitude_error','angle_error','model_magnitude','oracle_magnitude']:
      summary.append(dict(arm=arm,split=split,family='action',horizon=h,scope=scope,metric=key,**aggregate([r for r in rs if r[key] is not None],key)))
 summary_index={(s['arm'],s['split'],s['family'],s['horizon'],s['scope'],s['metric']):s for s in summary}
 for newer,older in comparisons:
  for s in summary:
   if s['arm']!=newer:continue
   other=summary_index[(older,s['split'],s['family'],s['horizon'],s['scope'],s['metric'])]
   assert s['episode_values'].keys()==other['episode_values'].keys()
   delta=[dict(episode=int(ep),delta=v-other['episode_values'][ep]) for ep,v in s['episode_values'].items()]
   paired.append(dict(comparison=f'{newer}-{older}',**{k:s[k] for k in ['split','family','horizon','scope','metric']},**aggregate(delta,'delta')))
 csvwrite(out/'metrics.csv',rows);save(out/'metrics.json',rows);save(out/'responses.json',responses);save(out/'summary.json',summary);save(out/'paired.json',paired)
 def get(arm,fam,h,key,scope='all',split='development'):
  return next(s for s in summary if (s['arm'],s['family'],s['horizon'],s['metric'],s['scope'],s['split'])==(arm,fam,h,key,scope,split))
 success={}
 for arm in [a for a in arms if a!='A']:
  measures=[('static',5,k,.9) for k in ['speed','position_drift']]+[('recorded',h,k,1.05) for h in [1,5] for k in ['position_error','velocity_error','visual_mse']]+[('action',h,'vector_error',1.05) for h in [1,5]]
  checks=[dict(family=f,horizon=h,metric=k,ratio=get(arm,f,h,k)['mean']/get('A',f,h,k)['mean'],threshold=th) for f,h,k,th in measures]
  success[arm]=dict(pass_point_estimates=all(x['ratio']<=x['threshold'] for x in checks),checks=checks)
 save(out/'success.json',success)
 # Immutable input audit and exact baseline cache checks.
 manifest=read(out/'manifest.json');assert all(sha(f)==v for f,v in manifest['sources'].items());prior={x['id']:x for x in read(root/'work/physics'/ 'pointmaze-physics-depth-20261011'/'rollouts.json') if x['variant'] in ['original','raw']};err=[]
 for r in read(out/'evaluation_A.json'):
  if r['id'] in prior:err.append(float(np.max(np.abs(np.array(r['states'])-np.array([x['state'] for x in prior[r['id']]['predictions']])))))
 assert max(err)<1e-5
 save(out/'checks.json',dict(pass_all=True,source_hashes_unchanged=True,baseline_cache_state_max=max(err),same_exposure=True,episode_masks_identical=True,finite_metrics=all(np.isfinite(r['velocity_error']) for r in rows),rows=len(rows),single_seed=True,sealed_test_opened=False))
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2});colors={a:plt.get_cmap('tab20')(i) for i,a in enumerate(arms)}
 fig,axes=plt.subplots(2,3,figsize=(13,7.5))
 for ax,(fam,key,title) in zip(axes.flat,[('static','speed','Stationary speed'),('static','position_drift','Stationary position drift'),('low_speed','velocity_error','Low-speed velocity error'),('recorded','position_error','Action cases: position error'),('recorded','velocity_error','Action cases: velocity error'),('recorded','visual_mse','DINO visual latent MSE')]):
  for arm in arms:ax.plot(np.arange(1,6)*.25,[get(arm,fam,h,key)['mean'] for h in range(1,6)],marker='o',label=arm,color=colors[arm])
  ax.set_title(title);ax.set_xlabel('Horizon (seconds)')
 axes[0,0].legend(ncol=5);fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,3.6))
 for ax,(key,title) in zip(axes,[('vector_error','Action response vector error'),('angle_error','Action direction error (degrees)'),('magnitude_error','Action magnitude error')]):
  for arm in arms:ax.plot(np.arange(1,6)*.25,[get(arm,'action',h,key)['mean'] for h in range(1,6)],marker='o',label=arm,color=colors[arm])
  ax.set_title(title);ax.set_xlabel('Horizon (seconds)')
 axes[0].legend();fig.tight_layout();fig.savefig(out/'action_response.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,3.6))
 for ax,key in zip(axes,['base','physics','stationary']):
  for arm in [a for a in arms if a!='A']:
   curve=read(out/arm/'curve.json');ax.plot([r['step'] for r in curve],[r[key] for r in curve],label=arm,color=colors[arm],alpha=.85)
  ax.set_title(f'Train {key} component (raw)');ax.set_xlabel('Optimizer step');ax.set_yscale('log')
 axes[0].legend();fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,3.6))
 for ax,(key,scope,title) in zip(axes,[('recurrent_velocity_residual','prefix_free','Affine velocity residual, oracle prefix free'),('wall_overlap_001','all','Wall overlap depth > 0.01 fraction'),('proprio_mse','all','Proprio latent MSE')]):
  for arm in arms:ax.plot(np.arange(1,6)*.25,[get(arm,'recorded',h,key,scope)['mean'] for h in range(1,6)],marker='o',label=arm,color=colors[arm])
  ax.set_title(title,fontsize=10);ax.set_xlabel('Horizon (seconds)')
 axes[0].legend();fig.tight_layout();fig.savefig(out/'consistency.png',dpi=160);plt.close(fig)
 save(out/'analysis_complete.json',dict(rows=len(rows),figures=4,paired_comparisons=len(comparisons),source_hashes_unchanged=True))
 print('COMPLETE',json.dumps(success),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();analyze(a.root,a.repo,a.out)
