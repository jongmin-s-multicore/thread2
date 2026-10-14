"""Independent reconstruction and preregistered equivalence checks."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from intervention import read,save,sha
from physics_attribution import ARMS,NEW

def verify(out):
 s=read(out/'summary.json');oracle=read(out/'oracles.json');cases={c['id']:c for c in read(out/'cases.json')};errors=[];details=[]
 for arm in ARMS:
  output=[r for r in read(out/f'evaluation_{arm}.json') if r['split']=='development']
  for family,metric,h in [('static','speed',5),('static','position_drift',5),('recorded','velocity_error',1),('recorded','velocity_error',5),('recorded','position_error',5),('low_speed','velocity_error',5)]:
   values={}
   for r in output:
    if r['family']!=family:continue
    pred=np.array(r['states'][h-1]);gt=np.array(oracle[r['id']][h-1]['trace'][-1]);start=np.array(cases[r['id']]['start'])
    if metric=='speed':delta=pred[2:]
    elif metric=='position_drift':delta=pred[:2]-start[:2]
    elif metric=='position_error':delta=pred[:2]-gt[:2]
    else:delta=pred[2:]-gt[2:]
    values.setdefault(r['episode'],[]).append(float(np.sqrt(sum(delta**2))))
   val=sum(sum(v)/len(v) for v in values.values())/len(values);expected=next(r for r in s if (r['arm'],r['split'],r['family'],r['scope'],r['horizon'],r['metric'])==(arm,'development',family,'all',h,metric));err=abs(val-expected['mean']);assert err<1e-12;errors.append(err);details.append(dict(arm=arm,family=family,metric=metric,horizon=h,mean=val,episodes=len(values)))
 baseline=read(out/'B/exposure.json');assert all(read(out/a/'exposure.json')==baseline for a in NEW)
 assert len({read(out/a/'complete.json')['frozen_hash'] for a in ARMS if a!='A'})==1
 src=Path(read(out/'attribution_manifest.json')['source']);src=src if src.exists() else out.parent/src.name;assert all(sha(out/f'evaluation_{a}.json')==sha(src/f'evaluation_{a}.json') for a in 'ABCDE')
 idx={(r['arm'],r['split'],r['family'],r['scope'],r['horizon'],r['metric']):r for r in s};eq=[]
 for key,g in idx.items():
  if key[0]!='G1' or key[1]!='development' or key[3]!='all' or key[5] not in ['position_error','velocity_error','visual_mse','proprio_mse','speed','position_drift','vector_error','angle_error','magnitude_error']:continue
  p=idx[('P1',)+key[1:]];a=idx[('A',)+key[1:]]
  if a['mean'] is None:continue
  ratio=abs(p['mean']-g['mean'])/max(abs(a['mean']),1e-12);eq.append(dict(family=key[2],horizon=key[4],metric=key[5],difference=float(p['mean']-g['mean']),relative_to_A=ratio,pass_threshold=ratio<.01))
 matched={}
 for arm in ['CM','DM']:
  curve=read(out/arm/'curve.json');maxerr=max(abs(r['aux_norm_ratio']-.1) for r in curve);assert maxerr<1e-6;matched[arm]=dict(ratio_error_max=maxerr,lambda_min=min(r['lambda_aux'] for r in curve),lambda_max=max(r['lambda_aux'] for r in curve))
 save(out/'independent_verification.json',dict(pass_all=True,reconstructed_metrics=len(errors),max_abs=max(errors),same_exposure=True,baseline_files_identical=True,same_frozen_hash=True,gradient_norm_matching=matched,details=details))
 save(out/'h1_evaluation_equivalence.json',dict(pass_all=all(r['pass_threshold'] for r in eq),max_relative_difference=max(r['relative_to_A'] for r in eq),threshold=.01,metrics=eq))
 print('VERIFIED',len(errors),'H1 equivalence',all(r['pass_threshold'] for r in eq),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();verify(a.out)
