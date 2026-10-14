"""Independent endpoint aggregation, exposure and checkpoint metadata checks."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def run(out):
 read=lambda p:json.loads(p.read_text());summary=read(out/'summary.json');oracle=read(out/'oracles.json');cases={c['id']:c for c in read(out/'cases.json')};errors=[];details=[]
 for arm in 'ABCDE':
  output=[r for r in read(out/f'evaluation_{arm}.json') if r['split']=='development']
  for family,metric,h in [('static','speed',5),('static','position_drift',5),('recorded','velocity_error',1),('recorded','velocity_error',5),('recorded','position_error',5),('low_speed','velocity_error',5)]:
   grouped={}
   for r in output:
    if r['family']!=family:continue
    state=np.array(r['states'][h-1]);truth=np.array(oracle[r['id']][h-1]['trace'][-1]);start=np.array(cases[r['id']]['start'])
    if metric=='speed':value=np.sqrt(sum(state[2:]**2))
    elif metric=='position_drift':value=np.sqrt(sum((state[:2]-start[:2])**2))
    elif metric=='position_error':value=np.sqrt(sum((state[:2]-truth[:2])**2))
    else:value=np.sqrt(sum((state[2:]-truth[2:])**2))
    grouped.setdefault(r['episode'],[]).append(value)
   value=sum(sum(v)/len(v) for v in grouped.values())/len(grouped)
   expected=next(x for x in summary if (x['arm'],x['family'],x['metric'],x['horizon'],x['scope'],x['split'])==(arm,family,metric,h,'all','development'))
   error=abs(value-expected['mean']);assert error<1e-12;assert expected['episodes']==len(grouped);errors.append(error);details.append(dict(arm=arm,family=family,metric=metric,horizon=h,value=value,episodes=len(grouped)))
 exposure=[read(out/a/'exposure.json') for a in 'BCDE'];assert all(e==exposure[0] for e in exposure);assert len(exposure[0])==120
 split=read(out/'split.json')['episodes'];assert len(set(sum(split.values(),[])))==sum(map(len,split.values()))
 assert all(cases[id]['split']=='calibration_fit' for pair in exposure[0] for id in pair)
 completes=[read(out/a/'complete.json') for a in 'BCDE'];assert len({r['frozen_hash'] for r in completes})==1
 assert all(r['trainable_only_predictor'] and r['steps']==120 for r in completes)
 result=dict(pass_all=True,reconstructed_metrics=len(errors),max_abs=max(errors),details=details,identical_exposure=True,exposed_unique_branches=len(set(sum(exposure[0],[]))),exposed_unique_episodes=len({cases[id]['episode'] for id in sum(exposure[0],[])}),same_frozen_hash=True,no_tune_or_development_in_training=True)
 with (out/'independent_verification.json').open('x') as f:json.dump(result,f,indent=2)
 print(json.dumps({k:v for k,v in result.items() if k!='details'}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();run(a.out)
