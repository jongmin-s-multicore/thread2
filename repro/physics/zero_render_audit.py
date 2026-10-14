"""Post hoc calibration-only check of rendering-domain sensitivity. No tuning."""
from pathlib import Path
import sys,subprocess
import numpy as np
from intervention import sources,sha,read,save
from zero_velocity import C_RUN
PARENT='pointmaze-zero-velocity-20261010'
def prepare_audit(root,repo,out):
 import torch
 from omegaconf import OmegaConf
 torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir()
 paths=sources(root,repo);c=paths['base']/C_RUN;parent=paths['base']/PARENT
 assert read(parent/'checks.json')['pass_all']
 cm=read(parent/'manifest.json');assert all(sha(f)==h for f,h in cm['sources'].items())
 hashes=cm['sources'].copy();hashes.update({str(parent/n):sha(parent/n) for n in ['checks.json','infer_complete.json','summary.json']})
 save(out/'manifest.json',dict(sources=hashes,baseline_reuse_source=str(parent/'infer_complete.json'),new_training=False,new_physics_loss=False,scope='Post hoc calibration renderer sensitivity, no new threshold tuning',code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),script_sha256=sha(__file__)))
 save(out/'normalization.json',read(c/'normalization.json'))
 sys.path.insert(0,str(repo));import gym,env,mujoco_py
 cfg=OmegaConf.load(paths['cfg']);e=gym.make(cfg.env.name,*cfg.env.args,**dict(cfg.env.kwargs)).unwrapped;e.prepare_for_render()
 states=torch.load(paths['data']/'states.pth',weights_only=False).float().numpy();actions=torch.load(paths['data']/'actions.pth',weights_only=False).float().numpy();rows=[];pixel=[]
 original=read(c/'predictions.json')
 for x in [z for z in read(c/'selected.json') if z['split']=='calibration_fit'][:2]:
  ep,k=x['episode'],x['raw_anchor'];ims=[];path=paths['data']/'obses'/f'episode_{ep:03d}.pth';hashes[str(path)]=sha(path);old=torch.load(path,weights_only=False).numpy()
  for n in [k-10,k-5,k]:
   st=states[ep,n].astype(float);q=st[:2]-.01*st[2:] if n>0 else st[:2]
   e.sim.reset();e.sim.set_state(mujoco_py.MjSimState(0.,q.copy(),st[2:].copy(),None,{}));e.sim.forward();e.set_marker();im=e._render_frame().copy();ims.append(im)
   pixel.append(dict(episode=ep,raw_frame=n,mae_rgb_255=float(np.mean(np.abs(im.astype(float)-old[n].astype(float))))))
  file=f'images/{ep}.npz';np.savez_compressed(out/file,images=np.stack(ims))
  rows.append(dict(episode=ep,split=x['split'],name='rerender_original_zero',context_states=states[ep,[k-10,k-5,k]].tolist(),raw_actions=np.r_[actions[ep,k-10:k],np.zeros((5,2))].tolist(),image_file=file,image_sha256=sha(out/file),reference=next(r['prediction'] for r in original if r['episode']==ep and r['condition']=='zero')))
 e.close();save(out/'branches.json',rows);save(out/'render_checks.json',pixel);save(out/'image_sources.json',{str(paths['data']/'obses'/f"episode_{x['episode']:03d}.pth"):sha(paths['data']/'obses'/f"episode_{x['episode']:03d}.pth") for x in rows})
 assert max(x['mae_rgb_255'] for x in pixel)<=3
 save(out/'prepare_complete.json',dict(rows=len(rows),prior_baseline_reused=True))

def analyze_audit(root,repo,out):
 p={(r['episode'],r['name']):r for r in read(out/'predictions.json')};rows=[]
 for r in read(out/'branches.json'):
  delta=np.array(p[r['episode'],r['name']]['prediction'])-r['reference'];rows.append(dict(episode=r['episode'],state_delta=delta.tolist(),velocity_difference=float(np.linalg.norm(delta[2:])),position_difference=float(np.linalg.norm(delta[:2]))))
 save(out/'summary.json',dict(calibration_only=True,rows=rows,mean_velocity_difference=float(np.mean([r['velocity_difference'] for r in rows])),max_velocity_difference=max(r['velocity_difference'] for r in rows),no_development_tuning=True))
 assert all(sha(f)==h for f,h in read(out/'manifest.json')['sources'].items());assert all(sha(f)==h for f,h in read(out/'image_sources.json').items())
 save(out/'checks.json',dict(pass_all=True,baseline_checks_reused=True,weights_unchanged=read(out/'infer_complete.json')['weights_unchanged']))
 save(out/'completion.json',dict(status='RENDER_SENSITIVITY_AUDIT_COMPLETE',new_training=False))
