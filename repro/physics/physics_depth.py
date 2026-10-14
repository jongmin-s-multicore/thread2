"""Rendering controls and cached-first-step frozen open-loop physics diagnostics."""
import argparse,os,sys,time,subprocess,traceback
from pathlib import Path
import numpy as np
from intervention import read,save,sha,sources,clearance,analytic
C='pointmaze-dynamics-c-20261010-v3';Z='pointmaze-zero-velocity-20261010';R='pointmaze-zero-velocity-render-audit-20261010'

def background_map(original,rendered):
 a=np.asarray(original,dtype=float);b=np.asarray(rendered,dtype=float)
 def green(x):return (x[...,1]>x[...,0]+15)&(x[...,1]>x[...,2]+15)
 mask=np.any(green(a)|green(b),axis=0);pad=np.pad(mask,3);dil=np.zeros_like(mask)
 for i in range(7):
  for j in range(7):dil|=pad[i:i+mask.shape[0],j:j+mask.shape[1]]
 correction=np.median(a-b,axis=0);correction[dil]=0
 return correction,dil

def match_background(images,correction):return np.clip(np.rint(np.asarray(images,float)+correction),0,255).astype(np.uint8)

def make_env(root,repo):
 sys.path.insert(0,str(repo));import gym,env
 from omegaconf import OmegaConf
 cfg=OmegaConf.load(sources(root,repo)['cfg']);e=gym.make(cfg.env.name,*cfg.env.args,**dict(cfg.env.kwargs)).unwrapped
 assert e.model.nq==e.model.nv==2 and e.model.na==e.model.nmocap==0 and e.frame_skip==5 and e.sim.nsubsteps==1
 assert e.model.opt.integrator==e.model.opt.disableflags==0 and abs(e.model.opt.timestep-.01)<1e-12
 return e

def simulate(e,st,u,t=0.):
 import mujoco_py
 s=e.sim;st=np.asarray(st,float);s.reset();s.set_state(mujoco_py.MjSimState(t,st[:2].copy(),st[2:].copy(),None,{}));s.forward();s.data.qacc_warmstart[:]=0
 particle=e.model.geom_name2id('particle_geom');walls={i for i in range(e.model.ngeom) if (e.model.geom_id2name(i) or '').startswith('wall_')}
 tr=[st.tolist()];contact=clip=0
 for _ in range(5):
  clip+=int(np.any(np.abs(s.data.qvel)>5));e.clip_velocity();s.data.ctrl[:]=u
  for _ in range(5):
   s.step();tr.append(np.r_[s.data.qpos.copy(),s.data.qvel.copy()].tolist());contact+=int(any((int(c.geom1)==particle and int(c.geom2) in walls) or (int(c.geom2)==particle and int(c.geom1) in walls) for c in s.data.contact[:s.data.ncon]))
 assert abs(s.data.time-t-.25)<1e-11
 return dict(trace=tr,contacts=contact,clips=clip)

def diagnose(trace,u,params,contacts=0,clips=0):
 tr=np.array(trace);M,D,G=[np.array(params[k]) for k in ['M','D','G']];work=loss=num=clip=0.
 for n in range(25):
  vbefore=tr[n,2:];v0=np.clip(vbefore,-5,5) if n%5==0 else vbefore;v1=tr[n+1,2:]
  work+=(G@u)@(tr[n+1,:2]-tr[n,:2]);loss+=.01*v1@D@v1;num+=.5*(v1-v0)@M@(v1-v0);clip+=.5*(vbefore@M@vbefore-v0@M@v0)
 residual=.5*(tr[-1,2:]@M@tr[-1,2:]-tr[0,2:]@M@tr[0,2:])-work+loss+num+clip
 margin=float(np.min(clearance(tr[:,:2],params)));free=contacts==clips==0 and margin>.004
 eq=None
 if free:
  theory,_=analytic(tr[0],np.tile(u,(5,1)),params);eq=float(np.max(np.abs(theory-tr)));assert eq<1e-10 and abs(residual)<1e-10
 return dict(free=free,contacts=int(contacts),clips=int(clips),min_clearance=margin,fine_energy_residual=float(residual),analytic_max_abs=eq)

def prepare(root,repo,out):
 import torch,mujoco_py
 torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=False);(out/'images').mkdir()
 paths=sources(root,repo);c,z,r=[paths['base']/x for x in [C,Z,R]]
 for p in [c,z,r]:assert read(p/'checks.json')['pass_all']
 hashes=read(z/'manifest.json')['sources'].copy()
 for p,names in [(c,['selected.json','predictions.json','oracles.json','normalization.json','infer_complete.json']),(z,['selected.json','branches.json','predictions.json','checks.json']),(r,['branches.json','predictions.json','checks.json'])]:
  for n in names:hashes[str(p/n)]=sha(p/n)
 assert all(sha(f)==h for f,h in hashes.items())
 params=read(c/'parameters.json');save(out/'parameters.json',params);save(out/'normalization.json',read(c/'normalization.json'))
 cs=read(c/'selected.json');zs=read(z/'selected.json');cp={(x['episode'],x['condition']):x for x in read(c/'predictions.json')};co={(x['episode'],x['condition']):x for x in read(c/'oracles.json')}
 zp={(x['episode'],x['name']):x for x in read(z/'predictions.json')};zb={(x['episode'],x['name']):x for x in read(z/'branches.json')}
 select=lambda rows:sum([[x for x in rows if x['split']==g][:10] for g in ['calibration_fit','development']],[])
 chosen=select(cs);static=select(zs);cases=[]
 def hashfile(f):hashes[str(f)]=sha(f);return str(f)
 for x in chosen:
  ep,k=x['episode'],x['raw_anchor'];context=hashfile(c/'token_cache'/f'context_{ep}_{k}.pth');prediction=hashfile(c/'token_cache'/f'predictions_{ep}_{k}.pth')
  for name in ['zero']+[f'r0.2_d{d}' for d in range(8)]:
   o=co[ep,name];cases.append(dict(id=f'C_{ep}_{name}',family='recorded',episode=ep,split=x['split'],condition=name,context_file=context,prediction_file=prediction,prediction_key=name,reference_h1=cp[ep,name]['prediction'],start=x['start'],u=o['requested_actions'][0],first_trace=o['trace'],first_contacts=o['wall_contact_substeps'],first_clips=o['clipped_boundaries']))
 for x in static:
  ep=x['episode'];b=zb[ep,'coast_zero'];token=hashfile(z/'token_cache'/f'{ep}_coast_zero.pth');image=hashfile(z/b['image_file'])
  cases.append(dict(id=f'Z_{ep}',family='static',episode=ep,split=x['split'],condition='zero',context_file=token,prediction_file=token,prediction_key='prediction',reference_h1=zp[ep,'coast_zero']['prediction'],start=b['start'],u=[0.,0.],first_trace=b['trace'][51:],first_contacts=0,first_clips=0,context_states=b['context_states'],source_image=image))
 # A fixed pixel residual is derived only from the two completed calibration controls.
 rendered_old={x['episode']:x for x in read(r/'branches.json')};rp={x['episode']:x for x in read(r/'predictions.json')}
 originals=[];renders=[];raw_images={};image_metadata=[]
 states=torch.load(paths['data']/'states.pth',weights_only=False,map_location='cpu').float().numpy();actions=torch.load(paths['data']/'actions.pth',weights_only=False,map_location='cpu').float().numpy()
 renderer=make_env(root,repo);renderer.prepare_for_render()
 for x in cs:
  ep,k=x['episode'],x['raw_anchor'];f=paths['data']/'obses'/f'episode_{ep:03d}.pth';hashfile(f);old=torch.load(f,weights_only=False,map_location='cpu')[[k-10,k-5,k]].numpy()
  if ep in rendered_old:
   f=r/rendered_old[ep]['image_file'];hashfile(f);im=np.load(f)['images']
   originals.extend(old);renders.extend(im)
  else:
   ims=[]
   for n in [k-10,k-5,k]:
    st=states[ep,n].astype(float);q=st[:2]-.01*st[2:] if n>0 else st[:2]
    renderer.sim.reset();renderer.sim.set_state(mujoco_py.MjSimState(0.,q.copy(),st[2:].copy(),None,{}));renderer.sim.forward();renderer.set_marker();ims.append(renderer._render_frame().copy())
   im=np.stack(ims)
  raw_images[ep]=(old,im);image_metadata.append(dict(episode=ep,split=x['split'],raw_anchor=k,context_states=states[ep,[k-10,k-5,k]].tolist(),raw_actions=np.r_[actions[ep,k-10:k],np.zeros((5,2))].tolist(),reference=cp[ep,'zero']['prediction'],oracle=co[ep,'zero']['endpoint'],reused_raw_prediction=rp[ep]['prediction'] if ep in rp else None))
 renderer.close();correction,mask=background_map(originals,renders);np.savez_compressed(out/'background_map.npz',correction=correction,excluded_green_mask=mask)
 render_rows=[]
 for row in image_metadata:
  ep=row['episode'];old,im=raw_images[ep];matched=match_background(im,correction);file=f'images/render_{ep}.npz';np.savez_compressed(out/file,raw=im,matched=matched,original=old)
  render_rows.append(dict(**row,image_file=file,image_sha256=sha(out/file),raw_pixel_mae=float(np.mean(np.abs(old.astype(float)-im))),matched_pixel_mae=float(np.mean(np.abs(old.astype(float)-matched)))))
 for case in cases:
  if case['family']=='static':
   ims=np.load(case['source_image'])['images'];file=f"images/static_{case['episode']}.npz";np.savez_compressed(out/file,matched=match_background(ims,correction));case['matched_image_file']=file;case['matched_image_sha256']=sha(out/file)
 save(out/'render_contexts.json',render_rows);save(out/'cases.json',cases)
 e=make_env(root,repo);oracles=[]
 for case in cases:
  tr=case['first_trace'];diag=diagnose(tr,case['u'],params,case['first_contacts'],case['first_clips']);seq=[dict(horizon=1,trace=tr,**diag)];st=tr[-1]
  for h in range(2,6):
   rr=simulate(e,st,case['u'],(h-1)*.25);diag=diagnose(rr['trace'],case['u'],params,rr['contacts'],rr['clips']);seq.append(dict(horizon=h,trace=rr['trace'],**diag));st=rr['trace'][-1]
  if case['family']=='static':assert np.max(np.abs(np.array(st)-case['start']))<1e-10
  oracles.append(dict(id=case['id'],segments=seq))
 e.close();save(out/'oracles.json',oracles)
 assert all(sha(f)==h for f,h in hashes.items())
 save(out/'manifest.json',dict(sources=hashes,previous_status=read(root/'work/STATUS.json'),code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),script_sha256=sha(__file__),protocol_sha256=sha(Path(__file__).with_name('physics_depth_protocol.md')),new_training=False,new_physics_loss=False,sealed_test_opened=False))
 save(out/'prepare_complete.json',dict(recorded_cases=sum(x['family']=='recorded' for x in cases),static_cases=sum(x['family']=='static' for x in cases),reused_first_step_oracles=len(cases),new_oracle_segments=len(cases)*4,render_contexts=len(cs),reused_raw_render_contexts=len(rendered_old),calibration_background_frames=len(originals),source_hashes_unchanged=True))
 print(read(out/'prepare_complete.json'),flush=True)

def infer(root,repo,out):
 import torch
 from frozen_runtime import load_frozen,weight_hash,inverse
 assert (out/'prepare_complete.json').exists();manifest=read(out/'manifest.json');assert all(sha(f)==h for f,h in manifest['sources'].items())
 wm,cfg=load_frozen(root,repo);before=weight_hash(wm);c=sources(root,repo)['base']/C
 assert before==read(c/'infer_complete.json')['weight_tensor_sha256']
 norm=read(out/'normalization.json');decode=inverse(norm);sm=torch.tensor(norm['state_mean']);ss=torch.tensor(norm['state_std']);am=torch.tensor(norm['action_mean']);ast=torch.tensor(norm['action_std'])
 from datasets.img_transforms import default_transform
 transform=default_transform(int(cfg.img_size));tokens=out/'token_cache';tokens.mkdir(exist_ok=False)
 def encode(images,states,rawactions):
  obs=dict(visual=transform(torch.from_numpy(images).float().permute(0,3,1,2)/255.)[None].to('cuda:0'),proprio=((torch.tensor(states,dtype=torch.float32)-sm)/ss)[None].to('cuda:0'))
  raw=torch.tensor(rawactions,dtype=torch.float32);act=((raw-am)/ast).reshape(1,3,10).to('cuda:0');assert torch.max(torch.abs(act.cpu().reshape(15,2)*ast+am-raw))<1e-6
  return wm.encode(obs,act)
 render_predictions=[];rollouts=[];h1errs=[];newsteps=0;reused=0;rollout_checks=[]
 with torch.inference_mode():
  for r in read(out/'render_contexts.json'):
   assert sha(out/r['image_file'])==r['image_sha256'];ims=np.load(out/r['image_file']);row=dict(episode=r['episode'],split=r['split'],original=r['reference'],oracle=r['oracle'])
   for variant in ['raw','matched']:
    if variant=='raw' and r['reused_raw_prediction'] is not None:row[variant]=r['reused_raw_prediction'];continue
    z=encode(ims[variant],r['context_states'],r['raw_actions']);y=wm.predict(z)[:,-1:];p,res=decode(wm,y);assert np.isfinite(p).all();row[variant]=p.tolist();torch.save(dict(context=z.cpu(),prediction=y.cpu()),tokens/f"render_{r['episode']}_{variant}.pth")
   render_predictions.append(row)
  for i,case in enumerate(read(out/'cases.json')):
   cache=torch.load(case['context_file'],weights_only=False,map_location='cpu');predcache=torch.load(case['prediction_file'],weights_only=False,map_location='cpu');act=(((torch.tensor(case['u'],dtype=torch.float32)-am)/ast).repeat(5)).reshape(1,1,10).to('cuda:0')
   variants=['original'] if case['family']=='recorded' else ['raw','matched']
   for variant in variants:
    if variant=='matched':
     assert sha(out/case['matched_image_file'])==case['matched_image_sha256'];z=encode(np.load(out/case['matched_image_file'])['matched'],case['context_states'],np.zeros((15,2)));y=wm.predict(z)[:,-1:];newsteps+=1
    else:
     z=cache['z' if case['family']=='recorded' else 'context'].to('cuda:0').clone();z[:,-1:]=wm.replace_actions_from_z(z[:,-1:].clone(),act)
     y=predcache[case['prediction_key']].to('cuda:0');p,_=decode(wm,y);err=float(np.max(np.abs(p-case['reference_h1'])));h1errs.append(err);assert err<1e-8;reused+=1
    states=[];saved=[]
    for h in range(1,6):
     if h>1:
      # Keep two past frames, append predicted full observation with chosen next action.
      incoming=wm.replace_actions_from_z(y.clone(),act);z=torch.cat([z[:,1:],incoming],dim=1);assert z.shape[1]==3
      y=wm.predict(z)[:,-1:];newsteps+=1
     p,res=decode(wm,y);assert np.isfinite(p).all();states.append(dict(horizon=h,state=p.tolist(),off_manifold=res));saved.append(y.cpu())
    torch.save(saved,tokens/f"rollout_{case['id']}_{variant}.pth");rollouts.append(dict(id=case['id'],family=case['family'],episode=case['episode'],split=case['split'],condition=case['condition'],variant=variant,predictions=states))
   if i%20==19:print('ROLLOUT_CASE',i+1,flush=True)
  # Independent check against the repository's public rollout API, calibration only.
  control_cases=[x for x in read(out/'cases.json') if x['family']=='recorded' and x['split']=='calibration_fit' and x['condition']=='zero'][:2]
  for case in control_cases:
   r=next(x for x in read(out/'render_contexts.json') if x['episode']==case['episode']);ims=np.load(out/r['image_file'])['original']
   obs=dict(visual=transform(torch.from_numpy(ims).float().permute(0,3,1,2)/255.)[None].to('cuda:0'),proprio=((torch.tensor(r['context_states'],dtype=torch.float32)-sm)/ss)[None].to('cuda:0'))
   aa=((torch.tensor(r['raw_actions'],dtype=torch.float32)-am)/ast).reshape(1,3,10).to('cuda:0');aa=torch.cat([aa,aa[:,-1:].repeat(1,4,1)],dim=1)
   _,full=wm.rollout(obs,aa);expected=next(x for x in rollouts if x['id']==case['id'])
   errs=[float(np.max(np.abs(decode(wm,full[:,3+h:4+h])[0]-expected['predictions'][h]['state']))) for h in range(5)]
   assert max(errs)<1e-5;rollout_checks.append(dict(episode=case['episode'],max_abs=max(errs)))
 assert before==weight_hash(wm) and all(not p.requires_grad for p in wm.parameters());assert all(sha(f)==h for f,h in manifest['sources'].items())
 save(out/'render_predictions.json',render_predictions);save(out/'rollouts.json',rollouts);save(out/'infer_complete.json',dict(weights_unchanged=True,weight_tensor_sha256=before,h1_cache_max_abs=max(h1errs),reused_h1_predictions=reused,new_rollout_steps=newsteps,rollout_sequences=len(rollouts),no_future_observations=True,new_training=False,raw_render_outputs_reused=2,new_render_outputs=len(render_predictions)*2-2,public_rollout_checks=rollout_checks,validation_only_recomputed_steps=10))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','infer','analyze'],required=True);a=p.parse_args();t=time.monotonic()
 try:
  if a.stage=='analyze':
   from analyze_physics_depth import analyze
   analyze(a.root.resolve(),a.repo.resolve(),a.out.resolve())
  else:globals()[a.stage](a.root.resolve(),a.repo.resolve(),a.out.resolve())
  save(a.out/f'{a.stage}_timing.json',dict(elapsed_seconds=time.monotonic()-t))
 except BaseException as ex:
  if a.out.exists():save(a.out/f'{a.stage}_failure_{time.time_ns()}.json',dict(reason=repr(ex)))
  traceback.print_exc();raise
if __name__=='__main__':main()
