"""Small predictor-only pilot; immutable input caches and fixed episode masks."""
import argparse, json, os, sys, time, subprocess, hashlib
from pathlib import Path
import numpy as np
from intervention import read, save, sha, sources, analytic
from physics_depth import C, Z, make_env, simulate, diagnose
DEPTH='pointmaze-physics-depth-20261011'

def prepare(root,repo,out):
 import torch,mujoco_py
 out.mkdir(exist_ok=False,parents=True);(out/'images').mkdir();base=sources(root,repo)['base'];c,z,d=[base/n for n in [C,Z,DEPTH]]
 hashes={}
 def remember(p):hashes[str(p)]=sha(p);return str(p)
 for folder in [c,z,d,base/'pointmaze-dynamics-b-20261010']:
  assert read(folder/'checks.json')['pass_all'];remember(folder/'checks.json')
 for folder,names in [(c,['selected.json','oracles.json','normalization.json','manifest.json']),(z,['selected.json','branches.json']),(d,['cases.json','oracles.json','rollouts.json'])]:
  for n in names:remember(folder/n)
 for f,h in read(c/'manifest.json')['sources'].items():
  if Path(f).exists():assert sha(f)==h;hashes[f]=h
 params=read(d/'parameters.json');save(out/'parameters.json',params);save(out/'normalization.json',read(c/'normalization.json'))
 cases=read(d/'cases.json');oracles={x['id']:x['segments'] for x in read(d/'oracles.json')}
 cs=read(c/'selected.json');zs=read(z/'selected.json');co={(x['episode'],x['condition']):x for x in read(c/'oracles.json')};zb={(x['episode'],x['name']):x for x in read(z/'branches.json')}
 selected_c=[x for x in cs if x['split']=='calibration_tune']
 for x in selected_c:
  ep,k=x['episode'],x['raw_anchor']
  for name in ['zero']+[f'r0.2_d{i}' for i in range(8)]:
   o=co[ep,name];cases.append(dict(id=f'C_{ep}_{name}',family='recorded',episode=ep,split=x['split'],condition=name,context_file=str(c/'token_cache'/f'context_{ep}_{k}.pth'),prediction_file=str(c/'token_cache'/f'predictions_{ep}_{k}.pth'),prediction_key=name,start=x['start'],u=o['requested_actions'][0],first_trace=o['trace'],first_contacts=o['wall_contact_substeps'],first_clips=o['clipped_boundaries']))
 chosen_z=sum([[x for x in zs if x['split']==s][:10] for s in ['calibration_fit','development']],[])+[x for x in zs if x['split']=='calibration_tune']
 for x in chosen_z:
  ep=x['episode'];names=([ 'coast_zero'] if x['split']=='calibration_tune' else [])+[f'coast_r0.1_{axis}_{sgn}' for axis in range(2) for sgn in [-1,1]]
  for name in names:
   b=zb[ep,name];token=str(z/'token_cache'/f'{ep}_{name}.pth');cases.append(dict(id=f'Z_{ep}' if name=='coast_zero' else f'Z_{ep}_{name}',family='static' if name=='coast_zero' else 'low_speed',episode=ep,split=x['split'],condition=name,context_file=token,prediction_file=token,prediction_key='prediction',start=b['start'],u=[0.,0.],first_trace=b['trace'][51:],first_contacts=0,first_clips=0))
 split={s:sorted({x['episode'] for x in cases if x['split']==s}) for s in ['calibration_fit','calibration_tune','development']}
 assert not set(split['calibration_fit'])&set(split['development']);assert not set(split['calibration_fit'])&set(split['calibration_tune']);assert not set(split['calibration_tune'])&set(split['development'])
 oldsplit=read(sources(root,repo)['src']/'split_manifest.json');assert not set(sum(split.values(),[]))&set(oldsplit['sealed_wm_test'])
 save(out/'split.json',dict(episodes=split,counts={s:sum(x['split']==s for x in cases) for s in split},relation='All chosen episodes in original WM train domain. Development held out only from this fine-tuning, previously inspected. No sealed data read.',prior_c_episodes=[x['episode'] for x in cs],prior_zero_episodes=[x['episode'] for x in zs]))
 e=make_env(root,repo);renderer=make_env(root,repo);renderer.prepare_for_render();reuse=0;new=0;eq=[]
 for i,case in enumerate(cases):
  for k in ['context_file','prediction_file']:remember(Path(case[k]))
  if case['id'] in oracles:seq=oracles[case['id']];reuse+=5
  else:
   tr=case['first_trace'];seq=[dict(horizon=1,trace=tr,**diagnose(tr,case['u'],params,case['first_contacts'],case['first_clips']))];reuse+=1
   for h in range(2,6):
    rr=simulate(e,seq[-1]['trace'][-1],case['u']);seq.append(dict(horizon=h,trace=rr['trace'],**diagnose(rr['trace'],case['u'],params,rr['contacts'],rr['clips'])));new+=1
  oracles[case['id']]=seq;images=[]
  for seg in seq:
   if seg['free']:eq.append(float(np.max(np.abs(analytic(seg['trace'][0],np.tile(case['u'],(5,1)),params)[0]-seg['trace']))))
   # Images precede state by one integrator step, including contact intervals.
   st=np.array(seg['trace'][-2]);renderer.sim.reset();renderer.sim.set_state(mujoco_py.MjSimState(0.,st[:2],st[2:],None,{}));renderer.sim.forward();renderer.set_marker();images.append(renderer._render_frame().copy())
  np.savez_compressed(out/'images'/f"{case['id']}.npz",images=np.stack(images))
  if i%40==0:print('PREPARE',i,len(cases),flush=True)
 e.close();renderer.close();assert max(eq)<1e-10
 save(out/'cases.json',cases);save(out/'oracles.json',oracles)
 save(out/'manifest.json',dict(sources=hashes,previous_status=read(root/'work/STATUS.json'),code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),protocol_sha256=sha(Path(__file__).with_name('physics_loss_protocol.md')),sealed_test_opened=False))
 save(out/'prepare_complete.json',dict(cases=len(cases),reused_oracle_segments=reuse,new_oracle_segments=new,oracle_equation_max=max(eq),source_hashes_unchanged=all(sha(f)==h for f,h in hashes.items())))

def runtime(root,repo,out):
 import torch
 from frozen_runtime import load_frozen,weight_hash
 wm,cfg=load_frozen(root,repo);assert weight_hash(wm)==read(sources(root,repo)['base']/C/'infer_complete.json')['weight_tensor_sha256'];wm.predictor.requires_grad_(True)
 import models.visual_world_model as m
 assert Path(m.__file__).resolve().is_relative_to(repo.resolve())
 n=read(out/'normalization.json');dev='cuda:0'
 tensors={k:torch.tensor(n[k],device=dev,dtype=torch.float32) for k in ['state_mean','state_std','action_mean','action_std','weight','bias']};tensors['pinv']=torch.linalg.pinv(tensors['weight'])
 return wm,cfg,tensors

def decode(wm,y,n,average=False):
 if average:
  prop=y[..., -(wm.proprio_dim+wm.action_dim):-wm.action_dim].mean(dim=2)[...,:wm.proprio_dim//wm.num_proprio_repeat]
 else:prop=wm.separate_emb(y)[0]['proprio']
 return ((prop-n['bias'])@n['pinv'].T)*n['state_std']+n['state_mean']

def transition(state,u,params):
 import torch
 a=torch.as_tensor(params['Ah'],dtype=state.dtype,device=state.device);b=torch.as_tensor(params['Bh'],dtype=state.dtype,device=state.device)
 q,v=state[...,:2],state[...,2:]
 for j in range(25):v=v@a.T+u@b.T;q=q+params['dt']*v
 return torch.cat([q,v],dim=-1)

def original_loss(wm,pred,target):
 return wm.emb_criterion(pred[...,:-wm.action_dim],target[...,:-wm.action_dim].detach())

def loss_parts(wm,b,n,params):
 import torch
 first=wm.predict(b['context']);y1=first[:,-1:];incoming=wm.replace_actions_from_z(y1.clone(),b['act']);y2=wm.predict(torch.cat([b['context'][:,1:],incoming],dim=1))[:,-1:]
 states=torch.cat([decode(wm,y1,n),decode(wm,y2,n)],dim=1)
 target=torch.cat([b['context'][:,1:],b['future'][:,:1]],dim=1);base=original_loss(wm,first,target)
 physical=torch.stack([transition(b['start'],b['u'],params),transition(states[:,0],b['u'],params)],dim=1)
 err=lambda a,c:((a-c)/n['state_std']).square().mean(dim=-1)
 masked=lambda v:(v*b['mask']).sum()/b['mask'].sum().clamp(min=1)
 supervised=masked(err(states,b['truth'][:,:2]));physics=masked(err(states,physical))
 stat=b['static'][:,None].expand(-1,2);stationary=(err(states,b['start'][:,None])*stat).sum()/stat.sum().clamp(min=1)
 return dict(base=base,supervised=supervised,physics=physics,stationary=stationary),states

def objective(parts,arm,lam,ls):
 loss=parts['base']
 if arm=='C':loss=loss+lam*parts['supervised']
 if arm in ['D','E']:loss=loss+lam*parts['physics']
 if arm=='E':loss=loss+ls*parts['stationary']
 return loss

def frozen_hash(wm):
 h=hashlib.sha256()
 for name,p in sorted(wm.state_dict().items()):
  if not name.startswith('predictor.'):h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def encode(root,repo,out):
 import torch
 from datasets import __name__ as unused # repo added by bootstrap
 from datasets.img_transforms import default_transform
 wm,cfg,n=runtime(root,repo,out);wm.requires_grad_(False);transform=default_transform(int(cfg.img_size));(out/'encoded').mkdir(exist_ok=False)
 cases=read(out/'cases.json');oracles=read(out/'oracles.json')
 with torch.inference_mode():
  for i,c in enumerate(cases):
   cache=torch.load(c['context_file'],map_location='cpu',weights_only=False);z=cache['z' if c['family']=='recorded' else 'context'].cuda().clone()
   u=torch.tensor(c['u'],device='cuda',dtype=torch.float32);act=((u-n['action_mean'])/n['action_std']).repeat(5).reshape(1,1,10);z[:,-1:]=wm.replace_actions_from_z(z[:,-1:].clone(),act)
   st=torch.tensor([s['trace'][-1] for s in oracles[c['id']]],dtype=torch.float32,device='cuda')
   images=np.load(out/'images'/f"{c['id']}.npz")['images'];im=transform(torch.tensor(images).float().permute(0,3,1,2)/255)[None].cuda()
   future=wm.encode(dict(visual=im,proprio=((st-n['state_mean'])/n['state_std'])[None]),act.repeat(1,5,1))
   mask=np.cumprod([s['free'] for s in oracles[c['id']]])[:2]
   torch.save(dict(context=z.cpu(),future=future.cpu(),act=act.cpu(),start=torch.tensor(c['start'],dtype=torch.float32)[None],u=u.cpu()[None],truth=st.cpu()[None],mask=torch.tensor(mask,dtype=torch.float32)[None],static=torch.tensor([c['family']=='static'],dtype=torch.float32)),out/'encoded'/f"{c['id']}.pth")
   if i%40==0:print('ENCODE',i,len(cases),flush=True)
 save(out/'encode_complete.json',dict(cases=len(cases),import_path=str(Path(sys.modules['models.visual_world_model'].__file__).resolve()),frozen_hash=frozen_hash(wm)))

def batch(out,cases):
 import torch
 rows=[torch.load(out/'encoded'/f"{x['id']}.pth",weights_only=False,map_location='cpu') for x in cases]
 return {k:torch.cat([r[k] for r in rows]).cuda() for k in rows[0]}

def initial(wm):return {k:p.detach().cpu().clone() for k,p in wm.predictor.state_dict().items()}

def optimize(wm):
 import torch
 return torch.optim.AdamW(wm.predictor.parameters(),lr=1e-5,betas=(.9,.999),weight_decay=.01)

def calibrate(root,repo,out):
 import torch
 wm,cfg,n=runtime(root,repo,out);params=read(out/'parameters.json');cases=[c for c in read(out/'cases.json') if c['split']=='calibration_fit'];b=batch(out,[next(c for c in cases if c['family']=='static'),next(c for c in cases if c['family']=='recorded')]);start=initial(wm);fh=frozen_hash(wm)
 # Deterministic eval-mode independent loss/gradient checks, still with gradients.
 parts,st=loss_parts(wm,b,n,params);ps=list(wm.predictor.parameters())
 grad=lambda loss:torch.autograd.grad(loss,ps,retain_graph=True)
 gb=grad(parts['base']);gz=grad(objective(parts,'E',0.,0.));g0=max(float((a-bb).abs().max()) for a,bb in zip(gb,gz));assert g0==0
 norms={k:float(torch.sqrt(sum(g.square().sum() for g in grad(v)))) for k,v in parts.items()};assert all(np.isfinite(v) and v>0 for v in norms.values())
 lam=float(np.clip(.1*norms['base']/norms['physics'],.1,100));ls=float(np.clip(.1*norms['base']/norms['stationary'],.1,100))
 eq=float((transition(b['start'].double(),b['u'].double(),params)-b['truth'][:,0].double()).abs().max());assert eq<1e-5
 q=wm.encode_proprio(((b['start']-n['state_mean'])/n['state_std'])[:,None])[:,0];back=((q-n['bias'])@n['pinv'].T)*n['state_std']+n['state_mean'];inverse_error=float((back-b['start']).abs().max());assert inverse_error<1e-5
 # Compare cached base implementation against actual repository forward on a coherent static context.
 from datasets.img_transforms import default_transform
 static_case=next(c for c in cases if c['family']=='static');zz=sources(root,repo)['base']/Z;bb=next(x for x in read(zz/'branches.json') if x['episode']==static_case['episode'] and x['name']=='coast_zero');im=np.load(zz/bb['image_file'])['images'];im=np.concatenate([im,im[-1:]],axis=0)
 obs=dict(visual=default_transform(int(cfg.img_size))(torch.tensor(im).float().permute(0,3,1,2)/255)[None].cuda(),proprio=((torch.tensor(bb['context_states']+[bb['start']],device='cuda',dtype=torch.float32)-n['state_mean'])/n['state_std'])[None]);acts=b['act'][:1].repeat(1,4,1)
 with torch.no_grad():
  actual=wm(obs,acts)[3];token=wm.encode(obs,acts);cached=original_loss(wm,wm.predict(token[:,:3]),token[:,1:]);base_error=float((actual-cached).abs());assert base_error<1e-7
 del parts,st,gb,gz,q,obs,token;wm.zero_grad(set_to_none=True)
 opt=optimize(wm);wm.predictor.train();times=[]
 for i in range(5):
  torch.cuda.synchronize();t=time.monotonic();opt.zero_grad(set_to_none=True);parts,st=loss_parts(wm,b,n,params);loss=objective(parts,'E',lam,ls);assert torch.isfinite(loss) and torch.isfinite(st).all();loss.backward();gn=torch.nn.utils.clip_grad_norm_(wm.predictor.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();times.append(time.monotonic()-t)
 assert frozen_hash(wm)==fh;delta=max(float((p.detach().cpu()-start[k]).abs().max()) for k,p in wm.predictor.state_dict().items());assert delta>0
 elapsed=sum(read(p)['elapsed_seconds'] for p in (root/'work'/f'{out.name}-jobs').glob('*/status.json') if read(p).get('gpu') and 'elapsed_seconds' in read(p))
 steps=min(120,int((2700-elapsed)/(4*2*max(times))));assert steps>=20
 save(out/'training_config.json',dict(seed=0,steps=steps,batch=2,lr=1e-5,lambda_state_or_physics=lam,lambda_stationary=ls,gradient_norms=norms,benchmark_step_seconds=times,selection='fixed train gradient scale; no metric tuning',checkpoint_selection='final step',budget_formula='min(120,floor((2700-prior_gpu_seconds)/(4*2*max_benchmark_step_seconds)))',prior_gpu_seconds=elapsed))
 save(out/'gradient_checks.json',dict(pass_all=True,zero_weight_gradient_max=g0,upstream_forward_loss_max=base_error,float_oracle_max=eq,inverse_roundtrip_max=inverse_error,predictor_update_max=delta,frozen_hash_unchanged=True,finite_loss_output_gradient=True,two_step_backprop=True,benchmark_discarded=True,import_path=str(Path(sys.modules['models.visual_world_model'].__file__).resolve())))

def schedule(cases,steps):
 rng=np.random.default_rng(0);static=[x for x in cases if x['family']=='static'];other=[x for x in cases if x['family']!='static'];a=rng.permutation(len(static));b=rng.permutation(len(other))
 return [[static[a[i%len(a)]],other[b[i%len(b)]]] for i in range(steps)]

def train(root,repo,out,arm):
 import torch
 wm,cfg,n=runtime(root,repo,out);fh=frozen_hash(wm);params=read(out/'parameters.json');conf=read(out/'training_config.json');assert read(out/'gradient_checks.json')['pass_all'];torch.manual_seed(0);np.random.seed(0);wm.predictor.train();opt=optimize(wm)
 cases=read(out/'cases.json');batches=schedule([c for c in cases if c['split']=='calibration_fit'],conf['steps']);curve=[];history=[];start=time.monotonic();folder=out/arm;folder.mkdir(exist_ok=False)
 for i,rows in enumerate(batches):
  b=batch(out,rows);opt.zero_grad(set_to_none=True);parts,st=loss_parts(wm,b,n,params);loss=objective(parts,arm,conf['lambda_state_or_physics'],conf['lambda_stationary']);assert torch.isfinite(loss) and torch.isfinite(st).all();loss.backward();gn=torch.nn.utils.clip_grad_norm_(wm.predictor.parameters(),1.,error_if_nonfinite=True);opt.step();curve.append(dict(step=i+1,loss=float(loss.detach()),grad_norm=float(gn),**{k:float(v.detach()) for k,v in parts.items()}));history.append([x['id'] for x in rows])
  if (i+1)%20==0:print('TRAIN',arm,i+1,curve[-1],flush=True)
 assert frozen_hash(wm)==fh
 torch.save(dict(predictor={k:p.detach().cpu() for k,p in wm.predictor.state_dict().items()},optimizer=opt.state_dict(),step=conf['steps'],arm=arm,config=conf),folder/'checkpoint.pth');save(folder/'curve.json',curve);save(folder/'exposure.json',history);save(folder/'complete.json',dict(steps=conf['steps'],frozen_hash=fh,checkpoint_sha256=sha(folder/'checkpoint.pth'),elapsed_seconds=time.monotonic()-start,trainable_parameters=sum(p.numel() for p in wm.parameters() if p.requires_grad),trainable_only_predictor=all(name.startswith('predictor.') for name,p in wm.named_parameters() if p.requires_grad)))

def evaluate(root,repo,out):
 import torch
 wm,cfg,n=runtime(root,repo,out);wm.requires_grad_(False);params=read(out/'parameters.json');cases=[c for c in read(out/'cases.json') if c['split']!='calibration_fit'];base=sources(root,repo)['base'];depth=base/DEPTH
 # Fix all trained checkpoints before opening evaluation metrics.
 complete={a:read(out/a/'complete.json') for a in 'BCDE'};assert len({json.dumps(read(out/a/'exposure.json')) for a in 'BCDE'})==1
 save(out/'evaluation_lock.json',dict(checkpoints=complete,protocol_sha256=sha(Path(__file__).with_name('physics_loss_protocol.md')),cases=[x['id'] for x in cases],settings=read(out/'training_config.json')))
 for arm in 'ABCDE':
  if arm!='A':wm.predictor.load_state_dict(torch.load(out/arm/'checkpoint.pth',map_location='cpu',weights_only=False)['predictor'])
  wm.eval();rows=[];reuse=0
  with torch.inference_mode():
   for i,c in enumerate(cases):
    b=batch(out,[c]);ctx=b['context'];act=b['act'];saved=None
    if arm=='A':
     f=depth/'token_cache'/f"rollout_{c['id']}_{'original' if c['family']=='recorded' else 'raw'}.pth"
     if f.exists():saved=torch.load(f,map_location='cpu',weights_only=False)
    states=[];pm=[];vm=[];off=[];avg=[];ys=[]
    for h in range(5):
     if saved is not None:y=saved[h].cuda();reuse+=1
     elif arm=='A' and h==0:y=torch.load(c['prediction_file'],map_location='cpu',weights_only=False)[c['prediction_key']].cuda();reuse+=1
     else:y=wm.predict(ctx)[:,-1:]
     st=decode(wm,y,n);states.append(st[0,0].cpu().tolist());avg.append(decode(wm,y,n,True)[0,0].cpu().tolist());truth=b['future'][:,h:h+1];pobs=wm.separate_emb(y)[0];tobs=wm.separate_emb(truth)[0];vm.append(float((pobs['visual']-tobs['visual']).square().mean()));pm.append(float((y[..., -(wm.proprio_dim+wm.action_dim):-wm.action_dim]-truth[..., -(wm.proprio_dim+wm.action_dim):-wm.action_dim]).square().mean()));prop=pobs['proprio'];reencoded=((st-n['state_mean'])/n['state_std'])@n['weight'].T+n['bias'];off.append(float((prop-reencoded).square().mean().sqrt()));assert torch.isfinite(y).all()
     ctx=torch.cat([ctx[:,1:],wm.replace_actions_from_z(y.clone(),act)],dim=1)
    rows.append(dict(id=c['id'],episode=c['episode'],split=c['split'],family=c['family'],condition=c['condition'],states=states,patch_mean_states=avg,visual_mse=vm,proprio_mse=pm,off_manifold=off))
   save(out/f'evaluation_{arm}.json',rows);print('EVALUATED',arm,len(rows),'REUSED',reuse,flush=True)
 save(out/'evaluation_complete.json',dict(arms=list('ABCDE'),cases_per_arm=len(cases),horizons=5,finite_outputs=True,same_exposure=True,sealed_test_opened=False))

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',required=True,choices=['prepare','encode','calibrate','train','evaluate']);p.add_argument('--arm',choices=list('BCDE'));a=p.parse_args();args=[a.root.resolve(),a.repo.resolve(),a.out.resolve()]
 if a.stage=='train':args.append(a.arm)
 globals()[a.stage](*args)
if __name__=='__main__':main()
