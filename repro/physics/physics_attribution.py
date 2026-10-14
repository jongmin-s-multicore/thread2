"""Controlled loss ablations reusing immutable pilot data and baselines."""
import argparse,json,time,sys,subprocess,shutil
from pathlib import Path
import numpy as np
from intervention import read,save,sha
import physics_loss_pilot as pilot
OLD='pointmaze-physics-loss-pilot-20261011'
NEW=['G1','P1','DS','CL','DL','CM','DM','CT','CI']
ARMS=list('ABCDE')+NEW
PAIRS=[('G1','B'),('P1','G1'),('C','G1'),('D','P1'),('DS','D'),('CL','C'),('DL','D'),('DM','CM'),('CM','C'),('DM','D'),('CT','C'),('E','D'),('CI','C'),('CT','CI')]

def source(out):return Path(read(out/'attribution_manifest.json')['source'])

def prepare(root,repo,out):
 src=root/'work/physics'/OLD;out.mkdir(parents=True,exist_ok=False);hashes={}
 for name in ['checks.json','gradient_checks.json','independent_verification.json','input_path_audit.json']:
  assert read(src/name)['pass_all'];hashes[str(src/name)]=sha(src/name)
 for f,h in read(src/'encoded_and_checkpoint_hashes.json').items():assert sha(src/f)==h;hashes[str(src/f)]=h
 for name in ['cases.json','oracles.json','parameters.json','normalization.json','split.json','training_config.json','manifest.json']:
  shutil.copy2(src/name,out/name);hashes[str(src/name)]=sha(src/name)
 for arm in 'ABCDE':
  f=src/f'evaluation_{arm}.json';shutil.copy2(f,out/f.name);hashes[str(f)]=sha(f)
  if arm!='A':
   (out/arm).mkdir()
   for name in ['complete.json','curve.json','exposure.json']:shutil.copy2(src/arm/name,out/arm/name);hashes[str(src/arm/name)]=sha(src/arm/name)
 shutil.copy2(Path(__file__).with_name('physics_attribution_protocol.md'),out/'protocol_ko.md')
 shutil.copy2(Path(__file__).with_name('physics_attribution_amendment.md'),out/'protocol_amendment_ko.md')
 save(out/'attribution_manifest.json',dict(source=str(src),sources=hashes,previous_status=read(root/'work/STATUS.json'),code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),protocol_sha256=sha(out/'protocol_ko.md'),new_arms=NEW,reused_arms=list('ABCDE'),sealed_test_opened=False))
 save(out/'prepare_complete.json',dict(cases_reused=len(read(out/'cases.json')),encoded_caches_reused=336,baseline_evaluations_reused=5,source_hashes_verified=True))

def parts(wm,b,n,params,teacher=False,detach_input=False):
 import torch
 first=wm.predict(b['context']);y1=first[:,-1:];incoming=b['future'][:,:1].clone() if teacher else y1.detach().clone() if detach_input else y1.clone();incoming=wm.replace_actions_from_z(incoming,b['act']);y2=wm.predict(torch.cat([b['context'][:,1:],incoming],dim=1))[:,-1:]
 st=torch.cat([pilot.decode(wm,y1,n),pilot.decode(wm,y2,n)],dim=1);den=b['mask'].sum().clamp(min=1)
 def terms(error):return (error/n['state_std']).square()
 def reduce(x,h):return (x.mean(-1)*b['mask'][:,h]).sum()/den
 errors={};errors['gt1']=terms(st[:,0]-b['truth'][:,0]);errors['ph1']=terms(st[:,0]-pilot.transition(b['start'],b['u'],params));errors['gt2']=terms(st[:,1]-b['truth'][:,1]);ph=pilot.transition(st[:,0],b['u'],params);errors['rec2']=terms(st[:,1]-ph);errors['stop2']=terms(st[:,1]-ph.detach())
 losses={k:reduce(v,0 if k.endswith('1') else 1) for k,v in errors.items()};losses['base']=pilot.original_loss(wm,first,torch.cat([b['context'][:,1:],b['future'][:,:1]],dim=1))
 for k in ['gt2','rec2']:
  for name,indices in [('q',slice(0,2)),('v',slice(2,4))]:losses[k+'_'+name]=reduce(errors[k][:,indices],1)*.5
 losses['C']=losses['gt1']+losses['gt2'];losses['D']=losses['ph1']+losses['rec2'];losses['DS']=losses['ph1']+losses['stop2'];stat=b['static'][:,None].expand(-1,2)
 losses['stationary']=((terms(st-b['start'][:,None])).mean(-1)*stat).sum()/stat.sum().clamp(min=1)
 return losses,st

def flatgrad(loss,wm):
 import torch
 return torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,list(wm.predictor.parameters()),retain_graph=True)])

def calibrate(root,repo,out):
 import torch
 src=source(out);wm,cfg,n=pilot.runtime(root,repo,out);params=read(out/'parameters.json');cases=[c for c in read(out/'cases.json') if c['split']=='calibration_fit'];batches=pilot.schedule(cases,120);fh=pilot.frozen_hash(wm);metrics=[];max_target=max_grad=0.;t=time.monotonic();wm.predictor.train()
 for i,cs in enumerate(batches[:12]):
  b=pilot.batch(src,cs);torch.manual_seed(0);old,oldst=pilot.loss_parts(wm,b,n,params);oldgrads={k:flatgrad(old[k],wm) for k in ['base','supervised','physics','stationary']};torch.manual_seed(0);new,st=parts(wm,b,n,params)
  for oldkey,newkey in [('base','base'),('supervised','C'),('physics','D'),('stationary','stationary')]:
   assert abs(float(old[oldkey]-new[newkey]))<1e-7
   g=flatgrad(new[newkey],wm);rel=float((g-oldgrads[oldkey]).norm()/oldgrads[oldkey].norm().clamp(min=1e-12));assert rel<1e-5
  if i==0:assert abs(float(new['base'])-read(src/'B/curve.json')[0]['base'])<1e-7
  del old,oldst,oldgrads,g
  gs={k:flatgrad(new[k],wm) for k in ['base','gt1','ph1','gt2','rec2','stop2','gt2_q','gt2_v','rec2_q','rec2_v']}
  target=float((pilot.transition(b['start'],b['u'],params)-b['truth'][:,0]).abs().max());relative=float((gs['gt1']-gs['ph1']).norm()/gs['gt1'].norm().clamp(min=1e-12));max_target=max(max_target,target);max_grad=max(max_grad,relative);assert target<1e-5 and relative<1e-3
  assert float((new['rec2']-new['stop2']).abs())==0
  cosine=lambda a,b:float((a@b)/(a.norm()*b.norm()).clamp(min=1e-12))
  metrics.append(dict(batch=i,cases=[c['id'] for c in cs],h1_target_max=target,h1_grad_relative=relative,losses={k:float(v.detach()) for k,v in new.items()},norms={k:float(v.norm()) for k,v in gs.items()},cos_base={k:cosine(gs['base'],v) for k,v in gs.items()},cos_gt2_rec2=cosine(gs['gt2'],gs['rec2']),cos_rec2_stop2=cosine(gs['rec2'],gs['stop2']),rec2_target_gradient_norm=float((gs['rec2']-gs['stop2']).norm())))
  del new,st,gs
 assert pilot.frozen_hash(wm)==fh
 # First-step unchanged RNG and frozen behavior are gated above; small bounded benchmark.
 torch.manual_seed(0);wm.zero_grad(set_to_none=True);opt=pilot.optimize(wm);bt=[];conf=read(out/'training_config.json')
 for i in range(3):
  b=pilot.batch(src,batches[i]);torch.cuda.synchronize();tt=time.monotonic();opt.zero_grad(set_to_none=True);ll,_=parts(wm,b,n,params);gb=flatgrad(ll['base'],wm);ga=flatgrad(ll['D'],wm);lam=float(.1*gb.norm()/ga.norm().clamp(min=1e-12));assert 0<lam<=10;del gb,ga;(ll['base']+lam*ll['D']).backward();torch.nn.utils.clip_grad_norm_(wm.predictor.parameters(),1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();bt.append(time.monotonic()-tt)
 assert pilot.frozen_hash(wm)==fh
 previous=90.81711977836676+sum(read(p).get('elapsed_seconds',0) for p in (root/'work'/f'{out.name}-jobs').glob('*/status.json') if read(p).get('gpu'))+time.monotonic()-t
 estimate=len(NEW)*120*max(bt)*2+120
 assert previous+estimate<3600,'Insufficient budget for all fixed arms'
 save(out/'gradient_audit.json',metrics);save(out/'attribution_checks.json',dict(pass_all=True,legacy_losses_and_gradients_equivalent=True,h1_target_max=max_target,h1_gradient_relative_max=max_grad,stop_target_forward_identical=True,frozen_hash_unchanged=True,benchmark_discarded=True,benchmark_seconds=bt,conservative_remaining_estimate_seconds=estimate,prior_gpu_seconds=previous,import_path=str(Path(sys.modules['models.visual_world_model'].__file__).resolve())))

def train(root,repo,out,arm):
 import torch
 assert arm in NEW;assert read(out/'attribution_checks.json')['pass_all'];src=source(out);wm,cfg,n=pilot.runtime(root,repo,out);fh=pilot.frozen_hash(wm);wm.predictor.train();torch.manual_seed(0);np.random.seed(0);opt=pilot.optimize(wm);conf=read(out/'training_config.json');params=read(out/'parameters.json');cases=[c for c in read(out/'cases.json') if c['split']=='calibration_fit'];schedule=pilot.schedule(cases,120);folder=out/arm;folder.mkdir();curve=[];t=time.monotonic()
 auxkey={'G1':'gt1','P1':'ph1','DS':'DS','CL':'C','DL':'D','CM':'C','DM':'D','CT':'C','CI':'C'}[arm]
 if arm=='CI':
  amendment=Path(__file__).with_name('physics_attribution_amendment.md');assert sha(amendment)==sha(out/'protocol_amendment_ko.md');b=pilot.batch(src,schedule[0]);torch.manual_seed(0);regular,_=parts(wm,b,n,params);g1=flatgrad(regular['C'],wm);torch.manual_seed(0);stopped,_=parts(wm,b,n,params,detach_input=True);g2=flatgrad(stopped['C'],wm);err=float((regular['C']-stopped['C']).abs());assert err==0;save(out/'input_stop_checks.json',dict(pass_all=True,forward_loss_abs=err,gradient_difference_norm=float((g1-g2).norm()),amendment_sha256=sha(amendment)));del regular,stopped,g1,g2;torch.manual_seed(0)
 for i,cs in enumerate(schedule):
  b=pilot.batch(src,cs);opt.zero_grad(set_to_none=True);ll,st=parts(wm,b,n,params,teacher=arm=='CT',detach_input=arm=='CI');lam=conf['lambda_state_or_physics']*(.1 if arm in ['CL','DL'] else 1);ratio=None
  if arm in ['CM','DM']:
   gb=flatgrad(ll['base'],wm);ga=flatgrad(ll[auxkey],wm);lam=float(.1*gb.norm()/ga.norm().clamp(min=1e-12));ratio=float(lam*ga.norm()/gb.norm());assert 0<lam<=10;del gb,ga
  loss=ll['base']+lam*ll[auxkey];assert torch.isfinite(loss) and torch.isfinite(st).all();loss.backward();gn=torch.nn.utils.clip_grad_norm_(wm.predictor.parameters(),1.,error_if_nonfinite=True);opt.step();curve.append(dict(step=i+1,loss=float(loss.detach()),base=float(ll['base'].detach()),physics=float(ll['D'].detach()),stationary=float(ll['stationary'].detach()),lambda_aux=lam,aux_norm_ratio=ratio,grad_norm=float(gn),components={k:float(v.detach()) for k,v in ll.items()}))
  if (i+1)%40==0:print(arm,i+1,curve[-1]['loss'],flush=True)
 assert pilot.frozen_hash(wm)==fh
 torch.save(dict(predictor=pilot.initial(wm),arm=arm,steps=120,seed=0,base_config=conf,aux_key=auxkey),folder/'checkpoint.pth');save(folder/'curve.json',curve);save(folder/'exposure.json',[[c['id'] for c in cs] for cs in schedule]);save(folder/'complete.json',dict(checkpoint_sha256=sha(folder/'checkpoint.pth'),steps=120,seed=0,frozen_hash=fh,elapsed_seconds=time.monotonic()-t,code_commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),trainable_only_predictor=all(k.startswith('predictor.') for k,p in wm.named_parameters() if p.requires_grad)))

def evaluate(root,repo,out):
 import torch
 src=source(out);wm,cfg,n=pilot.runtime(root,repo,out);wm.requires_grad_(False);wm.eval();cases=[c for c in read(out/'cases.json') if c['split']!='calibration_fit'];lock={a:read(out/a/'complete.json') for a in NEW};assert all(read(out/a/'exposure.json')==read(src/'B/exposure.json') for a in NEW)
 save(out/'evaluation_lock_attribution.json',dict(checkpoints=lock,cases=[c['id'] for c in cases],protocol_sha256=sha(out/'protocol_ko.md'),amendment_sha256=sha(out/'protocol_amendment_ko.md')))
 states_g=torch.load(out/'G1/checkpoint.pth',map_location='cpu',weights_only=False)['predictor'];states_p=torch.load(out/'P1/checkpoint.pth',map_location='cpu',weights_only=False)['predictor'];delta=max(float((states_g[k]-states_p[k]).abs().max()) for k in states_g);del states_g,states_p
 save(out/'h1_checkpoint_equivalence.json',dict(max_parameter_abs=delta,threshold=1e-4,pass_threshold=delta<1e-4))
 for arm in NEW:
  wm.predictor.load_state_dict(torch.load(out/arm/'checkpoint.pth',map_location='cpu',weights_only=False)['predictor']);rows=[]
  with torch.inference_mode():
   for c in cases:
    b=pilot.batch(src,[c]);ctx=b['context'];states=[];avg=[];vm=[];pm=[];off=[]
    for hi in range(5):
     y=wm.predict(ctx)[:,-1:];st=pilot.decode(wm,y,n);states.append(st[0,0].cpu().tolist());avg.append(pilot.decode(wm,y,n,True)[0,0].cpu().tolist());target=b['future'][:,hi:hi+1];ob=wm.separate_emb(y)[0];tar=wm.separate_emb(target)[0];vm.append(float((ob['visual']-tar['visual']).square().mean()));pm.append(float((y[...,-(wm.proprio_dim+wm.action_dim):-wm.action_dim]-target[...,-(wm.proprio_dim+wm.action_dim):-wm.action_dim]).square().mean()));rec=((st-n['state_mean'])/n['state_std'])@n['weight'].T+n['bias'];off.append(float((ob['proprio']-rec).square().mean().sqrt()));assert torch.isfinite(y).all();ctx=torch.cat([ctx[:,1:],wm.replace_actions_from_z(y.clone(),b['act'])],dim=1)
    rows.append(dict(id=c['id'],episode=c['episode'],split=c['split'],family=c['family'],condition=c['condition'],states=states,patch_mean_states=avg,visual_mse=vm,proprio_mse=pm,off_manifold=off))
  save(out/f'evaluation_{arm}.json',rows);print('EVALUATED',arm,len(rows),flush=True)
 assert all(sha(f)==h for f,h in read(out/'attribution_manifest.json')['sources'].items())
 save(out/'evaluation_complete_attribution.json',dict(new_arms=NEW,reused_arms=list('ABCDE'),same_exposure=True,frozen=True,no_future_observations=True,sources_unchanged=True))

def analyze(root,repo,out):
 from analyze_physics_loss import analyze as evaluate_metrics
 evaluate_metrics(root,repo,out,arms=ARMS,comparisons=PAIRS)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','calibrate','train','evaluate','analyze'],required=True);p.add_argument('--arm');a=p.parse_args();args=[a.root.resolve(),a.repo.resolve(),a.out.resolve()]
 if a.stage=='train':args.append(a.arm)
 globals()[a.stage](*args)
if __name__=='__main__':main()
