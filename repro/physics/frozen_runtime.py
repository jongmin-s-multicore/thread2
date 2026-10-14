"""Load the existing frozen WM and exact proprio inverse; no downloads."""
import hashlib,os,sys
import numpy as np
from intervention import sources,read

def load_frozen(root,repo):
 import torch,hydra
 from omegaconf import OmegaConf
 torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available();torch.cuda.set_per_process_memory_fraction(.6,0)
 sys.path.insert(0,str(repo));os.environ['TORCH_HOME']=str(root/'work/torch_home');paths=sources(root,repo)
 cfg=OmegaConf.load(paths['cfg']);original=torch.hub.load;hub=root/'work/torch_home/hub/facebookresearch_dinov2_85a24602099d397264d5b30461ad7f3bfd726ca1'
 def local(repo_or_dir,model,*args,**kw):
  assert str(repo_or_dir).startswith('facebookresearch/dinov2:')
  return original(str(hub),model,*args,source='local',**kw)
 torch.hub.load=local
 def no_download(*args,**kw):raise RuntimeError('No downloads allowed')
 torch.hub.download_url_to_file=no_download
 p=torch.load(paths['cfg'].parent/'checkpoints/model_latest.pth',map_location='cuda:0',weights_only=False);encoder=p.get('encoder')
 if encoder is None:encoder=hydra.utils.instantiate(cfg.encoder)
 wm=hydra.utils.instantiate(cfg.model,encoder=encoder,predictor=p['predictor'],decoder=None,proprio_encoder=p['proprio_encoder'],action_encoder=p['action_encoder'],proprio_dim=cfg.proprio_emb_dim,action_dim=cfg.action_emb_dim,concat_dim=cfg.concat_dim,num_action_repeat=cfg.num_action_repeat,num_proprio_repeat=cfg.num_proprio_repeat).to('cuda:0')
 wm.eval();wm.requires_grad_(False)
 for m in wm.modules():
  if isinstance(getattr(m,'bias',None),torch.Tensor) and 'bias' not in m._parameters and 'bias' not in m._buffers:m.bias=m.bias.to('cuda:0')
 return wm,cfg

def weight_hash(wm):
 h=hashlib.sha256()
 for k,v in sorted(wm.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def inverse(norm):
 w=np.array(norm['weight']);b=np.array(norm['bias']);pinv=np.linalg.pinv(w);sm=np.array(norm['state_mean']);ss=np.array(norm['state_std'])
 def decode(wm,y):
  prop=wm.separate_emb(y)[0]['proprio'][0,0].cpu().numpy().astype(float);n=(prop-b)@pinv.T
  return n*ss+sm,float(np.linalg.norm(prop-(n@w.T+b)))
 return decode
