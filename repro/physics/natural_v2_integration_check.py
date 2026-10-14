"""[repro] Train-cache-only integration candidate. No checkpoint output or comparison stage."""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

from natural_v2_guard import production_entry_guard, require, read, recorded_update, update_counts
from natural_v2_integration_control import (digest, pinned, OutputBudget, REQUIRED, strict_read,
    validate_manifest, validate_cases, check_access, update_once, finite_tree, verify_pins, validate_update_values)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','repo','out','job','uuid','execution-plan','plan-sha256'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--stage',required=True,choices=['integration'])
    return p


def main():
    a=parser().parse_args()
    # [repro] Actual production parsing/receipt/hash/single-use consumption precedes model imports.
    production_entry_guard(sys.argv)
    plan=validate_manifest(strict_read(pinned(a.execution_plan,a.plan_sha256)),a.execution_plan)
    require(plan['status']=='AUTHORIZED_INTEGRATION_ONLY' and plan['uuid']==a.uuid,
            'separate integration authorization required')
    out,job,repo=map(lambda p:Path(p).resolve(),(a.out,a.job,a.repo))
    budget=OutputBudget(out)
    admitted=read(out/'admitted.json')
    require(admitted['uuid']==a.uuid and admitted['plan_sha256']==a.plan_sha256,'admission identity')
    cg=Path(admitted['cgroup'])
    require(str(os.getpid()) in (cg/'cgroup.procs').read_text().split(),'child outside bounded tree')
    allow=plan['input_allowlist']
    validate_cases(allow,strict_read(plan['documents']['split']['path']))
    require(allow['natural_order']==['N_78_10','N_88_10'] and
            allow['synthetic_order']==['Z_529','C_311_r0.2_d0'],'fixed case order')
    require(allow['episode_ids']==[78,88,311,529],'fixed training IDs')
    require(set(allow['natural'])==set(allow['natural_order']) and set(allow['synthetic'])==set(allow['synthetic_order']),'extra case metadata')
    for spec in allow['natural'].values():
        require(isinstance(spec['image_shape'],list) and len(spec['image_shape'])==4 and all(type(x) is int and x>0 for x in spec['image_shape']),'image shape not approved')
    expected_tensors={allow['A_path']}|{spec['path'] for spec in allow['synthetic'].values()}
    require(set(allow['tensor_hashes'])==expected_tensors,'tensor load scope too broad')
    for path,h in allow['hashes'].items():pinned(path,h)
    verify_pins(plan)
    rows=[];access=[]
    def result(identifier,expected,actual,ok,tolerance=0):
        row=dict(id=identifier,expected=expected,actual=actual,tolerance=tolerance,passed=bool(ok),
                 evidence=[str(out/'run_lock.json'),str(out/'admitted.json')])
        rows.append(row);budget.write('check_'+identifier+'.json',row)
        require(ok,identifier+' failed')
    result('I01_admission','approved production context, one consumed receipt',
           dict(uuid=a.uuid,child_consumed=(job/'child_consumed.json').is_file()),
           (job/'child_consumed.json').is_file())
    # [repro] Narrow disk writes, forbid network/child workers; all artifacts small JSON only.
    original_open=io.open;opening=[False];written=[0]
    class Writer:
        def __init__(self,f):self.f=f
        def __enter__(self):return self
        def __exit__(self,*args):return self.f.__exit__(*args)
        def __getattr__(self,key):return getattr(self.f,key)
        def write(self,value):
            size=len(value.encode() if isinstance(value,str) else value)
            require(written[0]+size<=4*1024**2,'child output reservation')
            written[0]+=size;return self.f.write(value)
    def bounded_open(file,mode='r',*args,**kwargs):
        writing=any(k in mode for k in 'wax+')
        if writing:
            p=Path(file).resolve()
            require(out in p.parents and (p.suffix=='.json' or p==out/'budget.lock'),'only reserved evidence writes allowed')
        opening[0]=True
        try:f=original_open(file,mode,*args,**kwargs)
        finally:opening[0]=False
        return Writer(f) if writing else f
    def audit(event,args):
        if event.startswith('socket.') or event.startswith('subprocess.') or event in ('os.fork','os.system','os.exec','os.posix_spawn'):
            raise RuntimeError('network/extra process forbidden')
        if event=='open' and not isinstance(args[0],int) and args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT):
            require(opening[0],'unreserved write')
    import builtins
    io.open=bounded_open;builtins.open=bounded_open;sys.addaudithook(audit)
    # [repro] 승인 inventory 밖의 repo/third-party import는 module body 이전에 거부한다.
    import importlib.machinery
    approved_imports=plan['host']['import_inventory']
    class PinnedFinder:
        def find_spec(self,fullname,path=None,target=None):
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec is not None and spec.origin not in (None,'built-in','frozen'):
                origin=str(Path(spec.origin).resolve())
                if origin.startswith(str(repo)+'/') or 'site-packages/' in origin or fullname.startswith('dinov2'):
                    require(fullname in approved_imports and approved_imports[fullname]['path']==origin,'unapproved import: '+fullname)
                    pinned(origin,approved_imports[fullname]['sha256'])
            return spec
    sys.meta_path.insert(0,PinnedFinder())
    # No preflight import, WhitelistData instantiation, raw global tensors or new fitting.
    import torch
    import numpy as np
    import frozen_runtime
    import physics_loss_pilot as pilot
    from natural_v2_data import encode_case
    from natural_v2_core import parts, objective, flatgrad, upgrade_batch
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    torch.manual_seed(0);np.random.seed(0)
    load=torch.load;np_load=np.load
    permitted_tensors={str(Path(p).resolve()):h for p,h in allow['tensor_hashes'].items()}
    hub_weights={row['path']:row['sha256'] for row in plan['file_pins']['hub_weights']}
    permitted_tensors.update(hub_weights)
    def limited_load(path,*args,**kwargs):
        require(isinstance(path,(str,Path)),'file-object tensor load forbidden')
        canonical=str(Path(path).resolve());require(canonical in permitted_tensors,'tensor outside allowlist')
        pinned(canonical,permitted_tensors[canonical])
        kind='model' if canonical==allow['A_path'] else 'hub' if canonical in hub_weights else 'synthetic'
        limit=2 if kind=='model' else allow['hub_load_counts'][canonical] if kind=='hub' else 1
        require(sum(row['path']==canonical for row in access)<limit,'load count exceeded')
        access.append(dict(kind=kind,path=canonical,sha256=permitted_tensors[canonical],approved_sha256=permitted_tensors[canonical]))
        return load(canonical,*args,**kwargs)
    torch.load=limited_load
    def limited_np_load(path,*args,**kwargs):
        canonical=str(Path(path).resolve())
        permitted={row['path'] for row in allow['natural'].values()}
        require(canonical in permitted and not any(row['path']==canonical for row in access),'NPZ access scope/count')
        require(kwargs.get('allow_pickle') is False,'pickle forbidden')
        pinned(canonical,allow['hashes'][canonical])
        access.append(dict(kind='natural',path=canonical,sha256=allow['hashes'][canonical],
            id=next(k for k,v in allow['natural'].items() if v['path']==canonical)))
        return np_load(path,*args,**kwargs)
    np.load=limited_np_load
    def no_save(*args,**kwargs):raise RuntimeError('checkpoint output forbidden')
    torch.save=no_save
    # Shared loader's memory fraction is restricted, never enlarged beyond approved headroom.
    original_fraction=torch.cuda.set_per_process_memory_fraction
    def fraction(value,device=None):
        total=torch.cuda.get_device_properties(device).total_memory
        return original_fraction(min(value,6500*1024**2/total),device)
    torch.cuda.set_per_process_memory_fraction=fraction
    hub_files={row['path']:row['sha256'] for row in plan['file_pins']['hub_sources']}
    require(hub_files and hub_weights,'pinned local DINO unavailable')
    for path,h in hub_weights.items():pinned(path,h)
    wm,cfg=frozen_runtime.load_frozen(Path(a.root),repo)
    initial_hash=frozen_runtime.weight_hash(wm)
    wm.predictor.requires_grad_(True)
    norm_data=read(pinned(allow['normalization_path'],allow['hashes'][allow['normalization_path']]))
    n={k:torch.tensor(norm_data[k],device='cuda',dtype=torch.float32)
       for k in ('state_mean','state_std','action_mean','action_std','weight','bias')}
    n['pinv']=torch.linalg.pinv(n['weight'])  # same fixed inverse construction; no data fit
    params=read(pinned(allow['parameters_path'],allow['hashes'][allow['parameters_path']]))
    for tensor in n.values():tensor.requires_grad_(False)
    def frozen_signature():
        h=hashlib.sha256()
        for name,tensor in sorted(wm.state_dict().items()):
            if not name.startswith('predictor.'):
                h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        for name,tensor in sorted(wm.named_buffers()):
            h.update(('buffer:'+name).encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        for name,tensor in sorted(n.items()):
            h.update(name.encode());h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        for name,module in wm.named_modules():
            bias=getattr(module,'bias',None)
            if isinstance(bias,torch.Tensor) and 'bias' not in module._parameters and 'bias' not in module._buffers:
                h.update(name.encode());h.update(bias.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    frozen=frozen_signature();original=pilot.initial(wm)
    imports={name:str(Path(sys.modules[name].__file__).resolve()) for name in
             ('frozen_runtime','physics_loss_pilot','natural_v2_data','natural_v2_core','models.visual_world_model')}
    require(all(Path(path).is_relative_to(repo) for path in imports.values()),'unexpected model source import')
    hub_modules={name:str(Path(mod.__file__).resolve()) for name,mod in list(sys.modules.items())
                 if name.startswith('dinov2') and getattr(mod,'__file__',None)}
    require(all(path in hub_files for path in hub_modules.values()),'unexpected local hub import')
    count=sum(p.numel() for p in wm.predictor.parameters())
    result('I03_model','A tensor hash; 20122600 predictor-only parameters; frozen encoders',
           dict(tensor_sha256=initial_hash,predictor_parameters=count,imports=imports,hub_imports=hub_modules,frozen=frozen),
           initial_hash==allow['A_tensor_sha256'] and count==20122600 and
           all(name.startswith('predictor.') for name,p in wm.named_parameters() if p.requires_grad))
    encoded=[]
    shapes=dict(context_states=(3,4),truth=(5,4),start=(4,),controls=(5,5,2),
                context_actions=(3,5,2),future_actions=(5,5,2),mask=(5,))
    for case_id in allow['natural_order']:
        spec=allow['natural'][case_id];path=pinned(spec['path'],allow['hashes'][spec['path']])
        require(spec['id']==case_id and spec['episode'] in (78,88) and spec['raw_anchor']==10 and
                spec['frames']==[0,5,10,15,20,25,30,35],'natural metadata mismatch')
        with np.load(path,allow_pickle=False) as archive:
            require(set(archive.files)==set(shapes)|{'images'},'NPZ keys')
            data={k:archive[k] for k in archive.files}
        require(all(data[k].shape==shape for k,shape in shapes.items()),'natural cache shape')
        require(list(data['images'].shape)==spec['image_shape'],'unconfirmed image shape')
        require(data['images'].shape[0]==8 and data['images'].shape[-1]==3,'natural image layout')
        require(all(np.isfinite(v).all() for v in data.values()),'nonfinite cache')
        require(np.all(data['mask'][:2]==0),'natural mask changed; do not replace sample')
        encoded.append(encode_case(wm,cfg,n,dict(spec,**data,family='natural')))
    natural={k:torch.cat([row[k] for row in encoded]) for k in encoded[0]}
    require(all(bool(torch.isfinite(v).all()) for v in natural.values()),'nonfinite encoding')
    for key,episode in zip(allow['synthetic_order'],(529,311)):
        require(allow['synthetic'][key]['id']==key and allow['synthetic'][key]['episode']==episode and
                allow['synthetic'][key]['split']=='calibration_fit','synthetic metadata mismatch')
    cache_paths=[Path(allow['synthetic'][key]['path']) for key in allow['synthetic_order']]
    require(cache_paths[0].parent==cache_paths[1].parent and cache_paths[0].parent.name=='encoded','synthetic cache layout')
    synthetic=upgrade_batch(pilot.batch(cache_paths[0].parent.parent,[{'id':k} for k in allow['synthetic_order']]))
    check_access(access,allow)
    result('I02_input_scope','two fixed natural caches and two fixed synthetic caches only',
           dict(access=access,episode_ids=allow['episode_ids'],statistics_fit='static_not_called',raw_global_load='static_not_called'),
           [x['id'] for x in access if x['kind']=='natural']==allow['natural_order'])
    update_results=[];callback_count=0
    for step,(name,batch) in enumerate((('natural',natural),('synthetic',synthetic)),1):
        wm.predictor.load_state_dict(original);wm.eval();wm.predictor.train()
        torch.manual_seed(0);np.random.seed(0);wm.zero_grad(set_to_none=True)
        require(frozen_runtime.weight_hash(wm)==initial_hash,'A reset differs')
        opt=pilot.optimize(wm)
        losses,states=parts(wm,batch,n,params,stop_input=True)
        loss,scale=objective(losses,wm,batch['mask'],'P','norm',strength=1.)
        require(bool(torch.isfinite(states).all()) and all(bool(torch.isfinite(x).all()) for x in losses.values()) and
                bool(torch.isfinite(loss).all()),'nonfinite state/loss')
        basegrad=flatgrad(losses['base'],wm)
        require(bool(torch.isfinite(basegrad).all()) and float(basegrad.norm())>0,'invalid base gradient')
        require(all(math.isfinite(v) for v in scale.values() if type(v) in (int,float)),'nonfinite scale')
        if name=='natural':
            result('I04_natural_encode','finite encode, two-step q/v, losses and gradients',
                   dict(shapes={k:list(v.shape) for k,v in natural.items()},base_norm=float(basegrad.norm())),
                   bool(torch.isfinite(states).all()) and bool(torch.isfinite(loss)))
            result('I05_zero_mask','mask=0, lambda=0, aux_norm=0, aux_skipped',scale,
                   float(batch['mask'][:,:2].sum())==0 and scale['lambda']==0 and
                   scale['aux_norm']==0 and scale['aux_skipped'])
        else:
            result('I06_active_aux','active mask; aux norm>1e-12; lambda<=10; ratio=0.1',scale,
                   float(batch['mask'][:,:2].sum())>0 and scale['aux_norm']>1e-12 and
                   0<=scale['lambda']<=10 and abs(scale['ratio']-.1)<=1e-5,1e-5)
        loss.backward();torch.nn.utils.clip_grad_norm_(wm.predictor.parameters(),1.,error_if_nonfinite=True)
        require(update_counts(job)['attempted']==step-1,'duplicate/unknown prior update')
        verified={}
        def tensor_finite(value):
            if isinstance(value,torch.Tensor):
                require(bool(torch.isfinite(value).all()),'nonfinite tensor');return True
            return False
        def update():
            nonlocal callback_count
            require(callback_count<2,'optimizer callback limit');callback_count+=1;opt.step()
        def after_update():
            finite_tree(wm.predictor.state_dict(),tensor_finite)
            finite_tree(opt.state_dict(),tensor_finite)
            steps=[float(v['step']) for v in opt.state.values()]
            delta=max(float((v.detach().cpu()-original[k]).abs().max()) for k,v in wm.predictor.state_dict().items())
            validate_update_values(steps,delta)
            require(frozen_signature()==frozen,'frozen values changed')
            require(all(not module.training for module_name,module in wm.named_modules()
                        if module_name and not module_name.startswith('predictor')),'frozen mode changed')
            verified.update(steps=steps,delta=delta)
        # [repro] COMPLETE는 step, CUDA sync, finite/frozen/step1 검사 다음에만 기록된다.
        update_once(job,name,step,update,torch.cuda.synchronize,after_update)
        steps,delta=verified['steps'],verified['delta']
        update_results.append(dict(case=name,step=step,optimizer_steps=steps,delta=delta,frozen_unchanged=True))
        del opt,loss,losses,states,basegrad
    result('I07_update','two attempted/completed callbacks, fresh optimizer step1, frozen unchanged',
           dict(callbacks=callback_count,counts=update_counts(job),updates=update_results),
           callback_count==2 and update_counts(job)==dict(attempted=2,completed=2,uncertain=0,legacy_reserved=0))
    payload=torch.load(allow['A_path'],map_location='cpu',weights_only=False)
    wm.predictor.load_state_dict(payload['predictor'].state_dict());del payload
    restored=frozen_runtime.weight_hash(wm)
    verify_pins(plan)
    check_access(access,allow)
    require(sum(x['kind']=='model' for x in access)==2,'A initial/restore load count')
    require(all(sum(x['path']==path for x in access)==count for path,count in allow['hub_load_counts'].items()),'hub load multiset')
    approved_imports=plan['host']['import_inventory']
    for module_name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if not path:continue
        path=str(Path(path).resolve())
        if path.startswith(str(repo)+'/') or 'site-packages/' in path or path in hub_files:
            require(module_name in approved_imports and approved_imports[module_name]['path']==path,'unapproved imported source: '+module_name)
            pinned(path,approved_imports[module_name]['sha256'])
    result('I08_restore','original A tensor and all pinned bytes unchanged; no saved weights',
           dict(tensor_sha256=restored,checkpoints_saved=0),
           restored==initial_hash==allow['A_tensor_sha256'] and frozen_signature()==frozen)
    ordered=[next(row for row in rows if row['id']==name) for name in REQUIRED[:-1]]
    require(len(rows)==len(ordered) and len({r['id'] for r in rows})==len(rows),'duplicate/missing checks')
    budget.write('model_checks.json',dict(uuid=a.uuid,checks=ordered,access=access))


if __name__!='__main__':
    raise RuntimeError('unsupported import; approved integration CLI only')
main()
