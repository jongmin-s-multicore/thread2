"""Episode-level paired analysis of fixed-context action intervention."""
import argparse
from pathlib import Path
import time
import numpy as np
from intervention import read,save,csvwrite,sha,clearance


def aggregate(rows,field):
    valid=[r for r in rows if r.get(field) is not None]
    ids=sorted(set(r['episode'] for r in valid))
    if not ids:return dict(mean=None,ci95=None,episodes=0,rows=0)
    x=np.array([np.mean([r[field] for r in valid if r['episode']==ep]) for ep in ids])
    draws=np.random.default_rng(0).integers(len(x),size=(10000,len(x)))
    return dict(mean=float(x.mean()),ci95=np.quantile(x[draws].mean(1),[.025,.975]).tolist(),episodes=len(ids),rows=len(valid))


def norm(x):return float(np.linalg.norm(x))


def energy_terms(start,end,force,mass,damping,dt=.25):
    v0,v1=start[2:],end[2:]
    dk=float(.5*(v1@mass@v1-v0@mass@v0));work=float(force@(end[:2]-start[:2]))
    loss=float(dt/3*(v0@damping@v0+v0@damping@v1+v1@damping@v1))
    return dict(delta_k=dk,work=work,necessary_excess=dk-work,linear_damping=loss,coarse_residual=dk-work+loss,
                kinematic_error=norm(end[:2]-start[:2]-dt*(v0+v1)/2))


def main(root,repo,out):
    assert read(out/'infer_complete.json')['weights_unchanged']
    starttime=time.monotonic();params=read(out/'parameters.json');mass,damping,gear=[np.array(params[k]) for k in ['M','D','G']]
    predictions=read(out/'predictions.json');oracles=read(out/'oracles.json');conditions=read(out/'conditions.json');contexts=read(out/'selected.json')
    lookup={(r['episode'],r['condition']):r for r in predictions};truth={(r['episode'],r['condition']):r for r in oracles}
    assert len(lookup)==len(truth)==len(contexts)*42 and lookup.keys()==truth.keys()
    condition={r['name']:r for r in conditions}
    rows=[];zero=[];jacobians=[];symmetry=[];linearity=[];fine=[];substeps=[];recorded=[]
    for ctx in contexts:
        ep=ctx['episode'];st=np.array(ctx['start']);split=ctx['split']
        z=np.array(lookup[ep,'zero']['prediction']);ze=np.array(truth[ep,'zero']['endpoint'])
        common=all(truth[ep,c['name']]['free'] for c in conditions if not c['stress'])
        zero.append(dict(episode=ep,split=split,common_free=common,initial_speed=norm(st[2:]),model_speed=norm(z[2:]),
            environment_speed=norm(ze[2:]),velocity_error=norm(z[2:]-ze[2:]),position_error=norm(z[:2]-ze[:2]),
            speed_ratio=norm(z[2:])/norm(st[2:]) if norm(st[2:])>1e-6 else None,
            model_energy_increase=float(.5*(z[2:]@mass@z[2:]-st[2:]@mass@st[2:])),
            environment_energy_increase=float(.5*(ze[2:]@mass@ze[2:]-st[2:]@mass@st[2:]))))
        for c in conditions:
            name=c['name'];pred=lookup[ep,name];oracle=truth[ep,name];model=np.array(pred['prediction']);env=np.array(oracle['endpoint'])
            assert np.max(np.abs(np.array(pred['requested_actions'])-oracle['requested_actions']))==0,'Raw command mismatch'
            assert pred['context_sha256']==lookup[ep,'zero']['context_sha256']
            dv,dve=model[2:]-z[2:],env[2:]-ze[2:];radius=c['radius'];u=np.array(oracle['requested_actions'])[0]
            direction=u/np.linalg.norm(u) if radius else np.zeros(2)
            deltaerr=norm(dv-dve)
            angle=None
            if norm(dv)>1e-6 and norm(dve)>1e-6:angle=float(np.degrees(np.arccos(np.clip(dv@dve/(norm(dv)*norm(dve)),-1,1))))
            r=dict(episode=ep,split=split,raw_anchor=ctx['raw_anchor'],condition=name,radius=radius,direction=c['direction'],
                stress=c['stress'],common_free=common,branch_free=oracle['free'],pair_free=oracle['free'] and truth[ep,'zero']['free'],
                velocity_error=norm(model[2:]-env[2:]),position_error=norm(model[:2]-env[:2]),response_error=deltaerr,
                response_norm=norm(dv),environment_response_norm=norm(dve),angle_degrees=angle,
                parallel_gain=float(dv@direction/radius) if radius else None,
                perpendicular_response=float(dv@np.array([-direction[1],direction[0]])) if radius else None,
                environment_parallel_gain=float(dve@direction/radius) if radius else None,
                fixed_zero_response_error=norm(dve),fixed_zero_state_error=norm(z[2:]-env[2:]),
                hold_state_error=norm(st[2:]-env[2:]),off_manifold=pred['off_manifold'],
                patch_proprio_max_difference=pred['patch_proprio_max_difference'],
                model_endpoint_penetration=max(0.,-float(clearance(model[:2],params))),
                delta_vx=float(dv[0]),delta_vy=float(dv[1]),environment_delta_vx=float(dve[0]),environment_delta_vy=float(dve[1]))
            if radius:
                sign=lookup[ep,f'r{radius:g}_d{(c["direction"]+4)%8}'];rotate=lookup[ep,f'r{radius:g}_d{(c["direction"]+2)%8}']
                r['sign_flip_response_error']=norm(np.array(sign['prediction'][2:])-z[2:]-dve)
                r['wrong_branch_response_error']=norm(np.array(rotate['prediction'][2:])-z[2:]-dve)
            else:r.update(sign_flip_response_error=None,wrong_branch_response_error=None)
            for control in ['fixed_zero','sign_flip','wrong_branch']:
                val=r[f'{control}_response_error'];r[f'{control}_minus_model']=val-deltaerr if val is not None else None
            for label,end in [('model',model),('environment',env)]:
                r.update({f'{label}_{k}':v for k,v in energy_terms(st,end,gear@u,mass,damping).items()})
            r['coarse_residual_difference']=r['model_coarse_residual']-r['environment_coarse_residual']
            r['necessary_condition_violation']=int(r['model_necessary_excess']>1e-8) if oracle['free'] else None
            r['environment_necessary_violation']=int(r['environment_necessary_excess']>1e-8) if oracle['free'] else None
            rows.append(r)
            tr=np.array(oracle['trace']);work=loss=num=clip=0.
            for n in range(25):
                q0,vbefore=tr[n,:2],tr[n,2:];q1,v1=tr[n+1,:2],tr[n+1,2:]
                v0=np.clip(vbefore,-5,5) if n%5==0 else vbefore
                ww=float((gear@u)@(q1-q0));dd=float(.01*v1@damping@v1);nn=float(.5*(v1-v0)@mass@(v1-v0));cc=float(.5*(vbefore@mass@vbefore-v0@mass@v0))
                work+=ww;loss+=dd;num+=nn;clip+=cc
                substeps.append(dict(episode=ep,condition=name,substep=n,work=ww,damping=dd,numerical=nn,clip=cc))
            residual=r['environment_delta_k']-work+loss+num+clip
            fine.append(dict(episode=ep,split=split,condition=name,free=oracle['free'],work=work,damping_loss=loss,
                numerical_loss=num,clip_loss=clip,delta_k=r['environment_delta_k'],residual=residual))
            if oracle['free']:assert abs(residual)<1e-9 and r['environment_necessary_violation']==0
        for radius in [.05,.1,.2,.4,1.]:
            def d(direction,environment=False):
                source=truth if environment else lookup;field='endpoint' if environment else 'prediction';origin=ze if environment else z
                return np.array(source[ep,f'r{radius:g}_d{direction}'][field][2:])-origin[2:]
            jm=np.column_stack([(d(0)-d(4))/(2*radius),(d(2)-d(6))/(2*radius)])
            je=np.column_stack([(d(0,True)-d(4,True))/(2*radius),(d(2,True)-d(6,True))/(2*radius)])
            pairfree=all(truth[ep,f'r{radius:g}_d{i}']['free'] for i in [0,2,4,6])
            jacobians.append(dict(episode=ep,split=split,radius=radius,free=pairfree,model=jm.tolist(),environment=je.tolist(),
                frobenius_error=norm(jm-je),model_diagonal_mean=float(np.trace(jm)/2),cross_axis_norm=norm([jm[0,1],jm[1,0]])))
            for direction in range(4):
                symmetry.append(dict(episode=ep,split=split,radius=radius,direction=direction,
                     free=truth[ep,f'r{radius:g}_d{direction}']['free'] and truth[ep,f'r{radius:g}_d{direction+4}']['free'],
                     model_error=norm(d(direction)+d(direction+4)),environment_error=norm(d(direction,True)+d(direction+4,True))))
            if radius>.05:
                for direction in range(8):
                    low=np.array(lookup[ep,f'r0.05_d{direction}']['prediction'][2:])-z[2:]
                    lowe=np.array(truth[ep,f'r0.05_d{direction}']['endpoint'][2:])-ze[2:]
                    linearity.append(dict(episode=ep,split=split,radius=radius,direction=direction,
                       free=truth[ep,f'r{radius:g}_d{direction}']['free'] and truth[ep,f'r0.05_d{direction}']['free'],
                       model_error=norm(d(direction)-radius/.05*low),environment_error=norm(d(direction,True)-radius/.05*lowe)))
        pp=np.array(lookup[ep,'recorded']['prediction']);oo=np.array(truth[ep,'recorded']['endpoint'])
        recorded.append(dict(episode=ep,split=split,raw_anchor=ctx['raw_anchor'],free=truth[ep,'recorded']['free'],
                             velocity_error=norm(pp[2:]-oo[2:]),position_error=norm(pp[:2]-oo[:2])))
    csvwrite(out/'branch_metrics.csv',rows);csvwrite(out/'zero_input.csv',zero);save(out/'jacobians.json',jacobians)
    csvwrite(out/'symmetry.csv',symmetry);csvwrite(out/'linearity.csv',linearity);csvwrite(out/'energy_fine.csv',fine)
    csvwrite(out/'energy_substeps.csv',substeps);csvwrite(out/'recorded_baselines.csv',recorded)
    metrics=['response_error','velocity_error','position_error','response_norm','environment_response_norm','angle_degrees',
        'parallel_gain','perpendicular_response','fixed_zero_response_error','sign_flip_response_error','wrong_branch_response_error',
        'fixed_zero_minus_model','sign_flip_minus_model','wrong_branch_minus_model','necessary_condition_violation',
        'model_coarse_residual','environment_coarse_residual','coarse_residual_difference','off_manifold','patch_proprio_max_difference']
    summary=[]
    for split in ['calibration_fit','calibration_tune','development']:
        for layer in ['main_all','main_common_free','stress_all','stress_pair_free']:
            group=[r for r in rows if r['split']==split and r['radius']>0 and
                   (r['stress'] if layer.startswith('stress') else not r['stress']) and
                   (r['common_free'] if layer=='main_common_free' else r['pair_free'] if layer=='stress_pair_free' else True)]
            for radius in [None]+sorted(set(r['radius'] for r in group)):
                rr=[r for r in group if radius is None or r['radius']==radius]
                if not rr:continue
                summary.append(dict(split=split,layer=layer,radius=radius,rows=len(rr),episodes=len(set(r['episode'] for r in rr)),
                    angle_excluded=sum(r['angle_degrees'] is None for r in rr),**{f:aggregate(rr,f) for f in metrics}))
    save(out/'summary.json',summary)
    auxiliary=[]
    for split in ['calibration_fit','calibration_tune','development']:
        zz=[r for r in zero if r['split']==split and r['common_free']]
        auxiliary.append(dict(split=split,kind='zero',metrics={f:aggregate(zz,f) for f in ['velocity_error','position_error','speed_ratio','model_energy_increase','environment_energy_increase']},
                              positive_model_energy=sum(r['model_energy_increase']>1e-8 for r in zz),rows=len(zz)))
        for radius in [.05,.1,.2,.4,1.]:
            jj=[r for r in jacobians if r['split']==split and r['radius']==radius and r['free']]
            if jj:auxiliary.append(dict(split=split,kind='jacobian',radius=radius,metrics={f:aggregate(jj,f) for f in ['frobenius_error','model_diagonal_mean','cross_axis_norm']},
                 mean_model=np.mean([r['model'] for r in jj],axis=0).tolist(),mean_environment=np.mean([r['environment'] for r in jj],axis=0).tolist()))
            for kind,src in [('symmetry',symmetry),('linearity',linearity)]:
                rr=[r for r in src if r['split']==split and r['radius']==radius and r['free']]
                if rr:auxiliary.append(dict(split=split,kind=kind,radius=radius,metrics={f:aggregate(rr,f) for f in ['model_error','environment_error']}))
    save(out/'auxiliary_summary.json',auxiliary)
    # Clarify reuse from the preserved failed attempt, whose old generic label
    # did not distinguish a B trace from a successful partial C trace.
    reuse=read(out/'reuse_partial.json') if (out/'reuse_partial.json').exists() else None
    partial=read(reuse['source']) if reuse else []
    def k(r):return (r['episode'],r['raw_anchor'],r['condition'],np.array(r['requested_actions'],float).tobytes())
    partialkeys={k(r) for r in partial}
    save(out/'reuse_audit.json',dict(reused_partial=sum(r['provenance'].startswith('reused') and k(r) in partialkeys for r in oracles),
         reused_prior=sum(r['provenance'].startswith('reused') and k(r) not in partialkeys for r in oracles),
         new=sum(r['provenance']=='new_action_branch' for r in oracles),original_generic_label='reused_B_or_previous includes preserved partial C rows'))
    checks=dict(pass_all=True,paired_raw_actions_identical=True,context_hash_identical=True,source_rows=len(rows),
         contexts=len(contexts),free_fine_energy_max=max(abs(r['residual']) for r in fine if r['free']),
         baseline_reproduction_max=max(r['max_abs'] for r in read(out/'infer_complete.json')['baseline_checks']),
         main_nonfree=sum(not r['branch_free'] for r in rows if not r['stress']),stress_nonfree=sum(not r['branch_free'] for r in rows if r['stress']),
         angle_excluded_main=sum(r['angle_degrees'] is None for r in rows if r['radius']>0 and not r['stress']),
         main_predicted_endpoint_penetrations=sum(r['model_endpoint_penetration']>.01 for r in rows if not r['stress']),
         weights_unchanged=True,bootstrap_count=10000,bootstrap_seed=0,new_training=False)
    assert all(sha(p)==v for p,v in read(out/'manifest.json')['sources'].items())
    assert all(sha(p)==v for p,v in read(out/'image_sources.json').items())
    save(out/'checks.json',checks);save(out/'analysis_manifest.json',dict(script_sha256=sha(__file__),source_sha256={n:sha(out/n) for n in ['oracles.json','predictions.json','selected.json','conditions.json']},
        elapsed_seconds=time.monotonic()-starttime));save(out/'completion.json',dict(status='DYNAMICS_C_COMPLETE_DEVELOPMENT',new_training=False,new_physics_loss=False,phase_d='NOT_RUN'))
    print(checks,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.repo,a.out)
