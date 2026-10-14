"""Paired site-level diagnostics of coherent zero-input histories."""
import numpy as np
from pathlib import Path
from intervention import read,save,csvwrite,sha
from zero_velocity import decompose

def estimate(values):
    a=np.asarray(values,float);assert a.size and np.isfinite(a).all()
    ix=np.random.default_rng(0).integers(len(a),size=(10000,len(a)));ci=np.quantile(a[ix].mean(axis=1),[.025,.975],axis=0)
    return dict(mean=a.mean(axis=0).tolist(),ci95=ci.tolist(),episodes=len(a))

def grouped(rows,field):
    eps=sorted({r['episode'] for r in rows});return estimate([np.mean([r[field] for r in rows if r['episode']==ep],axis=0) for ep in eps])

def analyze(root,repo,out):
    params=read(out/'parameters.json');A=np.linalg.matrix_power(np.array(params['Ah']),25);alpha=float(A[0,0]);M=np.array(params['M'])
    np.testing.assert_allclose(A,alpha*np.eye(2),atol=1e-14)
    branches=read(out/'branches.json');preds=read(out/'predictions.json');ps={(r['episode'],r['name']):r for r in preds};bs={(r['episode'],r['name']):r for r in branches}
    assert len(ps)==len(bs)==len(branches);norm=lambda x:float(np.linalg.norm(x))
    metrics=[];coeffs=[];effects=[];biasrows=[];closure=0.
    for x in read(out/'selected.json'):
        ep=x['episode'];lookup={r['name']:np.array(ps[ep,r['name']]['prediction']) for r in branches if r['episode']==ep}
        b=lookup['coast_zero'][2:];biasrows.append(dict(episode=ep,split=x['split'],bias=b.tolist(),bias_norm=norm(b)))
        matrices={}
        for radius in [.1,.25,.5]:
            fp=[lookup[f'coast_r{radius:g}_{j}_1'][2:] for j in range(2)];fm=[lookup[f'coast_r{radius:g}_{j}_-1'][2:] for j in range(2)]
            J,E=decompose(b,fp,fm,radius,alpha);matrices[radius]=J
            coeffs.append(dict(episode=ep,split=x['split'],radius=radius,jacobian=J.tolist(),frobenius_error=norm(E),diagonal_mean=float(np.trace(J)/2),off_diagonal_norm=norm(J-np.diag(np.diag(J)))))
        for name,p in lookup.items():
            r=bs[ep,name];v=np.array(r['start'][2:]);q=np.array(r['start'][:2]);gt=np.array(r['endpoint']);f=p[2:];dK=float(.5*(f@M@f-v@M@v))
            row=dict(episode=ep,split=x['split'],name=name,history=r['history'],radius=r['radius'],axis=r['axis'],sign=r['sign'],velocity_error=norm(f-gt[2:]),position_error=norm(p[:2]-gt[:2]),delta_k=dK,energy_increase=int(dK>1e-8),start_speed=norm(v),end_speed=norm(f),speed_ratio=norm(f)/norm(v) if norm(v)>1e-6 else None,prediction=f.tolist(),reference=gt[2:].tolist())
            if r['history']=='coast' and r['radius']:
                J=matrices[r['radius']];slope=(J-A)@v;res=f-b-J@v;err=f-A@v;closure=max(closure,norm(err-(b+slope+res)))
                row.update(bias_term=norm(b),slope_term=norm(slope),nonlinear_term=norm(res),bias_removed_error=norm(f-b-gt[2:]),hold_error=norm(v-gt[2:]),fixed_output_error=norm(b-gt[2:]),sign_flip_error=norm(lookup[f"coast_r{r['radius']:g}_{r['axis']}_{-r['sign']}"][2:]-gt[2:]))
            metrics.append(row)
            if r['history']!='coast':
                name0='coast_'+name.split('_',2)[2];ref=lookup[name0]
                assert np.max(np.abs(np.array(bs[ep,name0]['start'])-r['start']))<1e-10
                effects.append(dict(episode=ep,split=x['split'],history=r['history'],radius=r['radius'],axis=r['axis'],sign=r['sign'],velocity_difference=norm(f-ref[2:]),signed_difference=(f-ref[2:]).tolist(),position_difference=norm(p[:2]-ref[:2]),oracle_difference=norm(gt-np.array(bs[ep,name0]['endpoint']))))
    summaries={}
    for split in sorted({r['split'] for r in metrics}):
        s=[r for r in metrics if r['split']==split];coast=[r for r in s if r['history']=='coast'];moving=[r for r in coast if r['radius']>0];zero=[r for r in coast if r['radius']==0]
        coef=[r for r in coeffs if r['split']==split and r['radius']==.25];ef=[r for r in effects if r['split']==split]
        summaries[split]=dict(episodes=len({r['episode'] for r in s}),branches=len(s),coast_moving={k:grouped(moving,k) for k in ['velocity_error','position_error','energy_increase','bias_term','slope_term','nonlinear_term','bias_removed_error','hold_error','fixed_output_error','sign_flip_error']},
            zero={k:grouped(zero,k) for k in ['velocity_error','position_error','energy_increase','delta_k']},bias=estimate([r['bias'] for r in biasrows if r['split']==split]),jacobian={k:grouped(coef,k) for k in ['jacobian','frobenius_error','diagonal_mean','off_diagonal_norm']},
            history={h:{k:grouped([r for r in ef if r['history']==h],k) for k in ['velocity_difference','signed_difference','position_difference','oracle_difference']} for h in ['push_positive','push_negative']},
            by_radius={str(radius):{k:grouped([r for r in moving if r['radius']==radius],k) for k in ['velocity_error','bias_removed_error','slope_term','nonlinear_term','energy_increase','speed_ratio']} for radius in [.1,.25,.5]})
        diffs=[dict(episode=r['episode'],improvement=r['velocity_error']-r['bias_removed_error'],hold_excess=r['hold_error']-r['velocity_error']) for r in moving]
        summaries[split]['paired_bias_removal']=grouped(diffs,'improvement');summaries[split]['paired_hold_excess']=grouped(diffs,'hold_excess')
    assert closure<1e-12
    checks=dict(pass_all=True,vector_decomposition_max_abs=closure,oracle_same_state_history_max=max(r['oracle_difference'] for r in effects),
                **{k:read(out/'prepare_complete.json')[k] for k in ['energy_max_abs','oracle_max_abs']},weights_unchanged=read(out/'infer_complete.json')['weights_unchanged'],baseline_reproduction_max=max(r['max_abs'] for r in read(out/'infer_complete.json')['baseline_checks']),new_training=False)
    assert checks['oracle_same_state_history_max']<1e-10 and checks['weights_unchanged'] and checks['baseline_reproduction_max']<1e-4
    assert all(sha(f)==h for f,h in read(out/'manifest.json')['sources'].items())
    save(out/'metrics.json',metrics);save(out/'coefficients.json',coeffs);save(out/'history_effects.json',effects);save(out/'biases.json',biasrows);save(out/'summary.json',dict(alpha=alpha,groups=summaries));save(out/'checks.json',checks)
    fields=list(dict.fromkeys(k for r in metrics for k in r));csvwrite(out/'metrics.csv',[{k:r.get(k) for k in fields} for r in metrics])
    save(out/'analysis_manifest.json',dict(script_sha256=sha(__file__),sources={n:sha(out/n) for n in ['branches.json','predictions.json','parameters.json']},bootstrap_replicates=10000,bootstrap_seed=0,unit='episode/site',coefficient_radius=.25,new_training=False))
    save(out/'completion.json',dict(status='ZERO_VELOCITY_COMPLETE_DEVELOPMENT',new_training=False,new_physics_loss=False,phase_d='NOT_RUN'))
    print(summaries.get('development'),flush=True)
