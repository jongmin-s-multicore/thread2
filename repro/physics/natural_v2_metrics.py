"""자연/합성 cohort를 분리하는 episode 집계와 실제 rollout 측정."""
import numpy as np
from intervention import clearance
from natural_v2_core import require,upgrade_batch,evaluation_mode
from natural_v2_data import numpy_transition


def rollout(wm,b,n):
    import torch
    import physics_loss_pilot as pilot
    b=upgrade_batch(b);ctx=b['context'];result={'states':[],'patch_mean_states':[],
        'visual_mse':[],'proprio_mse':[],'off_manifold':[]}
    with evaluation_mode(wm):
        for h in range(5):
            y=wm.predict(ctx)[:,-1:]
            require(bool(torch.isfinite(y).all()),'nonfinite prediction; entire run failed')
            st=pilot.decode(wm,y,n);avg=pilot.decode(wm,y,n,True)
            pred=wm.separate_emb(y)[0];truth=wm.separate_emb(b['future'][:,h:h+1])[0]
            reconstructed=((st-n['state_mean'])/n['state_std'])@n['weight'].T+n['bias']
            result['states'].append(st[0,0].cpu().tolist())
            result['patch_mean_states'].append(avg[0,0].cpu().tolist())
            result['visual_mse'].append(float((pred['visual']-truth['visual']).square().mean()))
            prop=slice(-(wm.proprio_dim+wm.action_dim),-wm.action_dim)
            result['proprio_mse'].append(float((y[...,prop]-b['future'][:,h:h+1,...,prop]).square().mean()))
            result['off_manifold'].append(float((pred['proprio']-reconstructed).square().mean().sqrt()))
            if h<4:ctx=torch.cat((ctx[:,1:],wm.replace_actions_from_z(y.clone(),b['actions'][:,h+1:h+2])),1)
    return result


def case_metrics(case,pred,params,std):
    truth=np.asarray(case['truth']);states=np.asarray(pred['states']);avg=np.asarray(pred['patch_mean_states'])
    require(states.shape==truth.shape==(5,4) and np.isfinite(states).all(),'invalid rollout')
    controls=np.asarray(case['controls']);start=np.asarray(case['start']);std=np.asarray(std)
    previous=start;oracle_previous=start;prefix=True;rows=[]
    for h in range(5):
        seg=case['segments'][h];prefix=prefix and seg['free']
        theoretical=numpy_transition(previous,controls[h],params)[-1]
        anchored=numpy_transition(oracle_previous,controls[h],params)[-1]
        minimum=float(clearance(states[h,:2],params));start_margin=float(clearance(previous[:2],params))
        values=dict(q=float(np.linalg.norm(states[h,:2]-truth[h,:2])),v=float(np.linalg.norm(states[h,2:]-truth[h,2:])),
            visual=pred['visual_mse'][h],proprio_full_patch=pred['proprio_mse'][h],
            patch_mean_q=float(np.linalg.norm(avg[h,:2]-truth[h,:2])),patch_mean_v=float(np.linalg.norm(avg[h,2:]-truth[h,2:])),
            off_manifold=pred['off_manifold'][h],signed_vx=float(states[h,2]-truth[h,2]),signed_vy=float(states[h,3]-truth[h,3]),
            recursive_q=float(np.linalg.norm(states[h,:2]-theoretical[:2])),recursive_v=float(np.linalg.norm(states[h,2:]-theoretical[2:])),
            recursive_mse=float(np.mean(((states[h]-theoretical)/std)**2)),
            gt_anchored_q=float(np.linalg.norm(states[h,:2]-anchored[:2])),gt_anchored_v=float(np.linalg.norm(states[h,2:]-anchored[2:])),
            gt_anchored_mse=float(np.mean(((states[h]-anchored)/std)**2)),
            wall_overlap=float(minimum<0),wall_depth_over_001=float(minimum<-.01),wall_depth=max(0.,-minimum),
            predicted_start_invalid=float(start_margin<=.004 or np.max(np.abs(previous[2:]))>5 or np.max(np.abs(controls[h]))>1))
        if case['family']=='static':values.update(speed=float(np.linalg.norm(states[h,2:])),drift=float(np.linalg.norm(states[h,:2]-start[:2])))
        rows.append(dict(id=case['id'],episode=case['episode'],family=case['family'],horizon=h+1,
            current_free=bool(seg['free']),prefix_free=bool(prefix),contact=seg['contacts']>0,
            clip=seg['clips']>0 or seg.get('control_clips',0)>0,values=values))
        previous=states[h];oracle_previous=truth[h]
    return rows


def action_metrics(cases,predictions):
    rows=[]
    for ep in sorted({c['episode'] for c in cases if c['family']=='intervention'}):
        group=[c for c in cases if c['family']=='intervention' and c['episode']==ep]
        zero=next((c for c in group if c['condition']=='zero'),None)
        require(zero is not None and len(group)==9,'incomplete action pairs')
        for c in group:
            if c is zero:continue
            for h in range(5):
                delta=np.asarray(predictions[c['id']]['states'])[h,2:]-np.asarray(predictions[zero['id']]['states'])[h,2:]
                truth=np.asarray(c['truth'])[h,2:]-np.asarray(zero['truth'])[h,2:]
                tn,dn=np.linalg.norm(truth),np.linalg.norm(delta)
                angle=None if tn<1e-8 else 180. if dn==0 else float(np.degrees(np.arccos(np.clip(delta@truth/(dn*tn),-1,1))))
                a,b=c['segments'],zero['segments']
                rows.append(dict(id=c['id']+'-zero',episode=ep,family='action',horizon=h+1,
                    current_free=bool(a[h]['free'] and b[h]['free']),prefix_free=all(x['free'] for x in a[:h+1]+b[:h+1]),
                    contact=any(x['contacts']>0 for x in (a[h],b[h])),clip=any(x['clips']>0 or x.get('control_clips',0)>0 for x in (a[h],b[h])),
                    values=dict(vector=float(np.linalg.norm(delta-truth)),magnitude=float(abs(dn-tn)),angle=angle)))
    return rows


def aggregate(rows):
    """빈 cohort는 N=0/None. branch 수와 episode 수를 함께 보존한다."""
    out={}
    for family in ('natural','static','low_speed','intervention','action'):
        family_rows=[r for r in rows if r['family']==family]
        metrics=sorted({k for r in family_rows for k in r['values']})
        for h in range(1,6):
            for cohort in ('all','current_free','prefix_free','contact','clip'):
                selected=[r for r in family_rows if r['horizon']==h and (cohort=='all' or r[cohort])]
                for key in metrics:
                    by_ep={}
                    for r in selected:
                        value=r['values'][key]
                        if value is None:continue  # 오직 oracle angle 정의불가
                        require(np.isfinite(value),'nonfinite metric')
                        by_ep.setdefault(str(r['episode']),[]).append(value)
                    ep={e:float(np.mean(v)) for e,v in by_ep.items()}
                    out[f'{family}.h{h}.{key}.{cohort}']=dict(episodes=ep,n_episodes=len(ep),
                        n_samples=sum(len(v) for v in by_ep.values()),mean=float(np.mean(list(ep.values()))) if ep else None)
    return out


def paired(first,second):
    require(set(first)==set(second),'different metric sets')
    result={}
    for key,x in first.items():
        y=second[key];require(set(x['episodes'])==set(y['episodes']),'paired episode mismatch')
        ids=sorted(x['episodes']);delta=np.array([x['episodes'][e]-y['episodes'][e] for e in ids])
        if len(ids):
            rng=np.random.default_rng(0);index=rng.integers(0,len(ids),(10000,len(ids)))
            ci=np.quantile(delta[index].mean(1),[.025,.975]).tolist();mean=float(delta.mean())
        else:ci=None;mean=None
        result[key]=dict(n_episodes=len(ids),mean_delta=mean,ci95=ci,episode_delta=dict(zip(ids,delta.tolist())))
    return result


def selection_row(summary,cases,**metadata):
    from natural_v2_selection import G,REST
    metrics={}
    for key in G+REST:
        if key+'.all' in summary:metrics[key]=summary[key+'.all']['mean']
    return dict(complete=True,verified=True,cohort={f:sorted(c['id'] for c in cases if c['family']==f)
                for f in sorted({c['family'] for c in cases})},metrics=metrics,**metadata)
