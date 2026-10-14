"""v2 자연 raw 자료·oracle. whitelist 밖 row 접근과 통계 적합은 금지한다."""
from pathlib import Path
import numpy as np
from intervention import read, sha, clearance, grid
from natural_v2_core import anchors, require


class WhitelistData:
    def __init__(self, data, allowed):
        import torch
        self.data = Path(data)
        self.allowed = set(allowed)
        self.access = []
        self.globals = {}
        for name in ('states','actions','seq_lengths'):
            self.globals[name] = torch.load(self.data/(name+'.pth'), map_location='cpu', weights_only=False)
        self.global_load_note = 'global serialized tensors loaded; no normalization/statistics; row access only through whitelist'

    def row(self, name, ep):
        require(ep in self.allowed, 'episode outside whitelist')
        self.access.append({'kind':name,'episode':int(ep)})
        value = self.globals[name][ep]
        return value.detach().numpy() if hasattr(value, 'detach') else np.asarray(value)

    def frames(self, ep, indices):
        import torch
        require(ep in self.allowed, 'image episode outside whitelist')
        self.access.append({'kind':'images','episode':int(ep),'frames':list(indices)})
        return torch.load(self.data/'obses'/f'episode_{ep:03d}.pth', map_location='cpu', weights_only=False)[indices].numpy()


def numpy_transition(state, controls, params):
    a, b = np.asarray(params['Ah']), np.asarray(params['Bh'])
    q, v = np.array(state[:2],dtype=float), np.array(state[2:],dtype=float)
    trace = [np.r_[q,v]]
    for u in controls:
        for _ in range(5):
            v = a@v+b@u
            q = q+params['dt']*v
            trace.append(np.r_[q,v])
    return np.array(trace)


def replay(env, state, controls, params, reset=True):
    import mujoco_py
    controls = np.asarray(controls)
    require(controls.shape == (5,2) and np.isfinite(controls).all(), 'raw control shape/finite')
    sim = env.sim
    if reset:
        st = np.asarray(state,dtype=float)
        sim.reset()
        sim.set_state(mujoco_py.MjSimState(0.,st[:2].copy(),st[2:].copy(),None,{}))
        sim.forward()
        sim.data.qacc_warmstart[:] = 0
    particle = env.model.geom_name2id('particle_geom')
    walls = {i for i in range(env.model.ngeom) if (env.model.geom_id2name(i) or '').startswith('wall_')}
    trace = [np.r_[sim.data.qpos.copy(),sim.data.qvel.copy()]]
    contacts, velocity_clips = 0, 0
    for u in controls:
        velocity_clips += int(np.any(np.abs(sim.data.qvel)>5))
        env.clip_velocity()
        sim.data.ctrl[:] = np.clip(u,-1,1)
        for _ in range(5):
            sim.step()
            trace.append(np.r_[sim.data.qpos.copy(),sim.data.qvel.copy()])
            contacts += int(any((int(c.geom1)==particle and int(c.geom2) in walls) or
                                (int(c.geom2)==particle and int(c.geom1) in walls)
                                for c in sim.data.contact[:sim.data.ncon]))
    trace = np.asarray(trace)
    margin = float(clearance(trace[:,:2],params).min())
    control_clips = int(np.any(np.abs(controls)>1,axis=1).sum())
    free = contacts==velocity_clips==control_clips==0 and margin>.004
    error = float(np.max(np.abs(numpy_transition(trace[0],controls,params)-trace))) if free else None
    require(error is None or error<1e-10, 'analytic/fine trace gate')
    return {'trace':trace.tolist(),'contacts':contacts,'clips':velocity_clips,
            'control_clips':control_clips,'min_clearance':margin,'free':free,'analytic_max_abs':error}


def natural_window(loader, ep, k, env, params, images=True):
    states, actions = loader.row('states',ep), loader.row('actions',ep)
    length = int(loader.row('seq_lengths',ep))
    require(k in anchors(length), 'unregistered anchor')
    frames = [k-10,k-5,k]+[k+5*h for h in range(1,6)]
    control = np.stack([actions[k+5*h:k+5*h+5] for h in range(5)])
    context_actions = np.stack([actions[r:r+5] for r in frames[:3]])
    future_actions = np.stack([actions[r:r+5] if h<4 else np.zeros((5,2),dtype=actions.dtype)
                               for h,r in enumerate(frames[3:])])
    segments = []
    for h in range(5):
        r = k+5*h
        seg = replay(env,states[r],control[h],params)
        seg['recorded_endpoint_max_abs'] = float(np.max(np.abs(np.array(seg['trace'][-1])-states[r+5])))
        seg['target_recorded_max_abs'] = float(np.max(np.abs(numpy_transition(states[r],control[h],params)[-1]-states[r+5]))) if seg['free'] else None
        segments.append(seg)
    # 실패시 caller가 모든 실제 값을 파일에 남긴 다음 중단한다.
    continuous = []
    for h in range(5):
        seg = replay(env,states[k],control[h],params,reset=h==0)
        continuous.append(float(np.max(np.abs(np.array(seg['trace'][-1])-states[k+5*(h+1)]))))
    return {'id':f'N_{ep}_{k}','episode':ep,'family':'natural','schedule_family':'N',
            'condition':'recorded_actions','raw_anchor':k,'frames':frames,
            'state_dtype':str(states.dtype),'action_dtype':str(actions.dtype),
            'context_states':states[frames[:3]],'truth':states[frames[3:]],'start':states[k],
            'controls':control,'context_actions':context_actions,'future_actions':future_actions,
            'images':loader.frames(ep,frames) if images else None,
            'segments':segments,'continuous_endpoint_max_abs':continuous,
            'mask':np.cumprod([s['free'] for s in segments]).astype(float),
            'image_state_fine_lag':1,'natural_targets_rerendered':False}


def alignment_pass(case):
    return all(s['recorded_endpoint_max_abs']<=1e-4 and
               (not s['free'] or s['target_recorded_max_abs']<1e-5) for s in case['segments'])


def render_states(renderer, states):
    import mujoco_py
    result=[]
    for st in states:
        st=np.asarray(st,float)
        renderer.sim.reset()
        renderer.sim.set_state(mujoco_py.MjSimState(0.,st[:2].copy(),st[2:].copy(),None,{}))
        renderer.sim.forward(); renderer.set_marker()
        result.append(renderer._render_frame().copy())
    return np.stack(result)


def synthetic_sites(loader, ep, env, renderer, params):
    """새 tune/eval 전용. 사전 oracle-only site 순서와 불가 사유를 반환한다."""
    from zero_velocity import coherent_trace
    states, actions = loader.row('states',ep), loader.row('actions',ep)
    length=int(loader.row('seq_lengths',ep)); rows=[]; audit=[]
    for group in ('coast','intervention'):
        chosen=None
        for k in range(10,71,5):
            require(k+25<length, 'synthetic frame bounds')
            proposals=[]
            if group=='coast':
                for name,v in [('zero',[0.,0.])]+[(f'v{axis}_{sign}',[sign*.1 if j==axis else 0. for j in range(2)]) for axis in range(2) for sign in (-1,1)]:
                    tr=coherent_trace(states[k,:2],v,np.zeros(2),params)
                    controls=np.zeros((5,5,2)); segments=[]; start=tr[51]
                    for h in range(5):
                        seg=replay(env,start,controls[h],params);segments.append(seg);start=np.array(seg['trace'][-1])
                    # backwards generated history must itself be valid; replay first two blocks.
                    history_ok=float(clearance(tr[:52,:2],params).min())>.004 and float(np.abs(tr[:52,2:]).max())<5
                    for idx in (1,26):
                        hist=replay(env,tr[idx],np.zeros((5,2)),params)
                        history_ok=history_ok and hist['free'] and np.max(np.abs(np.asarray(hist['trace'])-tr[idx:idx+26]))<1e-10
                    proposals.append(dict(condition=name,start=tr[51],context_states=tr[[1,26,51]],
                        image_states=tr[[0,25,50]],context_actions=np.zeros((3,5,2)),controls=controls,
                        segments=segments,valid=history_ok and all(s['free'] for s in segments),
                        family='static' if name=='zero' else 'low_speed'))
            else:
                for cc in [g for g in grid() if g['radius'] in (0.,.2)]:
                    # 기존 oracle 요청 control과 동일 double 배열.
                    controls=np.tile(np.asarray(cc['control']), (5,5,1));segments=[];start=states[k]
                    for h in range(5):
                        seg=replay(env,start,controls[h],params);segments.append(seg);start=np.array(seg['trace'][-1])
                    proposals.append(dict(condition=cc['name'],start=states[k],context_states=states[[k-10,k-5,k]],
                        context_actions=np.stack([actions[k-10:k-5],actions[k-5:k],controls[0]]),controls=controls,
                        segments=segments,valid=segments[0]['free'],family='intervention'))
            valid=all(p['valid'] for p in proposals)
            audit.append({'episode':ep,'group':group,'raw_anchor':k,'selected':valid})
            if valid: chosen=(k,proposals);break
        if chosen is None:
            audit.append({'episode':ep,'group':group,'structurally_unavailable':True});continue
        k,proposals=chosen
        for p in proposals:
            context_images=render_states(renderer,p.pop('image_states')) if group=='coast' else loader.frames(ep,[k-10,k-5,k])
            future_images=render_states(renderer,[s['trace'][-2] for s in p['segments']])
            p.update(id=f'{group}_{ep}_{p["condition"]}',episode=ep,raw_anchor=k,
                images=np.concatenate((context_images,future_images)),
                truth=np.array([s['trace'][-1] for s in p['segments']]),
                future_actions=np.concatenate((p['controls'][1:],np.zeros((1,5,2)))),
                mask=np.cumprod([s['free'] for s in p['segments']]).astype(float),
                schedule_family={'static':'S','low_speed':'L','intervention':'I'}[p['family']])
            rows.append(p)
    return rows,audit


def encode_case(wm,cfg,n,case):
    import torch
    from datasets.img_transforms import default_transform
    dev=n['state_mean'].device
    state=np.concatenate((case['context_states'],case['truth']))
    acts=np.concatenate((case['context_actions'],case['future_actions']))
    raw=torch.tensor(acts,dtype=torch.float32,device=dev)
    normalized=((raw-n['action_mean'])/n['action_std']).reshape(1,8,10)
    obs=dict(visual=default_transform(int(cfg.img_size))(
        torch.tensor(case['images'],dtype=torch.float32).permute(0,3,1,2)/255)[None].to(dev),
        proprio=((torch.tensor(state,dtype=torch.float32,device=dev)-n['state_mean'])/n['state_std'])[None])
    with torch.no_grad():
        z=wm.encode(obs,normalized)
    controls=torch.tensor(case['controls'],dtype=torch.float32,device=dev)[None]
    action=((controls-n['action_mean'])/n['action_std']).reshape(1,5,10)
    return dict(context=z[:,:3].detach(),future=z[:,3:].detach(),actions=action,
        controls=controls,start=torch.tensor(case['start'],dtype=torch.float32,device=dev)[None],
        truth=torch.tensor(case['truth'],dtype=torch.float32,device=dev)[None],
        mask=torch.tensor(case['mask'],dtype=torch.float32,device=dev)[None],
        static=torch.tensor([case['family']=='static'],dtype=torch.float32,device=dev))
