"""v2.0 전이·목적함수·고정 노출 순서. 기존 파일럿은 변경하지 않는다."""
from contextlib import contextmanager
import math
import numpy as np

LAMBDA = 0.14603252771223255
CELLS = {'K-F': (False, 'fixed'), 'K-N': (False, 'norm'),
         'I-F': (True, 'fixed'), 'I-N': (True, 'norm')}
MIXTURES = {
    'R06': ['N','N','N','I','N','N','L','N','N','N','I','N','N','N','S','N'],
    'R12': ['N','N','I','N','L','N','S','N','N','I','N','L','N','S','N','N'],
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def transition(state, controls, params):
    """원 단위 (...,5,2) control. clipping 없는 미분 가능한 affine 전이."""
    import torch
    require(controls.shape == state.shape[:-1] + (5, 2), 'action shape')
    a = torch.as_tensor(params['Ah'], dtype=state.dtype, device=state.device)
    b = torch.as_tensor(params['Bh'], dtype=state.dtype, device=state.device)
    q, v = state[..., :2], state[..., 2:]
    for j in range(5):
        for _ in range(5):
            v = v @ a.T + controls[..., j, :] @ b.T
            q = q + params['dt'] * v
    return torch.cat((q, v), -1)


def upgrade_batch(batch):
    """기존 constant-action cache를 수치 변환 없이 확장한다."""
    b = dict(batch)
    if 'controls' not in b:
        b['controls'] = b['u'][:, None, None, :].expand(-1, 5, 5, -1)
        b['actions'] = b['act'].expand(-1, 5, -1)
    return b


def parts(wm, batch, norm, params, stop_input=False, paths=False):
    import torch
    import physics_loss_pilot as pilot
    b = upgrade_batch(batch)
    first = wm.predict(b['context'])
    y1 = first[:, -1:]
    incoming = y1.detach().clone() if stop_input else y1.clone()
    incoming = wm.replace_actions_from_z(incoming, b['actions'][:, 1:2])
    y2 = wm.predict(torch.cat((b['context'][:, 1:], incoming), 1))[:, -1:]
    s1, s2 = pilot.decode(wm, y1, norm), pilot.decode(wm, y2, norm)
    states = torch.cat((s1, s2), 1)
    t1 = transition(b['start'], b['controls'][:, 0], params)
    t2 = transition(s1[:, 0], b['controls'][:, 1], params)  # attached target 유지
    mask = b['mask'][:, :2]
    den = mask.sum().clamp(min=1)
    error = lambda x, y: ((x-y)/norm['state_std']).square().mean(-1)
    # 원인분리 D의 H1/H2별 reduction 순서를 그대로 사용한다.
    phys = (error(s1[:, 0], t1)*mask[:, 0]).sum()/den
    phys = phys + (error(s2[:, 0], t2)*mask[:, 1]).sum()/den
    supervised = (error(s1[:, 0], b['truth'][:, 0])*mask[:, 0]).sum()/den
    supervised = supervised + (error(s2[:, 0], b['truth'][:, 1])*mask[:, 1]).sum()/den
    target = torch.cat((b['context'][:, 1:], b['future'][:, :1]), 1)
    losses = {'base': pilot.original_loss(wm, first, target),
              'physics': phys, 'state': supervised}
    require(bool(torch.isfinite(states).all()) and all(bool(torch.isfinite(x)) for x in losses.values()), 'nonfinite forward')
    if paths:
        return losses, states, {'y1': y1, 's2': s2, 't2': t2, 'mask': mask, 'den': den}
    return losses, states


def flatgrad(loss, wm):
    import torch
    return torch.cat([x.reshape(-1) for x in torch.autograd.grad(
        loss, list(wm.predictor.parameters()), retain_graph=True)])


def objective(losses, wm, mask, family, mode, strength=1.):
    """동일 graph에서 norm 측정. 반환 float 계수는 미분하지 않는다."""
    import torch
    require(family in ('B', 'P', 'S'), 'unknown arm')
    require(mode in ('fixed', 'norm') and strength in (0., .5, 1.), 'unknown scale')
    if family == 'B' or strength == 0:
        return losses['base'], {'lambda': 0., 'aux_norm': 0., 'ratio': 0., 'aux_skipped': True}
    aux = losses['physics' if family == 'P' else 'state']
    gb, ga = flatgrad(losses['base'], wm), flatgrad(aux, wm)
    nb, na = float(gb.norm()), float(ga.norm())
    require(math.isfinite(nb) and math.isfinite(na), 'nonfinite gradient')
    if float(mask[:, :2].sum()) == 0:
        require(na == 0, 'zero mask nonzero gradient')
        lam = 0.
    else:
        require(na > 1e-12, 'active mask degenerate auxiliary gradient')
        lam = strength * (.1*nb/max(na, 1e-12) if mode == 'norm' else LAMBDA)
        require(math.isfinite(lam) and lam <= 10, 'invalid auxiliary coefficient')
    info = {'lambda': lam, 'base_norm': nb, 'aux_norm': na,
            'ratio': lam*na/max(nb, 1e-12),
            'cosine': float((gb@ga)/(gb.norm()*ga.norm()).clamp(min=1e-12)),
            'aux_skipped': float(mask[:, :2].sum()) == 0}
    return losses['base'] + lam*aux, info


def anchors(length):
    k = list(range(10, int(length)-25))
    require(len(k) >= 4, 'insufficient natural anchors')
    result = [k[j*(len(k)-1)//3] for j in range(4)]
    require(len(set(result)) == 4, 'duplicate natural anchors')
    return result


def schedule(cases, mixture, steps=480):
    require(mixture in MIXTURES and steps == 480, 'unregistered schedule')
    groups = {}
    for family in 'NILS':
        ids = sorted(c['id'] for c in cases if c['schedule_family'] == family)
        require(ids and len(ids) == len(set(ids)), 'empty/duplicate family')
        groups[family] = list(np.random.default_rng(0).permutation(ids))
    positions = dict.fromkeys(groups, 0)
    exposure = []
    for i in range(steps*2):
        f = MIXTURES[mixture][i % 16]
        exposure.append(str(groups[f][positions[f] % len(groups[f])]))
        positions[f] += 1
    return [exposure[i:i+2] for i in range(0, len(exposure), 2)]


@contextmanager
def evaluation_mode(wm):
    """평가가 train dropout RNG 및 모든 module mode를 바꾸지 않는다."""
    import torch
    cpu, cuda, numpy = torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state()
    modes = [(m, m.training) for m in wm.modules()]
    try:
        wm.eval()
        with torch.no_grad():
            yield
    finally:
        for m, mode in modes:
            m.training = mode
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        np.random.set_state(numpy)
