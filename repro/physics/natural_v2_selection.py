"""v2.0 tune 선택 규칙. 실제 지표와 합성 fixture가 동일 함수를 사용한다."""
import math
from natural_v2_core import require

NATURAL = tuple(f'natural.h{h}.{m}' for h in (1,5) for m in ('q','v','visual'))
INTERVENTION = tuple(f'intervention.h{h}.{m}' for h in (1,5) for m in ('q','v','visual'))
ACTION = ('action.h1.vector','action.h5.vector')
REST = ('static.h5.speed','static.h5.drift')
G = NATURAL + INTERVENTION + ACTION + ('low_speed.h5.v',)
STAGE1 = REST + INTERVENTION + ACTION
EPS = 1e-6


def check(row, keys):
    require(row.get('complete') is True and row.get('verified') is True, 'technical failure')
    require(row.get('cohort'), 'empty cohort')
    for k in keys:
        require(k in row['metrics'] and math.isfinite(row['metrics'][k]) and row['metrics'][k] >= 0, 'missing/nonfinite metric '+k)


def comparable(rows, keys):
    for r in rows: check(r, keys)
    require(all(r['cohort'] == rows[0]['cohort'] for r in rows), 'cohort mismatch')


def ratios(row, base, keys):
    return [row['metrics'][k]/max(base['metrics'][k], 1e-12) for k in keys]


def near_min(rows, score):
    low = min(score(r) for r in rows)
    return [r for r in rows if score(r) <= low + EPS]


def select_stage1(a, cells):
    order = ['I-N','K-N','I-F','K-F']
    require(set(cells) == set(order), 'incomplete four cells')
    comparable([a]+list(cells.values()), STAGE1)
    rows = [dict(r, cell=k) for k,r in cells.items()]
    rows = near_min(rows, lambda r: max(ratios(r,a,STAGE1)))
    rows = near_min(rows, lambda r: sum(ratios(r,a,STAGE1))/10)
    return min(rows, key=lambda r: order.index(r['cell']))


def preserving(r, references):
    return all(r['metrics'][k] <= 1.05*b['metrics'][k] for b in references for k in G)


def select_base(a, candidates):
    require(len(candidates) == 6 and {(r['mixture'],r['step']) for r in candidates} ==
            {(m,s) for m in ('R06','R12') for s in (120,240,480)}, 'incomplete base candidates')
    comparable([a]+candidates, G+REST)
    rows = [r for r in candidates if preserving(r,[a])]
    if not rows: return {'status': 'NO_PRESERVING_BASELINE'}
    rows = near_min(rows, lambda r: sum(ratios(r,a,NATURAL))/6)
    rows = near_min(rows, lambda r: r['metrics'][REST[1]])
    choice = min(rows, key=lambda r: (r['step'],r['mixture']))
    return dict(choice, status='BASE_CHOSEN')


def rest_score(r,a,b):
    return max(ratios(r,a,REST) + ratios(r,b,(REST[1],)))


def select_aux(a,b,candidates,family):
    require(b.get('status') == 'BASE_CHOSEN' and family in ('P','S'), 'base lock required')
    require(len(candidates)==2 and {r['strength'] for r in candidates}=={.5,1.}, 'incomplete strengths')
    require(all(r['step']==b['step'] and r['mixture']==b['mixture'] for r in candidates), 'common checkpoint mismatch')
    comparable([a,b]+candidates, G+REST)
    rows = [r for r in candidates if preserving(r,[a,b])]
    passed = bool(rows)
    if not rows:
        rows = near_min(candidates, lambda r: max(
            r['metrics'][k]/max(1.05*ref['metrics'][k],1e-12) for ref in (a,b) for k in G))
    rows = near_min(rows, lambda r: rest_score(r,a,b))
    chosen = min(rows, key=lambda r:r['strength'])
    return dict(chosen, preserving=passed, family=family,
                incremental_gate=passed and rest_score(chosen,a,b)<=.90)


def final_success(a,b,p,drift_delta_ci):
    comparable([a,b,p],G+REST)
    require(len(drift_delta_ci)==2 and all(math.isfinite(x) for x in drift_delta_ci), 'invalid CI')
    return preserving(b,[a]) and preserving(p,[a,b]) and rest_score(p,a,b)<=.9 and drift_delta_ci[1]<0
