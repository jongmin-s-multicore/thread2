"""합성 fixture만 사용하는 선택/집계/승인 및 입력 정렬 단위검사."""
import copy
import tempfile
from pathlib import Path
import numpy as np
from natural_v2_core import require
from natural_v2_preflight import selection_tests
from natural_v2_metrics import aggregate,paired
from natural_v2_comparison import locked_save,locked_read


def run():
    results=selection_tests()
    rows=[]
    for ep,values in ((1,[1.,3.]),(2,[10.])):
        for i,value in enumerate(values):
            rows.append(dict(id=str((ep,i)),episode=ep,family='natural',horizon=1,current_free=False,
                prefix_free=False,contact=True,clip=False,values={'q':value,'gt_anchored_mse':value/10}))
    summary=aggregate(rows)
    require(summary['natural.h1.q.all']['mean']==6.,'episode equal weighting')
    require(summary['natural.h1.q.prefix_free']['n_episodes']==0 and summary['natural.h1.q.prefix_free']['mean'] is None,'empty cohort')
    delta=paired(summary,summary)
    require(delta['natural.h1.gt_anchored_mse.all']['ci95']==[0.,0.],'anchored paired aggregation')
    bad=copy.deepcopy(rows);bad[0]['values']['q']=float('inf')
    try:aggregate(bad)
    except RuntimeError:pass
    else:raise RuntimeError('nonfinite not rejected')
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'choice.json';locked_save(path,{'chosen':'fixture'})
        require(locked_read(path)['chosen']=='fixture','lock read')
        try:locked_save(path,{'chosen':'replacement'})
        except FileExistsError:pass
        else:raise RuntimeError('lock overwrite')
        path.write_text('{}')
        try:locked_read(path)
        except RuntimeError:pass
        else:raise RuntimeError('lock tampering')
    results['fixtures']+=['episode_equal_weight','empty_cohort_NA','anchored_paired_bootstrap',
        'nonfinite_run_failure','exclusive_selection_lock','selection_lock_hash']
    return results

if __name__=='__main__':
    import json
    print(json.dumps(run(),indent=2))
