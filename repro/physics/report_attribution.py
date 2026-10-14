"""Report fixed ablations and distinguish observed causes from untested mechanisms."""
import argparse,json,html
from pathlib import Path
import numpy as np
from intervention import read,save
from physics_attribution import ARMS,NEW,PAIRS
LABELS={'A':'Frozen','B':'Base loss','C':'GT 2-step','D':'Physics 2-step','E':'Physics + rest','G1':'GT 1-step','P1':'Physics 1-step','DS':'Stop target grad','CL':'GT weight x0.1','DL':'Physics weight x0.1','CM':'GT norm matched','DM':'Physics norm matched','CI':'Stop input grad','CT':'True input, stopped'}
ORDER=['A','B','G1','P1','C','D','DS','E','CL','DL','CM','DM','CI','CT']

def report(out):
 s=read(out/'summary.json');p=read(out/'paired.json');grad=read(out/'gradient_audit.json');verify=read(out/'independent_verification.json');eq=read(out/'h1_evaluation_equivalence.json');cp=read(out/'h1_checkpoint_equivalence.json');checks=read(out/'attribution_checks.json')
 index={(r['arm'],r['family'],r['horizon'],r['metric'],r['scope'],r['split']):r for r in s}
 def metric(a,f,m,h=5,scope='all',split='development'):return index[a,f,h,m,scope,split]
 def value(a,f,m,h=5,scope='all',split='development'):return metric(a,f,m,h,scope,split)['mean']
 def delta(comp,f,m):return next(r for r in p if (r['comparison'],r['family'],r['horizon'],r['metric'],r['scope'],r['split'])==(comp,f,5,m,'all','development'))
 rows=['| 조건 | 정지 속력 H5 | 정지 drift H5 | 행동개입 속도오차 H5 | 행동반응 vector오차 H5 | 시각 latent MSE H5 |','|---|---:|---:|---:|---:|---:|']
 specs=[('static','speed'),('static','position_drift'),('recorded','velocity_error'),('action','vector_error'),('recorded','visual_mse')]
 for a in ORDER:rows.append('| '+a+' | '+' | '.join(f'{value(a,f,m):.5f}' for f,m in specs)+' |')
 contrasts=['| 비교 | 정지 drift 차이 | 행동개입 속도오차 차이 | 행동반응 오차 차이 |','|---|---:|---:|---:|']
 for a,b in PAIRS:
  cols=[]
  for f,m in [('static','position_drift'),('recorded','velocity_error'),('action','vector_error')]:
   r=delta(a+'-'+b,f,m);cols.append(f"{r['mean']:+.5f} [{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}]")
  contrasts.append('| '+a+'−'+b+' | '+' | '.join(cols)+' |')
 jobs=[read(f) for f in (out/'jobs').glob('*/status.json')];assert all(r['state']=='COMPLETE' for r in jobs);resources=dict(new_gpu_seconds=sum(r['elapsed_seconds'] for r in jobs if r['gpu']),prior_gpu_seconds=90.81711977836676,peak_gpu_device_mib=max(r['peak_gpu_device_mib'] for r in jobs),peak_rss_gib=max(r['peak_rss_bytes'] for r in jobs)/1024**3,cpu_cores=2);resources['combined_gpu_seconds']=resources['new_gpu_seconds']+resources['prior_gpu_seconds'];save(out/'resource_summary.json',resources)
 ratios={}
 for a in ['CM','DM']:
  curve=read(out/a/'curve.json');weights=[r['lambda_aux'] for r in curve];ratios[a]=dict(min=min(weights),median=float(np.median(weights)),max=max(weights),norm_ratio_error=max(abs(r['aux_norm_ratio']-.1) for r in curve))
 grad_table=['| 初期12学習batchの項 | baseとの平均cosine | 負のbatch数 |'.replace('初期12学習batchの項','초기12학습 batch 항').replace('baseとの平均cosine','base와 평균 cosine').replace('負のbatch数','음수 batch수'),'|---|---:|---:|']
 for k in ['gt1','gt2','rec2','stop2','gt2_q','gt2_v','rec2_q','rec2_v']:
  vals=[r['cos_base'][k] for r in grad];grad_table.append(f"| {k} | {np.mean(vals):+.4f} | {sum(v<0 for v in vals)}/12 |")
 text=f'''# DINO-WM physics loss 악화 원인 분리 결과

**이 설정에서 악화의 핵심은 잘못된 물리 target보다 2단계 보조항의 재귀 역전파와 보조 gradient 크기의 균형이었다.** 실제 정답을 쓰는 상태감독에서도 같은 악화가 나타났고, 재귀 입력의 gradient 경로를 차단하거나 보조 gradient 크기를 매 batch 제한하면 악화가 크게 줄었다. 다만 두 개입은 gradient의 크기와 방향에 함께 영향을 주므로 서로 독립적인 기여율까지 분해한 것은 아니다. 물리식 일반의 우열이나 모든 원인을 확정한 결과로 확대하지 않는다.

## 무엇을 실제로 비교했는가

기존 A–E의 데이터·checkpoint·평가를 재사용하고 새 9조건을 각각 seed0·120step으로 학습했다. 학습12episode의 같은140후보에서 동일240노출(고유130branch), 같은 AdamW lr1e-5, 초기 checkpoint, predictor만 학습, encoder/readout 동결을 유지했다. 새 9조건과 기존 5조건을 합쳐14조건이다. 새로운 학습seed·episode·sealed test는 추가하지 않았다. 최종 checkpoint 고정 후196개 tune/dev case×5horizon을 평가했다. 기존 A–E를 다시 추론/학습하지 않았다.

| 조건 | 바꾼 요소 |
|---|---|
| A / B | 원래 동결 / 기존 loss 추가 학습 |
| G1 / P1 | 1단계 실제 상태 target / 1단계 물리 target |
| C / D | 1단계+2단계 실제 상태감독 / 물리 전이 일관성 |
| DS | D에서2단계 physical target 쪽 gradient만 차단 |
| E | D에 정지해 강조항 추가 |
| CL / DL | C/D의 고정 보조 가중치를0.1배 |
| CM / DM | 매 batch 보조 gradient norm을 base norm의10%로 맞춤 |
| CI | C에서2단계 입력값은 예측값 그대로, 첫 예측으로 돌아가는 입력 gradient만 차단 |
| CT | CI와 같은 차단 상태에서2단계 입력값만 실제oracle token으로 교체 |

G1/P1은2단계항을 제거할 때 분모를 바꾸지 않아 기존1단계항의 가중치를 유지했다. 모든 조건은 같은2회 forward를 수행한다. CT만 학습 중 oracle token을 입력으로 사용하며, 평가에서는 CT도 미래정답 없이 자유rollout한다. CI는 CT의 입력값 효과와 gradient 경로 효과를 구분하기 위해 **새 평가 전에** 별도 amendment로 추가했다. 기존 8조건을 바꾸거나 평가를 보고 가중치를 고르지 않았다.

## 원인에 대한 직접 증거

### 물리식이나 상태 역변환 오류라는 설명은 지지되지 않았다

G1과 P1의120step 학습 후 최대 파라미터 차이는 {cp['max_parameter_abs']:.3e}였다. 사전기준1e-4를 통과했고, 개발평가 주요오차 차이의 최대 A대비 비율은 {eq['max_relative_difference']:.3e}로1% 기준도 통과했다. 정지 drift와 다단계 오차는 표시된 자릿수에서 동일하다. 초기target 최대차이 {checks['h1_target_max']:.3e}, gradient 상대차이 {checks['h1_gradient_relative_max']:.3e}였다. 따라서 검증한 자유공간·단위·정규화·고정readout 범위에서 물리식 오류가 주원인이라는 근거는 없다. 이 결과가 모든 접촉식이나 별도 물리모델 구현까지 검증하는 것은 아니다.

### 큰 악화는2단계항을 추가하면서 나타났다

G1→C의 정지 drift는 {value('G1','static','position_drift'):.5f}→{value('C','static','position_drift'):.5f}, 행동개입H5 속도오차는 {value('G1','recorded','velocity_error'):.5f}→{value('C','recorded','velocity_error'):.5f}였다. P1→D도 각각 {value('P1','static','position_drift'):.5f}→{value('D','static','position_drift'):.5f}, {value('P1','recorded','velocity_error'):.5f}→{value('D','recorded','velocity_error'):.5f}로 악화됐다. 같은 초기 모델·순서에서2단계 보조항 추가가 악화의 발생 위치임을 확인했다. 실제 정답을 쓴 C도 악화되므로 잘못된 자기 target만으로 설명할 수 없다.

### 두 번째 loss가 첫 예측을 수정하는 경로가 중요했다

C→CI는2단계 입력 token의 **값을 유지하면서 gradient만 차단**했다. 정지 drift는 {value('C','static','position_drift'):.5f}→{value('CI','static','position_drift'):.5f}(약62% 감소), H5 속도오차는 {value('C','recorded','velocity_error'):.5f}→{value('CI','recorded','velocity_error'):.5f}, 행동 반응오차는 {value('C','action','vector_error'):.5f}→{value('CI','action','vector_error'):.5f}로 줄었다. 세 paired CI 모두0보다 작다.

CI는 unroll 경로의 gradient만 차단한다. 두 단계가 predictor 파라미터를 공유하므로 첫 단계 모델 자체를 동결한 것은 아니다.

반면 CI→CT에서 입력을 실제 정답으로 바꾼 추가효과는 작고 세 paired CI가0을 포함했다. 이 설정에서는 입력예측의 부정확성 자체보다 **2단계loss가 첫 예측을 거슬러 수정하도록 만든 학습경로**가 더 유력한 설명이다. 경로 차단은 gradient의 크기와 방향도 바꾸므로 독립적인 방향효과로만 단정하지 않는다. CT와 CI가 통계적으로 동등하다는 검정까지 한 것은 아니다.

### physical target의 gradient만 끊는 것으로는 해결되지 않았다

D→DS에서 forward 값은 같고 target 쪽 gradient만 제거했다. 정지 drift는 {value('D','static','position_drift'):.5f}→{value('DS','static','position_drift'):.5f}로 일부 줄었지만 H5 속도오차는 {value('D','recorded','velocity_error'):.5f}→{value('DS','recorded','velocity_error'):.5f}, 행동 오차는 {value('D','action','vector_error'):.5f}→{value('DS','action','vector_error'):.5f}로 개선되지 않았다. 첫 예측이 만든 physical target이 움직이는 효과는 전체 악화의 충분한 설명이 아니다. DS는 재귀 입력 쪽 gradient를 유지한다는 점에서 CI와 다르다.

### 최초 한 batch에서 정한 가중치는 전체 학습을 대표하지 못했다

고정lambda를0.1배한 CL/DL은 일부 악화를 줄였지만 충분하지 않았다. 매 batch 보조 gradient norm을 base의10%로 맞춘 CM/DM은 정지 drift {value('CM','static','position_drift'):.5f}/{value('DM','static','position_drift'):.5f}, H5 속도오차 {value('CM','recorded','velocity_error'):.5f}/{value('DM','recorded','velocity_error'):.5f}로 큰 악화를 줄였다.

원래lambda=.14603에 비해 CM의 실측lambda 중앙값은 {ratios['CM']['median']:.5f}, 범위는 {ratios['CM']['min']:.5f}..{ratios['CM']['max']:.5f}; DM은 중앙값 {ratios['DM']['median']:.5f}, 범위 {ratios['DM']['min']:.5f}..{ratios['DM']['max']:.5f}였다. 매 batch norm비10%가 수치적으로 유지됐다. 이 계수범위는 각 matched모델 학습궤적에서 측정한 값이며 원래C/D의 모든step gradient비를 역산한 값은 아니다.

동일 gradient 규모로 맞춘 DM−CM의 정지 drift·속도오차·행동반응 차이는 모두 paired CI가0을 포함했다. 이전 D가 C보다 크게 나빴던 결과를 물리식의 본질적 열세로 해석하기 어렵다. 다만 동적가중은 batch별 상대가중까지 바꾸므로 고정가중의 전체평균 크기만 조정한 개입과 같지는 않다.

## 전체 결과와 불확실성

H5=1.25초, 작을수록 좋다. family별 episode평균이다. 정지10branch/10episodes, 행동개입90/10, 저속40/10이며 합집합은12episode다. 이미분석한 원 WM train 영역의 개발표본이며 추가학습에서만 보류됐다.

{chr(10).join(rows)}

![원인분리 조건의 주요결과](attribution_overview.png)

아래는 동일episode에서 새조건−비교조건의 차이와10000회 episode bootstrap의 탐색적95% CI다. 음수는 개선이다. 다중비교 보정과 seed간변동 추정은 없다.

{chr(10).join(contrasts)}

![사전 정의 비교의 paired 효과](attribution_paired.png)

추가 학습한 13조건을 기존 판정식으로 확인했을 때 새조건도 완전한 성공 기준을 통과하지 못했다. CI/CT는 정지 drift와 여러예측오차를 안정화했지만 H5행동 반응오차는 A의 {value('A','action','vector_error'):.5f}에 비해 CI {value('CI','action','vector_error'):.5f}, CT {value('CT','action','vector_error'):.5f}로5% 보존기준을 넘었다. G1/P1도 정지 drift10%감소 기준을 충족하지 못했다. 따라서 원인분리의 성과와 개선모델 채택을 구분하며 어떤checkpoint도 원모델을 대체하지 않는다.

전체/현재free/prefixfree/contact 결과와 전수예측은 summary.json,paired.json,metrics.csv에 유지했다. 행동개입H5까지 계속자유공간인 것은5branch/5episode뿐이며, 행동과zero가 모두계속자유공간인 쌍은0이다. 전체H5행동결과에는접촉이 포함된다. 마스크는oracle만 사용했고 모델 실패는 제외하지 않았다.

## gradient 감사가 말해주는 범위

처음 12개의 고정 학습 batch에서 아래cosine을 측정했다. 음수는 그지점에서 두gradient 방향이 반대성분을 가진다는 뜻이며 실제 학습 전체의 충돌빈도가 아니다. 여러항의norm은 서로더해 기여율로해석할 수 없다.

{chr(10).join(grad_table)}

gt1=첫단계 상태감독, gt2=두번째 상태감독, rec2=재귀물리전이, stop2=물리 target detach, q/v=위치/속도 성분이다. 물리항 전체가 항상base와충돌하는 패턴은 없었다. 특히rec2는12 batch 중2개에서만 음수였다. 따라서 gradient 충돌을 일반적인단일원인으로단정하지 않는다. 경로 차단과 규모 통제의 실제학습 개입이 초기 cosine보다 강한 증거다.

## 검증과 보존

분해한 loss와 gradient가 기존 C/D/E 구현과 일치했고, CI의 forward loss도 C와 차이 0임을 확인했다. 모든 새 조건은 120 step, 동일 sample 순서와 동결 hash를 유지했다. G1/P1 파라미터·예측 동등성, CM/DM의 norm matching, 원본 A–E 평가 파일의 SHA 동일성을 검증했다. 독립 84개 집계 재계산의 최대 차이는 {verify['max_abs']:.3e}였다. 새 학습·평가 중 검증 실패는 없었다.

새 GPU 작업은 {resources['new_gpu_seconds']:.2f}초, 이전 파일럿을 포함하면 {resources['combined_gpu_seconds']:.2f}초로 누적 3600초 이내였다. 이번 장치 전체 최대 메모리는 {resources['peak_gpu_device_mib']}MiB, process RAM은 {resources['peak_rss_gib']:.2f}GiB였고 CPU 2개를 사용했다. 새 checkpoint 9개와 기록은 원본 밖 고유 폴더에 저장하고 encoded 자료는 읽기 참조로 재사용했다. 기존 결과·환경·재현 repo와 thread0 push 차단을 유지했다. thread0에는 게시하지 않았다.

최초 정의는 [protocol_ko.md](protocol_ko.md), 평가 전 입력 gradient 분리 보완은 [protocol_amendment_ko.md](protocol_amendment_ko.md)에 있다. 준비·감사·첫 8조건 학습은 `ca01370`, CI 학습·평가·분석은 `d3d5914`였다. 정확한 명령과 자원은 jobs, 원입력 hash는 attribution_manifest.json, 기존 원본 hash는 manifest.json, checkpoint 목록은 checkpoint_catalog.json, 재현 명령은 reproduce.sh에 있다. Mac에서 index.html과 보고서·그림을 열 수 있으며 대용량 checkpoint는 서버에 보존했다.

## 남아 있는 원인 구분

확인한 것은 이 고정 설정에서 2단계항의 추가, 재귀 입력 gradient 경로, 보조 gradient 규모를 바꾸면 성능이 달라진다는 점이다. CI가 좋아진 정확한 메커니즘을 gradient 방향과 크기의 독립 효과로 완전히 분리하지는 못했다. 이를 위해서는 **CM과 같은 norm 통제 하에서 입력 gradient 차단을 교차하는 대조**가 다음 우선순위다. 평가 후 조건 추가 금지 규칙에 따라 이번에는 실행하지 않았다.

단일 seed, 짧은 120 step, 이미 노출된 개발 표본, 정지 50% oversampling, 합성 미래영상의 renderer 차이, 접촉 구간의 작은 자유공간 분모는 그대로 남아 있다. 학습 부족 대 과적합, 자연 분포 보존, 미관측 WM 일반화, 시각 DINO 표현의 물리 이해는 확정하지 못했다. 여러 seed 확장·장기 학습·sealed test·에너지/벽 loss는 실행하지 않았다. 결과를 본 뒤 좋아 보이는 조건을 추가 탐색하지 않았다.
'''
 with (out/'report_ko.md').open('x') as f:f.write(text)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2});colors=[plt.get_cmap('tab20')(i) for i in range(len(ORDER))]
 fig,axes=plt.subplots(1,3,figsize=(14,8),sharey=True)
 for ax,(f,m,title) in zip(axes,[('static','position_drift','Stationary drift at H5'),('recorded','velocity_error','Velocity error at H5'),('action','vector_error','Action response error at H5')]):
  vals=[value(a,f,m) for a in ORDER];ax.barh(range(len(ORDER)),vals,color=colors);ax.set_yticks(range(len(ORDER)),[a+'  '+LABELS[a] for a in ORDER]);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('Episode mean; lower is better')
 fig.tight_layout();fig.savefig(out/'attribution_overview.png',dpi=160);plt.close(fig)
 comps=[a+'-'+b for a,b in PAIRS if a!='P1'];fig,axes=plt.subplots(1,3,figsize=(14,8),sharey=True)
 for ax,(f,m,title) in zip(axes,[('static','position_drift','Change in stationary drift'),('recorded','velocity_error','Change in velocity error'),('action','vector_error','Change in action error')]):
  for i,comp in enumerate(comps):
   r=delta(comp,f,m);ax.plot(r['ci'],[i,i],color='#58728a',lw=2);ax.plot(r['mean'],i,'o',color='#007c83')
  ax.axvline(0,color='#a44538',lw=1);ax.set_yticks(range(len(comps)),comps);ax.invert_yaxis();ax.set_title(title);ax.set_xlabel('New - reference; negative is better')
 fig.suptitle('Paired episode bootstrap 95% intervals; development, single seed',fontsize=12);fig.tight_layout();fig.savefig(out/'attribution_paired.png',dpi=160);plt.close(fig)
 table_html='<table><tr><th>조건</th>'+''.join('<th>'+t+'</th>' for t in ['정지 속력 H5','정지 drift H5','속도오차 H5','행동 오차 H5','시각 MSE H5'])+'</tr>'+''.join('<tr><td>'+a+'</td>'+''.join(f'<td>{value(a,f,m):.5f}</td>' for f,m in specs)+'</tr>' for a in ORDER)+'</table>'
 page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DINO-WM physics loss 원인 분리</title><style>body{font:16px/1.75 system-ui,sans-serif;background:#f4f6f8;color:#17283a}main{max-width:1160px;margin:30px auto;background:white;padding:30px}h1{font-size:28px}img{width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}.note{background:#edf6f5;padding:16px;border-left:5px solid #00867e}a{color:#005d97}</style><main><h1>DINO-WM physics loss 원인 분리</h1><p class="note"><b>물리식 오류보다 2단계 재귀 역전파와 보조 gradient 규모의 균형이 유력했습니다.</b> 입력값을 바꾸지 않고 gradient 경로만 차단해도 큰 악화가 줄었습니다. 두 효과의 독립적 기여율과 단일 seed 밖의 일반화까지 확정한 것은 아닙니다.</p><p>기존 A–E 재사용 · 새 9조건 각120step · 같은 seed와 데이터 · sealed test 미사용</p><p><a href="report_ko.md">한국어 상세 보고서와 조건 정의</a> · <a href="protocol_ko.md">프로토콜</a> · <a href="protocol_amendment_ko.md">평가 전 보완</a> · <a href="checkpoint_catalog.json">checkpoint 위치</a></p><p>CI: 예측 입력값을 유지하고 gradient만 차단. CT: 같은 차단 상태에서 실제 입력 사용. CM/DM: 보조 gradient norm을 매 batch 10%로 통제. 추가 학습한 모든 조건은 원래 성공 기준을 완전히 통과하지 못했고 checkpoint를 채택하지 않았습니다.</p>'''+table_html+'''<h2>주요 결과</h2><img src="attribution_overview.png" alt="14조건 정지 drift 및 속도와 행동 오차"><h2>같은 episode에서의 변화</h2><p>음수는 개선. 여러 비교를 포함한 단일 seed 개발평가의 탐색적 구간입니다.</p><img src="attribution_paired.png" alt="사전정의 비교의 episode paired 차이와 구간"><p>정지10branch/10episodes, 행동개입90/10. 조건별 지표는 summary.json과paired.json, 원예측은evaluation 파일에 보존했습니다.</p></main></html>'''
 with (out/'index.html').open('x') as f:f.write(page)
 save(out/'report_complete.json',dict(attribution_complete=True,checkpoint_adopted=False,all_original_success_checks_failed=all(not r['pass_point_estimates'] for r in read(out/'success.json').values()),gradient_scale_ranges=ratios,remaining_identification='Gradient direction versus magnitude under input-stop not crossed; no claims of universal causality.',single_seed=True,sealed_test_opened=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();report(a.out)
