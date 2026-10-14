"""Korean pilot report from locked evaluation; no fitting or model execution."""
import argparse,json,html,datetime,hashlib
from pathlib import Path
from intervention import read,save,sha

def report(root,repo,out):
 s=read(out/'summary.json');pairs=read(out/'paired.json');cfg=read(out/'training_config.json');split=read(out/'split.json');checks=read(out/'gradient_checks.json');audit=read(out/'independent_verification.json');success=read(out/'success.json')
 def get(a,f,h,m,scope='all',split='development'):
  return next(r for r in s if (r['arm'],r['family'],r['horizon'],r['metric'],r['scope'],r['split'])==(a,f,h,m,scope,split))
 def val(a,f,h,m,scope='all'):return get(a,f,h,m,scope)['mean']
 def table(specs,split='development'):
  lines=['| 지표 | A 동결 | B 기존 loss | C 상태 지도 | D 물리 전이 | E 전이와 정지 |','|---|---:|---:|---:|---:|---:|']
  for label,f,h,m in specs:lines.append('| '+label+' | '+' | '.join(f"{get(a,f,h,m,split=split)['mean']:.5f}" for a in 'ABCDE')+' |')
  return '\n'.join(lines)
 specs=[('정지 H5 속력','static',5,'speed'),('정지 H5 위치 drift','static',5,'position_drift'),('저속 H5 속도오차','low_speed',5,'velocity_error'),('행동개입 H1 위치오차','recorded',1,'position_error'),('행동개입 H1 속도오차','recorded',1,'velocity_error'),('행동개입 H5 위치오차','recorded',5,'position_error'),('행동개입 H5 속도오차','recorded',5,'velocity_error'),('행동반응 H1 vector오차','action',1,'vector_error'),('행동반응 H5 vector오차','action',5,'vector_error'),('행동반응 H5 방향오차 도','action',5,'angle_error'),('행동개입 H5 시각 latent MSE','recorded',5,'visual_mse'),('행동개입 H5 proprio MSE','recorded',5,'proprio_mse')]
 t=table(specs);pt=['| paired 차이 | 지표 H5 | 평균 차이 [탐색적 95% CI] |','|---|---|---:|']
 for comp,f,m,label in [('B-A','static','speed','정지 속력'),('B-A','static','position_drift','정지 drift'),('D-C','static','position_drift','정지 drift'),('D-C','recorded','velocity_error','행동개입 속도오차'),('D-C','recorded','visual_mse','시각 latent MSE'),('E-D','static','position_drift','정지 drift'),('E-D','recorded','velocity_error','행동개입 속도오차'),('E-A','static','position_drift','정지 drift'),('E-A','recorded','velocity_error','행동개입 속도오차'),('E-A','action','vector_error','행동반응 vector오차')]:
  r=next(r for r in pairs if (r['comparison'],r['split'],r['family'],r['horizon'],r['scope'],r['metric'])==(comp,'development',f,5,'all',m));pt.append(f"| {comp} | {label} | {r['mean']:+.5f} [{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}] |")
 counts=['| horizon | 전체 | 해당 구간 자유공간 | 계속 자유공간 | 실제 접촉 구간 |','|---|---:|---:|---:|---:|']
 for h in range(1,6):
  counts.append('| '+str(h)+' | '+' | '.join(f"{get('A','recorded',h,'velocity_error',scope)['branches']} / {get('A','recorded',h,'velocity_error',scope)['episodes']}" for scope in ['all','current_free','prefix_free','contact'])+' |')
 jobs=[read(p) for p in (out/'jobs').glob('*/status.json')];gpu=sum(j['elapsed_seconds'] for j in jobs if j['gpu']);peakgpu=max(j['peak_gpu_device_mib'] for j in jobs);peakram=max(j['peak_rss_bytes'] for j in jobs)/1024**3;disk=max(j['peak_disk_bytes'] for j in jobs)/1024**3
 resources=dict(gpu_wall_seconds_including_failed_check=gpu,peak_gpu_device_mib=peakgpu,peak_process_tree_ram_gib=peakram,measured_new_artifact_gib=disk,cpu_cores=2,all_training_complete=True)
 save(out/'resource_summary.json',resources)
 ckpts={a:read(out/a/'complete.json') for a in 'BCDE'};ckrows='\n'.join(f"| {a} | `{a}/checkpoint.pth` | `{v['checkpoint_sha256']}` |" for a,v in ckpts.items())
 text=f'''# DINO-WM 첫 physics loss 비교 파일럿 결과

**이번 설정의 physics loss는 채택 기준을 통과하지 못했다.** 짧은 구간의 속도·행동 반응은 개선됐지만 C·D·E의 장기 위치 drift와 행동 예측은 악화됐다. E의 정지 속력 감소는 기존 loss만 추가 학습한 B에서도 나타났으므로 physics loss 고유 효과로 설명할 수 없다. D보다 E가 여러 지표를 개선했지만 A나 B를 대신할 안정적인 모델은 아니었다.

## 실제 실행한 비교

A는 원래 동결 모델, B는 기존 latent MSE, C는 B+실제 위치·속도 지도, D는 B+물리 전이, E는 D+정지 강조다. B–E는 같은 초기 checkpoint에서 predictor {ckpts['B']['trainable_parameters']:,}개 파라미터만 각각 {cfg['steps']}step 학습했다. DINO·proprio·action encoder 및 기존 선형 역변환은 동결했다. AdamW lr=1e-5, weight decay=.01, batch2, seed0, gradient clip1, 학습 dropout=.1이다. 기존 decoder의 loss는 predictor와 분리된 경로이므로 decoder 학습은 생략했다.

같은 정지1+비정지1 batch를 네 조건에 제공했다. 학습후보140branch 중 실제120batch에 노출된 고유 branch는 {audit['exposed_unique_branches']}개, episode는 {audit['exposed_unique_episodes']}개다. 정지는120회, 비정지는120회 노출되어 정지를 의도적으로 oversampling했고 B에도 동일하게 적용했다. 10개 비정지 후보는120step 상한 때문에 노출되지 않았으며 조건 간 차이는 없다. 평가 이후 표본·지표·가중치·checkpoint를 변경하지 않았다.

가중치는 평가 성능을 사용하지 않고 최초 학습batch의 gradient 크기로 고정했다. C/D/E의 상태·전이 가중치는 {cfg['lambda_state_or_physics']:.8f}, E 정지항은 {cfg['lambda_stationary']:.8f}다. E의5step 시간측정 가중치는 폐기하고 모든 조건을 원본에서 다시 시작했다. 조정자료는 최종 horizon별 결과만 기록했고 checkpoint 선택·가중치 탐색에는 쓰지 않았다.

## 데이터 분할과 평가 범위

학습12episodes/140branch, 조정4/56, 평가12/140이며 episode는 분할 간 겹치지 않는다. 각 평가 family는 원영상 행동개입10episodes×9=90branch, 정지10×1=10branch, 저속10×4=40branch다. family끼리 위치 episode 일부가 달라 합집합이12다. 기존 C/depth의 fit/dev 첫10 context와 무입력 fit/dev 첫10 위치를 유지했고 tune은 기존4개를 썼다. 저속은 일관된 coast 이력의±x/±y 속도0.1이다.

학습 episode: {split['episodes']['calibration_fit']}. 조정: {split['episodes']['calibration_tune']}. 평가: {split['episodes']['development']}.

**평가는 이번 추가 학습에서만 보류됐다. 원래 WM 학습 영역이며 과거 개발 분석에서 이미 많이 본 표본이다.** 미관측 WM 일반화나 sealed test 결과가 아니다. 최종 sealed test는 열지 않았다. 정지·저속·개입 자료를 B에도 제공했으므로 B–E 차이를 추가 데이터 접근 차이로 설명할 수 없지만, A–B는 추가 학습과 데이터 증강이 함께 바뀐다.

## loss의 의미와 한계

원래 loss는 과거3frame의 shifted target에 대한 visual+proprio latent MSE이며 action 차원은 제외한다. 예측한 full token을 다시 넣는2step 경로를 모든 학습 조건에서 계산했다. B의 두번째 예측은 보조 gradient에 기여하지 않는다.

물리식은 검증된 `v_next=Ah v+Bh u`, `q_next=q+0.01 v_next`를25substep 적용한다. raw control5개를 각5substep 유지하므로 macro 간격은0.25초다. 모델 action은 기존 평균·표준편차로 정규화한 시간순10성분이고, 식에는 원 단위 control을 쓴다. 위치·속도 오차는 기존 state_std `[1.005501, 0.963009, 1.680277, 1.931931]`로 나눈4성분 MSE다. 이는 simulator 단위에 대한 명시적 가중이며 SI 단위를 주장하지 않는다.

C는 H1/H2 예측을 실제 oracle state에 맞춘다. D는 H1에서 실제 출발상태를, H2에서 첫 예측상태를 전이식에 넣는다. H2는 양쪽 예측으로 gradient가 흐르는 자기일치성이다. **H1의 C/D target은 자유공간에서 사실상 같은 정답이다. H1 개선을 새로운 물리 제약의 증거로 해석하지 않는다.** E는 정지branch의 H1/H2 위치유지·속도0을 추가 가중한다. 전이식의 정지해를 강조하는 항이며 독립적인 법칙이 아니다.

oracle 접촉0·속도clip0·전구간margin>.004인 prefix만 보조항 대상이다. 고정 oracle 마스크를 쓰므로 모델이 벽으로 들어가거나 실패해도 학습항을 숨기지 못한다. 실제 접촉 oracle 구간에는 자유공간식을 적용하지 않았다. 다만 H2의 예측 출발상태가 물리 영역 밖이면 affine 식을 연장한 regularizer이며 그 지점의 실제 접촉 동역학이라고 볼 수 없다. 평가에서도 이 잔차와 실제 궤적오차를 분리했다. 에너지 loss와 벽 loss는 넣지 않았다.

## 동일 개발평가 결과

아래는 episode 내 branch평균 후 episode 동일가중 평균이다. H1=.25초, H5=1.25초. 방향오차 외 값은 simulator 단위 또는 latent MSE이며 모두 작을수록 좋다. 방향은 oracle 반응norm<1e-8만 정의불가로 기록하고 모델반응이0이면180도 실패 처리했다.

{t}

![조건별 오차와 정지 drift](comparison.png)

B는 정지H5 속력을 약44% 줄였지만 위치 drift는 거의 그대로다. E 역시 속력은 약44% 줄였으나 위치 drift는 A의 약2.53배다. 속도 출력이0에 가까워졌다는 사실만으로 위치가 안정적으로 유지된다고 판단할 수 없다. C·D·E 모두1step 속도 및 행동 방향은 개선됐지만5step에서 큰 악화가 발생했다.

사전 성공 기준은 정지H5 속력·drift 모두10% 이상 감소하면서 행동개입 H1/H5 위치·속도·시각 latent 및 행동반응 vector오차가 A 대비5% 이내인 것이었다. **B/C/D/E 모두 이를 충족하지 못했다.** 이 기준은 점추정 screening이며 통계적 비열등성 검정이 아니다.

![행동 반응](action_response.png)

## physics loss 효과를 구분할 수 있었는가

각 차이는 같은 episode의 paired 차이다. 음수는 오차 감소다. bootstrap10000/seed0의 탐색적95% CI이며 다중비교 보정이나 seed간 변동 추정은 아니다.

{chr(10).join(pt)}

D−C에서 정지 drift·다단계 위치/속도·시각 latent가 악화됐다. 이 설정에서 재귀 물리 일관성이 실제 상태감독보다 유리하다는 근거는 없다. E−D는 정지 drift와 다단계 속도/행동오차를 줄였다. 하지만 정지해의 추가 가중 효과이며 E가 A/B보다 좋은 모델이 됐다는 의미는 아니다.

동일 coefficient를 사용해도 최초 보조 gradient norm은 C={cfg['gradient_norms']['supervised']:.5f}, D={cfg['gradient_norms']['physics']:.5f}로 다르다. 따라서 목적함수 형태와 실제 gradient 크기 효과를 완전히 분리한 설계는 아니다. 이번 결과는 고정된 한 설정의 비교이며 physics loss 일반의 가능성/불가능성을 증명하지 않는다.

## 자유공간과 접촉을 나누어 보면

각 칸은 branch수 / episode수다. 모든 조건에서 동일한 oracle 기준과 분모를 사용했다. 자유공간과 접촉의 합이 전체와 다를 수 있는데, margin 경계의 무접촉 구간은 자유공간 조건을 통과하지 않기 때문이다.

{chr(10).join(counts)}

H5까지 계속 자유공간인 고정5branch/5episode에서 E의 재귀 속도잔차는 A의 {val('A','recorded',5,'recurrent_velocity_residual','prefix_free'):.5f}에서 {val('E','recorded',5,'recurrent_velocity_residual','prefix_free'):.5f}로 줄었다. 그러나 같은 cohort의 실제 속도오차는 {val('A','recorded',5,'velocity_error','prefix_free'):.5f}→{val('E','recorded',5,'velocity_error','prefix_free'):.5f}, 위치오차는 {val('A','recorded',5,'position_error','prefix_free'):.5f}→{val('E','recorded',5,'position_error','prefix_free'):.5f}로 커졌다. **자기일치성 개선이 실제 궤적 개선을 보장하지 않는 구체적 사례다.** 5개 소표본으로 일반화하지 않는다. H5 행동branch와 zero branch가 모두 계속 자유공간인 행동쌍은0이므로 전체 H5 행동오차는 접촉 포함 결과다.

H5 원영상90branch의 벽겹침 깊이>.01은 A24, B20, C12, D16, E7개이고 oracle은1개다. E의 벽겹침 감소와 실제 위치오차 악화가 동시에 나타났다. 이것만으로 벽 물리를 학습했다고 말할 수 없다. H5 실제 접촉75branch의 속도오차도 A {val('A','recorded',5,'velocity_error','contact'):.5f}, E {val('E','recorded',5,'velocity_error','contact'):.5f}로 악화됐다. 예측 실패는 제외하지 않았다.

![자기일치성과 벽겹침](consistency.png)

첫 패널의 자유공간 cohort는 horizon별90→54→21→10→5branch로 달라진다. 곡선의 단순 하락을 동일표본 개선으로 해석하면 안 된다. 정지10branch와 저속40branch oracle은 모든 horizon에서 자유공간이다. D의H5 정지10개 중1개는 예측 출발위치가 안전영역을 벗어났고 그대로 보고했다.

## 시각 표현과 상태 출력

원영상 행동개입 H5 시각 latent MSE는 B에서 A보다 약34% 감소했지만 E는 A와 비슷하고 D는 증가했다. E의 proprio latent MSE는 A보다 크게 증가했다. 첫patch 고정 역변환뿐 아니라 patch평균의 같은 역변환에서도 H5 속도오차가 A {val('A','recorded',5,'patch_mean_velocity_error'):.5f}→E {val('E','recorded',5,'patch_mean_velocity_error'):.5f}로 악화됐다. 첫patch만의 평가 착시로 장기 악화를 설명하기 어렵다.

DINO encoder와 readout 자체는 학습하지 않았다. 학습한 것은 predictor이며, 상태지도 보조항은 고정 readout이 읽는 첫patch proprio에 작용한다. 시각 latent MSE는 그 readout과 별도로 측정한 예측 정확도지만 내부에 물리법칙이 표현됐다는 검사는 아니다. 미래 시각 target은 oracle fine trace의 마지막 전substep을 raw renderer로 생성했다. 원영상 이력과 미래 합성영상 사이의 domain 차이는 모든 조건에 공통이며 renderer의 영향을 완전히 배제하지 못한다. 별도 시각-state probe는 새로 적합하지 않았다.

## 학습 곡선과 조정자료

![학습 loss 성분](training_curves.png)

곡선은 동일 batch의 raw 성분이며 B도 진단용 성분을 계산하지만 보조항으로 역전파하지 않는다. loss 감소가 rollout 성공과 같지 않았다. 조정4episode의 최종 결과도 아래에 함께 보존한다. 이를 보고 재학습·가중치 변경을 하지 않았다.

{table([x for x in specs if x[0] in ['정지 H5 속력','정지 H5 위치 drift','행동개입 H5 속도오차','행동반응 H5 vector오차']],split='calibration_tune')}

## 구현 검증과 자원

원본 forward와 cached 기존 loss 차이0, physics 가중치0의 gradient 차이0, inverse 왕복 최대 {checks['inverse_roundtrip_max']:.3e}, batch oracle 최대 {checks['float_oracle_max']:.3e}였다. 전체 자유공간 oracle 식은 {read(out/'prepare_complete.json')['oracle_equation_max']:.3e}로 재현됐다. 실제 predictor 업데이트·유한2step gradient·동결 hash 불변을 확인했다. 별도2개 단위검사에서 닫힌형 전이식과 finite-difference gradient, 정지 고정점을 검사했다.

학습 전에 검사 코드의 proprio 시간축 누락으로1회 실패했다. 비교 학습은 시작하지 않은 상태였고 입력축을 수정한 뒤 동일 관문을 다시 통과했다. 실패 로그와 사용 GPU시간도 보존·포함했다. 검증 기준을 완화하지 않았다.

기존 oracle1136구간과 A의 유효 full token 예측을 재사용했고 필요한 oracle544구간만 추가했다. 모든 직접 입력SHA가 유지됐으며 기존 C/depth 분석을 다시 실행하지 않았다. A 캐시 state 차이 {read(out/'checks.json')['baseline_cache_state_max']:.3e}는 기존float64 inverse와 이번differentiable float32 inverse 차이로1e-5 관문 안이다. 30개 집계를 독립 재계산한 최대차이는 {audit['max_abs']:.3e}. 동일 batch 노출, split분리, frozen hash와4900개 endpoint 유한성을 확인했다.

GPU job 누적 wall time은 실패검사·encoding·평가 포함 {gpu:.2f}초(상한3600초), 장치전체 최대 {peakgpu}MiB(11000), process tree RAM최대 {peakram:.2f}GiB(12), 측정 신규 산출물 약 {disk:.2f}GiB(10)였다. CPU affinity2개로 순차 실행했다. 다른 작업을 중단하지 않았다. 실제 import는 `{checks['import_path']}`이며 기존 재현용 repo는 바꾸지 않았다.

## 산출물과 재현

실행 폴더 `{out.name}`. 학습·평가 코드 `59fb4eb`, 분석 `4b4f476`; 준비 코드와 manifest HEAD는 `fa08db0`였다. 원본 source SHA·checkpoint SHA는 manifest.json, 분할은 split.json, 설정은 training_config.json, 실행 명령과 자원은 jobs에 있다. 최종step checkpoint는 서버 실행 폴더에 보존했다. A는 manifest의 원본 model_latest.pth를 참조하며 복제/수정하지 않는다.

| 조건 | 실행 폴더 상대경로 | SHA256 |
|---|---|---|
{ckrows}

Mac에는 report_ko.md, index.html, 비교 그림, JSON/CSV, 학습 곡선·실행 기록을 복사했다. 대용량 encoded token과 checkpoint는 서버에 남겼다. 정확한 명령은 [재현 명령](reproduce.sh), 최초 고정 정의는 [프로토콜](protocol_ko.md), 전수 수치는 summary.json / paired.json / metrics.csv를 참고한다. 기존 데이터·checkpoint·결과·환경을 보존했고 thread0에 push/PR/issue/comment/업로드하지 않았다.

## 결론과 다음 실험

이 작은 단일seed 파일럿에서는 physics 전이항의 우월성이 확인되지 않았다. 정지 강조항 E−D의 일부 개선은 관측됐지만, 장기 위치와 행동을 보존하지 못했으므로 배포/대체 모델로 채택하지 않는다. B가 비교군 중 상대적으로 안정적이지만 B도 사전 성공 기준을 통과하지 못했다.

다음 우선순위는 큰 학습이나 여러 seed 확장보다 **H1에서 C/D가 동등한지 확인하는 학습 대조와 H2 자기일치성의 영향 분리**다. 학습/조정 자료에서 위치·속도별 gradient 및 base와의 방향 충돌을 기록하고, 작은 보조 가중치와 GT anchoring을 사전 고정해 비교할 수 있다. 정지 oversampling 비율과 자연 행동 궤적 보존도 별도 요인으로 분리하는 것이 필요하다. 현재 개발 결과를 보고 바꾼 설계는 새 프로토콜로 남겨야 한다.

여러 seed, 장기 학습, 광범위 가중치 탐색, 원 WM 미관측 데이터, sealed test, 자연 행동분포 전체 재평가, 에너지/벽 loss, 새 시각 probe는 이번에 실행하지 않았다. 추가 실행은 하지 않고 이 파일럿 결과를 보존한다.
'''
 with (out/'report_ko.md').open('x') as f:f.write(text)
 # Small offline entry point; scientific figures remain standalone PNGs.
 headers=['지표','A 동결','B 기존 loss','C 상태 지도','D 물리 전이','E 전이+정지'];htmlrows=''.join('<tr><td>'+html.escape(label)+'</td>'+''.join(f'<td>{val(a,f,h,m):.5f}</td>' for a in 'ABCDE')+'</tr>' for label,f,h,m in specs)
 page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>DINO-WM 첫 physics loss 파일럿</title><style>body{font:16px/1.7 system-ui,sans-serif;color:#182333;background:#f6f8fa;margin:0}main{max-width:1120px;margin:35px auto;padding:28px;background:white}h1{font-size:28px}h2{margin-top:36px}table{border-collapse:collapse;font-size:14px;width:100%}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%;height:auto}.lead{border-left:5px solid #c24740;padding:15px;background:#fff3f1}a{color:#0062a8}.scroll{overflow:auto}</style><main><h1>DINO-WM 첫 physics loss 파일럿</h1><p class="lead"><b>이번 설정은 성공 기준을 통과하지 못했습니다.</b> 정지 속력은 줄었지만, physics loss 조건의 장기 위치 drift와 행동 반응이 악화됐습니다. 같은 개선 일부는 기존 loss만 학습한 B에서도 나타났습니다.</p><p>단일 seed · 조건별120step · 최종 sealed test 미사용 · 추가 학습에서만 보류한 개발평가</p><p><a href="report_ko.md">한국어 전체 보고서</a> · <a href="protocol_ko.md">실행 전 프로토콜</a> · <a href="metrics.csv">전수 지표 CSV</a> · <a href="paired.json">episode paired 비교</a></p><h2>조건별 결과</h2><p>H1=.25초, H5=1.25초. episode 내 branch평균 후 episode 동일가중. 행동개입90branch/10episodes, 정지10/10, 저속40/10. 작을수록 좋습니다.</p><div class="scroll"><table><thead><tr>'''+''.join('<th>'+h+'</th>' for h in headers)+'</tr></thead><tbody>'+htmlrows+'''</tbody></table></div><h2>정지와 궤적 정확도</h2><img src="comparison.png" alt="다섯 조건의 정지 속력, 위치 drift, 저속 및 행동개입 오차"><h2>행동 반응</h2><img src="action_response.png" alt="행동 반응 크기와 방향 오차"><h2>자기일치성과 접촉 실패</h2><p>첫 패널 자유공간 분모는 H1..H5에서90,54,21,10,5branch이며 같은 cohort가 아닙니다. H5의5개 자유공간 branch에서 E는 자기일치 속도잔차를 줄였지만 실제 위치·속도오차는 증가했습니다. 벽겹침 감소도 위치 정확도 개선을 뜻하지 않습니다.</p><img src="consistency.png" alt="자유공간 잔차와 벽겹침 및 proprio 오차"><h2>학습 곡선</h2><img src="training_curves.png" alt="기존 loss와 물리 전이 및 정지 성분의 학습 곡선"><p>조건별 checkpoint는 서버의 새 실행 폴더에 보존했습니다. 원본과 기존 연구 결과는 유지했습니다.</p></main></html>'''
 with (out/'index.html').open('x') as f:f.write(page)
 save(out/'report_complete.json',dict(verified_metrics=30,success={a:r['pass_point_estimates'] for a,r in success.items()},resources=resources,new_training_complete=True,adopt_checkpoint=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();report(a.root,a.repo,a.out)
