"""한국어 결과 보고서, 정적 연구 그림, 네트워크 없는 궤적 뷰어 생성."""
import csv
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/dinowm-c-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from intervention import read,save,sha

p=Path(sys.argv[1]);summary=read(p/'summary.json');aux=read(p/'auxiliary_summary.json');checks=read(p/'checks.json')
params=read(p/'parameters.json');contexts=read(p/'selected.json');conditions=read(p/'conditions.json')
oracles=read(p/'oracles.json');preds=read(p/'predictions.json');rows=list(csv.DictReader((p/'branch_metrics.csv').open()))
lookup={(r['episode'],r['condition']):r for r in preds};truth={(r['episode'],r['condition']):r for r in oracles}
def get(radius=None,split='development',layer='main_common_free'):
    return next(r for r in summary if r['split']==split and r['layer']==layer and r['radius']==radius)
def ci(x):return f"{x['mean']:.5f} [{x['ci95'][0]:.5f}, {x['ci95'][1]:.5f}]"
main=get();stress=get(1.,layer='stress_all');zero=next(r for r in aux if r['split']=='development' and r['kind']=='zero')
radii=[.05,.1,.2,.4];gain=float(params['macro_B'][0][0]);dev=[r for r in rows if r['split']=='development']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
for field,label,color in [('response_norm','Frozen WM','#286fba'),('environment_response_norm','Simulator','#d14b40')]:
    mean=np.array([get(r)[field]['mean'] for r in radii]);bounds=np.array([get(r)[field]['ci95'] for r in radii])
    axs[0].errorbar(radii,mean,yerr=[mean-bounds[:,0],bounds[:,1]-mean],marker='o',label=label,color=color,capsize=4)
axs[0].set(xlabel='Constant action norm',ylabel='Response norm relative to zero action',title=f'Same-context action response / development n={main["episodes"]}');axs[0].legend()
fields=['response_error','fixed_zero_response_error','sign_flip_response_error','wrong_branch_response_error']
mean=[main[f]['mean'] for f in fields];bounds=np.array([main[f]['ci95'] for f in fields])
axs[1].bar(range(4),mean,color=['#286fba','#b4b8bf','#b4b8bf','#b4b8bf']);axs[1].errorbar(range(4),mean,yerr=[np.array(mean)-bounds[:,0],bounds[:,1]-mean],fmt='none',color='black',capsize=4)
axs[1].set_xticks(range(4),['Correct','Fixed zero','Sign flip','Wrong direction']);axs[1].set(ylabel='Response error L2',title='Paired negative controls / main radii')
fig.savefig(p/'action_response.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(10,8),layout='constrained')
for ax,radius in zip(axs.flat,radii):
    rr=[r for r in dev if float(r['radius'])==radius]
    for fieldx,fieldy,color,label in [('environment_delta_vx','environment_delta_vy','#d14b40','Simulator'),('delta_vx','delta_vy','#286fba','Frozen WM')]:
        means=np.array([[np.mean([float(r[f]) for r in rr if int(r['direction'])==d]) for f in [fieldx,fieldy]] for d in range(8)])
        ax.quiver(np.zeros(8),np.zeros(8),means[:,0],means[:,1],angles='xy',scale_units='xy',scale=1,color=color,alpha=.8,label=label,width=.006)
        ax.scatter(means[:,0],means[:,1],color=color,s=12)
    limit=radius*gain*1.28;ax.set(xlim=(-limit,limit),ylim=(-limit,limit),xlabel='Delta vx',ylabel='Delta vy',title=f'Action norm {radius:g} / mean of {main["episodes"]} contexts');ax.set_aspect('equal');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(p/'response_vectors.png',dpi=170);plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
j=next(r for r in aux if r['split']=='development' and r['kind']=='jacobian' and r['radius']==.1)
for ax,matrix,title in zip(axs,[j['mean_model'],j['mean_environment'],(np.array(j['mean_model'])-j['mean_environment']).tolist()],['WM Jacobian','Simulator Jacobian','WM - simulator']):
    a=np.array(matrix);lim=.3 if title=='WM - simulator' else 6.
    ax.imshow(a,cmap='coolwarm',vmin=-lim,vmax=lim);ax.set_title(title+' / r=0.1');ax.set_xticks([0,1],['action x','action y']);ax.set_yticks([0,1],['velocity x','velocity y'])
    for iy in range(2):
        for ix in range(2):ax.text(ix,iy,f'{a[iy,ix]:.3f}',ha='center',va='center',color='black')
fig.savefig(p/'action_jacobian.png',dpi=170);plt.close(fig)

zeros=list(csv.DictReader((p/'zero_input.csv').open()));zz=sorted([r for r in zeros if r['split']=='development'],key=lambda r:int(r['episode']))
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
axs[0].axhline(0,color='black',lw=1);axs[0].scatter(range(len(zz)),[float(r['model_energy_increase']) for r in zz],label='WM',color='#286fba')
axs[0].scatter(range(len(zz)),[float(r['environment_energy_increase']) for r in zz],label='Simulator',color='#d14b40',marker='x')
axs[0].set_xticks(range(len(zz)),[r['episode'] for r in zz],rotation=60);axs[0].set(xlabel='Episode',ylabel='Delta kinetic energy / zero action',title='Zero-input energy change');axs[0].legend()
rr=[r for r in dev if r['stress']=='False'];x=np.array([float(r['environment_necessary_excess']) for r in rr]);y=np.array([float(r['model_necessary_excess']) for r in rr])
axs[1].scatter(x,y,s=14,alpha=.5);axs[1].axhline(0,color='#d14b40',lw=1);axs[1].set(xlabel='Simulator: Delta K - actuator work',ylabel='WM endpoints: Delta K - actuator work',title='Conditional endpoint energy diagnostic')
fig.savefig(p/'energy_diagnostic.png',dpi=170);plt.close(fig)

# All contexts and branches, not just a cherry-picked figure, remain inspectable.
viewer=dict(contexts=contexts,conditions=conditions,walls=params['walls'],radius=params['particle_radius'],
    branches=[dict(episode=r['episode'],condition=r['condition'],trace=r['trace'],free=r['free'],
                   contacts=r['wall_contact_substeps'],clips=r['clipped_boundaries'],prediction=lookup[r['episode'],r['condition']]['prediction']) for r in oracles])
template='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DINO-WM 행동 개입 C 결과</title><style>body{font:16px system-ui;margin:24px;background:#f4f6f8;color:#172033}header,section{background:white;padding:20px;border-radius:12px;margin:14px 0}select{font:inherit;padding:7px;margin:5px}svg{width:100%;max-width:760px;background:#fff}p{line-height:1.65}.blue{color:#286fba}.red{color:#d14b40}</style>
<header><h1>같은 관측 이력에서 행동만 변경</h1><p>파랑은 동결 WM의 다음 상태, 빨강은 같은 행동을 적용한 simulator입니다. 선은 simulator의 실제 substep 궤적이며 모델은 끝점만 표시합니다. 예측점까지의 선을 모델의 실제 경로로 해석하지 마세요.</p>
<label>Context <select id="ctx"></select></label><label>행동 <select id="cond"></select></label></header>
<section><svg id="plot" viewBox="0 0 620 580" role="img" aria-label="환경 궤적과 모델 예측 끝점"></svg><p id="details"></p></section>
<section><h2>읽는 방법</h2><p>같은 context에서 zero와 다른 행동을 번갈아 선택하세요. 위치·속도는 같은 시점의 값입니다. stress(r=1)는 접촉/속도제한이 자주 발생하므로 main(r≤0.4)과 구분합니다. 운동에너지 필요조건은 예측 경로에도 무접촉 등의 가정이 성립할 때만 해석할 수 있습니다.</p></section>
<script>const DATA=__DATA__;const cs=document.querySelector('#ctx'),bs=document.querySelector('#cond');
for(const c of DATA.contexts){let o=new Option(c.split+' · episode '+c.episode+' · raw '+c.raw_anchor,c.episode);cs.add(o)}
for(const c of DATA.conditions){bs.add(new Option(c.name==='zero'?'무입력':(c.stress?'stress ':'')+'크기 '+c.radius+' · '+c.direction*45+'°',c.name))}bs.add(new Option('원 기록 action','recorded'));
const X=x=>55+x*125,Y=y=>525-y*125;
function draw(){let c=DATA.contexts.find(c=>c.episode===+cs.value),b=DATA.branches.find(b=>b.episode===c.episode&&b.condition===bs.value);let st=c.start,gt=b.trace.at(-1),pr=b.prediction;
let s='<defs><marker id="red" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#d14b40"/></marker><marker id="blue" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="#286fba"/></marker></defs>';
for(const w of DATA.walls)s+=`<rect x="${X(w.center[0]-w.halfsize[0])}" y="${Y(w.center[1]+w.halfsize[1])}" width="${w.halfsize[0]*250}" height="${w.halfsize[1]*250}" fill="#dfe3e8" stroke="#a6b0bd"/>`;
s+=`<polyline points="${b.trace.map(p=>X(p[0])+','+Y(p[1])).join(' ')}" fill="none" stroke="#d14b40" stroke-width="2"/><circle cx="${X(st[0])}" cy="${Y(st[1])}" r="5" fill="#172033"/>`;
for(const [p,col,mark] of [[gt,'#d14b40','red'],[pr,'#286fba','blue']])s+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="${DATA.radius*125}" fill="none" stroke="${col}" stroke-width="2"/><line x1="${X(p[0])}" y1="${Y(p[1])}" x2="${X(p[0]+.12*p[2])}" y2="${Y(p[1]+.12*p[3])}" stroke="${col}" stroke-width="2" marker-end="url(#${mark})"/>`;
document.querySelector('#plot').innerHTML=s;document.querySelector('#details').textContent=`0.25초 · 자유공간 ${b.free?'예':'아니오'} · 벽 접촉 substeps ${b.contacts} · 속도 clip ${b.clips}회 · 속도오차 ${Math.hypot(pr[2]-gt[2],pr[3]-gt[3]).toFixed(4)} · 위치오차 ${Math.hypot(pr[0]-gt[0],pr[1]-gt[1]).toFixed(4)}. 화살표: 끝점 속도, 표시길이=속도×0.12.`}
cs.onchange=draw;bs.onchange=draw;draw();</script></html>'''
(p/'viewer.html').write_text(template.replace('__DATA__',json.dumps(viewer,ensure_ascii=False)))

prepare=read(p/'prepare_complete.json');reuse=read(p/'reuse_audit.json');infer=read(p/'infer_complete.json');manifest=read(p/'manifest.json');dist=read(p/'action_distribution.json')
nonzero=[r for r in dev if r['stress']=='False' and float(r['radius'])>0]
devmain=[r for r in dev if r['stress']=='False']
violations=sum(int(r['necessary_condition_violation']) for r in nonzero)
allviolations=sum(int(r['necessary_condition_violation']) for r in devmain)
penetrations=sum(float(r['model_endpoint_penetration'])>.01 for r in devmain)
def av(kind,radius):return next(r for r in aux if r['split']=='development' and r['kind']==kind and r['radius']==radius)['metrics']['model_error']['mean']
jobs={}
for stage in ['prepare','infer','analyze']:
    name=f'{p.name}-{stage}-job/status.json'
    candidates=[p/'provenance'/name,p.parent.parent/name]
    jobs[stage]=read(next(f for f in candidates if f.exists()))
figs=['action_response.png','response_vectors.png','action_jacobian.png','energy_diagnostic.png']
lines=['# DINO-WM 동일 context 행동 개입 결과',
f'C 단일 단계 pilot을 완료했다. **관측 이력을 고정하고 행동만 바꿨을 때 모델은 행동의 방향과 크기에 맞는 속도 변화를 보였다.** 개발{main["episodes"]}개 context의 반응 오차는{main["response_error"]["mean"]:.5f}, 평균 방향 오차는{main["angle_degrees"]["mean"]:.2f}°다. 입력고정·부호반전·잘못된 방향 대조군보다 정확했다. 다만 무입력 감쇠, 축간 결합, 대칭성·선형성, 일부 끝점 에너지 조건에서는 불일치가 남는다.',
'## 실행과 표본',
f"실행 결과: `{p.name}`. 후보40episodes에서 사전 기하/속도 규칙을 통과한{len(contexts)}contexts(선택분할: {prepare['selected_by_split']})를 선택했다. 각 context에 main33, stress8, 기록action1을 비교했다. main{len(contexts)*33}branch, stress{len(contexts)*8}branch, 기록baseline{len(contexts)}개다. 유효 표본을 늘리기 위해 제외 기준을 완화하거나 다른 episode로 대체하지 않았다.",
'각 context는 원 관측 영상·proprio 3시점과 과거 action 2blocks를 공유한다. 미래를 만드는 마지막 action block만 바꾸고 raw5회, 0.25초 동안 같은 control을 유지했다. 실제 미래 관측은 넣지 않았다. episode당 anchor는1개이며 기존 raw10이 부적합하면 macro2..14의 첫 적합 anchor를 썼다. 따라서 B와 평가 시점·선택표본이 다르고 수치를 직접 성능 향상으로 비교하지 않는다.',
'모델·encoder·proprio 역변환을 동결했다. 새 WM 추론은 했지만 모델/probe 재학습, physics loss, D 다단계/보류평가, 환경 설치, thread0 업로드는 하지 않았다. WM train 영역의 개발 자료이므로 미관측 WM 일반화 실험이 아니다.',
'## 행동 반응의 크기와 방향',
f'반응은 `δv(u)=v_next(u)−v_next(0)`이다. 초기속도가0이 아닐 수 있으므로 최종속도 자체가 행동 방향과 평행해야 한다고 요구하지 않는다. 아래는 development{main["episodes"]}contexts의 비영 main{main["rows"]}branches다. 평균은 episode 내 branch평균→episode 동일가중, CI는 episode bootstrap10000/seed0의 탐색적95% 구간이다. 각 branch를 독립표본으로 세지 않았다.',
'| 지표 | 평균과 95% CI |','|---|---|',
f"| 반응 벡터 L2 오차 | {ci(main['response_error'])} |",
f"| 반응 방향 오차 도 | {ci(main['angle_degrees'])} |",
f"| 행동 평행 방향 이득 | {ci(main['parallel_gain'])} |",
f"| 최종속도 오차 | {ci(main['velocity_error'])} |",
f'환경의 상수 control 이득은{gain:.6f}이다. 모델의 평균 이득{main["parallel_gain"]["mean"]:.5f}은 약{(main["parallel_gain"]["mean"]/gain-1)*100:.1f}% 크지만 CI가 환경 이득을 포함한다. 이는 모든 방향·context에서 정확하다는 뜻은 아니다. 각도에서 두 반응 norm≥1e-6을 요구했고 main 비영branch 제외는0개다.',
'| 행동 크기 | 반응 오차 | 평균 방향 오차 도 | 모델 평행 이득 |','|---|---:|---:|---:|']
for r in radii:
    s=get(r);lines.append(f"| {r:.2f} | {s['response_error']['mean']:.5f} | {s['angle_degrees']['mean']:.3f} | {s['parallel_gain']['mean']:.5f} |")
lines+=['![행동 반응과 음성 대조](action_response.png)','![방향별 평균 속도 반응](response_vectors.png)',
'## 음성 대조군',
'모든 branch에 zero 출력을 붙이면 행동에 반응하지 않는 모델이 된다. 부호반전 대조는 반대방향 행동의 모델 출력을, 잘못된 방향 대조는 같은 크기에서90° 회전한 branch 출력을 붙인다. 정상 결과와 동일한 context·행동 집합에서 비교했다.',
'| 조건 | 반응 오차 | 정상보다 커진 오차와 paired 95% CI |','|---|---:|---|',
f"| 정상 대응 | {main['response_error']['mean']:.5f} | 기준 |"]
for name,key in [('zero 출력 고정','fixed_zero'),('부호반전','sign_flip'),('다른 방향 대응','wrong_branch')]:lines.append(f"| {name} | {main[key+'_response_error']['mean']:.5f} | {ci(main[key+'_minus_model'])} |")
lines+=['세 대조군 모두 정상보다 오차가 크고 paired CI가0을 포함하지 않는다. B의 기록 action 순서 대조보다 이번 검사는 행동 방향·크기에 대한 모델의 인과적 반응을 직접 확인한다. 다만 모델 전체의 물리 이해나 독립적으로 초기속도를 바꿨을 때의 마찰/질량 식별까지 입증하지는 않는다.',
'## 무입력 감쇠와 Jacobian',
f"무입력에서 모델의 속도 오차는 {ci(zero['metrics']['velocity_error'])}, 속력비 평균은 {ci(zero['metrics']['speed_ratio'])}다. 환경 속력비{params['macro_A'][0][0]:.6f}과 비교하면 평균 감쇠가 약해 보이지만 비의 CI는 넓다. 모델의 운동에너지가 증가한 context는 **{zero['positive_model_energy']}/{zero['rows']}개**다. 실제 자유공간 무입력 환경에서는 모든 에너지가 감소했다.",
'r=0.1의 ±x/±y 중앙차분으로 구한 평균 행동 Jacobian은 다음과 같다. 행=속도 출력축, 열=행동 입력축이다.',
f"```text\n모델:    [{j['mean_model'][0][0]:.5f}, {j['mean_model'][0][1]:.5f}]\n         [{j['mean_model'][1][0]:.5f}, {j['mean_model'][1][1]:.5f}]\n환경:    [{gain:.5f}, 0]\n         [0, {gain:.5f}]\n```",
f"개별 context의 Jacobian Frobenius 오차 평균은 {ci(j['metrics']['frobenius_error'])}다. 평균 행렬의 차이만 보면 context별 오차가 상쇄될 수 있으므로 둘을 구분한다. 교차축 반응도 남는다. 이는 유한 간격 중앙차분이며 모델 내부의 정확한 미분이나 물리 파라미터가 아니다.",
'![행동 Jacobian](action_jacobian.png)',
f'대칭성 `||δv(+u)+δv(−u)||`은 r=.05에서{av("symmetry",.05):.5f}, r=.4에서{av("symmetry",.4):.5f}이다. r=.05 응답을 비례 확대했을 때 r=.4의 불일치는{av("linearity",.4):.5f}다. 환경은 주 자유공간에서 두 성질을 수치 정밀도로 만족한다. 따라서 모델이 거의 선형적인 행동 반응을 보이더라도 완전히 선형·대칭인 것은 아니다.',
'## 큰 행동과 에너지 진단',
f"stress r=1.0은 전체{len(contexts)*8}branch 중{checks['stress_nonfree']}개가 접촉·clip·기하 여유 조건 중 하나를 충족하지 못했다. development 전체{stress['rows']} stress branches의 반응 오차는{stress['response_error']['mean']:.5f}, 방향 오차는{stress['angle_degrees']['mean']:.2f}°다. main의 정확도를 큰 행동·벽 근처로 그대로 일반화할 수 없다. 조건별 구분은 원시 CSV와 viewer에 보존했다.",
'상수힘에서 구동 일은 끝점으로 `W=(Gu)·(q_next−q_start)`를 계산할 수 있다. 다만 WM의 중간 경로는 관측되지 않으므로 `ΔK≤W`는 예측 경로에도 무접촉·기타 에너지원 없음 등의 가정이 성립할 때의 필요조건 진단이다. oracle이 자유공간이라는 사실만으로 예측 경로까지 보증하지 않는다.',
f"development 비영 main{main['rows']}branch 중 **{violations}개({100*violations/len(nonzero):.2f}%)**가 이 필요조건의 양의 초과를 보였다(수치 구분1e-8). episode별 비율의 CI는 [{main['necessary_condition_violation']['ci95'][0]*100:.2f}%, {main['necessary_condition_violation']['ci95'][1]*100:.2f}%]다. zero까지 포함하면{allviolations}/{len(devmain)}개다. 예측 끝점 벽겹침은 development main에서{penetrations}개였고 이 실패를 제외하지 않았다. 조건부 진단을 모델의 실제 substep 에너지 위반량으로 해석하지 않는다.",
f"환경 fine 에너지 수지의 최대 잔차는{checks['free_fine_energy_max']:.3e}이다. 선형속도를 가정한 모델−환경 coarse 수지 차이는{ci(main['coarse_residual_difference'])}로 CI가0을 포함한다. 평균 coarse 잔차가 작다는 것과 일부 끝점 필요조건 불일치가 존재한다는 것은 서로 다른 사실이다.",
'![무입력과 끝점 에너지 진단](energy_diagnostic.png)',
'## 분포와 해석 범위',
f"기존 train100의 raw action norm 중앙값은{dist['norm_quantiles'][3]:.3f}다. norm≤0.05/0.1/0.2/0.4의 비율은 각각{[round(dist['intervention_norm_percentiles'][str(r)]*100,2) for r in radii]}%이며 정확히 같은 action을5회 반복한 block 비율은{dist['constant_5action_fraction']:.4f}이다. 작은 상수 control 개입은 actuator 범위 안이지만 기록 action의 크기·시간 패턴과 다르다. 이 조건에서도 관측된 좋은 행동 반응은 의미가 있으나 전체 상태·행동 분포에 대한 일반화는 확인하지 않았다.",
'calibration fit/tune과 development의 결과는 `summary.json`에서 별도로 제공한다. development 결과로 모델·임계값·행동 grid를 튜닝하지 않았다. angle 기준1e-6과 에너지 수치 구분1e-8은 수치정의 기준이며 후자는 CPU 환경 floor보다 충분히 크지만 과학적 허용오차나 실용적 손실 한계가 아니다. 다중비교 보정 없는 소표본 탐색 결과다.',
'## 실행 기록과 재현',
f"초기 두 준비 시도는 기존 trace의 raw float64 action을 모델용 float32 key로 재구성한 오류 때문에 검증에서 중단됐다. float32 정밀도 수준의 차이를 발견했고 tolerance1e-10을 완화하지 않았다. simulator에는 원 action dtype, 모델에는 기존 float32 전처리를 유지하도록 수정했다. main의 새 상수 control은 동일 float32 raw값을 양쪽에 전달했다. 실패 폴더와 진단을 보존했으며, v3는 통과한 partial C {reuse['reused_partial']}개와 기존 B/전이 {reuse['reused_prior']}개를 재사용하고 {reuse['new']}개만 새로 계산했다. 재사용 원본의 모호한 provenance 표기는 `reuse_audit.json`으로 보완했다.",
f"원 H1 baseline2개 재추론 차이는{checks['baseline_reproduction_max']:.1e}, 기록 action GT재시작 최대성분오차는{prepare['recorded_gt_max_abs']:.3e}, action 정규화 왕복 최대차이는{infer['action_roundtrip_max']:.3e}다. 과거 action/관측 context 동일, 미래 관측 미사용, split중복0, 입력/가중치 SHA 불변을 확인했다. 전체{infer['rows']}개 모델 조건 중 기록baseline{infer['cached_recorded_outputs']}개는 cache를 재사용했다. 주 결과는 첫patch proprio 역변환이며 열공간 잔차와 patch간 차이도 저장했다.",
f'v3 CPU 준비{jobs["prepare"]["elapsed_seconds"]:.2f}초, GPU 추론{jobs["infer"]["elapsed_seconds"]:.2f}초, CPU 분석{jobs["analyze"]["elapsed_seconds"]:.2f}초. 표본 최대 RAM은 GPU 단계 약{jobs["infer"]["peak_rss_bytes"]/1024**3:.2f}GiB, 장치전체 GPU 메모리는{jobs["infer"]["peak_gpu_device_mib"]}MiB로 설정 상한12GiB/11000MiB 이내였다. simulator와 GPU 단계는 순차 실행했다. 두 실패 준비 시도 시간은 별도 보존 로그에 있으며 위 v3 시간에 포함하지 않는다.',
f"추론 코드 commit `{manifest['code_commit']}`; 분석 코드 commit은 별도 delivery manifest에 기록한다. `manifest.json`에 checkpoint/config/상태/action/기존 cache/환경소스 SHA, `image_sources.json`에 사용 영상 SHA가 있다. CPU raw trace, branch별 예측·정규화action·context hash, 계측 수지 및 episode CI를 보존했다. 서버의 `token_cache/`에는 context와 새 예측 full tokens를 남겨 후속 다단계의 첫 step을 재사용할 수 있다. 이 대용량 cache는 Mac 보기용 사본에 포함하지 않았다.",
'[전체 context와 행동을 고르는 오프라인 뷰어](viewer.html). PNG 네 장은 오른쪽 파일 탐색기의 `연구결과_C`에도 제공한다. 원 결과와 이전 문서는 보존한다. 다음 D는 다단계 누적오차와 사전 고정 보류표본 검증이며 이번 C 작업에서 실행하지 않았다.']
body='\n\n'.join(lines)+'\n';body=body.replace('|\n\n|','|\n|')
(p/'report_ko.md').write_text(body)
save(p/'report_manifest.json',dict(script_sha256=sha(__file__),source_checks_sha256=sha(p/'checks.json'),
     figures=figs,viewer='viewer.html',scope='Read-only postprocessing; no new inference or fit'))
print('Korean report, four PNG figures, and offline viewer written.')
