"""Read-only reporting of coherent velocity intervention artifacts."""
import argparse, json, os
from pathlib import Path
import numpy as np
from intervention import read,save,sha

def report(out):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib'))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    S=read(out/'summary.json');d=S['groups']['development'];m=[r for r in read(out/'metrics.json') if r['split']=='development'];co=[r for r in m if r['history']=='coast'];alpha=S['alpha']
    coef=[r for r in read(out/'coefficients.json') if r['split']=='development' and r['radius']==.25];bs=[r for r in read(out/'biases.json') if r['split']=='development'];effects=[r for r in read(out/'history_effects.json') if r['split']=='development']
    figures=[]
    def finish(fig,name):fig.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(fig);figures.append(name)
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for j,ax in enumerate(axs):
        for ep in sorted({r['episode'] for r in co}):
            rows=sorted([r for r in co if r['axis']==j or r['radius']==0],key=lambda r:r['radius']*r['sign'])
            rows=[r for r in rows if r['episode']==ep];x=[r['radius']*r['sign'] for r in rows]
            ax.plot(x,[r['prediction'][j] for r in rows],color='#2875b8',alpha=.2,lw=1)
        for axis,color,label in [(j,'#2875b8','Model parallel mean'),(1-j,'#d39424','Model cross-axis mean')]:
            x=sorted({r['radius']*r['sign'] for r in co if r['axis']==j or r['radius']==0})
            y=[np.mean([r['prediction'][axis] for r in co if (r['axis']==j or r['radius']==0) and r['radius']*r['sign']==v]) for v in x];ax.plot(x,y,'o-',c=color,label=label)
        ax.plot([-.5,.5],[-.5*alpha,.5*alpha],'--',c='black',label='Environment parallel');ax.axhline(0,c='gray',lw=.6);ax.set(xlabel=f'Initial velocity {"xy"[j]}',ylabel='Next velocity',title=f'Coherent coast history | {"xy"[j]} axis');ax.legend(fontsize=8)
    fig.suptitle('Zero future action | 0.25 s | thin lines: development sites');finish(fig,'decay_curves.png')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained');b=np.array([r['bias'] for r in bs])
    axs[0].scatter(b[:,0],b[:,1],c='#2875b8');axs[0].axhline(0,c='gray',lw=.7);axs[0].axvline(0,c='gray',lw=.7);axs[0].set(xlabel='Predicted vx at rest',ylabel='Predicted vy at rest',title='Rest drift by site')
    for ax,J,title in [(axs[1],np.mean([r['jacobian'] for r in coef],axis=0),'Model central difference'),(axs[2],alpha*np.eye(2),'Environment')]:
        im=ax.imshow(J,cmap='coolwarm',vmin=-.15,vmax=1.15)
        for i in range(2):
            for j in range(2):ax.text(j,i,f'{J[i,j]:.4f}',ha='center',va='center')
        ax.set(xticks=[0,1],xticklabels=['vx input','vy input'],yticks=[0,1],yticklabels=['vx next','vy next'],title=title)
    fig.colorbar(im,ax=list(axs[1:]),shrink=.8);finish(fig,'bias_and_damping.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');dm=d['coast_moving']
    keys=['velocity_error','bias_removed_error','hold_error','fixed_output_error'];labels=['Model','Subtract f(0)\n(diagnostic)','Hold velocity','Fixed f(0)']
    axs[0].bar(range(4),[dm[k]['mean'] for k in keys],color=['#2875b8','#28866b','#8c8c8c','#ba7b3d']);axs[0].set(xticks=range(4),xticklabels=labels,ylabel='Velocity L2 error',title='Paired performance references')
    keys=['bias_term','slope_term','nonlinear_term'];axs[1].bar(range(3),[dm[k]['mean'] for k in keys],color=['#ba7b3d','#2875b8','#28866b']);axs[1].set(xticks=range(3),xticklabels=['Intercept norm','Slope error norm','Even residual norm'],ylabel='Vector norm',title='Error terms: norms are NOT additive');finish(fig,'error_decomposition.png')
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for i,(h,label) in enumerate([('coast','Coast'),('push_positive','Past +x force'),('push_negative','Past -x force')]):
        rows=[r for r in m if r['history']==h and r['radius']==0];v=np.array([r['prediction'] for r in rows]);axs[0].scatter(v[:,0],v[:,1],label=label,alpha=.7)
    axs[0].axvline(0,c='gray',lw=.7);axs[0].axhline(0,c='gray',lw=.7);axs[0].set(xlabel='Predicted vx at rest',ylabel='Predicted vy at rest',title='Same current state, different past');axs[0].legend(fontsize=8)
    radii=[.1,.25,.5];rates=[d['by_radius'][str(r)]['energy_increase']['mean']*100 for r in radii];axs[1].bar(range(3),rates,color='#ba7b3d');axs[1].set(xticks=range(3),xticklabels=radii,xlabel='Initial speed',ylabel='Energy-increasing endpoints (%)',ylim=(0,100),title='Coast history | zero future force');finish(fig,'history_and_energy.png')
    branches=read(out/'branches.json');preds={(r['episode'],r['name']):r for r in read(out/'predictions.json')};first=next(r['episode'] for r in branches if r['split']=='development')
    fig,axs=plt.subplots(3,3,figsize=(8,8),layout='constrained')
    for i,h in enumerate(['coast','push_positive','push_negative']):
        r=next(r for r in branches if r['episode']==first and r['name']==h+'_zero');ims=np.load(out/r['image_file'])['images']
        for j in range(3):axs[i,j].imshow(ims[j]);axs[i,j].axis('off');axs[i,j].set_title(f'{h} | t={[-.51,-.26,-.01][j]:.2f}s',fontsize=9)
    fig.suptitle(f'Coherent image histories | site {first} | current v=0');finish(fig,'context_examples.png')
    def fmt(x):return f"{x['mean']:.5f} [{x['ci95'][0]:.5f}, {x['ci95'][1]:.5f}]"
    J=np.array(d['jacobian']['jacobian']['mean']);prep=read(out/'prepare_complete.json');checks=read(out/'checks.json');n0=sum(r['energy_increase'] for r in co if r['radius']==0);nm=sum(r['energy_increase'] for r in co if r['radius']>0);N=sum(r['radius']>0 for r in co)
    audit=read(out/'render_sensitivity.json')
    reduction=100*(1-dm['bias_removed_error']['mean']/dm['velocity_error']['mean'])
    text=f'''# DINO-WM 무입력 초기속도와 이력 실험 결과

초기속도를 바꾸면서 과거 상태·영상도 함께 물리적으로 재생한 단일 전이 실험이다. 현재 위치와 미래 무입력은 고정했다. 정지 상태 예측의 편향, 초기속도에 대한 유한차분 반응, 같은 현재 상태에서 과거 이력의 효과를 분리해 측정했다. 모델·encoder·proprio 역변환은 동결했고 별도 학습이나 physics loss는 추가하지 않았다.

## 주요 해석

이번 저속·무입력 범위에서는 초기속도에 비례하는 감쇠 반응의 오차보다 위치별 정지 출력 편향이 크게 관측됐다. 정지 편향을 빼는 진단으로 평균 속도오차가 {dm['velocity_error']['mean']:.5f}에서 {dm['bias_removed_error']['mean']:.5f}로 약 {reduction:.1f}% 감소했다. 이는 같은 위치의 정지 출력까지 사용한 사후 진단이며 일반화 성능 개선이나 실제 모델 수정이 아니다.

평균 편향의 y성분은 음수이고95% CI가0을 포함하지 않는다. 반면 평균 대각 반응계수의 CI는 환경 감쇠계수를 포함하므로 감쇠율이 체계적으로 잘못됐다고 확정할 수 없다. 위치별 교차축·비선형 오차는 여전히 남는다. 원 모델의 무입력 오차는 단순 속도유지 기준보다 크다. 행동 반응이 좋았던 C 결과를 정지·저속 유지 능력으로 확대할 수 없다.

같은 현재 상태에서 과거힘을 바꾼 효과는 {d["history"]["push_positive"]["velocity_difference"]["mean"]:.5f} / {d["history"]["push_negative"]["velocity_difference"]["mean"]:.5f}로 측정됐다. 이 범위에서는 {dm["bias_term"]["mean"]:.5f} 수준의 정지 편향보다 작지만, 과거힘은±0.05x만 검사했으므로 모든 이력 효과가 작다고 일반화하지 않는다.

추가 calibration2개 렌더 민감도 감사에서 기존 영상 대신 같은 시점의 재렌더 영상을 쓰면 속도 출력 차이는 평균 {audit['mean_velocity_difference']:.5f}, 최대 {audit['max_velocity_difference']:.5f}였다. RGB 차이가 작아도 예측 영향은0이 아니다. 이번 정지 편향을 전부 모델 고유의 동역학 오류로 귀속하거나 렌더 영향이 없다고 단정하지 않는다. 이 검사는 결과 확인 후 추가한 민감도 감사이며 모델·임계값을 조정하지 않았다.

## 표본과 검증

기존 C의 후보40episodes를 유지하고 raw10..70 중 처음 안전한 위치를 선택했다. {prep['sites']}개 위치, {prep['branches']}개 조건을 실행했다. 분할은 {prep['by_split']}이다. 최초 C 위치 그대로의 기하 검사에서는 fit3개만 가능해 모델 추론 전에 위치 탐색을 보완했다. 자유공간 기준·속도grid·후보episode는 완화/교체하지 않았다. 변경 전 설계도 Git에 보존했다.

주 coast 이력은 지난0.5초와 미래0.25초의 힘이0이다. v=0, ±x/±y 크기0.1/0.25/0.5의13조건을 썼다. 이력 대조는 현재 v=0 또는 ±0.25축속도에 대해 과거힘만 ±0.05x로 바꾼10조건이다. 미래힘은 모두0이다. 과거영상을 고정한 채 속도채널만 조작하지 않았다. 이미지의0.01초 지연도 원자료 규칙과 맞췄다.

모든 선택 조건의 전체 과거·미래 trace가 무접촉·무clip이며 margin>0.004다. 환경식 최대차이 {checks['oracle_max_abs']:.3e}, fine 에너지 잔차 {checks['energy_max_abs']:.3e}, 원 C 무입력 baseline 재현 최대차이 {checks['baseline_reproduction_max']:.3e}로 통과했다. 렌더 비교와 입력·가중치 SHA 검증도 통과했다.

## 초기속도 반응과 편향

아래는 development {d['episodes']}개 위치의 episode 동일가중 평균과 탐색적95% bootstrap CI다. 같은 위치의 분기를 독립 표본으로 세지 않았다.

| 지표 | 평균 [95% CI] |
|---|---|
| 비영 초기속도 최종속도 오차 | {fmt(dm['velocity_error'])} |
| 정지 상태에서 예측된 속력 ‖f(0)‖ | {fmt(d['zero']['velocity_error'])} |
| 정지 편향을 뺀 진단 오차 | {fmt(dm['bias_removed_error'])} |
| 속도유지 기준 오차 | {fmt(dm['hold_error'])} |
| f(0) 출력고정 기준 오차 | {fmt(dm['fixed_output_error'])} |
| 반대 초기속도 출력 대조 오차 | {fmt(dm['sign_flip_error'])} |

환경은 v_next={alpha:.8f}v이고 정지시0을 유지한다. 모델 f(0)의 평균 벡터는 {d['bias']['mean']}이다. 이는 현재 위치별로 다른 context에서 측정한 출력 편향이며 전체 위치에 통용되는 상수라고 가정하지 않았다.

![무입력 속도 곡선](decay_curves.png)

r=0.25의 중앙차분 J 평균(행=출력속도 축, 열=초기속도 축):

```text
{np.array2string(J,precision=5)}
환경: {alpha:.5f} I
```

개별 J의 환경 대비 Frobenius 오차는 {fmt(d['jacobian']['frobenius_error'])}, 대각 평균은 {fmt(d['jacobian']['diagonal_mean'])}다. 현재속도와 물리적으로 일관된 과거 이력을 함께 바꾼 계수이며, 입력 velocity 채널만의 편미분이나 모델 내부 마찰계수가 아니다.

![정지 편향과 감쇠 행렬](bias_and_damping.png)

## 오차 분해와 에너지

각 축의 ±r 쌍에서 오차벡터를 f(0), (J_r−αI)v, 남은 짝수 비선형항으로 정확히 분해했다. 각 항 norm 평균은 {dm['bias_term']['mean']:.5f}, {dm['slope_term']['mean']:.5f}, {dm['nonlinear_term']['mean']:.5f}다. 벡터 합 closure 오차는 {checks['vector_decomposition_max_abs']:.3e}다. **norm들은 서로 더해 총오차가 되지 않으므로 기여율로 해석하지 않는다.** f(0)을 빼는 조작은 원인 진단이며 모델을 교정하거나 새 성능으로 보고하지 않는다. paired 오차 감소는 {fmt(d['paired_bias_removal'])}다.

![오차 분해](error_decomposition.png)

coast 이력 정지 {d['episodes']}개 중 {n0}개, 비영속도 {N}개 중 {nm}개에서 예측 끝점 운동에너지가 증가했다(양의 초과 수치기준1e-8). 이는 자유공간·무외력 경로를 가정한 필요조건 진단이다. 실제 모델 내부 경로의 에너지 적분을 측정한 것은 아니다. 원 C와 초기속도·위치·과거힘이 달라 비율을 직접 개선/악화로 비교하지 않는다.

## 같은 현재 상태에서 이력 효과

현재 q/v와 미래힘을 같게 유지해도 과거에 +x힘을 준 경우 coast 대비 속도예측 차이는 {fmt(d['history']['push_positive']['velocity_difference'])}, −x힘은 {fmt(d['history']['push_negative']['velocity_difference'])}다. 같은 상태에서 환경의 미래 차이는 최대 {checks['oracle_same_state_history_max']:.3e}다. 이 효과는 영상·상태·action 과거가 함께 다른 물리적 이력에 대한 반응이며 한 입력 채널의 원인으로 단정하지 않는다.

![과거 이력과 에너지](history_and_energy.png)

![실제 생성된 입력 영상](context_examples.png)

## 해석 범위와 재사용

개발 표본이며 미관측 WM test나 다단계 D 평가가 아니다. 상수·저속 이력은 기존 데이터의 시간 패턴과 다르다. 새 렌더와 원영상 RGB 차이는 render_checks.json에 남겼다. 정규화는 기존 C 통계를 그대로 사용했다. 기존 A/B/C 검사와 C 원 무입력28개 결과를 재사용했으며, 최대2개 calibration C 출력만 구현 정렬 검사를 위해 다시 추론했다. 결과 분해·bootstrap은 동결 출력 후처리이고 학습이 아니다.

원자료는 branches.json, predictions.json, coefficients.json, history_effects.json, summary.json에 있다. 서버 token_cache는 새 context/예측 재사용용이다. [조건별 궤적 보기](viewer.html). 기존 데이터·체크포인트·결과를 보존했고 thread0 게시는 하지 않았다.
'''
    (out/'report_ko.md').write_text(text)
    payload=[]
    for r in branches:
        payload.append(dict(episode=r['episode'],split=r['split'],name=r['name'],path=np.array(r['trace'])[51:,:2].tolist(),start=r['start'],end=r['endpoint'],prediction=preds[r['episode'],r['name']]['prediction']))
    template='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DINO-WM 무입력 초기속도 실험</title><style>body{font:16px system-ui;margin:32px;background:#f4f6fa;color:#162334}section{background:white;padding:24px;margin:16px 0;border-radius:16px}select{font-size:16px;padding:8px}svg{width:min(800px,100%);height:auto}p{line-height:1.6}</style><section><h1>초기속도와 과거 이력에 따른 무입력 예측</h1><p>빨강은 실제 환경 궤적·끝점, 파랑은 모델 끝점입니다. 화살표는 끝점 속도입니다. 모델 중간 경로는 표시하지 않습니다.</p><label>위치 <select id="ep"></select></label> <label>조건 <select id="cond"></select></label></section><section><svg id="plot" viewBox="0 0 600 600" role="img" aria-label="환경 궤적과 모델 끝점 비교"></svg><p id="stats"></p></section><script>const data=__DATA__,walls=__WALLS__;const ep=document.getElementById('ep'),cond=document.getElementById('cond'),svg=document.getElementById('plot');for(const n of [...new Set(data.map(r=>r.episode))]){let r=data.find(r=>r.episode===n);ep.add(new Option(r.split+' · '+n,n))}for(const n of [...new Set(data.map(r=>r.name))])cond.add(new Option(n,n));const X=x=>30+x*140,Y=y=>570-y*140;function draw(){const r=data.find(r=>r.episode===+ep.value&&r.name===cond.value);let html='<defs><marker id="red" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0L6,3L0,6" fill="#cc4d4d"/></marker><marker id="blue" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto"><path d="M0,0L6,3L0,6" fill="#2676b8"/></marker></defs>';for(const w of walls)html+=`<rect x="${X(w.center[0]-w.halfsize[0])}" y="${Y(w.center[1]+w.halfsize[1])}" width="${280*w.halfsize[0]}" height="${280*w.halfsize[1]}" fill="#d7dbe2"/>`;html+=`<polyline points="${r.path.map(p=>X(p[0])+','+Y(p[1])).join(' ')}" fill="none" stroke="#cc4d4d" stroke-width="2"/>`;for(const [p,c,id] of [[r.end,'#cc4d4d','red'],[r.prediction,'#2676b8','blue']]){html+=`<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="8" fill="none" stroke="${c}" stroke-width="2"/><line x1="${X(p[0])}" y1="${Y(p[1])}" x2="${X(p[0]+.3*p[2])}" y2="${Y(p[1]+.3*p[3])}" stroke="${c}" stroke-width="2" marker-end="url(#${id})"/>`}html+=`<circle cx="${X(r.start[0])}" cy="${Y(r.start[1])}" r="3" fill="#162334"/>`;svg.innerHTML=html;document.getElementById('stats').textContent='미래힘 0 · 0.25초 · 초기속도 ['+r.start.slice(2).map(v=>v.toFixed(3))+'] · 속도오차 '+Math.hypot(r.end[2]-r.prediction[2],r.end[3]-r.prediction[3]).toFixed(4)}ep.onchange=cond.onchange=draw;draw();</script></html>'''
    (out/'viewer.html').write_text(template.replace('__DATA__',json.dumps(payload)).replace('__WALLS__',json.dumps(read(out/'parameters.json')['walls'])))
    save(out/'report_manifest.json',dict(script_sha256=sha(__file__),source_summary_sha256=sha(out/'summary.json'),figures=figures,viewer='viewer.html',new_training=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);report(ap.parse_args().out)
