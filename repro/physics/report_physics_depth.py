"""Plots and Korean interpretation for the frozen physical consistency study."""
import argparse,json,os
from pathlib import Path
import numpy as np
from intervention import read,save,sha,clearance

def report(out):
 os.environ.setdefault('MPLCONFIGDIR',str(out/'matplotlib'));import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 S=read(out/'summary.json');params=read(out/'parameters.json');m=read(out/'metrics.json');cases=read(out/'cases.json');oracles={r['id']:r for r in read(out/'oracles.json')};t=np.arange(1,6)*.25;figs=[]
 def stat(family,variant,h,subset='all'):
  return next(r for r in S['rollout'] if (r['split'],r['family'],r['variant'],r['horizon'],r['subset'])==('development',family,variant,h,subset))
 def response(h,subset='all'):return next(r for r in S['response'] if (r['split'],r['horizon'],r['subset'])==('development',h,subset))
 def plot(ax,family,variant,key,label,subset='all',color=None):
  stats=[stat(family,variant,h,subset)['metrics'][key] for h in range(1,6)];v=[s['mean'] for s in stats];ci=np.array([s['ci95'] if s['mean'] is not None else [np.nan,np.nan] for s in stats]);line=ax.plot(t,v,'o-',label=label,color=color)[0];ax.fill_between(t,ci[:,0],ci[:,1],alpha=.12,color=line.get_color());ax.set_xlabel('Prediction horizon (s)')
 def finish(fig,name):fig.savefig(out/name,dpi=170,bbox_inches='tight');plt.close(fig);figs.append(name)
 rd=S['render']['development'];fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
 for ax,keys,labels,ylabel,title in [(axs[0],['raw_pixel_mae','matched_pixel_mae'],['Raw render','Background matched'],'Mean absolute RGB difference (0..255)','Pixel match to original images'),(axs[1],['raw_velocity_difference','matched_velocity_difference'],['Raw render','Background matched'],'Velocity output difference to original','Same state / actions, different images')]:
  vals=[rd[k]['mean'] for k in keys];err=np.array([[rd[k]['mean']-rd[k]['ci95'][0],rd[k]['ci95'][1]-rd[k]['mean']] for k in keys]).T;ax.bar(range(2),vals,yerr=err,capsize=5,color=['#c18843','#2879b4']);ax.set(xticks=range(2),xticklabels=labels,ylabel=ylabel,title=title)
 finish(fig,'rendering_controls.png')
 audit=[]
 for h in range(1,6):
  rr=[r for r in m if r['family']=='recorded' and r['split']=='development' and r['horizon']==h]
  audit.append(dict(horizon=h,rows=len(rr),episodes=len({r['episode'] for r in rr}),oracle_prefix_free=sum(r['oracle_prefix_free'] for r in rr),local_free=sum(r['local_free'] for r in rr),model_penetration_gt001=sum(r['endpoint_penetration']>.01 for r in rr),oracle_penetration_gt001=int(sum(clearance(r['oracle_state'][:2],params)<-.01 for r in rr)),model_max_penetration=max(r['endpoint_penetration'] for r in rr),local_energy_failures=sum(r['energy_necessary_failure']==1 for r in rr)))
 fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
 plot(axs[0,0],'recorded','original','velocity_error','All branches');plot(axs[0,0],'recorded','original','velocity_error','Fixed H5-free cohort','fixed_h5_free');axs[0,0].set(ylabel='Velocity L2 error',title='Original-image open-loop error');axs[0,0].legend(fontsize=8)
 plot(axs[0,1],'recorded','original','position_error','All branches');plot(axs[0,1],'recorded','original','position_error','Fixed H5-free cohort','fixed_h5_free');axs[0,1].set(ylabel='Position L2 error',title='Position error');axs[0,1].legend(fontsize=8)
 plot(axs[1,0],'recorded','original','local_velocity_error','All model starting states');plot(axs[1,0],'recorded','original','local_velocity_error','Local free substitute oracle','local_free');axs[1,0].set(ylabel='One-step velocity mismatch',title='From model state: WM vs simulator');axs[1,0].legend(fontsize=8)
 axs[1,1].plot(t,[100*(1-r['oracle_prefix_free']/r['rows']) for r in audit],'o-',label='Oracle left free-space regime');axs[1,1].plot(t,[100*r['model_penetration_gt001']/r['rows'] for r in audit],'o-',label='Model wall overlap >0.01');axs[1,1].plot(t,[100*r['oracle_penetration_gt001']/r['rows'] for r in audit],'--',label='Oracle wall overlap >0.01');axs[1,1].set(xlabel='Prediction horizon (s)',ylabel='Branches (%)',ylim=(-2,102),title='Contact regime and geometry failures');axs[1,1].legend(fontsize=8)
 finish(fig,'rollout_errors.png')
 fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
 for ax,key,title in zip(axs,['predicted_speed','position_error','kinetic_energy'],['Drift speed','Position drift','Kinetic energy']):
  for variant,label in [('raw','Raw renderer'),('matched','Background matched')]:plot(ax,'static',variant,key,label)
  ax.axhline(0,c='black',ls='--',label='Stationary oracle');ax.set_ylabel(title);ax.set_title(title);ax.legend(fontsize=8)
 finish(fig,'stationary_rollout.png')
 fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
 for key,label in [('response_error','Correct action'),('fixed_zero_error','Fixed zero output'),('sign_flip_error','Opposite action output'),('wrong_branch_error','90-degree action output')]:axs[0].plot(t,[response(h)['metrics'][key]['mean'] for h in range(1,6)],'o-',label=label)
 axs[0].set(xlabel='Prediction horizon (s)',ylabel='Action-induced velocity response error',title='All branches (contact regimes included)');axs[0].legend(fontsize=8)
 axs[1].plot(t,[response(h)['metrics']['angle_degrees']['mean'] for h in range(1,6)],'o-',c='#2879b4');axs[1].set(xlabel='Prediction horizon (s)',ylabel='Response angle error (degrees)',title='Angle after subtracting zero-input output')
 finish(fig,'action_response_horizons.png')
 fig,axs=plt.subplots(1,2,figsize=(10,5),layout='constrained')
 examples=[next(c for c in cases if c['split']=='development' and c['family']=='recorded' and c['condition']=='r0.2_d0'),next(c for c in cases if c['split']=='development' and c['family']=='static')]
 from matplotlib.patches import Rectangle
 for ax,c in zip(axs,examples):
  for w in params['walls']:ax.add_patch(Rectangle(np.array(w['center'])-w['halfsize'],2*w['halfsize'][0],2*w['halfsize'][1],fc='#dde1e7'))
  tr=np.concatenate([np.array(s['trace']) for s in oracles[c['id']]['segments']]);ax.plot(tr[:,0],tr[:,1],c='#c45252',label='Simulator substeps')
  for variant in (['original'] if c['family']=='recorded' else ['raw','matched']):
   rr=sorted([r for r in m if r['id']==c['id'] and r['variant']==variant],key=lambda r:r['horizon']);xy=np.array([c['start']]+[r['state'] for r in rr]);ax.plot(xy[:,0],xy[:,1],'o--',label='Model macro endpoints: '+variant)
  ax.set(xlim=(.3,3.5),ylim=(.3,3.5),aspect='equal',xlabel='x',ylabel='y',title=f"{c['family']} | site {c['episode']}");ax.legend(fontsize=7)
 fig.suptitle('Dashed segments connect macro endpoints; they are not observed model paths');finish(fig,'trajectory_examples.png')
 # Reproducibility and interpretation are kept separate from scientific performance.
 c=read(out/'checks.json');inf=read(out/'infer_complete.json');prep=read(out/'prepare_complete.json')
 one=stat('recorded','original',1);five=stat('recorded','original',5);fixed1=stat('recorded','original',1,'fixed_h5_free');fixed5=stat('recorded','original',5,'fixed_h5_free');st1=stat('static','raw',1);st5=stat('static','raw',5);mt5=stat('static','matched',5);rp=response(5);paired=next(r for r in S['static_paired'] if r['split']=='development' and r['horizon']==5)
 val=lambda r,k:r['metrics'][k]['mean']
 def fmt(s):return '표본 없음' if s['mean'] is None else f"{s['mean']:.5f} [{s['ci95'][0]:.5f}, {s['ci95'][1]:.5f}]"
 table='\n'.join(f"| {h*.25:.2f}초 | {val(stat('recorded','original',h),'position_error'):.5f} | {val(stat('recorded','original',h),'velocity_error'):.5f} | {audit[h-1]['oracle_prefix_free']}/{audit[h-1]['rows']} | {audit[h-1]['model_penetration_gt001']}/{audit[h-1]['rows']} |" for h in range(1,6))
 text=f'''# DINO-WM 물리적 일관성의 범위와 한계

동결 모델은 짧은 시간의 행동 방향·크기에 반응하지만, 정지 유지와 장기 rollout의 물리적 일관성은 충분하지 않다. 원영상에서도 음의 y속도 편향이 관측되므로 새 렌더링만으로 이전 오차를 설명할 수 없다. 이번 검사는 개발 데이터의 제한된 PointMaze 조건이며, 범용 물리 이해나 미관측 일반화를 검증한 결과가 아니다.

## 이번에 추가한 검사

렌더 대조는 기존 C28개 context(개발12개)에서 현재 상태·과거action·미래zero action을 고정했다. 원영상 출력은 재사용했고 raw 재렌더의 기존 calibration2개도 재사용했다. calibration2개×3frames에서 얻은 고정 배경 RGB차이만 적용한 보조 조건을 추가했다. 녹색 물체 영역과3pixel 주변은 지도에서 제외했다. 개발 출력으로 보정 지도를 조정하지 않았다.

원영상 다단계 검사는 calibration10개·개발10개에서 zero와 크기0.2의8방향을1.25초 유지했다. 정지 검사는 기존 무입력 실험의 calibration10개·개발10개에서 raw/배경일치 renderer를 비교했다. H1 예측 {inf['reused_h1_predictions']}개와 oracle {prep['reused_first_step_oracles']}개를 재사용했고, 이후에는 예측된 full visual/proprio token으로 과거3frames window를 굴렸다. 미래 정답 상태나 영상을 주입하지 않았다. 새 모델/probe학습·physics loss·sealed test 개방은 없다.

## 렌더링만으로 편향을 설명할 수 있는가

개발12개에서 원영상 출력의 oracle 속도오차는 {fmt(rd['original_velocity_error'])}, y속도 signed bias는 {fmt(rd['original_vy_bias'])}다. 기존 원영상 자체에서도 음의 편향이 나타난다.

| 영상 조건 | 원영상 대비 RGB MAE (0..255) | 원영상 출력 대비 속도 차이 | oracle 대비 속도오차 |
|---|---:|---:|---:|
| raw 재렌더 | {rd['raw_pixel_mae']['mean']:.5f} | {rd['raw_velocity_difference']['mean']:.5f} | {rd['raw_velocity_error']['mean']:.5f} |
| 고정 배경 일치 | {rd['matched_pixel_mae']['mean']:.5f} | {rd['matched_velocity_difference']['mean']:.5f} | {rd['matched_velocity_error']['mean']:.5f} |

픽셀 차이는 약 {100*(1-rd['matched_pixel_mae']['mean']/rd['raw_pixel_mae']['mean']):.1f}% 줄었지만 속도 출력 차이의 paired 감소는 {fmt(rd['paired_matching_reduction'])}로0을 포함한다. 이 배경 보정으로 출력 차이가 해소됐다는 증거는 없다. 물체 렌더 차이·시각 특징 민감도 등 남은 요인은 구분하지 못했다. 고정 배경 지도는 원인을 조사하는 대조이며 모델 개선으로 보고하지 않는다.

![렌더 대조](rendering_controls.png)

## 원영상에서 장기 예측하면 무엇이 달라지는가

개발10episodes×9행동의90branch를 모두 유지했다. 표의 오차는 episode 내 행동평균 후 episode 동일가중 평균이다. 자유공간 잔존은 시작부터 해당 horizon까지 실제 oracle에 접촉/clip이 없고 margin>0.004인 branch다. 예측 오차로 사례를 제외하지 않았다.

| horizon | 위치오차 | 속도오차 | 계속 자유공간인 oracle | 모델 벽겹침 깊이>0.01 |
|---|---:|---:|---:|---:|
{table}

전체 속도오차는0.25초 {val(one,'velocity_error'):.5f}에서1.25초 {val(five,'velocity_error'):.5f}로 증가했다. H5까지 계속 자유공간인 고정 cohort는 {fixed5['rows']}branches/{fixed5['episodes']}episodes뿐이다. 이 같은 cohort의 H1/H5 속도오차도 {val(fixed1,'velocity_error'):.5f} → {val(fixed5,'velocity_error'):.5f}였다. 소표본이며 전체 조건을 대표하지 않는다.

H5 모델 벽겹침 깊이>0.01은 {audit[-1]['model_penetration_gt001']}/{audit[-1]['rows']}개였고 같은 시점 oracle은 {audit[-1]['oracle_penetration_gt001']}/{audit[-1]['rows']}개였다. 이0.01 기준은 기존 시작상태 진단과 같은 기하 보고기준이며 새 학습 임계값이 아니다. 단순한 수치 접촉 penetration과 구분하려고 원래의 >0 결과도 metrics.json에 보존했다.

이전 모델 상태에서 실제 simulator를 한step 실행한 결과와 다음 모델 상태를 비교한 local 속도 불일치는 H1 {val(one,'local_velocity_error'):.5f}, H5 {val(five,'local_velocity_error'):.5f}다. 이는 초기 GT에서부터 누적된 전체오차와 다른 지표다. local substitute oracle의 자유공간/접촉 조건은 별도로 기록했다.

![오차 누적과 기하 실패](rollout_errors.png)

## 행동 반응은 얼마나 유지되는가

무입력 출력을 뺀 행동 반응 오차는 H1 {response(1)['metrics']['response_error']['mean']:.5f}, H5 {rp['metrics']['response_error']['mean']:.5f}다. 방향오차는 {response(1)['metrics']['angle_degrees']['mean']:.2f}° → {rp['metrics']['angle_degrees']['mean']:.2f}°다. H5의 입력고정/부호반전/90도 다른branch 대조 오차는 각각 {rp['metrics']['fixed_zero_error']['mean']:.5f}/{rp['metrics']['sign_flip_error']['mean']:.5f}/{rp['metrics']['wrong_branch_error']['mean']:.5f}였다. 모델은 긴 구간에서도 행동 정보를 사용하지만 반응의 정확도는 떨어졌다.

H5까지 행동branch와 zero branch가 모두 자유공간인 쌍은 {response(5,'pair_prefix_free')['metrics']['response_error']['rows']}개다. 따라서 H5 전체 결과를 자유공간 선형 물리식의 정확도라고 해석하지 않는다. 충돌을 포함한 행동 결과 예측 성능이다.

![시간에 따른 행동 반응](action_response_horizons.png)

## 정지 상태를 유지하는가

실제 환경은 무입력 정지상태를 계속 유지한다. 개발10개의 raw renderer 모델 예측 속력은 H1 {val(st1,'predicted_speed'):.5f}에서 H5 {val(st5,'predicted_speed'):.5f}로 증가했고, 위치 drift는 H5 {val(st5,'position_error'):.5f}였다. 배경 일치 renderer에서도 H5 속력 {val(mt5,'predicted_speed'):.5f}, 위치 drift {val(mt5,'position_error'):.5f}가 남았다.

H5 raw→배경일치의 paired 속력 감소는 {fmt(paired['metrics']['speed_reduction'])}다. 이 대조에서는 정지 drift가 해소되지 않았다. 같은 물리 정지상태에 대한 원래 데이터 영상은 없으므로, 합성영상의 모든 영향을 제거했다고 주장하지 않는다.

에너지 조건은 모델 출발점의 local substitute oracle이 자유공간인 경우에만 ΔK−W>1e-8을 표시했다. 이 조건은 모델 내부 경로까지 무접촉·기타 에너지원 없음이라고 가정한 필요조건이다. 모델의 정확한 마찰 적분을 관측한 것은 아니다. 평균값의 분모는 조건별로 달라 summary의 rows/episodes와 함께 보아야 한다.

![정지 상태의 drift와 에너지](stationary_rollout.png)

![사전 순서로 선택한 예시 궤적](trajectory_examples.png)

## 물리 이해에 대해 말할 수 있는 수준

| 항목 | 이번까지의 근거 | 판정 범위 |
|---|---|---|
| 행동 방향·크기의 활용 | 동일 context 행동 개입과 음성대조에서 차이 | 짧은 구간의 행동 반응을 학습했다는 근거가 있음 |
| 감쇠·관성 관계 | 무입력 초기속도 반응의 평균 계수는 환경 근처이나 정지 편향이 남음 | 일부 관계를 근사하지만 정확한 물리 파라미터 식별은 입증 안 됨 |
| 정지·에너지 일관성 | 무입력 정지 drift와 조건부 에너지 초과 | 검사한 조건에서 일관성이 충분하지 않음 |
| 장기 전이·벽 제약 | 누적오차와 기하적 벽겹침, 접촉 이후 반응 악화 | 신뢰할 수 있는 장기 물리 시뮬레이터로 보기 어려움 |
| 미관측 환경 일반화 | 이번에 새로운 WM test를 평가하지 않음 | 판단 불가 |

종합하면 **지역적 행동–전이 관계는 포착했지만, 상태를 안정적으로 유지하고 여러 단계에 걸쳐 제약을 지키는 능력은 불완전하다.** 관측된 기능을 모델 내부의 명시적 물리법칙 표현이나 범용 물리 이해의 증명으로 확대하지 않는다. 후속 연구에서는 별도 보류표본과 시각/상태 출력 경로의 원인 분리를 우선 검토할 수 있다.

## 검증과 재현

H1 캐시 역변환 차이 {c['h1_cache_max_abs']:.3e}, calibration2개 원래 wm.rollout API와 새 continuation 비교 차이 {c['public_rollout_max_abs']:.3e}, local 자유공간 fine 에너지 잔차 최대 {c['free_local_energy_max_abs']:.3e}로 통과했다. 원 API 비교10step 재추론은 구현 검증용으로 별도 기록했다. 입력과 가중치 SHA 불변, 유한 출력, no future observation, 새 학습 없음 검사를 통과했다. bootstrap10000/seed0은 episode 단위의 탐색적95% CI이며 다중비교 보정 없는 소규모 개발 실험이다.

원자료는 render_metrics.json, metrics.csv, self_oracles.json, rollouts.json, summary.json에 있고 코드·실행·재사용 정보는 manifest/checks/infer_complete/provenance에 있다. 기존 원결과·데이터·checkpoint를 보존했고 thread0에는 게시하지 않았다. [조건별 다단계 뷰어](viewer.html).
'''
 (out/'report_ko.md').write_text(text);save(out/'interpretation_audit.json',dict(penetration_and_free_counts=audit,source_metrics_sha256=sha(out/'metrics.json')))
 payload=[]
 for r in read(out/'rollouts.json'):
  case=next(c for c in cases if c['id']==r['id']);payload.append(dict(**r,start=case['start'],segments=oracles[r['id']]['segments']))
 html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>DINO-WM 다단계 물리 검사</title><style>body{font:16px system-ui;background:#f3f5f9;color:#17273c;margin:28px}section{background:white;padding:24px;border-radius:16px;margin:16px 0}select{padding:8px;font-size:15px}svg{width:min(700px,100%)}p{line-height:1.6}</style><section><h1>다단계 예측과 실제 환경 비교</h1><p>빨강은 simulator의 실제 경로, 파랑 점은 모델의0.25초 간격 예측입니다. 파란 점선은 점을 연결한 것으로 모델의 실제 중간 경로가 아닙니다.</p><label>조건 <select id="case"></select></label> <label>시간 <select id="h"></select></label></section><section><svg id="plot" viewBox="0 0 600 600" role="img" aria-label="시뮬레이터 궤적과 모델 다단계 예측"></svg><p id="stats"></p></section><script>const rows=__DATA__,walls=__WALLS__,s=document.getElementById('case'),h=document.getElementById('h'),plot=document.getElementById('plot');rows.forEach((r,i)=>s.add(new Option((r.family==='static'?'정지':'원영상')+' · '+r.split+' · '+r.episode+' · '+r.condition+' · '+r.variant,i)));for(let n=1;n<=5;n++)h.add(new Option((n*.25).toFixed(2)+'초',n));const X=x=>30+x*140,Y=y=>570-y*140;function draw(){let r=rows[+s.value],n=+h.value,p=r.predictions[n-1].state,g=r.segments[n-1].trace.at(-1),path=r.segments.slice(0,n).flatMap(x=>x.trace),pts=[r.start,...r.predictions.slice(0,n).map(x=>x.state)],svg='';for(let w of walls)svg+=`<rect x="${X(w.center[0]-w.halfsize[0])}" y="${Y(w.center[1]+w.halfsize[1])}" width="${280*w.halfsize[0]}" height="${280*w.halfsize[1]}" fill="#dce0e6"/>`;svg+=`<polyline points="${path.map(x=>X(x[0])+','+Y(x[1])).join(' ')}" fill="none" stroke="#c45252" stroke-width="3"/><polyline points="${pts.map(x=>X(x[0])+','+Y(x[1])).join(' ')}" fill="none" stroke="#2879b4" stroke-width="2" stroke-dasharray="6 5"/>`;for(let x of pts)svg+=`<circle cx="${X(x[0])}" cy="${Y(x[1])}" r="4" fill="#2879b4"/>`;plot.innerHTML=svg;document.getElementById('stats').textContent='위치오차 '+Math.hypot(p[0]-g[0],p[1]-g[1]).toFixed(4)+' · 속도오차 '+Math.hypot(p[2]-g[2],p[3]-g[3]).toFixed(4)+' · 해당 환경 구간 자유공간 '+(r.segments[n-1].free?'예':'아니오')}s.onchange=h.onchange=draw;draw();</script></html>'''
 (out/'viewer.html').write_text(html.replace('__DATA__',json.dumps(payload)).replace('__WALLS__',json.dumps(params['walls'])))
 save(out/'report_manifest.json',dict(script_sha256=sha(__file__),summary_sha256=sha(out/'summary.json'),figures=figs,viewer='viewer.html',new_training=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);report(p.parse_args().out)
