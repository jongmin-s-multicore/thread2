# 첫 predictor physics loss 파일럿 (실행 전 고정)

목적: 정지/저속 편향과 재귀 drift를 줄이되 행동 반응, 상태 및 DINO latent 예측을 보존하는지 비교한다. 단일 seed=0, 개발 파일럿이다. 기존 결과를 많이 본 설계이며 sealed test는 열지 않는다. 기존 WM train 영역의 calibration_fit / calibration_tune / development 분할을 유지한다. 개발 episode는 이번 추가 학습에서만 보류됐고 원 WM 미관측 자료가 아니다. 기존 depth의 C fit/dev 각 첫10 context 및 Z fit/dev 각 첫10 위치, C/Z tune 전체4 위치를 사용한다. C는 zero+0.2의8방향, Z는 coast_zero 및 속도0.1의±x/±y. 모든 branch를 유지한다. 정확한 episode·교집합은 준비 단계 split.json에 저장한다.

A 동결. B 기존 latent MSE. C B+상태 지도. D B+전이. E D+정지 강조. B–E 모두 같은 입력, oracle, 정지 및 개입 데이터와 batch 순서에 접근한다. 같은 초기 predictor, AdamW(lr=1e-5, betas=.9/.999, weight_decay=.01), gradient norm clip1, batch=2(순환 정지1+비정지1), seed0. DINO/proprio/action encoder 및 선형 상태 역변환은 고정. predictor dropout은 학습 때0.1로 원 설정 유지, 평가 때off. Decoder는 predictor로 역전파하지 않는 기존 경로이므로 로드/학습하지 않는다. 미니배치별 기존 latent loss는 원 forward와 같은 3개의 shifted target MSE(action 제외). 모든 조건은 같은 2회 predictor forward를 실행한다. B의 두번째 forward는 gradient에 기여하지 않는다.

시간: raw action 5개×각5 substep×0.01=0.25초. action 입력은 기존 평균/표준편차로 정규화 후 시간순 (u0x,u0y,...,u4y)이다. 물리식은 원 단위 control에 적용한다. v'=Ah v+Bh u, q'=q+dt v'를25회 적용한다. 위치/속도는 simulator 단위이고 loss는 기존 state_std로 나눈4성분 MSE(위치2/속도2 동일 성분 가중)다. 학습 control은0 또는 크기0.2, 속도clip 없는 oracle만 물리항 대상이다.

고정 마스크: oracle fine trace의 접촉0·clip0·margin>.004인 prefix만 해당 horizon의 보조 loss에 사용한다. 모델 출력/오차로 마스크를 바꾸지 않는다. 모델이 벽으로 들어가도 제외하지 않는다. 2단계 D는 GT 자유공간 branch에서 affine 자유공간 전이를 재귀 예측 상태에 연장한 일관성 regularizer다. 예측 상태가 허용 영역 밖이면 실제 접촉 물리식이라고 주장하지 않으며, 전체 정확도와 실패율을 함께 평가한다. 실제 접촉 oracle 구간에는 이 식을 적용하지 않는다. 에너지/벽 loss 없음.

C: H1/H2 재귀 예측과 각각 실제 oracle state MSE의 평균. D: H1은 실제 출발 상태의 물리 target, H2는 첫 예측 상태로부터 물리 target(양쪽 gradient 전달)의 MSE 평균. 동일 고정 prefix mask 사용. H1의 C/D target은 자유공간에서 수치적으로 동등하다. 차이는 H2 truth anchoring 대 self consistency다. E는 같은 정지 branch에서 H1/H2 위치=실제 시작위치, 속도=0을 추가 강조한다. 새 물리법칙이 아니라 전이식 정지해의 추가 가중이다.

가중치: 학습용 고정 최초 batch에서 base / D / stationary gradient norm을 측정한다. lambda=clip(0.1*||grad base||/||grad D||,0.1,100); C/D/E 동일 lambda, stationary는 별도 lambda_s=clip(0.1*||grad base||/||grad stationary||,0.1,100). 결과 성능으로 조정/탐색하지 않는다. tune는 검증 및 곡선 기록용이며 checkpoint 선택에 쓰지 않는다. 평가 전 최종 step checkpoint를 고정한다.

실행량: 최대120step/조건. 최초 검증+5step E 시간측정 후 그 가중치는 폐기한다. t=max(측정step시간), steps=min(120,floor((2700-사전GPU초)/(4*2*t))), 최소20 미달이면 학습 중단. 총 GPU job wall time 누적3600초(측정/encoding/평가 포함), 장치전체11000MiB, process tree RAM12GiB, CPU2개, 신규10GiB. 준비 CPU는900초, 분석900초. 매1초 감독하며 본인 process group만 종료한다. 타 작업 중단 없음. 학습 종료조건: 상한, 비유한 loss/gradient/output, gradient 불통/동결값 변경/검증 실패. 성능에 따른 조기종료나 조건별 학습량 변경 없음. 실패 후 기준 완화 금지; 관련 수정/검증 기록 보존.

검증: 실제 개발 worktree import 경로, 기존 SHA, oracle double 정밀식<1e-10, differentiable float 구현<1e-5, 상태 inverse 왕복<1e-5, 0가중치 loss/gradient 원계산 동등, gradient 유한·비영 및 predictor 변경, frozen hash 불변, 2step 역전파 유한, no future GT autoregressive input. baseline full tokens/상태는 유효 캐시 재사용; 새 low-speed H2..H5만 추론. 기존 C/depth 평가 자체를 반복하지 않는다. 새 latent target encoding과 학습 경로 검증은 별도 기록한다.

평가: 동일 dev case H1..H5 전체/현재oracle free/prefix free/contact를 분리. 위치/속도 L2, 정지 속력/위치 drift, signed bias, 자유공간 GT anchored 및 recurrent affine 잔차, 행동 δv 크기·방향·vector오차, geometry overlap>0 및>.01, DINO visual latent MSE, proprio全patch MSE와 첫patch固定inverse 상태오차 및 patch평균inverse 진단. 후자는 새 학습readout 아님. 미래 이미지 target은 oracle fine trace의 마지막 전substep 위치(0.01초 이미지 지연)를 raw renderer로 생성하므로 latent 평가의 renderer 한계를 명시한다. 기존 원영상→새미래renderer 전환은 모든 조건 동일.

모든 episode 내 branch 평균 후 episode 동일가중, paired delta와 episode bootstrap10000 seed0 CI. branch를 독립표본으로 세지 않는다. 방향: true δv norm<1e-8만 사전 정의불가, model norm<1e-8이면180도 실패 처리. 표본 사후제외 없음. 정지H5속력·drift와 recorded H1/H5위치·속도 및 행동/latent 지표를 함께 본다. 성공 기준: 정지H5속력/drift 모두A보다10% 이상 감소, recorded H1/H5 위치·속도/행동vector오차/visual latent MSE가A 대비5% 이내 비열등; paired CI로 불확실성도 제시한다. D-C, E-D, B-A로 효과를 구분하고 E가좋아도 정지 재가중 효과 이상 주장하지 않는다. 단일 seed/노출된 개발평가로 확정적 일반화 주장하지 않는다.
