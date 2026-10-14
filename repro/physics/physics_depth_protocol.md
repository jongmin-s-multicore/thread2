# 물리적 일관성의 범위와 렌더 민감도 검사

동결 모델을 대상으로 렌더 영향을 통제하고 5-step open-loop에서 행동 반응·상태 전이·접촉·에너지의 일관성을 분석한다. 재학습/probe 학습/physics loss/새 환경 설치/다운로드를 하지 않는다. A/B/C와 무입력 검사를 SHA 확인 후 재사용한다. 새 고유 폴더만 사용한다. sealed WM test와 추가 보류표본은 열지 않으며 개발 분석으로 한정한다.

## 렌더 대조

C의28개 context에서 q/v/과거action/미래zero action을 모두 고정하고 원영상과 원시점 재렌더 영상만 비교한다. 완료된 calibration2개 재렌더와 추론은 재사용한다. 나머지26개만 새로 생성한다. raw0 영상은 현재시점이고 raw>0 영상은 state보다0.01초 과거인 기존 규칙을 유지한다.

보조 배경 일치 대조: 기존 calibration2개×3frames의 original−rerender RGB 차이의 pixel별 median을 구한다. 어느 calibration frame에서든 녹색 물체(g>r+15,g>b+15)를 포함한 pixel 및3pixel 주변은 보정0으로 둔다. 이 고정 배경차이 지도를 모든28개 재렌더와 정지 context에 적용하고 [0,255]로 clip/반올림한다. 정규화/WM 가중치·물리 상태·행동은 바꾸지 않는다. 이것은 이미지 생성 차이의 진단용 보정이며 새 물리모델 학습이나 실용 성능 개선 주장이 아니다. dev 결과로 지도/마스크/규칙을 조정하지 않는다. 원영상·raw 재렌더·배경 일치 영상을 모두 보존하고 전체/물체영역 pixel 오차를 기록한다. 원영상 대조의 출력차이와 실제 oracle 오차를 구분한다.

## 자유 rollout

C calibration_fit 첫10개와 development 첫10개를 기존 순서 그대로 사용한다. zero 및 r=.2의8방향(9조건)을1.25초 유지한다. H1은 저장된 full prediction token과 oracle trace를 재사용한다. H2..H5는 예측한 visual/proprio full token을 다음 입력으로 쓰고 과거3frames window를 굴린다. 예측 token의 action 부분만 다음 고정 control로 교체한다. 미래 실제영상/state 주입 없음. H1 캐시 역변환을 기존 state와 비교하여1e-8 이하인지 확인한다.

무입력 실험의 calibration_fit 첫10개와 development 첫10개 정지 coast context도 재사용한다. raw renderer H1은 완료된 캐시를 사용하고 배경 일치 renderer H1은 새로 추론한다. 이후 H2..H5를 같은 방식으로 굴린다. 원영상 C와 합성정지 결과를 합쳐 평균내지 않는다. 환경 oracle은 동일 현재상태/action이면 렌더 variant 간 공유한다. 미래 정지 oracle은 계속 정지여야 한다.

기존 첫step 자유공간 선택을 유지하며 장기 미래 충돌 여부나 예측 오차로 표본을 추가 제외하지 않는다. 전체 / 각 step의 oracle 자유공간 / 시작부터 해당 horizon까지 자유공간을 분리하고 각 분모를 공개한다. H5까지 자유공간인 고정 branch cohort에서 H1↔H5 paired 차이도 보조 기록한다. 선택된 branch가 적으면 그대로 표시한다.

## 지표와 대조

각 horizon의 위치/속도 L2, signed velocity bias, 이전 모델 endpoint에서 출발한 동일 action simulator와의 한step 자기일치성, 예측 벽 겹침을 측정한다. 동일 action oracle의 완전일치기준과 속도유지/등속 기준을 함께 둔다. 접촉·clip·margin을 전substep 검사한다. 모델이 벽을 예측했다는 이유로 원래 자유공간인 사례를 제거하지 않는다.

모델의 local 출발상태→다음모델상태에서 상수힘의 W=(Gu)·Δq, ΔK, 조건부 ΔK−W>1e-8 빈도를 계산한다. local 대체oracle도 자유공간인 경우의 필요조건 진단이며 모델의 실제 중간 경로/마찰적분을 관측했다고 주장하지 않는다. 환경 fine 에너지 항등식은1e-10 내인지 검사한다. 부호반전/90도 다른 branch/zero 출력고정 대조의 행동반응 오차를 각 horizon에서 비교한다. 충돌 이후의 반응은 자유공간 선형 법칙으로 해석하지 않는다.

coast 정지조건은 속도/위치 drift와 에너지증가를 raw/배경일치 renderer에서 paired 비교한다. 이미지 보정이 결과에 영향이 있어도 원모델 성능으로 바꿔 보고하지 않는다. 배경지도는 calibrated images 차이만 사용하고 물리 출력은 지도 추정에 사용하지 않는다.

집계는 episode내 branch평균→episode동일가중. bootstrap10000 seed0의 탐색적95% CI, 다중검정 보정 없음. 정확도·인과적 입력 반응·일관성·장기 누적·미관측 일반화를 구분해 결론을 쓴다. 물리 이해를 임의의 단일 점수로 환산하지 않는다.

## 검증과 상한

동결 가중치·원입력 SHA, oracle/모델 H1 cache 재사용, 정규화 왕복, elapsed horizon, no future-state input, 전출력 유한성을 검사한다. MuJoCo parameter/source는 기존 버전과 일치해야 한다. raw 재렌더 calibration검사는 재사용하고 pixel 차이를 모든 context에 기록한다. 단계별 CPU2/RAM8GiB/900초 준비, GPU1/RAM12GiB/장치전체11000MiB/1800초 추론, CPU2/RAM8GiB/600초 분석, 신규5GiB. CPU simulation과GPU는 순차 실행한다. 감독기는 자기 job만 중단한다. 결과·실패·이전 STATUS/문서를 보존하고 thread0에 게시하지 않는다.

새 rollout window의 독립 검증을 위해 calibration2개 zero 조건만 저장소의 원래 `wm.rollout` API로 H1..H5를 비교한다. 검증용10step 재추론은 캐시 재사용량과 분리해 기록하며1e-5 관문을 쓴다. 개발 결과를 보면서 기준을 바꾸지 않는다.
