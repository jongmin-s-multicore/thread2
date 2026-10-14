# C 동일 관측 이력 행동 개입 프로토콜

2026-10-10 B 이후 단일 단계 pilot. 기존 WM·encoder·proprio 역변환은 고정한다. 새로운 학습·physics loss·D 다단계/보류평가를 하지 않는다. 기존 데이터·checkpoint·결과는 읽기 전용으로 유지하고 고유 결과 폴더를 생성한다.

## 분할과 선택

기존 inner fit80에서16, tune20에서4를 seed20261009의 동일 RNG로 순차 비복원 선택하고 fresh development20을 모두 후보로 둔다. 선택 실패 episode를 다른 episode로 대체하지 않는다. WM train 영역의 개발 실험이며 미관측 test가 아니다. calibration fit/tune과 development 집계를 구분한다.

episode당 macro2..14 순서에서 최초의 적합 anchor를 선택한다. context는 원 영상/state의 k−10,k−5,k이며 미래 H5까지 원 자료가 있는 시점만 허용한다. main33개 모든 action의 검증된 자유공간 식 경로가 벽에서0.004보다 멀고 raw경계 clip이 없는 상태를 먼저 고른다. 모델 출력은 선택에 사용하지 않는다. 이후 실제 MuJoCo substep에서 접촉·clip·margin을 검증한다. 부적합 branch도 결과에서 삭제하지 않는다. 보수적 선택으로 얻은 표본수와 제외 사유를 공개한다.

## 행동과 모델 입력

영입력1개와 r=0.05/0.10/0.20/0.40의8방향으로 main33조건, r=1.0의8방향을 별도 stress로 고정한다. 대각선도 norm=r이다. 각 control을 raw5회, 총0.25초 유지한다. 원 기록 action열은 별도 참고 조건이다.

원 영상·proprio context와 첫2개 과거 action block을 고정한다. 세 번째 block(raw k..k+4)만 변경한다. raw action으로 개입을 정의하고 모델 입력만 기존 full-dataset 평균/표준편차로 정규화한다. 실제 미래 관측은 입력하지 않는다. 기존 full visual cache는 없으므로 context별1회만 encode하고 branch별 action encoder/predictor를 사용한다. 원 H1 baseline은 calibration fit 첫2개 context만 정렬 검사 목적으로 재추론한다. 기존 시점의 기록 action 출력은 teacher-forced cache를 재사용한다. 새 anchor에 없는 baseline만 새 추론한다. 첫 patch proprio를 주 결과로 유지하고 열공간 잔차·patch 간 차이를 보조 저장한다. context/full prediction token cache를 보존해 후속 단계 중복 계산을 피한다.

## 측정과 대조

같은 context의 zero 출력을 뺀 δv(u)를 환경과 비교한다. 총속도 자체와 행동의 평행성을 강제하지 않는다. 반응 L2, 평행 이득, 수직 반응, 각도, 반대방향 대칭성, r=.05 대비 비례성, 축방향 중앙차분 Jacobian을 기록한다. 각도는 두 응답 norm이 각각1e-6 이상일 때만 계산하며 제외수를 공개한다(수치정의 기준, 성능 선택 아님).

주집계는 같은 context의 main33조건이 모두 자유공간인 공통표본에서 수행하며 전체·개별 자유공간·stress 층도 보존한다. 모델 예측 끝점의 벽 겹침·오차로 사례를 제외하지 않는다. branch/frame을 독립표본으로 세지 않고 episode 내 평균→episode 동일가중, bootstrap10000/seed0의95% percentile CI 및 paired 차이를 보고한다. calibration과 development를 합쳐 확증 결과로 제시하지 않는다.

대조군은 모델 출력을 모든 branch에서 zero에 고정, action 부호 반전 모델 출력, 동일 크기에서90도 회전한 다른 branch 출력이다. 환경 기준은 검증된 이산식이며 무입력에서는 v_next=0.9421300323v_start와 비교한다. 원 action 크기·방향·연속성 분포를 train100에서 요약하고 상수 control의 분포 차이를 명시한다. 같은 context의 행동 개입은 행동반응을 검사하지만 초기속도까지 독립적으로 바꾸지 않으므로 내부 마찰/질량의 인과 식별을 주장하지 않는다.

상수 control·oracle 자유공간에서만 끝점 일과 ΔK를 계산한다. 모델 경로의 무접촉/무clip은 관측하지 못하므로 endpoint 에너지 조건은 해당 조건을 가정한 필요조건 진단이다. 모델의 정확한 감쇠 적분을 주장하지 않는다. 모델·환경의 동일 선형속도 근사, 환경의 fine 수지·수치항·clip손실을 분리한다.

## 관문과 상한

A/B 완료·입력 SHA·코드 일치·split중복0·GT재시작≤1e-4·baseline 재현≤1e-4·TF=OL H1 cache 동치·action 왕복≤1e-6·관측/과거 action 불변·미래 관측 미사용·동결 가중치 hash·유한 출력 검사. 자유공간 analytic↔MuJoCo trace≤1e-10. 관문 실패 시 자료를 보존하고 원인을 기록한다. 원결과 덮어쓰기, 자동 표본확장, 새 설치/다운로드 없음.

순차 CPU 준비: CPU2/RAM8GiB/900초. GPU 추론: 지정GPU1/장치전체11000MiB/RAM12GiB/CPU2/1800초. 후처리: CPU2/RAM8GiB/600초. 전체 신규5GiB. GPU와 CPU simulator는 동시 실행하지 않는다. 감독기는 자기 process group만 종료한다. 서버 주소·GPU UUID 등 환경값은 실행 인자/환경변수로 전달하며 코드에 저장하지 않는다.

정밀도 관문 수정: 원 기록 action은 simulator의 원 저장 dtype을 유지한다. WM에는 기존 구현과 동일하게 float32로 변환한 후 정규화한다. 요청값과 모델 float32값을 별도 보존한다. 새 상수개입은 float32로 표현한 동일 raw값을 두 경로에 전달한다. 원 trace의 cache key를 모델 float32 action으로 재구성하지 않는다. 실패 시도는 보존하고 입력 hash가 같은 완료 partial oracle만 정확한 key로 재사용한다. 관문 임계값과 표본 선택 규칙은 바꾸지 않는다.
