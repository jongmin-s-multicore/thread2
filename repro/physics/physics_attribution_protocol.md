# Physics loss 악화 원인 분리 실험

이전 파일럿 결과에 근거한 후속 개발 실험이다. 원인 분리는 이 데이터/seed/optimizer 범위의 개입 비교를 뜻하며 보편적 원인을 확정하지 않는다. sealed test를 열지 않고 기존 fit/tune/dev와 고정 oracle 마스크·encoded cache·모든 A–E 평가/학습 결과를 재사용한다. source 해시를 확인하고 기존 파일은 쓰지 않는다. 모델/encoder/readout 구성, seed0, AdamW lr1e-5, batch2(정지1+비정지1), dropout.1, clip1,120step, 최종 checkpoint를 유지한다. 표본/지표/제외 기준은 이전 그대로이며 추가 결과에 따른 선택/탐색/조기종료는 없다.

정의: N=sum(기존2step oracle prefix mask), L1=sum(mask1*normalized_state_MSE1)/N, L2=sum(mask2*normalized_state_MSE2)/N. H1만 쓸 때도 같은 N을 유지해 2단계항 제거 외의 재가중을 피한다. 이전 lambda=.14603252771223255, stationary=.1을 고정한다.

기존 A 동결, B base, C base+lambda*(L1_GT+L2_GT), D base+lambda*(L1_phys+L2_recursive), E D+정지항은 재사용한다. 새8조건:
- G1: base+lambda*L1_GT. P1: base+lambda*L1_phys. 1단계 동등성 대조.
- DS: D와 forward값 동일하되2단계 physical target의 첫 예측상태를 detach. recursive input 쪽 gradient는 유지. D→DS는 target 쪽 gradient만 제거한다.
- CL, DL: C/D의 lambda를 정확히0.1배. loss 형태를 유지한 규모 개입이며 최적 가중치 탐색이 아니다.
- CM, DM: 매 training batch에서 base와 보조항의 predictor gradient norm을 계산하고 lambda=.1*norm(base)/max(norm(aux),1e-12), 그 계수는 detach. 합성 전 보조 gradient norm을 같은10%로 맞춰 gradient 규모 차이를 통제한다. lambda>10/비유한이면 중단하고 기준을 바꾸지 않는다. 동일 sample/step, 추가 gradient 계산 비용은 별도 기록한다.
- CT: C에서2단계 incoming full observation token만 oracle 첫 미래 token으로 교체(동일action). 학습의2단계입력노출 효과를 분리한다. 평가에서는 모든 조건을 원래대로 autoregressive 실행하고 미래 GT를 주입하지 않는다.

같은조건간 첫 forward와 dropout RNG 소비 순서를 유지한다. G1/P1도 두번forward를 하며2단계항만0이다. source의 기존 loss_parts와 새 decomposed C/D/E의 loss/gradient를 같은 RNG로 비교한다. H1 GT/phys target오차<1e-5, gradient 상대차<1e-3를 학습 전 검사한다. 120step 뒤 G1/P1 max parameter차1e-4, 대표 평가오차 차이 A수치의1%를 동등성 진단기준으로 미리 고정한다. 후자의 미통과는 수치적 학습 민감성 증거이며 기준을 완화하지 않는다. 원본값/통과여부를 그대로 보고한다. DS forward=D, gradient 차이도 기록한다. frozen hash·정규화·oracle검사는 이전 유효 검사를 재사용하고 각 새checkpoint에서 frozen 불변을 확인한다.

초기 gradient 감사는 기존 fit batch schedule 첫12개를 사용한다. base, GT H1, GT H2, recursive H2, target-detached H2, 위치/속도 성분의 norm·base와 cosine·동일data 각 항끼리 cosine을 기록한다. 학습 중 CM/DM은 모든step의 계수와 실제 norm ratio, 모든조건은 train loss 성분/gradient norm을 기록한다. 비용 때문에 gradient 계산 결과를 CPU로 장기간 보관하지 않고 통계만 저장한다. 기하 마스크는 oracle만 쓰며 예측 실패로 바꾸지 않는다.

평가는 기존196 tune/dev case×5horizon 전체/현재free/prefixfree/contact, family별 episode평균, paired episode bootstrap10000 seed0를 유지한다. 기본비교 G1-B, P1-G1, C-G1, D-P1, DS-D, CL-C, DL-D, DM-CM, CM-C, DM-D, CT-C, E-D. 주 관심은 정지H5 drift/speed, recorded H1/H5 q/v, action H5 vector/angle, visual latent, prefixfree잔차와 실제오차다. 비교의 판정은 여러metric과paired CI를 함께 보며 단일seed·이미노출된dev·다중비교 한계를 명시한다. 학습 부족/과적합의 확정 구분, seed간재현성, 자연데이터분포, encoder 시각물리표현은 이번 설계의 식별범위 밖이다.

자원은 기존 파일럿 GPU90.81711977836676초를 포함해 누적3600초,장치전체11000MiB,RAM12GiB,CPU2,기존신규1.856GB+이번신규 합계10GiB 이하로 보수적으로 제한한다. 새 폴더의 pretrained/encoded는 원본의 read-only 경로참조로 재사용하고 새predictor만 저장한다. 초기gradient감사에서 실측시간을 기록하며 남은예산으로120step×8이 불가능하면 시작하지 않는다. 상한/비유한/코드검증실패 시 자기process group만 중지한다. 수정과실패로그는 보존하고 관련검사만 다시 실행한다. 에너지/벽loss, 여러seed, 데이터확장, thread0게시 없음.
