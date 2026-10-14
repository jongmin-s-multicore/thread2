# 평가 전 입력값과 gradient 경로의 분리 보완

새 조건의 개발/조정 평가를 실행하기 전에 CT의 해석을 점검했다. CT-C는2단계 입력값을 예측에서정답으로 바꾸면서 첫예측을 통한 gradient 경로도 제거하므로 두요인을 혼동한다. 처음8조건의 설계/학습을 바꾸지 않고 CI 1조건을 추가한다. 새성능 결과를 근거로 선택한 조건이 아니다. 초기gradient 감사와 학습진행 로그만 확인했고 새 rollout평가는 아직하지 않았다.

CI는 C의2단계 incoming token 값을 그대로 유지하고 그 token만 detach한다. 물리 target을 바꾸지 않고 실제oracle state를 정답으로 쓴다. CI-C는 재귀 입력 gradient 경로의 효과, CT-CI는 같은 gradient 차단 상태에서 예측입력 대 oracle입력 값의 효과다. CT-C도 전체효과로 유지한다. CI의 forward loss=C를 동일 RNG 첫학습batch에서 검사하고 gradient차이를 기록한다. 검증용 forward는 RNG를 원래seed로 다시설정하며 optimizer업데이트를 하지 않는다. CI는 같은초기checkpoint·동결·batch순서·seed0·120step·optimizer·lambda·마스크를 유지한다.

새학습 총9조건이며 대규모 탐색이 아니다. 기존+현재실제GPU누적 및9조건 총량 모두3600초 상한,메모리/CPU/디스크상한은동일하다. 원protocol 파일을 보존하고 이별도 amendment hash를 evaluation lock에 추가한다. 이미완료된조건을 다시학습하지 않는다. 평가이후 추가조건을 만들거나 가중치를조정하지 않는다.
