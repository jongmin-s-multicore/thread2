#!/bin/bash
# [repro] 기존 root 소유 service 안에서만 사용. JSON을 source/eval하지 않는다.
set -euo pipefail
umask 077
[[ $# == 6 ]] || exit 64
trust=$1; plan=$2; plan_hash=$3; approval=$4; approval_hash=$5; unit=$6
# trust 파일: anchor, unit, bootstrap, Python, watchdog의 다섯 canonical 절대경로/값.
[[ -f "$trust" && ! -L "$trust" && $(stat -c %u "$trust") == 0 ]] || exit 65
mode=$(stat -c %a "$trust")
(( (8#$mode & 8#022) == 0 )) || exit 65
mapfile -t binding < "$trust"
[[ ${#binding[@]} == 5 && ${binding[1]} == "$unit" && ${binding[2]} == "$0" ]] || exit 65
anchor=${binding[0]}; python=${binding[3]}; watchdog=${binding[4]}
[[ "$anchor" == /* && -d "$anchor" && ! -L "$anchor" ]] || exit 65
[[ ${#plan} -le 4096 && ${#approval} -le 4096 && ${#unit} -le 256 ]] || exit 65
[[ "$plan_hash" =~ ^[a-f0-9]{64}$ && "$approval_hash" =~ ^[a-f0-9]{64}$ ]] || exit 65
# [repro] 영구 소비/긴 해시 전에 실제 소속과 이미 적용된 외부 제한을 확인한다.
# 임의 shell을 sandbox하는 기능이 아니다. 직접 잘못 부른 경우 marker를 만들지 않는다.
property() { timeout 2s systemctl show "$unit" -p "$1" --value; }
[[ $(uname -s) == Linux ]] || exit 65
unit_group=$(property ControlGroup)
[[ "$unit_group" == /* && "$unit_group" != / && "$unit_group" != *..* ]] || exit 65
current_group=''
while IFS=: read -r hierarchy controllers group; do
  [[ "$hierarchy" == 0 && -z "$controllers" ]] && current_group=$group
done < /proc/self/cgroup
[[ "$current_group" == "$unit_group/supervisor" ]] || exit 65
cg="/sys/fs/cgroup$unit_group"
[[ $(<"$cg/memory.max") == 12884901888 && $(<"$cg/memory.swap.max") == 0 ]] || exit 65
[[ $(<"$cg/supervisor/memory.max") == 268435456 && $(<"$cg/worker/memory.max") == 12348030976 ]] || exit 65
read -r cpu_quota cpu_period < "$cg/cpu.max"
[[ "$cpu_quota" =~ ^[0-9]+$ && "$cpu_period" =~ ^[0-9]+$ ]] || exit 65
(( cpu_quota > 0 && cpu_period > 0 && cpu_quota <= 2*cpu_period )) || exit 65
allowed=''
while read -r key value rest; do [[ "$key" == Cpus_allowed_list: ]] && allowed=$value; done < /proc/self/status
IFS=, read -ra chunks <<< "$allowed"
count=0
for chunk in "${chunks[@]}"; do
  [[ "$chunk" =~ ^([0-9]+)(-([0-9]+))?$ ]] || exit 65
  first=${BASH_REMATCH[1]}; last=${BASH_REMATCH[3]:-$first}
  (( last >= first )) || exit 65
  count=$((count+last-first+1))
done
(( count == 2 )) || exit 65
fragment=$(property FragmentPath)
[[ -f "$fragment" && ! -L "$fragment" && $(stat -c %u "$fragment") == 0 ]] || exit 65
mode=$(stat -c %a "$fragment"); (( (8#$mode & 8#022) == 0 )) || exit 65
[[ $(property Type) == simple && $(property RuntimeMaxUSec) == 2min\ 59s ]] || exit 65
[[ $(property KillMode) == control-group && $(property SendSIGKILL) == yes && $(property Restart) == no ]] || exit 65
[[ -z $(property ExecStartPre) && -z $(property ExecStartPost) && -z $(property ExecStopPost) ]] || exit 65
stop=$(property TimeoutStopUSec)
if [[ "$stop" == 1s ]]; then stop_us=1000000
elif [[ "$stop" =~ ^([0-9]+)ms$ ]]; then stop_us=$((${BASH_REMATCH[1]}*1000))
else exit 65; fi
(( stop_us <= 1000000 )) || exit 65
t0_us=$(property ExecMainStartTimestampMonotonic); active_us=$(property ActiveEnterTimestampMonotonic)
[[ "$t0_us" =~ ^[0-9]+$ && "$active_us" =~ ^[0-9]+$ ]] || exit 65
(( t0_us > 0 && active_us > 0 && active_us+179000000+stop_us <= t0_us+180000000 )) || exit 65
scope="$anchor/natural-v2-integration-once-v1"
[[ -d "$scope" && ! -L "$scope" ]] || exit 65
mkdir "$scope/attempt-1" || exit 73
marker="$scope/attempt-1"
trap 'printf "%s\n" "UNKNOWN bootstrap failure; full reservation retained" > "$marker/bootstrap_failure"; sync -f "$marker"' ERR
printf '%s\n' "$unit" "$plan" "$plan_hash" "$approval" "$approval_hash" > "$marker/raw_intent"
printf '%s\n' "$t0_us" > "$marker/entry_clock_us"
sync -f "$marker/raw_intent"; sync -f "$marker/entry_clock_us"; sync -f "$scope"
# 큰 입력 해시는 watchdog의 t0+90 alarm이 적용된 경로에서만 읽는다.
exec "$python" -B "$watchdog" --trust "$trust" --execution-plan "$plan" --plan-sha256 "$plan_hash" --resource-approval "$approval" --approval-sha256 "$approval_hash" --unit "$unit"
