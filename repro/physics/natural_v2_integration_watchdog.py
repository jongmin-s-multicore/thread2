"""[repro] 독립 종료 기록자. 기존 Linux service/cgroup/전용 파일시스템 없으면 거부한다."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import selectors
import shlex
import signal
import subprocess
import sys
import time

from natural_v2_integration_control import (
    CAPS, REQUIRED, SCOPE, RESERVATION, strict_read, pinned, require, validate_manifest,
    make_argv, verify_pins, reconcile_ledger, validate_checks, digest, canonical)
from natural_v2_guard import update_counts


def bounded_command(argv, deadline):
    # helper도 supervisor 메모리/CPU 제한 안에서 하나씩 실행, 출력 pipe 즉시 drain.
    require(time.monotonic()<deadline,'helper deadline')
    proc=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
    result=bytearray();limit=min(deadline,time.monotonic()+2)
    try:
        while proc.poll() is None or selector.get_map():
            require(time.monotonic()<limit,'metadata command timeout')
            for key,_ in selector.select(.01):
                chunk=os.read(key.fd,4096)
                if not chunk:selector.unregister(key.fileobj)
                result.extend(chunk);require(len(result)<=65536,'metadata command output')
        require(proc.returncode==0,'metadata command failure')
        return result.decode().strip()
    finally:
        if proc.poll() is None:proc.kill()
        proc.wait(timeout=max(.001,min(1,deadline-time.monotonic())))


def seconds_property(value):
    # 지원되는 systemctl 형식만 처리. 해석 불가능하면 기능 미확인으로 거부.
    if value.endswith('us'):return float(value[:-2])/1000000
    if value.endswith('ms'):return float(value[:-2])/1000
    if value.endswith('s') and ' ' not in value:return float(value[:-1])
    if value=='2min 59s':return 179.
    raise RuntimeError('unsupported systemd time format')


def deadlines(t0,now,model_start=None,ended=None):
    require(t0<=now<t0+180,'whole deadline')
    if model_start is None:return t0+90
    require(t0<=model_start<=t0+90,'late model start')
    gpu=min(model_start+60,t0+150)
    if ended is None:return gpu
    require(model_start<=ended<=gpu,'invalid model end')
    return min(ended+30,t0+180)


def inspect_envelope(plan,command=bounded_command):
    b=plan['bootstrap'];deadline=time.monotonic()+2
    properties=command(['systemctl','show',b['unit'],'--no-pager'],deadline)
    props=dict(line.split('=',1) for line in properties.splitlines() if '=' in line)
    require(props['KillMode']=='control-group' and props['SendSIGKILL']=='yes' and props['Restart']=='no','external kill contract')
    require(seconds_property(props['RuntimeMaxUSec'])==179 and seconds_property(props['TimeoutStopUSec'])<=1,'external deadline')
    require(props.get('ExecStopPost','')=='' and props.get('ExecStartPre','')=='' and props.get('ExecStartPost','')=='','unsupported service lifecycle')
    require(props['ControlGroup']==b['control_group'],'external cgroup binding')
    # systemctl의 runtime PID/start 필드는 고정 argv와 구분한다. 모호한 표현은 거부한다.
    text=props['ExecStart'].split('argv[]=',1)[1].split(' ;',1)[0]
    actual_argv=shlex.split(text)
    actual_argv=[('<PLAN_SHA256>' if x==digest(b['plan_path']) else
                  '<APPROVAL_SHA256>' if x==digest(b['approval_path']) else x) for x in actual_argv]
    require(actual_argv==b['exec_start'],'external argv mismatch')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip()==b['boot_id'],'boot mismatch')
    t0=int(props['ExecMainStartTimestampMonotonic'])/1000000
    require(b['start_clock']=='ExecMainStartTimestampMonotonic' and t0>0,'start clock unavailable')
    active=int(props['ActiveEnterTimestampMonotonic'])/1000000
    require(props['Type']=='simple' and active>0 and
            active+179+seconds_property(props['TimeoutStopUSec'])<=t0+180,
            'service activation/start gap exceeds whole cap')
    require(t0<=time.monotonic()<t0+90,'metadata window expired')
    fragment=Path(props['FragmentPath']);mode=fragment.stat()
    require(mode.st_uid==0 and not mode.st_mode&0o022 and not fragment.is_symlink(),'untrusted unit')
    group=Path('/sys/fs/cgroup'+b['control_group'])
    require(group.is_dir() and str(group.resolve())==str(group),'cgroup path')
    current=Path('/sys/fs/cgroup'+Path('/proc/self/cgroup').read_text().strip().split('0::',1)[1])
    require(current==group/'supervisor','bootstrap not in supervisor leaf')
    require((group/'memory.max').read_text().strip()==str(12*1024**3) and
            (group/'memory.swap.max').read_text().strip()=='0','parent RAM')
    quota,period=map(int,(group/'cpu.max').read_text().split())
    require(0<quota<=2*period and set(os.sched_getaffinity(0))==set(plan['host']['cpu_affinity']),'CPU cap')
    worker=group/'worker';supervisor=group/'supervisor'
    require((supervisor/'memory.max').read_text().strip()==str(256*1024**2) and
            (worker/'memory.max').read_text().strip()==str(11776*1024**2),'RAM partition')
    require((worker/'cgroup.kill').is_file() and not (worker/'cgroup.procs').read_text().strip(),'worker kill/empty capability')
    # 전용 영속 소형 파일시스템의 block/inode 상한을 kernel에서 읽는다. tmpfs는 영구 marker가 아니다.
    quota_path=Path(b['quota_path']);v=os.statvfs(quota_path)
    fs=command(['findmnt','-n','-o','FSTYPE,TARGET','--target',str(quota_path)],t0+90).split()
    require(len(fs)==2 and fs[0] in ('ext4','xfs') and fs[1]==str(quota_path),'persistent dedicated quota mount required')
    require(v.f_blocks*v.f_frsize<=64*1024**2 and 0<v.f_files<=4096,'kernel output/inode cap unavailable')
    require(Path(b['scope_anchor']).is_relative_to(quota_path) and Path(plan['out']).is_relative_to(quota_path),'all evidence must share quota')
    return dict(start_monotonic=t0,worker_cgroup=str(worker),parent_cgroup=str(group),properties=props)


def reserve_scope(anchor):
    # Testable counterpart to bootstrap's mkdir; both refuse even empty old directories.
    marker=Path(anchor)/SCOPE/'attempt-1';marker.mkdir()
    fd=os.open(marker.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return marker


def exclusive_json(path,value,limit):
    data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
    require(len(data)<=limit,'receipt partition exceeded')
    with Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    fd=os.open(Path(path).parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def terminate_worker(worker,proc,deadline):
    (worker/'cgroup.kill').write_text('1')
    if proc is not None:
        proc.wait(timeout=max(.001,min(2,deadline-time.monotonic())))
    while time.monotonic()<deadline:
        events=dict(line.split() for line in (worker/'cgroup.events').read_text().splitlines())
        if events.get('populated')=='0':return True
        time.sleep(.01)
    return False


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('trust','execution-plan','plan-sha256','resource-approval','approval-sha256','unit'):
        p.add_argument('--'+key,required=True)
    return p


def bootstrap_context(a):
    # [repro] 파싱 실패에도 shell marker는 남는다. 가능한 경우에만 최종 JSON을 보완한다.
    trust=Path(a.trust);st=trust.stat()
    require(st.st_uid==0 and not st.st_mode&0o022 and not trust.is_symlink(),'untrusted anchor config')
    lines=trust.read_text().splitlines();require(len(lines)==5 and lines[1]==a.unit,'trust binding')
    marker=Path(lines[0])/SCOPE/'attempt-1'
    require(marker.is_dir() and (marker/'raw_intent').read_text().splitlines()==
            [a.unit,a.execution_plan,a.plan_sha256,a.resource_approval,a.approval_sha256],'missing bootstrap intent')
    require(str(Path(sys.executable).resolve())==lines[3] and str(Path(__file__).resolve())==lines[4],'direct watchdog invocation')
    t0=int((marker/'entry_clock_us').read_text().strip())/1000000
    require(t0>0,'bootstrap clock missing')
    return lines,marker,t0


class Runtime:
    """[repro] main이 사용하는 좁은 clock/I/O/process seam. 운영 기본값은 실제 OS다."""
    def __init__(self):
        self.phase=None;self.deadline=None;self.selector=None
    now=staticmethod(time.monotonic)
    read=staticmethod(strict_read)
    pin=staticmethod(pinned)
    prelude=staticmethod(bootstrap_context)
    envelope=staticmethod(inspect_envelope)
    verify=staticmethod(verify_pins)
    ledger=staticmethod(reconcile_ledger)
    kill=staticmethod(terminate_worker)
    write_io=staticmethod(exclusive_json)
    def arm(self,phase,deadline):
        if phase==self.phase:deadline=min(deadline,self.deadline)
        self.phase,self.deadline=phase,deadline
        self.check()
        def expired(*_):raise TimeoutError('absolute '+phase+' deadline')
        signal.signal(signal.SIGALRM,expired)
        signal.setitimer(signal.ITIMER_REAL,deadline-self.now())
    def check(self):
        require(self.deadline is not None and self.now()<self.deadline,'phase deadline exhausted')
    def write(self,path,value,limit):
        self.check();self.write_io(path,value,limit);self.check()
    def launch(self,args,repo,worker):
        self.check()
        def join(): (worker/'cgroup.procs').write_text(str(os.getpid()))
        proc=subprocess.Popen(args,cwd=repo,preexec_fn=join,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        self.selector=selectors.DefaultSelector();self.selector.register(proc.stdout,selectors.EVENT_READ)
        return proc
    def running(self,proc):return proc.poll() is None or bool(self.selector.get_map())
    def drain(self):
        for key,_ in self.selector.select(.02):
            block=os.read(key.fd,4096)
            if not block:self.selector.unregister(key.fileobj)
            yield block


def main(argv=None,runtime=None):
    r=Runtime() if runtime is None else runtime
    a=parser().parse_args(argv)
    lines,marker,t0=r.prelude(a)
    proc=None;worker=None;envelope=None;out=None;result=None;failure=None;reaped=False;lock=None;phase_end=None
    try:
        r.arm('metadata',t0+90)
        plan=validate_manifest(r.read(r.pin(a.execution_plan,a.plan_sha256)),a.execution_plan)
        require(plan['bootstrap']['trust']==a.trust and plan['bootstrap']['unit']==a.unit and
                plan['bootstrap']['scope_anchor']==lines[0] and plan['bootstrap']['approval_path']==a.resource_approval,'trusted scope mismatch')
        envelope=r.envelope(plan);require(envelope['start_monotonic']==t0,'bootstrap/service clock mismatch')
        worker=Path(envelope['worker_cgroup']);r.check()
        r.verify(plan);r.check()
        approval=r.read(r.pin(a.resource_approval,a.approval_sha256));r.check()
        require(approval['status']=='APPROVED' and approval['issuer']=='user' and approval['fixture'] is False and
                approval['plan_sha256']==a.plan_sha256 and approval['uuid']==plan['uuid'],'separate production approval')
        lock=(Path(lines[0])/'budget.lock').open('a+b');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # global root binding comes from immutable trusted file, not a replaceable output UUID.
        ledger=r.ledger(plan);r.check()
        require(set(x.name for x in (Path(lines[0])/SCOPE).iterdir())=={'attempt-1'},'extra scope consumption')
        r.write(marker/'reservation.json',dict(scope_id=SCOPE,uuid=plan['uuid'],plan_sha256=a.plan_sha256,
            approval_sha256=a.approval_sha256,candidate=plan['candidate_code_commit'],out=plan['out'],job=plan['job'],
            unit=a.unit,boot_id=plan['bootstrap']['boot_id'],t0=t0,deadline=t0+180,caps=CAPS,ledger=ledger),1024**2)
        out=Path(plan['out']);out.mkdir();(out/'jobs').mkdir();Path(plan['job']).mkdir()
        r.write(out/'supervisor.json',dict(envelope,uuid=plan['uuid'],plan_sha256=a.plan_sha256,scope_marker=str(marker)),1024**2)
        args=make_argv(plan,'controller',a.execution_plan,a.plan_sha256)
        args=[a.approval_sha256 if x=='<APPROVAL_SHA256>' else x for x in args]
        r.check();proc=r.launch(args,plan['repo'],worker);log=bytearray()
        while r.running(proc):
            now=r.now();start=r.read(out/'model_start.json')['monotonic'] if (out/'model_start.json').exists() else None
            end=r.read(out/'model_end.json')['monotonic'] if (out/'model_end.json').exists() else None
            limit=deadlines(t0,now,start,end)
            if end is not None:
                phase_end=end if phase_end is None else min(phase_end,end)
                r.arm('final',min(phase_end+30,t0+180))
            elif start is not None:r.arm('gpu',limit)
            else:r.arm('metadata',t0+90)
            for block in r.drain():
                log.extend(block);require(len(log)<=65536,'watchdog log quota')
            r.check()
        require(proc.returncode==0,'controller lost/failed')
        if phase_end is None and (out/'model_end.json').exists():
            phase_end=r.read(out/'model_end.json')['monotonic']
            require(t0<=phase_end<=r.now(),'invalid terminal clock')
        phase_end=r.now() if phase_end is None else phase_end
        r.arm('final',min(phase_end+30,t0+180))
        result=r.read(out/'controller_result.json');r.check()
        require(result['uuid']==plan['uuid'] and result['status']=='READY_FOR_FINALIZATION','controller result identity')
        validate_checks(result['checks'],REQUIRED[:-1])
    except BaseException as ex:
        phase_end=r.now() if phase_end is None else phase_end
        failure=dict(type=type(ex).__name__,message=str(ex)[:1500])
    finally:
        # watchdog itself/OS/storage failure can prevent this receipt. Marker still consumes attempt.
        # 절대 final deadline은 한 번 정한 종료시각 기준이며 재진입해도 늘지 않는다.
        phase_end=r.now() if phase_end is None else phase_end
        deadline=min(phase_end+30,t0+180)
        try:
            r.arm('final',deadline)
            if out is not None and (out/'model_end.json').exists():
                end=r.read(out/'model_end.json')['monotonic'];r.check()
                require(t0<=end<=r.now(),'invalid terminal clock')
                phase_end=min(phase_end,end);deadline=min(phase_end+30,t0+180)
                r.arm('final',deadline)
        except BaseException:
            if lock is not None:lock.close()
            return 1  # 이미 기한 소진: 저장 시간을 새로 주지 않음, marker는 UNKNOWN
        if worker is not None:
            try:reaped=r.kill(worker,proc,deadline-1);r.check()
            except BaseException as ex:failure=dict(type='UNKNOWN',message='worker reap: '+str(ex)[:1000])
        if not reaped:failure=failure or dict(type='UNKNOWN',message='worker termination not established')
        checks=[] if result is None else result['checks']
        if failure is None:
            try:
                require(update_counts(plan['job'])==dict(attempted=2,completed=2,uncertain=0,legacy_reserved=0),'final update counts')
                r.verify(plan);r.check()
                events=dict(line.split() for line in (worker/'memory.events').read_text().splitlines())
                require(int(events['oom'])==0 and int(events['oom_kill'])==0,'worker OOM')
                r.check()
                checks.append(dict(id='I09_completion',expected='one attempt; bounded worker reaped; exact updates',
                    actual=dict(reaped=True,updates=update_counts(plan['job'])),tolerance=0,
                    evidence=[str(marker/'reservation.json'),str(out/'supervisor.json')],passed=True))
                validate_checks(checks)
            except BaseException as ex:failure=dict(type='UNKNOWN',message=str(ex)[:1000])
        receipt=dict(status='COMPLETE' if failure is None else 'FAILED_OR_UNKNOWN',failure=failure,
            checks=checks,scope_id=SCOPE,plan_sha256=a.plan_sha256,charged=RESERVATION,
            reaped=reaped,job_id=None if out is None else plan['uuid'],
            host_id=None if envelope is None else plan['host']['actual_context']['hostname_sha256'],
            invocation_id=None if out is None else plan['uuid'],remaining_attempts=0,monotonic=r.now(),
            whole_wall_seconds=r.now()-t0,
            controller_resources=None if result is None else result['resources'],
            actual_cgroup_cpu=None if worker is None else (worker/'cpu.stat').read_text(),
            actual_cgroup_memory=None if worker is None else (worker/'memory.peak').read_text())
        try:
            # One final writer, separate 4MiB headroom; no success claim if fsync fails.
            target=(out/'completion.json') if out is not None else (marker/'completion.json')
            r.check();payload=json.dumps(receipt,ensure_ascii=False,allow_nan=False).encode();r.check()
            import hashlib
            r.write(marker/'settlement.json',dict(receipt_path=str(target),sha256=hashlib.sha256(payload).hexdigest(),
                charged=RESERVATION,commit_rule='receipt hash plus successful service exit required'),1024**2)
            r.write(target,receipt,3*1024**2);r.check()
        except BaseException:
            return 1  # durable marker/reservation remains UNKNOWN, never retry
        finally:
            if lock is not None:lock.close()
    return 0 if failure is None else 1


if __name__=='__main__':
    sys.exit(main())
