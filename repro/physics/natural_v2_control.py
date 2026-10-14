"""프로토콜/코드 잠금과 누적 자원 supervisor. 원본 결과는 읽기 전용이다."""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import re
import shutil
import subprocess
import sys
import time
# [repro] 운영 검사는 모델 의존성 없이 공유한다.
from natural_v2_guard import (read, save, sha, require, document_manifest, verify_documents,
    budget_for_run, update_counts, verify_execution_authorization, verify_constants, capture_context, bind_actual)




def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def source_hashes(repo):
    return {str(p.relative_to(repo)):sha(p) for p in sorted((repo/'repro/physics').glob('*.py'))}


def initialize(root,repo,out,registry_path):
    raw=registry_path.read_bytes();r=json.loads(raw)
    require(r['protocol']['id']=='DINO-WM-PHYSICS-NATURAL-2.0','wrong protocol')
    require(r['phase'] in ('PROTOCOL_FINALIZED','IMPLEMENTATION_REVIEWED'),'unrecognized registry phase')
    require(not out.exists(), 'output already exists')
    require(not git(repo,'diff','HEAD','--'), 'tracked code dirty')
    docs={}
    pinned_documents=document_manifest(r)
    for d in r['documents']:
        p=Path(d['server_path'])
        require(sha(p)==d['sha256'],'document mismatch '+str(p));docs[p.name]=p.read_bytes()
    out.mkdir(parents=True)
    with (out/'registry_snapshot.json').open('xb') as f:f.write(raw)
    for name,content in docs.items():
        with (out/name).open('xb') as f:f.write(content)
    hashes=source_hashes(repo)
    save(out/'source_hashes.json',hashes)
    lock=dict(run_id=out.name,started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        registry_revision=r['revision'],registry_sha256=hashlib.sha256(raw).hexdigest(),
        document_sha256=pinned_documents,
        protocol_id=r['protocol']['id'],protocol_version=r['protocol']['version'],
        protocol_sha256=r['protocol']['sha256'],protocol_snapshot_path=str(out/'physics_natural_protocol_v2_0.md'),
        code_commit=git(repo,'rev-parse','HEAD'),code_dirty_diff_sha256_or_clean='clean',
        approved_scope_snapshot=r['execution_scope'],input_manifest_sha256=sha(out/'physics_natural_evidence_v2_0.json'),
        source_hash_manifest_sha256=sha(out/'source_hashes.json'),output_path=str(out))
    save(out/'run_lock.json',lock)
    return lock


def verify_lock(repo,out):
    lock=read(out/'run_lock.json')
    verify_documents(out)  # [repro] 전체 복사 문서와 입력 manifest를 사용 전에 검사.
    require(git(repo,'rev-parse','HEAD')==lock['code_commit'],'code version changed')
    require(not git(repo,'diff','HEAD','--'),'tracked code dirty')
    require(sha(out/'registry_snapshot.json')==lock['registry_sha256'],'registry snapshot altered')
    require(sha(out/'physics_natural_protocol_v2_0.md')==lock['protocol_sha256'],'protocol altered')
    require(sha(out/'source_hashes.json')==lock['source_hash_manifest_sha256'],'source manifest altered')
    require(source_hashes(repo)==read(out/'source_hashes.json'),'source/import code changed')
    return lock


def approval(repo,out,record,stage):
    lock=verify_lock(repo,out)
    require(record is not None,'comparison requires reviewed approval file')
    a=read(record)
    require(a.get('status')=='APPROVED' and a.get('code_commit')==lock['code_commit'] and
            a.get('protocol_sha256')==lock['protocol_sha256'],'approval version mismatch')
    require(stage in a.get('approved_stages',[]),'stage not approved')
    r=read(out/'registry_snapshot.json')
    require(r['phase']=='IMPLEMENTATION_REVIEWED' and r['code_review']['reviewed_implementation_commit']==lock['code_commit'] and
            stage in r['code_review']['approved_comparison_stages'],'registry does not approve comparison')
    require(a.get('preflight_completion_sha256') and Path(a['preflight_completion_path']).is_file(),'missing preflight review evidence')
    require(sha(a['preflight_completion_path'])==a['preflight_completion_sha256'],'review evidence changed')
    require(read(a['preflight_completion_path'])['status']=='AWAITING_CODE_REVIEW_NO_COMPARISON_TRAINING','preflight incomplete')
    return a


def disk_bytes(paths):
    seen=set();total=0
    for parent in paths:
        if not parent.exists():continue
        for p in parent.rglob('*'):
            if not p.is_file() or p.is_symlink():continue
            st=p.stat();key=(st.st_dev,st.st_ino)
            if key not in seen:total+=st.st_size;seen.add(key)
    return total


def ledger(root,registry):
    jobs=[]
    for p in sorted((root/'work/physics').glob('pointmaze-physics-natural-v2-*/jobs/*/status.json')):
        j=read(p);j['path']=str(p);j['job_id']=str(p.parent)
        require(j['state']!='RUNNING','another v2 job running or interrupted ledger; reconcile before launch')
        jobs.append(j)
    return dict(known_prior_gpu_seconds=registry['execution_scope']['known_prior_gpu_seconds'],
        new_gpu_seconds=sum(j['elapsed_seconds'] for j in jobs if j['gpu']),
        cpu_job_seconds=sum(j['elapsed_seconds'] for j in jobs if not j['gpu']),
        optimizer_updates=sum(j.get('optimizer_updates',0) for j in jobs),jobs=jobs)


def supervise(root,repo,out,stage,preflight=True,approval_path=None,resource_approval=None,execution_plan=None):
    lock=verify_lock(repo,out);registry=read(out/'registry_snapshot.json')
    review=approval(repo,out,approval_path,stage) if not preflight else None
    require(resource_approval is not None and execution_plan is not None,'separate user approval and plan required')
    plan=read(execution_plan);user_approval=read(resource_approval)
    spec=plan['stages'][stage]
    require(plan['code_commit']==lock['code_commit'],'execution plan code mismatch')
    gpu=stage in ('gpu','extra','stage1','stage2-base','stage2-aux','final-eval','encode-tune','encode-eval')
    led=ledger(root,registry)
    for row in led['jobs']:
        row['comparison_stage']=Path(row['path']).parents[2]==out and row['stage'] in ('stage1','prepare-tune','encode-tune','stage2-base','stage2-aux','prepare-eval','encode-eval','final-eval')
    forecast=budget_for_run(out,review,led['jobs'],led['known_prior_gpu_seconds'],stage) if not preflight else None
    remaining=(120-led['new_gpu_seconds']) if gpu and preflight else (900-led['cpu_job_seconds']) if not gpu and preflight else 3600-led['known_prior_gpu_seconds']-led['new_gpu_seconds']
    require(remaining>0,'time budget exhausted')
    remaining=min(remaining,3600-led['known_prior_gpu_seconds']-led['new_gpu_seconds']) if gpu else remaining
    require(not preflight or led['optimizer_updates']<12,'update budget exhausted')
    old=[root/'work/physics'/n for n in ('pointmaze-physics-loss-pilot-20261011','pointmaze-physics-attribution-20261011')]
    new=list((root/'work/physics').glob('pointmaze-physics-natural-v2-*'))
    initial_new=disk_bytes(new);require(initial_new<1024**3 if preflight else True,'preflight disk exhausted')
    require(disk_bytes(old+new)<10*1024**3,'research disk exhausted')
    smi=os.environ.get('DINO_NVIDIA_SMI') or shutil.which('nvidia-smi')
    gpu_uuid=re.search(r'GPU-[0-9a-f-]{36}',(out/'physics_natural_protocol_v2_0.md').read_text()).group(0)
    # [repro] 계획의 job 경로를 사용해 실행 argv를 사전에 정확히 승인할 수 있다.
    require(re.fullmatch(r'[A-Za-z0-9_.-]+',spec['job_name']) is not None,'invalid planned job name')
    job=out/'jobs'/spec['job_name'];require(not job.exists(),'job already used; no implicit retry')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu_uuid if gpu else '',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',WANDB_MODE='disabled')
    python=root/'work/env/bin/python'
    if stage in ('cpu','prepare-tune','prepare-eval'):
        setup=root/'work/replay-setup-20261008';replay=root/'work/replay-env';python=replay/'bin/python'
        env.update(MUJOCO_PY_FORCE_CPU='1',MUJOCO_PY_MUJOCO_PATH=str(setup/'mujoco210'),LD_LIBRARY_PATH=f'{setup}/mujoco210/bin:{replay}/lib',CFLAGS='-Wno-error=incompatible-pointer-types -fno-strict-aliasing -include stdio.h',CPATH=str(replay/'include'),LIBRARY_PATH=f'{setup}/native-links:{replay}/lib',LDFLAGS=f'-L{setup}/native-links -L{replay}/lib',D4RL_DATASET_DIR=str(setup/'d4rl-cache'),TMPDIR=str(setup/'tmp'),XDG_CACHE_HOME=str(setup/'cache'))
    script=repo/'repro/physics'/('natural_v2_preflight.py' if preflight else 'natural_v2_comparison.py')
    bootstrap='import sys,runpy;root,repo,script,*args=sys.argv[1:];sys.path.append(root+"/work/env/lib/python3.10/site-packages");sys.path.insert(0,repo);sys.path.insert(0,repo+"/repro/physics");sys.argv=[script]+args;runpy.run_path(script,run_name="__main__")'
    cmd=[str(python),'-c',bootstrap,str(root),str(repo),str(script),'--root',str(root),'--repo',str(repo),'--out',str(out),'--stage',stage,'--job',str(job)]
    if preflight:cmd.append('--preflight-only')
    else:cmd+=['--approval',str(approval_path)]
    # [repro] 장치 조회 및 process launch보다 먼저 별도 사용자 승인을 검사한다.
    remaining=min(remaining,spec['limits']['wall_seconds'])
    if gpu:remaining=min(remaining,spec['limits']['gpu_seconds'])
    require(remaining>0,'approved time exhausted')
    ram_cap=min(12*1024**3,spec['limits']['ram_bytes'])
    disk_cap=min(10*1024**3,spec['limits']['disk_bytes'])
    device_cap=min(11000,spec['limits']['gpu_device_mib'])
    core_cap=min(2,spec['limits']['cpu_cores'])
    require(type(core_cap) is int and core_cap>0,'invalid CPU limit')
    request=dict(plan_sha256=sha(execution_plan),code_commit=lock['code_commit'],argv=cmd,
        data_scope=dict(stage=stage,preflight=preflight,protocol_sha256=lock['protocol_sha256'],
            split_sha256=lock['document_sha256']['physics_natural_split_v2_0.json']),
        limits=dict(gpu_seconds=remaining if gpu else 0,cpu_cores=core_cap,ram_bytes=ram_cap,
            wall_seconds=remaining,disk_bytes=disk_cap,gpu_device_mib=device_cap),max_executions=1,max_retries=0)
    require(spec['argv']==cmd and spec['data_scope']==request['data_scope'],'actual request differs from plan')
    actual=capture_context(root,repo,out,stage,job,[str(script)]+cmd[cmd.index('--root'):])
    bind_actual(spec['actual_context'],actual)
    require(user_approval.get('actual_context')==actual,'actual context not user approved')
    request['actual_context']=actual
    receipt=verify_execution_authorization(user_approval,request)
    require(disk_bytes(old+new)<disk_cap,'approved disk exhausted')
    if gpu:
        require(smi is not None,'nvidia-smi path required')
        info=subprocess.check_output([smi,'--id=0','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(',')
        require(info[0].strip()==gpu_uuid,'GPU UUID mismatch')
        require(int(info[1])+6500<=device_cap,'insufficient approved device headroom')
    job.mkdir(parents=True)
    save(job/'execution_authorization.json',dict(receipt,plan_path=str(execution_plan),
        approval_path=str(resource_approval),approval_sha256=sha(resource_approval)))
    save(job/'budget_at_launch.json',dict(stage=stage,remaining_seconds=remaining,ledger=led,
        remaining_forecast=forecast,old_and_new_bytes=disk_bytes(old+new)))
    cores=sorted(os.sched_getaffinity(0))[:core_cap]
    def child_setup():os.sched_setaffinity(0,cores);os.nice(10)
    base=dict(gpu=gpu,stage=stage,command=cmd,code_commit=lock['code_commit'],protocol_sha256=lock['protocol_sha256'],cpu_cores=cores)
    start=time.monotonic();reason=None;peak_rss=peak_gpu=0
    with (job/'stdout.log').open('x') as log,(job/'resources.jsonl').open('x') as samples:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=repo,env=env,start_new_session=True,preexec_fn=child_setup)
        save(job/'status.json',dict(base,state='RUNNING',pid=proc.pid))
        try:
            while proc.poll() is None:
                rss=0
                for d in Path('/proc').iterdir():
                    if not d.name.isdigit():continue
                    try:
                        if int((d/'stat').read_text().rsplit(')',1)[1].split()[2])==proc.pid:rss+=int((d/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                    except (OSError,ValueError,IndexError):pass
                memory=int(subprocess.check_output([smi,'--id=0','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=2).strip()) if gpu else 0
                elapsed=time.monotonic()-start;size=disk_bytes(new)
                peak_rss=max(peak_rss,rss);peak_gpu=max(peak_gpu,memory)
                samples.write(json.dumps(dict(seconds=elapsed,rss_bytes=rss,gpu_device_mib=memory,new_bytes=size))+'\n');samples.flush()
                if elapsed>=remaining:reason='TIME_LIMIT'
                elif rss>ram_cap:reason='RAM_LIMIT'
                elif memory>device_cap:reason='DEVICE_LIMIT'
                elif size>1024**3 and preflight:reason='PREFLIGHT_DISK_LIMIT'
                elif disk_bytes(old+new)>disk_cap:reason='RESEARCH_DISK_LIMIT'
                if reason:break
                time.sleep(min(1,max(.01,remaining-elapsed)))
        except BaseException as ex:reason=repr(ex)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    rc=proc.wait();updates=update_counts(job)
    status=dict(base,state='COMPLETE' if rc==0 and reason is None else 'FAILED',returncode=rc,reason=reason,
        elapsed_seconds=time.monotonic()-start,peak_rss_bytes=peak_rss,peak_gpu_device_mib=peak_gpu,
        optimizer_updates=updates['attempted'],update_accounting=updates,new_bytes=disk_bytes(new))
    (job/'status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2));return rc if rc else int(reason is not None)


def reuse_cpu(repo,out,previous):
    """완료 CPU 검증을 재실행하지 않고 검증된 함수/의존 소스가 같을 때만 재사용."""
    import ast
    import shutil
    old=read(previous/'run_lock.json')
    require(read(previous/'cpu_tests.json')['pass_all'],'previous CPU failed')
    for filename in ('natural_v2_data.py','natural_v2_selection.py','physics_loss_pilot.py','physics_depth.py','intervention.py','zero_velocity.py'):
        original=git(repo,'show',old['code_commit']+':repro/physics/'+filename)
        require(original==(repo/'repro/physics'/filename).read_text().strip(),'CPU dependency changed '+filename)
    for filename,names in {'natural_v2_preflight.py':('cpu','metadata','selection_tests','check','write','serial'),
                           'natural_v2_core.py':('transition','anchors','schedule','require')}.items():
        original=ast.parse(git(repo,'show',old['code_commit']+':repro/physics/'+filename))
        current=ast.parse((repo/'repro/physics'/filename).read_text())
        extract=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
        require(extract(original)==extract(current),'CPU checked functions changed')
    constants=verify_constants(git(repo,'show',old['code_commit']+':repro/physics/natural_v2_core.py'),
        (repo/'repro/physics/natural_v2_core.py').read_text(),('LAMBDA','CELLS','MIXTURES'))
    require(old['protocol_sha256']==read(out/'run_lock.json')['protocol_sha256'],'CPU protocol changed')
    files=['cpu_tests.json','selection_tests.json','data_manifest.json','reuse_audit.json','alignment_checks.json','preflight_natural_cases.json']
    files += [c['id']+'.npz' for c in read(previous/'preflight_natural_cases.json')]
    for name in files:
        require(not (out/name).exists(),'CPU output exists');shutil.copyfile(previous/name,out/name)
    save(out/'cpu_reuse_provenance.json',dict(source_run=str(previous),source_code_commit=old['code_commit'],
        run_lock_sha256=sha(previous/'run_lock.json'),artifacts={name:sha(previous/name) for name in files},
        checked_functions_and_dependencies_unchanged=True,core_schedule_constant_values_unchanged=constants['equal'],constant_evidence=constants))


def reuse_gpu(repo,out,previous):
    import ast
    import shutil
    old=read(previous/'run_lock.json')
    require(read(previous/'gpu_tests.json')['pass_all'],'previous GPU failed')
    require(old['protocol_sha256']==read(out/'run_lock.json')['protocol_sha256'],'GPU protocol changed')
    for rel,h in read(previous/'source_hashes.json').items():
        if Path(rel).name in ('natural_v2_preflight.py','natural_v2_control.py'):continue
        require(sha(repo/rel)==h,'GPU dependency changed '+rel)
    old_tree=ast.parse(git(repo,'show',old['code_commit']+':repro/physics/natural_v2_preflight.py'))
    new_tree=ast.parse((repo/'repro/physics/natural_v2_preflight.py').read_text())
    extract=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('gpu','unit')}
    require(extract(old_tree)==extract(new_tree),'GPU validation body changed')
    files=['gpu_tests.json','gradient_paths.json','resource_forecast.json','unit_tests.json','approval_gate_test.json','schedule_manifest.json']
    for name in files:shutil.copyfile(previous/name,out/name)
    shutil.copyfile(previous/'implementation_manifest.json',out/'gpu_execution_manifest.json')
    save(out/'gpu_reuse_provenance.json',dict(source_run=str(previous),source_code_commit=old['code_commit'],
        artifacts={name:sha(previous/name) for name in files},checked_functions_and_dependencies_unchanged=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--init',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--stage');p.add_argument('--approval',type=Path);p.add_argument('--reuse-cpu',type=Path);p.add_argument('--reuse-gpu',type=Path);p.add_argument('--resource-approval',type=Path);p.add_argument('--execution-plan',type=Path);a=p.parse_args()
    if a.init:
        initialize(a.root,a.repo,a.out,a.root/'CURRENT_PROTOCOL.json')
        if a.reuse_cpu:reuse_cpu(a.repo,a.out,a.reuse_cpu)
        if a.reuse_gpu:reuse_gpu(a.repo,a.out,a.reuse_gpu)
    else:sys.exit(supervise(a.root,a.repo,a.out,a.stage,a.preflight_only,a.approval,a.resource_approval,a.execution_plan))
if __name__=='__main__':main()
