"""Sequential bounded execution; environment paths are rooted by CLI arguments."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True)
p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',choices=['prepare','infer','analyze'],required=True)
p.add_argument('--experiment',choices=['action','zero_velocity','zero_render','physics_depth'],default='action');p.add_argument('--gpu',default='');p.add_argument('--reuse-oracles',type=Path);a=p.parse_args();root=a.root.resolve();repo=a.repo.resolve();result=a.out.resolve()
job=root/'work'/f'{result.name}-{a.stage}-job';job.mkdir(exist_ok=False)
gpu=a.stage=='infer';ram=12 if gpu else 8;timeout=1800 if gpu else 900 if a.stage=='prepare' else 600
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu if gpu else '',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
 PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',MPLCONFIGDIR=str(job/'matplotlib'))
if gpu:assert a.gpu,'GPU identifier is required for inference'
python=root/'work/env/bin/python'
if a.stage=='prepare' or (a.experiment=='physics_depth' and a.stage=='analyze'):
    setup=root/'work/replay-setup-20261008';replay=root/'work/replay-env';python=replay/'bin/python'
    env.update(MUJOCO_PY_FORCE_CPU='1',MUJOCO_PY_MUJOCO_PATH=str(setup/'mujoco210'),
      LD_LIBRARY_PATH=f'{setup}/mujoco210/bin:{replay}/lib',CFLAGS='-Wno-error=incompatible-pointer-types -fno-strict-aliasing -include stdio.h',
      CPATH=str(replay/'include'),LIBRARY_PATH=f'{setup}/native-links:{replay}/lib',LDFLAGS=f'-L{setup}/native-links -L{replay}/lib',
      D4RL_DATASET_DIR=str(setup/'d4rl-cache'),TMPDIR=str(setup/'tmp'),XDG_CACHE_HOME=str(setup/'cache'))
script=repo/'repro/physics'/('physics_depth.py' if a.experiment=='physics_depth' else 'zero_velocity.py' if a.experiment in ['zero_velocity','zero_render'] else 'analyze_intervention.py' if a.stage=='analyze' else 'intervention.py')
bootstrap='import sys,runpy; root,script,*args=sys.argv[1:]; sys.path.append(root+"/work/env/lib/python3.10/site-packages"); sys.path.insert(0,str(__import__("pathlib").Path(script).parent)); sys.argv=[script]+args; runpy.run_path(script,run_name="__main__")'
cmd=[str(python),'-c',bootstrap,str(root),str(script),'--root',str(root),'--repo',str(repo),'--out',str(result)]
if a.stage!='analyze' or a.experiment in ['zero_velocity','zero_render','physics_depth']:cmd+=['--stage',a.stage]
if a.experiment=='zero_render':cmd+=['--render-audit']
if a.reuse_oracles is not None:cmd+=['--reuse-oracles',str(a.reuse_oracles)]
cores=sorted(os.sched_getaffinity(0))[:2]
def setup():os.sched_setaffinity(0,cores);os.nice(10)
base=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cpu_cores=cores,ram_limit_gib=ram,
          disk_limit_gib=5,timeout_seconds=timeout,gpu_memory_limit_mib=11000 if gpu else None,command=cmd)
def status(**kw):(job/'status.json').write_text(json.dumps(dict(base,**kw),indent=2))
start=time.monotonic();reason=None;peak=0;peak_gpu=0
with (job/'stdout.log').open('x') as log,(job/'resources.csv').open('x') as res:
    proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,preexec_fn=setup,cwd=repo)
    status(state='RUNNING',pid=proc.pid);writer=csv.writer(res);writer.writerow(['seconds','rss_bytes','disk_bytes','gpu_device_mib'])
    try:
        while proc.poll() is None:
            rss=0
            for d in Path('/proc').iterdir():
                if not d.name.isdigit():continue
                try:
                    if int((d/'stat').read_text().rsplit(')',1)[1].split()[2])==proc.pid:rss+=int((d/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
                except (OSError,ValueError,IndexError):pass
            disk=sum(f.stat().st_size for f in result.rglob('*') if f.is_file()) if result.exists() else 0
            memory=0
            if gpu:
                memory=int(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip())
            elapsed=time.monotonic()-start;peak=max(peak,rss);peak_gpu=max(peak_gpu,memory)
            writer.writerow([elapsed,rss,disk,memory]);res.flush()
            if rss>ram*1024**3:reason='RAM_LIMIT'
            elif disk>5*1024**3:reason='DISK_LIMIT'
            elif memory>11000:reason='GPU_DEVICE_LIMIT'
            elif elapsed>timeout:reason='TIME_LIMIT'
            if reason:break
            time.sleep(1)
    except BaseException as ex:reason=repr(ex)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    rc=proc.wait();status(state='COMPLETE' if rc==0 and reason is None else 'FAILED',returncode=rc,reason=reason,
        peak_rss_bytes=peak,peak_gpu_device_mib=peak_gpu,elapsed_seconds=time.monotonic()-start)
sys.exit(0 if rc==0 and reason is None else 1)
