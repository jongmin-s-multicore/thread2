"""Cumulative GPU wall-time and per-job resource guard; never kills other jobs."""
import argparse, csv, datetime, json, os, signal, subprocess, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--stage',required=True,choices=['prepare','encode','calibrate','train','evaluate','analyze']);p.add_argument('--experiment',choices=['pilot','attribution'],default='pilot');p.add_argument('--arm',default='');p.add_argument('--gpu',required=True);a=p.parse_args();root=a.root.resolve();repo=a.repo.resolve();out=a.out.resolve();gpu=a.stage not in ['prepare','analyze']
jobs=root/'work'/f'{out.name}-jobs';jobs.mkdir(exist_ok=True);job=jobs/f'{a.stage}{a.arm}-{time.time_ns()}';job.mkdir()
prior=sum(json.loads(f.read_text()).get('elapsed_seconds',0) for f in jobs.glob('*/status.json') if json.loads(f.read_text()).get('gpu'));old_gpu=90.81711977836676 if a.experiment=='attribution' else 0;limit=3600-prior-old_gpu if gpu else 900
assert limit>0,'Cumulative GPU budget exhausted'
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.gpu if gpu else '',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',WANDB_MODE='disabled',MPLCONFIGDIR=str(job/'mpl'))
python=root/'work/env/bin/python'
if a.stage=='prepare':
 setup=root/'work/replay-setup-20261008';replay=root/'work/replay-env';python=replay/'bin/python';env.update(MUJOCO_PY_FORCE_CPU='1',MUJOCO_PY_MUJOCO_PATH=str(setup/'mujoco210'),LD_LIBRARY_PATH=f'{setup}/mujoco210/bin:{replay}/lib',CFLAGS='-Wno-error=incompatible-pointer-types -fno-strict-aliasing -include stdio.h',CPATH=str(replay/'include'),LIBRARY_PATH=f'{setup}/native-links:{replay}/lib',LDFLAGS=f'-L{setup}/native-links -L{replay}/lib',D4RL_DATASET_DIR=str(setup/'d4rl-cache'),TMPDIR=str(setup/'tmp'),XDG_CACHE_HOME=str(setup/'cache'))
script=repo/'repro/physics'/('physics_attribution.py' if a.experiment=='attribution' else 'analyze_physics_loss.py' if a.stage=='analyze' else 'physics_loss_pilot.py');bootstrap='import sys,runpy; root,repo,script,*args=sys.argv[1:];sys.path.append(root+"/work/env/lib/python3.10/site-packages");sys.path.insert(0,repo);sys.path.insert(0,str(__import__("pathlib").Path(script).parent));sys.argv=[script]+args;runpy.run_path(script,run_name="__main__")'
cmd=[str(python),'-c',bootstrap,str(root),str(repo),str(script),'--root',str(root),'--repo',str(repo),'--out',str(out)]
if a.stage!='analyze' or a.experiment=='attribution':cmd+=['--stage',a.stage]
if a.arm:cmd+=['--arm',a.arm]
cores=sorted(os.sched_getaffinity(0))[:2]
def setup():os.sched_setaffinity(0,cores);os.nice(10)
base=dict(gpu=gpu,prior_gpu_seconds=prior,limit_seconds=limit,cpu_cores=cores,command=cmd,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
def status(**kw):(job/'status.json').write_text(json.dumps(dict(base,**kw),indent=2))
start=time.monotonic();reason=None;peak=peakgpu=peakdisk=0
with (job/'stdout.log').open('x') as log,(job/'resources.csv').open('x') as res:
 proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True,preexec_fn=setup,cwd=repo);status(state='RUNNING',pid=proc.pid);w=csv.writer(res);w.writerow(['seconds','rss_bytes','gpu_device_mib','disk_bytes'])
 try:
  while proc.poll() is None:
   rss=0
   for d in Path('/proc').iterdir():
    if not d.name.isdigit():continue
    try:
     if int((d/'stat').read_text().rsplit(')',1)[1].split()[2])==proc.pid:rss+=int((d/'statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
    except (OSError,ValueError,IndexError):pass
   disk=sum(f.stat().st_size for parent in [out,jobs] if parent.exists() for f in parent.rglob('*') if f.is_file());memory=int(subprocess.check_output(['/usr/lib/wsl/lib/nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).strip()) if gpu else 0
   elapsed=time.monotonic()-start;peak=max(peak,rss);peakgpu=max(peakgpu,memory);peakdisk=max(peakdisk,disk);w.writerow([elapsed,rss,memory,disk]);res.flush()
   if rss>12*1024**3:reason='RAM_LIMIT'
   elif disk+(1856039710 if a.experiment=='attribution' else 0)>10*1024**3:reason='DISK_LIMIT'
   elif memory>11000:reason='GPU_DEVICE_LIMIT'
   elif elapsed>limit:reason='CUMULATIVE_GPU_TIME_LIMIT' if gpu else 'CPU_TIME_LIMIT'
   if reason:break
   time.sleep(1)
 except BaseException as ex:reason=repr(ex)
 finally:
  if proc.poll() is None:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
 rc=proc.wait();status(state='COMPLETE' if rc==0 and reason is None else 'FAILED',returncode=rc,reason=reason,elapsed_seconds=time.monotonic()-start,peak_rss_bytes=peak,peak_gpu_device_mib=peakgpu,peak_disk_bytes=peakdisk)
print((job/'status.json').read_text());sys.exit(0 if rc==0 and reason is None else 1)
