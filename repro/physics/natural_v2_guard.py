"""[repro] 연구 수식과 독립적인 표준 라이브러리 실행 관문."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def document_manifest(registry):
    result = {}
    for doc in registry['documents']:
        name = Path(doc['server_path']).name
        require(name not in result, 'duplicate document name')
        require(re.fullmatch(r'[0-9a-f]{64}', doc['sha256']), 'invalid document hash')
        result[name] = doc['sha256']
    require(result, 'empty document registry')
    return result


def verify_documents(out):
    out = Path(out)
    lock = read(out / 'run_lock.json')
    require(sha(out / 'registry_snapshot.json') == lock['registry_sha256'], 'registry changed')
    expected = document_manifest(read(out / 'registry_snapshot.json'))
    require(lock.get('document_sha256') == expected, 'document lock differs from registry')
    for name, digest in expected.items():
        path = out / name
        require(path.is_file() and not path.is_symlink(), 'document missing or symlink: ' + name)
        require(sha(path) == digest, 'document changed: ' + name)
    manifest = out / 'physics_natural_evidence_v2_0.json'
    require(sha(manifest) == lock['input_manifest_sha256'] == expected[manifest.name], 'input manifest changed')
    return expected


def guarded_input(out, callback):
    verify_documents(out)
    return callback()


def read_document(out, name):
    return guarded_input(out, lambda: read(Path(out)/name))


STAGES = ('stage1', 'prepare-tune', 'encode-tune', 'stage2-base',
          'stage2-aux', 'prepare-eval', 'encode-eval', 'final-eval')


def number(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'invalid nonnegative cost')
    return value


def stage_costs(forecast):
    """기존 총량을 단계에 분배. 시점0 평가 여유도 기존 총량에 유지한다."""
    step = max(number(x['seconds']) for x in forecast['measured_step'])
    enc = max(number(x) for x in forecast['measured_encoding_seconds_per_window'])
    ev = number(forecast['measured_h5_rollout_seconds_per_window'])
    counts = {
        'stage1': (240, 0, 112), 'prepare-tune': (0, 0, 0),
        'encode-tune': (0, 344, 0), 'stage2-base': (960, 0, 1728),
        'stage2-aux': (1920, 0, 3456), 'prepare-eval': (0, 0, 0),
        'encode-eval': (0, 360, 0), 'final-eval': (0, 0, 1440)}
    costs = {s: dict(updates=u, encode_windows=e, evaluation_windows=v,
                     raw_seconds=u*step+e*enc+v*ev) for s, (u, e, v) in counts.items()}
    require(math.isclose(sum(c['raw_seconds'] for c in costs.values()),
                         forecast['raw_estimate_seconds'], rel_tol=1e-10), 'forecast totals differ')
    return dict(stages=costs, factor=2, reserve_seconds=120, cap_seconds=3600)


def remaining_budget(costs, jobs, prior, stage):
    require(set(costs['stages']) == set(STAGES), 'incomplete stage cost table')
    require((costs['factor'], costs['reserve_seconds'], costs['cap_seconds']) == (2, 120, 3600), 'budget constants changed')
    require(stage in STAGES, 'unknown stage')
    done = set(); seen = set(); actual = number(prior)
    for job in jobs:
        identity = job['job_id']
        require(identity not in seen, 'duplicate job'); seen.add(identity)
        require(job['state'] in ('COMPLETE', 'FAILED', 'CANCELLED'), 'running or ambiguous job')
        require(type(job['gpu']) is bool, 'missing GPU accounting')
        elapsed = number(job['elapsed_seconds'])
        if job['gpu']:
            actual += elapsed
        if job.get('comparison_stage'):
            s = job['stage']; require(s in STAGES, 'unknown completed stage')
            if job['state'] == 'COMPLETE':
                require(job.get('returncode') == 0 and job.get('reason') is None, 'ambiguous completion')
                require(s not in done, 'stage completed twice'); done.add(s)
    require(stage not in done, 'completed stage cannot run again')
    require(all(s in done for s in STAGES[:STAGES.index(stage)]), 'preceding stage not complete')
    remaining = {s: number(c['raw_seconds']) for s, c in costs['stages'].items() if s not in done}
    reserve = costs['reserve_seconds'] if remaining else 0
    projected = actual + costs['factor']*sum(remaining.values()) + reserve
    require(projected <= costs['cap_seconds'], 'remaining work exceeds GPU cap')
    return dict(actual_gpu_seconds=actual, completed_stages=sorted(done),
                remaining_raw_by_stage=remaining, factor=costs['factor'], reserve_seconds=reserve,
                projected_gpu_seconds=projected, cap_seconds=costs['cap_seconds'])


def budget_for_run(out, approval_record, jobs, prior, stage):
    out = Path(out)
    path = Path(approval_record['preflight_completion_path']).parent / 'resource_forecast.json'
    require(sha(path) == approval_record.get('forecast_sha256'), 'forecast not review-pinned')
    costs = stage_costs(read(path))
    bound = dict(forecast_sha256=sha(path), costs=costs)
    target = out / 'stage_costs.json'
    if target.exists():
        require(read(target) == bound, 'fixed stage costs changed')
    else:
        save(target, bound)
    return remaining_budget(costs, jobs, prior, stage)


def event_path(job, arm, step, kind):
    require(re.fullmatch(r'[A-Za-z0-9_.-]+', arm) and type(step) is int and step > 0, 'bad update identity')
    require(kind in ('attempt', 'complete'), 'bad event kind')
    return Path(job) / 'updates' / f'{arm}.{step}.{kind}.json'


def update_event(job, arm, step, kind):
    path = event_path(job, arm, step, kind)
    path.parent.mkdir(exist_ok=True)
    event = dict(job_id=Path(job).name, arm=arm, step=step, event=kind)
    if kind == 'complete':
        require(read(event_path(job, arm, step, 'attempt')) == dict(event, event='attempt'), 'completion without attempt')
    save(path, event)
    return event


def recorded_update(job, arm, step, callback):
    update_event(job, arm, step, 'attempt')
    callback()
    return update_event(job, arm, step, 'complete')


def update_counts(job, arm=None):
    events = {}; job = Path(job)
    for p in sorted((job / 'updates').glob('*.json')):
        event = read(p)
        require(event['job_id'] == job.name, 'foreign job update')
        require(p == event_path(job, event['arm'], event['step'], event['event']), 'event filename mismatch')
        key = (event['arm'], event['step'], event['event'])
        require(key not in events, 'duplicate event'); events[key] = event
    attempts = {(a, s) for a, s, k in events if k == 'attempt' and (arm is None or arm == a)}
    complete = {(a, s) for a, s, k in events if k == 'complete' and (arm is None or arm == a)}
    require(complete <= attempts, 'orphan completion')
    # 과거 preflight reservation은 완료 여부를 추정하지 않는다.
    legacy = len(list(job.glob('update_*.json'))) if arm is None else 0
    return dict(attempted=len(attempts)+legacy, completed=len(complete),
                uncertain=len(attempts-complete)+legacy, legacy_reserved=legacy)


def reconcile_curve(job, arm, curve):
    counts = update_counts(job, arm)
    steps = [r['step'] for r in curve]
    expected = sorted(read(p)['step'] for p in (Path(job)/'updates').glob('*.complete.json') if read(p)['arm'] == arm)
    require(len(steps) == len(set(steps)) and sorted(steps) == expected, 'curve and update events differ')
    require(counts['uncertain'] == 0, 'uncertain update at completion')
    return counts


LIMIT_KEYS = ('gpu_seconds', 'cpu_cores', 'ram_bytes', 'wall_seconds', 'disk_bytes', 'gpu_device_mib')


def verify_execution_authorization(approval, request, allow_fixture=False):
    require(isinstance(approval, dict), 'separate user execution approval required')
    require(approval.get('status') == 'APPROVED' and approval.get('issuer') == 'user', 'not user approval')
    require(type(approval.get('fixture')) is bool, 'approval fixture marker required')
    require(approval['fixture'] is False or allow_fixture, 'fixture is not execution authority')
    for key in ('plan_sha256', 'code_commit', 'argv', 'data_scope'):
        require(key in approval and key in request and approval[key] == request[key], 'approval mismatch: ' + key)
    require(bool(request['argv']) and isinstance(request['argv'], list) and all(isinstance(x,str) for x in request['argv']), 'invalid argv')
    require(bool(request['data_scope']), 'empty data scope')
    for key in LIMIT_KEYS:
        require(key in approval['limits'] and key in request['limits'], 'missing limit: ' + key)
        require(number(request['limits'][key]) <= number(approval['limits'][key]), 'limit exceeds approval: ' + key)
    for key in ('max_executions', 'max_retries'):
        require(type(request.get(key)) is int and type(approval.get(key)) is int, 'missing execution count')
        require(0 <= request[key] <= approval[key], 'execution count exceeds approval')
    require(request['max_executions'] > 0, 'no execution allowed')
    return dict(verified=True, fixture=approval['fixture'], request=request)


def authorized_call(approval, request, callback, allow_fixture=False):
    receipt = verify_execution_authorization(approval, request, allow_fixture)
    return callback(), receipt


def capture_context(root, repo, out, stage, job, argv):
    """실제 호스트와 Git/문서 상태. fixture는 이 production 관측기를 호출하지 않는다."""
    import platform
    import subprocess
    import socket
    import sys
    root,repo,out,job=map(lambda p:Path(p).resolve(),(root,repo,out,job))
    require(platform.system()=='Linux','server-only execution')
    git=lambda *a:subprocess.check_output(['git','-C',str(repo),*a])
    dirty=git('diff','HEAD','--')
    require(not dirty and not git('diff','--cached','--'),'dirty code')
    return dict(system=platform.system(),hostname_sha256=hashlib.sha256(socket.gethostname().encode()).hexdigest(),
        root=str(root),repo=str(repo),out=str(out),stage=stage,job=str(job),argv=list(argv),
        code_commit=git('rev-parse','HEAD').decode().strip(),branch=git('branch','--show-current').decode().strip(),
        dirty_sha256=hashlib.sha256(dirty).hexdigest(),registry_sha256=sha(root/'CURRENT_PROTOCOL.json'),
        protocol_sha256=sha(repo/'repro/physics/physics_natural_protocol_v2_0.md'),
        input_sha256=sha(repo/'repro/physics/physics_natural_evidence_v2_0.json'),
        policy_sha256=sha(root/'orchestration/SERVER_EXECUTION_POLICY.txt'))


CONTEXT_KEYS=('system','hostname_sha256','root','repo','out','stage','job','argv',
    'code_commit','branch','dirty_sha256','registry_sha256','protocol_sha256','input_sha256','policy_sha256')


def bind_actual(expected, actual):
    require(actual.get('system')=='Linux','server-only execution')
    for key in CONTEXT_KEYS:
        require(key in expected and key in actual and actual[key]==expected[key],'actual context mismatch: '+key)
    for key in ('root','repo','out','job'):
        require(str(Path(actual[key]).resolve())==actual[key],'noncanonical path: '+key)
    require(actual['dirty_sha256']==hashlib.sha256(b'').hexdigest(),'dirty execution')
    return True


def consume_context(expected, actual, marker, callback):
    bind_actual(expected,actual)
    require(Path(marker).resolve().parent==Path(actual['job']).resolve(),'receipt copied to other job')
    save(marker,dict(consumed=True,context=actual))
    return callback()


def verify_job_authorization(out, job, actual, allow_fixture=False):
    receipt=read(Path(job)/'execution_authorization.json')
    require(receipt['fixture'] is False or allow_fixture,'fixture receipt in production')
    request=receipt['request'];lock=read(Path(out)/'run_lock.json')
    require(request['code_commit']==lock['code_commit'],'job code changed')
    require(sha(receipt['plan_path'])==request['plan_sha256'],'execution plan changed')
    require(sha(receipt['approval_path'])==receipt['approval_sha256'],'user approval changed')
    approval=read(receipt['approval_path'])
    verify_execution_authorization(approval,request,allow_fixture)
    require(approval.get('actual_context')==request.get('actual_context') and bool(request.get('actual_context')),'missing approved actual context')
    consume_context(request['actual_context'],actual,Path(job)/'child_consumed.json',lambda:None)
    return receipt


def admission_pipeline(argv, observe, callback, allow_fixture=False):
    # [repro] CLI와 가짜 callback 검사 모두 같은 파싱/파일/승인/소비 경로를 사용한다.
    values={}
    for key in ('root','repo','out','stage','job'):
        flag='--'+key
        require(argv.count(flag)==1,'missing/duplicate '+flag)
        index=argv.index(flag)+1
        require(index<len(argv) and not argv[index].startswith('--'),'missing value '+flag)
        values[key]=argv[index]
    actual=observe(values, list(argv))
    for key in values:require(actual.get(key)==values[key],'parsed context mismatch: '+key)
    require(actual.get('argv')==list(argv),'parsed argv mismatch')
    receipt=verify_job_authorization(Path(values['out']),Path(values['job']),actual,allow_fixture)
    return callback(),receipt


def production_entry_guard(argv):
    return admission_pipeline(argv,lambda v,a:capture_context(v['root'],v['repo'],v['out'],v['stage'],v['job'],a),lambda:None)


def static_constants(source, names):
    """literal/list만 제한 해석한다. 과거 모듈을 실행하지 않는다."""
    values = {}
    def literal(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'list' and len(node.args) == 1 and not node.keywords:
            value = ast.literal_eval(node.args[0]); require(isinstance(value, (str, list, tuple)), 'unknown list argument')
            return list(value)
        if isinstance(node, ast.Dict):
            return {literal(k): literal(v) for k,v in zip(node.keys,node.values)}
        return ast.literal_eval(node)
    try:
        nodes=ast.parse(source).body
        # [repro] Whole-module binding preflight: later literals never erase unsafe bindings.
        protected=set(names)|{'list'}
        for node in nodes:
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                for alias in node.names:
                    require(alias.name!='*','manual_review_required: wildcard import')
                    bound=alias.asname or (alias.name.split('.')[0] if isinstance(node,ast.Import) else alias.name)
                    require(bound not in protected,'manual_review_required: imported protected binding '+bound)
            elif isinstance(node,ast.FunctionDef):
                require(node.name not in protected,'manual_review_required: function protected binding')
            elif isinstance(node,ast.Assign):
                require(not any(isinstance(n,ast.Name) and n.id=='list' for target in node.targets for n in ast.walk(target)), 'manual_review_required: constructor binding')
        for node in nodes:
            if isinstance(node, ast.Assign):
                require(len(node.targets)==1 and isinstance(node.targets[0],ast.Name), 'manual_review_required: indirect/multiple assignment')
                require(not any(isinstance(n,ast.Name) and n.id in names for n in ast.walk(node.value)), 'manual_review_required: alias')
                value=literal(node.value)
                if node.targets[0].id in names:values[node.targets[0].id]=value
            elif isinstance(node,(ast.Import,ast.ImportFrom)):
                continue
            elif isinstance(node,ast.FunctionDef):
                require(not node.decorator_list and not node.args.defaults and not any(node.args.kw_defaults), 'manual_review_required: definition side effect')
                require(not node.returns and not any(a.annotation for a in node.args.args+node.args.kwonlyargs+node.args.posonlyargs), 'manual_review_required: annotation')
            elif isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
                continue
            else:
                raise RuntimeError('manual_review_required: unsupported top-level effect')
        require(set(values) == set(names), 'missing constant: manual review required')
    except (ValueError, TypeError, KeyError, SyntaxError) as ex:
        raise RuntimeError('constant unresolved: manual review required') from ex
    return values


def verify_constants(before, after, names):
    a = static_constants(before, names); b = static_constants(after, names)
    require(a == b, 'referenced constants changed')
    return dict(names=list(names), values=b, equal=True)


EPISODE_KEYS = ('episode', 'episode_id', 'ep', 'traj_idx')
GROUP_KEYS = ('train', 'development', 'validation', 'dev', 'fit', 'tune', 'fresh_development', 'old_development', 'inner_fit', 'inner_tune')


def audit_manifests(paths, target_ids, excluded=()):
    records = []; hits = []; unknown = []; targets = set(target_ids)
    for raw in sorted(map(str, paths)):
        path = Path(raw); found = set(); keys = []; unparsed = []
        try:
            data = read(path)
            def walk(v, at=''):
                if isinstance(v, dict):
                    for k,x in v.items():
                        loc = at+'/'+k
                        if (k in EPISODE_KEYS and type(x) is not int) or (('episode' in k or k in GROUP_KEYS) and isinstance(x,list) and not all(type(i) is int for i in x)):
                            unparsed.append(loc)
                        if k in EPISODE_KEYS and type(x) is int:
                            found.add(x); keys.append(loc)
                        if isinstance(x,list) and ('episode' in k or k in GROUP_KEYS) and all(type(i) is int for i in x):
                            found.update(x); keys.append(loc)
                        # cache 사용 manifest의 경로 이름만 검사한다. tensor는 열지 않는다.
                        for match in re.finditer(r'(?:C_|Z_|context_|episode_)(\d+)(?:_|\.|/|$)', k):
                            found.add(int(match.group(1))); keys.append(loc)
                        walk(x,loc)
                elif isinstance(v,list):
                    for i,x in enumerate(v):walk(x,at+'/'+str(i))
            walk(data)
            row=dict(path=raw,sha256=sha(path),episode_ids=sorted(found),recognized_keys=keys)
            records.append(row)
            if unparsed:unknown.append(dict(path=raw,reason='unparsed episode keys',keys=unparsed))
            if not keys:unknown.append(dict(path=raw,reason='no recognized episode metadata'))
            overlap=sorted(found & targets)
            if overlap:hits.append(dict(path=raw,episode_ids=overlap))
        except (OSError,ValueError,TypeError) as ex:
            unknown.append(dict(path=raw,reason=type(ex).__name__))
    return dict(status='OVERLAP_STOP' if hits else 'INCOMPLETE' if unknown or excluded else 'NO_OVERLAP_IN_SCANNED_METADATA',
                scanned=records,overlap=hits,unknown=unknown,excluded=list(excluded),
                recognized_scalar_keys=list(EPISODE_KEYS),recognized_group_keys=list(GROUP_KEYS),
                scope='explicit origin-labelled metadata only; no observations/tensors; no global novelty guarantee')


def prior_charge(record, upper=120):
    value=record.get('wall_seconds')
    if type(value) not in (int,float) or not math.isfinite(value) or value<0:return upper
    return max(upper,value)


def coverage(required, rows):
    require(required and all(required.values()),'empty required coverage')
    expected={(group,name) for group,names in required.items() for name in names}
    require(sum(map(len,required.values()))==len(expected),'duplicate required check')
    actual=[(r['id'],r['name']) for r in rows]
    invalid=len(actual)!=len(set(actual)) or any(k not in expected for k in actual)
    result={}
    for group,names in required.items():
        got=[r for r in rows if r['id']==group and r['name'] in names]
        passed=sum(r.get('passed') is True for r in got)
        result[group]=dict(expected=len(names),completed=len(got),passed=passed,
            failed=sum(r.get('passed') is not True for r in got),not_run=len(set(names)-{r['name'] for r in got}),
            status='passed' if not invalid and len(got)==len(names) and passed==len(names) else 'not_run' if not got else 'failed')
    return dict(valid=not invalid,groups=result,passed=not invalid and all(r['status']=='passed' for r in result.values()))


# [repro] 가짜 검사의 실제 생명주기에도 쓰는 제한된 기록/실패 경로.
class BoundedStore:
    def __init__(self, root, quota, reserve=8192):
        self.root=Path(root).resolve();self.quota=quota;self.reserve=reserve;self.written=0
    def write(self, path, value, terminal=False):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        path=Path(path).resolve()
        require(self.root in path.parents,'write outside result root')
        require(self.written+len(data)<=self.quota-(0 if terminal else self.reserve),'write reserve exhausted')
        self.written+=len(data)
        temporary=path.with_name(path.name+'.pending')
        with temporary.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.link(temporary,path);temporary.unlink()
        return len(data)


def protected_lifecycle(folder, validate, body, finalize, seconds, store):
    import signal,time
    folder=Path(folder);folder.mkdir()
    start=time.monotonic();store.write(folder/'reserved.json',dict(state='RESERVED',monotonic=start))
    old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
    def expired(*args):raise TimeoutError('protected lifecycle deadline')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,min(seconds,old_timer[0]) if old_timer[0] else seconds)
    phase='validation'
    try:
        validate();phase='body';result=body();phase='finalization';finalize(result)
        store.write(folder/'complete.json',dict(state='COMPLETE',duration=time.monotonic()-start))
        return dict(status='succeeded',duration=time.monotonic()-start)
    except (Exception,MemoryError) as ex:
        # Scoped deadline expired: independent enclosing supervisor still bounds cleanup.
        signal.setitimer(signal.ITIMER_REAL,0.01)
        result=dict(status='failed',phase=phase,error=type(ex).__name__,duration=time.monotonic()-start)
        store.write(folder/'failed.json',result,terminal=True)
        return result
    finally:
        elapsed=time.monotonic()-start
        signal.signal(signal.SIGALRM,old_handler)
        signal.setitimer(signal.ITIMER_REAL,max(.000001,old_timer[0]-elapsed) if old_timer[0] else 0,old_timer[1])


def reconcile_receipt(intent, receipt):
    if receipt is None:return dict(status='UNKNOWN',attempt_consumed=True,relaunch=False)
    for key in ('uuid','invocation','candidate','manifest_sha256','plan_sha256','input_sha256'):
        require(receipt.get(key)==intent.get(key) and key in intent,'receipt identity mismatch: '+key)
    require(receipt.get('duration_monotonic_seconds',0)>=0,'invalid local duration')
    return dict(status=receipt.get('status','UNKNOWN'),attempt_consumed=True,relaunch=False)


def verify_copy(data, digest):
    require(hashlib.sha256(data).hexdigest()==digest,'copy hash mismatch')
    return True
