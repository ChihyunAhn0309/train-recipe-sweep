#!/usr/bin/env python3
"""Bounded single-host reference study controller. Python 3.11+, standard library.

This is not a cluster scheduler. GPU-specific acceptance is a separate measured
gate. See the accompanying controller protocol for accounting and crash limits.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import time
import uuid

VERSION = 1
TERMINAL = {'completed', 'saturated', 'right_censored', 'budget_exhausted',
            'interrupted', 'pruned', 'failed'}
SUCCESS = {'completed', 'saturated'}
EXTENSION_JOURNAL = 'study-extension.pending.json'


def unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('Duplicate JSON key: ' + key)
        value[key] = item
    return value


def reject_constant(value):
    raise ValueError('Invalid non-finite JSON value: ' + value)


def canonical(value):
    # Match sweep_guard.trial_id: canonical JSON is encoded directly as UTF-8.
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'),
                       object_pairs_hook=unique, parse_constant=reject_constant)
    canonical(value)  # Also rejects exponent overflow to infinity.
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def number(value, name, positive=False):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(name + ' must be finite numeric')
    if value < 0 or (positive and value == 0):
        raise ValueError(name + ' is outside its allowed range')
    return float(value)


class Busy(RuntimeError):
    pass


class FileLock:
    def __init__(self, path, timeout=0.0):
        self.path, self.stream = Path(path), None
        self.timeout = timeout

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b'0')
            self.stream.flush()
        self.stream.seek(0)
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if time.monotonic() < deadline:
                    time.sleep(min(0.01, max(0, deadline-time.monotonic())))
                    continue
                self.stream.close()
                self.stream = None
                raise Busy('Lock is held: ' + str(self.path)) from exc
        return self

    def __exit__(self, *_):
        if self.stream:
            self.stream.close()
            self.stream = None


def artifacts(manifest, base, required=()):
    if not isinstance(manifest, dict):
        raise ValueError('Artifact manifest must be an object')
    items = manifest.get('artifacts')
    if not isinstance(items, list) or not items:
        raise ValueError('Nonempty artifacts are required')
    base = Path(base).resolve()
    roles, seen = set(), set()
    for entry in items:
        if not isinstance(entry, dict) or set(entry) != {'path', 'sha256', 'bytes', 'role'}:
            raise ValueError('Invalid artifact entry')
        if not isinstance(entry['role'], str) or not entry['role']:
            raise ValueError('Artifact role must be a nonempty string')
        rel = entry['path']
        if not isinstance(rel, str) or not rel or '\\' in rel or ':' in rel or Path(rel).is_absolute() or '..' in Path(rel).parts:
            raise ValueError('Artifact path must be relative POSIX and contained')
        path = (base / rel).resolve()
        path.relative_to(base)
        if path in seen or not path.is_file():
            raise ValueError('Duplicate or missing artifact')
        seen.add(path)
        if isinstance(entry['bytes'], bool) or not isinstance(entry['bytes'], int) or entry['bytes'] < 0:
            raise ValueError('Artifact byte count must be a nonnegative integer')
        if path.stat().st_size != entry['bytes'] or file_hash(path) != entry['sha256']:
            raise ValueError('Artifact integrity mismatch: ' + rel)
        roles.add(entry['role'])
    if not set(required) <= roles:
        raise ValueError('Required artifact roles missing: ' + repr(set(required) - roles))
    return True


def verify_resume(path, trial_id):
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError('Resume manifest must be an object')
    if data.get('trial_id') != trial_id or data.get('schema_version') != VERSION:
        raise ValueError('Resume identity/schema mismatch')
    artifacts(data, Path(path).parent, ('checkpoint', 'training_state', 'metrics'))
    return data


def validate_metric(metric, objective):
    """Validate every reported metric, including evaluated partial outcomes."""
    if not isinstance(metric, dict) or not {'name', 'direction', 'value'} <= metric.keys():
        raise ValueError('Metric must be an object with name, direction and value')
    if not isinstance(metric['name'], str) or not metric['name'] or metric['direction'] not in ('min', 'max'):
        raise ValueError('Invalid selection metric name/direction')
    value = metric['value']
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Metric value must be finite numeric')
    number(abs(value), 'metric magnitude')
    if {key: metric[key] for key in ('name', 'direction')} != objective:
        raise ValueError('Result metric differs from immutable objective')


def validate_result(result, trial_id, attempt_id, objective, base):
    if not isinstance(result, dict):
        raise ValueError('Result must be an object')
    status = result.get('status')
    if (result.get('trial_id') != trial_id or result.get('attempt_id') != attempt_id
            or not isinstance(status, str) or status not in TERMINAL):
        raise ValueError('Result identity/status mismatch')
    artifacts(result, base, ('checkpoint', 'metrics') if status in SUCCESS else ('metrics',))
    if 'metric' in result:
        validate_metric(result['metric'], objective)
    elif status in SUCCESS:
        raise ValueError('Successful result requires the immutable objective metric')
    return status


GATES = {'hardware_environment', 'assets_and_gt', 'model_smoke', 'controller_recovery',
         'performance_profile', 'budget_feasibility'}


def verify_evidence(entry, base):
    if not isinstance(entry, dict) or set(entry) != {'path', 'sha256', 'bytes'}:
        raise ValueError('Evidence entries require path, sha256 and bytes')
    artifacts({'artifacts': [dict(entry, role='evidence')]}, base, ('evidence',))


def acceptance_scope(study):
    return digest({'controller_file_sha256': file_hash(Path(__file__).resolve()),
                   'mode': study['mode'], 'devices': study['devices'], 'worker': study['worker'],
                   'budgets': study['budgets'], 'trials': study['trials']})


def verify_acceptance(study, source):
    link = study.get('target_acceptance')
    if not isinstance(link, dict) or set(link) != {'path', 'sha256'}:
        raise ValueError('GPU launch requires hashed target_acceptance')
    path = (source.parent / link['path']).resolve()
    if file_hash(path) != link['sha256']:
        raise ValueError('Target acceptance hash mismatch')
    data = read_json(path)
    if data.get('schema_version') != VERSION or data.get('kind') != 'gpu_acceptance' or data.get('execution_mode') != 'gpu':
        raise ValueError('CPU/schema fixture cannot satisfy GPU target acceptance')
    if data.get('scope_sha256') != acceptance_scope(study):
        raise ValueError('Target acceptance scope is stale or belongs to other config/code/budget')
    if data.get('device_ids') != sorted(d['id'] for d in study['devices']):
        raise ValueError('Target acceptance physical device IDs differ')
    gates = data.get('gates')
    if not isinstance(gates, dict) or set(gates) != GATES:
        raise ValueError('Target acceptance requires all six declared gates')
    for name, gate in gates.items():
        if not isinstance(gate, dict) or gate.get('status') != 'passed':
            raise ValueError('Unresolved GPU acceptance gate: ' + name)
        evidence = gate.get('evidence')
        if not isinstance(evidence, list) or not evidence:
            raise ValueError('GPU gate needs nonempty evidence artifacts: ' + name)
        for entry in evidence:
            verify_evidence(entry, path.parent)
        subchecks = gate.get('subchecks', [])
        if not isinstance(subchecks, list):
            raise ValueError('Gate subchecks must be a list')
        for item in subchecks:
            if not isinstance(item, dict) or not isinstance(item.get('name'), str) or not item['name'] or item.get('status') not in ('passed', 'not_applicable'):
                raise ValueError('Invalid optional GPU subcheck')
            if item['status'] == 'not_applicable' and (not isinstance(item.get('reason'), str) or not item['reason'].strip()):
                raise ValueError('N/A subcheck needs specific reason')
            if not isinstance(item.get('evidence'), list) or not item['evidence']:
                raise ValueError('Optional subcheck needs supporting evidence')
            for entry in item['evidence']:
                verify_evidence(entry, path.parent)


def load_study(path, acceptance_required=True):
    source = Path(path).resolve()
    study = read_json(source)
    if study.get('schema_version') != VERSION or study.get('mode') not in ('cpu_fixture', 'gpu'):
        raise ValueError('Study needs schema_version=1 and mode cpu_fixture/gpu')
    if not isinstance(study.get('study_id'), str) or not study['study_id']:
        raise ValueError('study_id is a required display identifier')
    for name in ('output_root', 'device_lock_root'):
        value = study.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(name + ' is required')
        study[name] = str((source.parent / value).resolve())
    worker = study.get('worker', {}).get('argv')
    if not isinstance(worker, list) or not worker or not all(isinstance(x, str) and x for x in worker):
        raise ValueError('worker.argv must be a nonempty string array')
    # Portable expansion: interpreter comes from this invocation; {plan_dir} is
    # resolved from the supplied study, never from the author's environment.
    study['worker']['argv'] = [x.replace('{python}', sys.executable).replace('{plan_dir}', str(source.parent)) for x in worker]
    if study['mode'] == 'gpu':
        code = study['worker'].get('code_artifacts')
        if not isinstance(code, list) or not code:
            raise ValueError('GPU worker requires nonempty hashed code_artifacts')
        for entry in code:
            verify_evidence(entry, source.parent)
    b = study['budgets']
    for key in ('total_device_seconds', 'poll_seconds', 'grace_seconds', 'terminate_seconds'):
        number(b[key], key, True)
    number(b['confirmation_reserve_device_seconds'], 'confirmation reserve')
    if b['confirmation_reserve_device_seconds'] >= b['total_device_seconds']:
        raise ValueError('Confirmation reserve must be smaller than total budget')
    if not isinstance(b['per_method_device_seconds'], dict) or not b['per_method_device_seconds']:
        raise ValueError('Per-method budgets required')
    for key, value in b['per_method_device_seconds'].items():
        number(value, 'method budget ' + key, True)
    devices = study['devices']
    if not isinstance(devices, list) or not devices:
        raise ValueError('At least one explicit device required')
    names = set()
    for device in devices:
        name, slots = device.get('id'), device.get('slots')
        if study['mode'] == 'gpu':
            if not isinstance(name, str) or not re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', name):
                raise ValueError('GPU mode requires unique physical GPU UUIDs, not indices or MIG partitions')
            name = 'GPU-' + name[4:].lower()
            device['id'] = name
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Device IDs must be unique nonempty strings')
        names.add(name)
        if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
            raise ValueError('Device slots must be positive integer')
        if slots > 1:
            evidence = device.get('packing_evidence')
            if not isinstance(evidence, dict) or set(evidence) != {'path', 'sha256'}:
                raise ValueError('Packing needs a hashed measurement file')
            p = (source.parent / evidence['path']).resolve()
            if file_hash(p) != evidence['sha256']:
                raise ValueError('Packing evidence hash mismatch')
            data = read_json(p)
            if data.get('device_id') != name or data.get('slots') != slots:
                raise ValueError('Packing measurement device/slot mismatch')
            if data.get('kind') != ('cpu_fixture' if study['mode'] == 'cpu_fixture' else 'gpu_measurement'):
                raise ValueError('Packing measurement is not appropriate to execution mode')
            for field in ('peak_memory_bytes', 'available_memory_bytes', 'aggregate_samples_per_second'):
                number(data[field], field, True)
            if data['peak_memory_bytes'] >= data['available_memory_bytes']:
                raise ValueError('Measured packing leaves no memory margin')
            device['packing_evidence'] = {'sha256': evidence['sha256'], 'measurement': data}
    resolved = {}
    for trial in study['trials']:
        if not isinstance(trial.get('config'), dict) or not trial['config']:
            raise ValueError('Every trial needs an immutable config')
        tid = digest(trial['config'])
        if trial.get('trial_id', tid) != tid:
            raise ValueError('Trial ID is not the full config SHA-256')
        trial['trial_id'] = tid
        for field in ('method', 'family'):
            if not isinstance(trial.get(field), str) or trial[field] != trial['config'].get(field):
                raise ValueError(field + ' must match immutable config')
        metric = trial['config'].get('metric')
        if not isinstance(metric, dict) or set(metric) != {'name', 'direction'} or not isinstance(metric['name'], str) or not metric['name'] or metric['direction'] not in ('min', 'max'):
            raise ValueError('Immutable config requires metric {name,direction}')
        if trial['method'] not in b['per_method_device_seconds']:
            raise ValueError('Missing method budget')
        if (not isinstance(trial.get('roles'), list) or not trial['roles']
                or not all(isinstance(role, str) for role in trial['roles'])
                or not set(trial['roles']) <= {'baseline', 'anchor', 'candidate', 'confirmation'}):
            raise ValueError('Invalid trial roles')
        trial['roles'] = sorted(set(trial['roles']))
        if 'confirmation' in trial['roles'] and len(set(trial['roles'])) != 1:
            raise ValueError('Confirmation must have its own trial/seed identity')
        if trial.get('device_count', 1) != 1:
            raise ValueError('This reference supports one device per worker only')
        number(trial['max_seconds'], 'max_seconds', True)
        fallback = trial.setdefault('anchor_fallback', 'reject')
        if fallback not in ('reject', 'allow_censored'):
            raise ValueError('Invalid anchor fallback')
        if tid in resolved:
            left, right = dict(resolved[tid]), dict(trial)
            left.pop('roles'); right.pop('roles')
            if left != right:
                raise ValueError('Duplicate config has conflicting execution metadata')
            resolved[tid]['roles'] = sorted(set(resolved[tid]['roles'] + trial['roles']))
        else:
            resolved[tid] = trial
    if not resolved:
        raise ValueError('No trials')
    study['trials'] = list(resolved.values())
    for trial in study['trials']:
        if 'confirmation' in trial['roles'] and trial['roles'] != ['confirmation']:
            raise ValueError('Deduplicated confirmation must have its own trial/seed identity')
    if study['mode'] == 'gpu' and acceptance_required:
        verify_acceptance(study, source)
    study['_source'] = str(source)
    # The binding includes resolved paths/command/environment choices. Relocation
    # requires a newly created output; it must never reinterpret existing runs.
    study['_hash'] = digest({k: v for k, v in study.items() if not k.startswith('_')})
    return study


SCHEMA = '''
CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE trials(id TEXT PRIMARY KEY CHECK(length(id)=64), spec TEXT NOT NULL,
 state TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', last_attempt TEXT,
 result_path TEXT,result_sha TEXT,resume_path TEXT,resume_sha TEXT);
CREATE TABLE attempts(id TEXT PRIMARY KEY,trial_id TEXT NOT NULL REFERENCES trials(id),
 token TEXT NOT NULL,state TEXT NOT NULL,device TEXT NOT NULL,slot INTEGER NOT NULL,
 method TEXT NOT NULL,confirmation INTEGER NOT NULL,reserve REAL NOT NULL CHECK(reserve>=0),
 max_seconds REAL NOT NULL CHECK(max_seconds>0),admitted REAL NOT NULL,started REAL,
 ended REAL,output TEXT NOT NULL,lease TEXT NOT NULL,receipt TEXT NOT NULL);
CREATE UNIQUE INDEX occupied_slot ON attempts(device,slot) WHERE state IN ('admitted','running');
CREATE UNIQUE INDEX one_trial_attempt ON attempts(trial_id) WHERE state IN ('admitted','running');
CREATE TABLE segments(id INTEGER PRIMARY KEY,start REAL NOT NULL,end REAL NOT NULL,
 device TEXT NOT NULL,owners TEXT NOT NULL,method_cost TEXT NOT NULL,previous_hash TEXT NOT NULL,hash TEXT NOT NULL,
 CHECK(end>=start));
'''


class Registry:
    def __init__(self, study, extension_recovery=None):
        self.study = study
        self.root = Path(study['output_root'])
        self.path = self.root / 'registry.sqlite3'
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / 'study-identity.json'
        journal = self.root / EXTENSION_JOURNAL
        allowed_hashes = {study['_hash']}
        if journal.exists():
            # Only study_extend may bridge the marker/SQLite commit boundary.
            # It validates the complete fixed old/new plans and this journal;
            # Registry still validates SQLite against exactly `study` below.
            if extension_recovery is None or canonical(read_json(journal)) != canonical(extension_recovery):
                raise ValueError('Pending study extension; retry the same study_extend command')
            record = extension_recovery.get('record')
            if (not isinstance(record, dict) or extension_recovery.get('sha256') != digest(record)
                    or record.get('kind') != 'append_only_study_extension'):
                raise ValueError('Invalid study extension recovery journal')
            allowed_hashes = {record.get('old_study_hash'), record.get('new_study_hash')}
            if study['_hash'] not in allowed_hashes:
                raise ValueError('Study extension recovery identity mismatch')
        elif extension_recovery is not None:
            raise ValueError('Study extension recovery requires its pending journal')
        if marker.exists():
            identity = read_json(marker)
            if not any(identity == {'schema_version': VERSION, 'study_hash': value} for value in allowed_hashes):
                raise ValueError('Study output identity mismatch')
            if not self.path.is_file():
                raise ValueError('Bound study lost registry/accounting; explicit recovery required')
            fresh = False
        else:
            if extension_recovery is not None:
                raise ValueError('Study extension cannot recreate a missing identity marker')
            if any(p.name != '.controller.lock' for p in self.root.iterdir()):
                raise ValueError('Refusing to adopt nonempty unbound study output')
            atomic_json(marker, {'schema_version': VERSION, 'study_hash': study['_hash']})
            fresh = True
        self.db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        if fresh:
            self.db.executescript(SCHEMA)
            self.db.execute('BEGIN IMMEDIATE')
            self.db.executemany('INSERT INTO meta VALUES (?,?)', [('schema', str(VERSION)), ('hash', study['_hash']), ('last_time', repr(time.time())), ('segment_count', '0'), ('segment_head', '0'*64)])
            for trial in study['trials']:
                self.db.execute('INSERT INTO trials(id,spec,state) VALUES (?,?,?)', (trial['trial_id'], canonical(trial), 'pending'))
            self.db.commit()
        self.db.execute('BEGIN')
        try:
            self.validate()
            self.db.commit()
        except BaseException:
            self.db.rollback()
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def validate(self):
        if self.db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or self.db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('SQLite registry integrity failure')
        meta = dict(self.db.execute('SELECT key,value FROM meta'))
        if set(meta) != {'schema', 'hash', 'last_time', 'segment_count', 'segment_head'} or meta['schema'] != str(VERSION) or meta['hash'] != self.study['_hash']:
            raise ValueError('Registry identity/accounting metadata is missing or corrupt')
        number(float(meta['last_time']), 'last accounting time')
        rows = self.db.execute('SELECT * FROM trials').fetchall()
        if {row['id'] for row in rows} != {t['trial_id'] for t in self.study['trials']}:
            raise ValueError('Registry trial set changed')
        for row in rows:
            trial = json.loads(row['spec'], object_pairs_hook=unique, parse_constant=reject_constant)
            if digest(trial['config']) != row['id'] or canonical(trial) != canonical(next(t for t in self.study['trials'] if t['trial_id'] == row['id'])):
                raise ValueError('Registry immutable trial config corrupt')
            if row['state'] not in TERMINAL | {'pending', 'running'}:
                raise ValueError('Unknown trial state')
        for row in self.db.execute('SELECT * FROM attempts'):
            for field in ('reserve', 'max_seconds', 'admitted'):
                number(row[field], 'attempt ' + field, field == 'max_seconds')
            if row['state'] not in ('admitted', 'running', 'ended'):
                raise ValueError('Unknown attempt state')
            for field in ('started', 'ended'):
                if row[field] is not None:
                    number(row[field], 'attempt ' + field)
            if row['state'] == 'running' and row['started'] is None:
                raise ValueError('Running attempt has no start accounting')
        self.costs()  # Validates each persisted physical-allocation segment.

    def costs(self):
        own_transaction = not self.db.in_transaction
        if own_transaction:
            self.db.execute('BEGIN')
        try:
            result = self._costs()
            if own_transaction:
                self.db.commit()
            return result
        except BaseException:
            if own_transaction:
                self.db.rollback()
            raise

    def _costs(self):
        total, methods = 0.0, {m: 0.0 for m in self.study['budgets']['per_method_device_seconds']}
        previous = {}
        count, head = 0, '0'*64
        for row in self.db.execute('SELECT * FROM segments ORDER BY id'):
            start, end = number(row['start'], 'segment start'), number(row['end'], 'segment end')
            if end < start or start < previous.get(row['device'], 0) - 1e-6:
                raise ValueError('Corrupt or overlapping physical accounting segments')
            previous[row['device']] = end
            owners, shares = json.loads(row['owners']), json.loads(row['method_cost'])
            if not isinstance(owners, list) or not owners or len(set(owners)) != len(owners):
                raise ValueError('Invalid allocation owners')
            for owner in owners:
                attempt = self.db.execute('SELECT * FROM attempts WHERE id=?', (owner,)).fetchone()
                if attempt is None or attempt['device'] != row['device']:
                    raise ValueError('Allocation references missing or wrong-device ownership')
            if not isinstance(shares, dict) or not set(shares) <= set(methods):
                raise ValueError('Invalid method cost attribution')
            for method, cost in shares.items():
                methods[method] += number(cost, 'method charge')
            if not math.isclose(sum(shares.values()), end-start, abs_tol=1e-6, rel_tol=1e-9):
                raise ValueError('Method charges disagree with physical interval')
            expected_hash = digest({'start': start, 'end': end, 'device': row['device'],
                                    'owners': row['owners'], 'method_cost': row['method_cost'], 'previous_hash': head})
            count += 1
            if row['id'] != count or row['previous_hash'] != head or row['hash'] != expected_hash:
                raise ValueError('Physical accounting hash chain is corrupt or incomplete')
            head = row['hash']
            total += end-start
        meta = dict(self.db.execute("SELECT key,value FROM meta WHERE key IN ('segment_count','segment_head')"))
        if meta != {'segment_count': str(count), 'segment_head': head}:
            raise ValueError('Physical accounting count/head disagrees with records')
        return total, methods

    def begin(self):
        self.db.execute('BEGIN IMMEDIATE')

    def settle(self, now=None):
        now = time.time() if now is None else now
        previous = float(self.db.execute("SELECT value FROM meta WHERE key='last_time'").fetchone()[0])
        if now < previous:
            raise ValueError('Wall clock moved backwards; accounting recovery required')
        active = self.db.execute("SELECT * FROM attempts WHERE state='running'").fetchall()
        for device in sorted({a['device'] for a in active}):
            group = [a for a in active if a['device'] == device]
            shares = {}
            for item in group:
                shares[item['method']] = shares.get(item['method'], 0) + (now-previous)/len(group)
            owners, method_cost = canonical([a['id'] for a in group]), canonical(shares)
            old_head = self.db.execute("SELECT value FROM meta WHERE key='segment_head'").fetchone()[0]
            new_hash = digest({'start': float(previous), 'end': float(now), 'device': device, 'owners': owners,
                               'method_cost': method_cost, 'previous_hash': old_head})
            self.db.execute('INSERT INTO segments(start,end,device,owners,method_cost,previous_hash,hash) VALUES (?,?,?,?,?,?,?)',
                            (previous, now, device, owners, method_cost, old_head, new_hash))
            self.db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='segment_count'")
            self.db.execute("UPDATE meta SET value=? WHERE key='segment_head'", (new_hash,))
        self.db.execute("UPDATE meta SET value=? WHERE key='last_time'", (repr(now),))

    def admission(self, trial, device, slot):
        b = self.study['budgets']
        tail = b['grace_seconds'] + 2*b['terminate_seconds'] + 2*b['poll_seconds']
        reserve = trial['max_seconds'] + tail
        self.begin()
        try:
            self.settle()
            current = self.db.execute('SELECT * FROM trials WHERE id=?', (trial['trial_id'],)).fetchone()
            if current['state'] != 'pending':
                self.db.rollback(); return None, 'not_pending'
            active = self.db.execute("SELECT * FROM attempts WHERE state IN ('admitted','running')").fetchall()
            if any(a['device'] == device and a['slot'] == slot for a in active):
                self.db.rollback(); return None, 'device_busy'
            total, methods = self.costs()
            # Full per-attempt remaining wall reservation is conservative even
            # when packed. Actual billing uses physical union intervals above.
            now = time.time()
            remaining = lambda a: max(0.0, a['reserve'] - (now-a['started'] if a['started'] is not None else 0))
            held = sum(remaining(a) for a in active)
            held_method = sum(remaining(a) for a in active if a['method'] == trial['method'])
            confirmation = trial['roles'] == ['confirmation']
            limit = b['total_device_seconds'] - (0 if confirmation else b['confirmation_reserve_device_seconds'])
            reason = ('total_budget_or_confirmation_reserve' if total+held+reserve > limit else
                      'method_budget' if methods[trial['method']]+held_method+reserve > b['per_method_device_seconds'][trial['method']] else None)
            if reason:
                self.db.execute('UPDATE trials SET reason=? WHERE id=?', (reason, trial['trial_id']))
                self.db.commit(); return None, reason
            aid, token = uuid.uuid4().hex, uuid.uuid4().hex
            out = self.root / 'trials' / trial['trial_id'] / 'attempts' / aid
            values = (aid, trial['trial_id'], token, 'admitted', device, slot, trial['method'], int(confirmation), reserve,
                      trial['max_seconds'], time.time(), str(out), str(out / '.worker.lock'), str(out / 'termination.json'))
            self.db.execute('INSERT INTO attempts(id,trial_id,token,state,device,slot,method,confirmation,reserve,max_seconds,admitted,output,lease,receipt) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', values)
            self.db.execute("UPDATE trials SET state='running',reason='',last_attempt=? WHERE id=?", (aid, trial['trial_id']))
            self.db.commit()
            return dict(self.db.execute('SELECT * FROM attempts WHERE id=?', (aid,)).fetchone()), None
        except BaseException:
            self.db.rollback(); raise

    def start(self, aid):
        self.begin()
        try:
            now = time.time()
            self.settle(now)
            self.db.execute("UPDATE attempts SET state='running',started=? WHERE id=? AND state='admitted'", (now, aid))
            self.db.commit()
        except BaseException:
            self.db.rollback(); raise

    def end(self, aid):
        self.begin()
        try:
            now = time.time()
            self.settle(now)
            self.db.execute("UPDATE attempts SET state='ended',ended=? WHERE id=?", (now, aid))
            self.db.commit()
        except BaseException:
            self.db.rollback(); raise


def device_paths(study, device, slot):
    base = Path(study['device_lock_root']) / hashlib.sha256(device.encode()).hexdigest()
    return base, base / ('slot-' + str(slot) + '.lock'), base / ('slot-' + str(slot) + '.owner.json')


def clear_owner(owner):
    if not Path(owner).exists():
        return not Path(str(owner) + '.initialized').exists()
    data = read_json(owner)
    receipt = read_json(data['receipt'])
    return receipt.get('attempt_id') == data['attempt_id'] and receipt.get('token') == data['token'] and receipt.get('tree_terminated') is True


def slot_available(study, device, slot):
    _, lockpath, owner = device_paths(study, device, slot)
    try:
        with FileLock(lockpath):
            return clear_owner(owner)
    except Busy:
        return False
    except (OSError, ValueError, KeyError):
        return False


def device_policy(study):
    for device in study['devices']:
        base, _, _ = device_paths(study, device['id'], 0)
        with FileLock(base / 'policy.lock'):
            path = base / 'policy.json'
            data = {'id': device['id'], 'slots': device['slots'], 'packing_evidence': device.get('packing_evidence')}
            if path.exists():
                if read_json(path) != data:
                    raise ValueError('Shared device slot policy differs; reconcile measured policy first')
            else:
                atomic_json(path, data)


def receipt_valid(attempt):
    receipt = read_json(attempt['receipt'])
    if receipt.get('attempt_id') != attempt['id'] or receipt.get('token') != attempt['token'] or receipt.get('tree_terminated') is not True:
        raise ValueError('Missing acknowledged process-tree termination')
    return receipt


def reconcile(reg, attempt):
    # OS lock and receipt must both agree before success/reuse/recovery.
    with FileLock(attempt['lease']):
        receipt = receipt_valid(attempt)
        if attempt['state'] != 'ended':
            # Conservatively charge until recovery if the final DB event was lost.
            reg.end(attempt['id'])
        out = Path(attempt['output'])
        result_path = out / 'result.json'
        result_exists = result_path.exists()
        status, reason, result_sha = 'failed', 'missing_or_invalid_result', None
        try:
            result = read_json(result_path)
            config = json.loads(reg.db.execute('SELECT spec FROM trials WHERE id=?', (attempt['trial_id'],)).fetchone()[0])['config']
            status = validate_result(result, attempt['trial_id'], attempt['id'], config['metric'], out)
            if status in SUCCESS:
                if receipt['returncode'] != 0:
                    raise ValueError('Successful result requires zero worker return code')
            reason = str(result.get('reason', 'worker_reported'))
            result_sha = file_hash(result_path)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            status, reason = 'failed', 'invalid_result: ' + str(exc)
        if not result_exists and receipt.get('stop_reason') in ('max_seconds', 'study_budget'):
            status, reason = 'budget_exhausted', receipt['stop_reason'] + '; ' + reason
        elif not result_exists and receipt.get('stop_reason') == 'controller_interrupt':
            status, reason = 'interrupted', reason
        resume = out / 'resume.json'
        resume_path, resume_sha = None, None
        if resume.exists():
            try:
                verify_resume(resume, attempt['trial_id'])
                resume_path, resume_sha = str(resume), file_hash(resume)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                status, reason = 'failed', 'invalid_resume: ' + str(exc)
                result_sha = None
        reg.db.execute('UPDATE trials SET state=?,reason=?,result_path=?,result_sha=?,resume_path=?,resume_sha=? WHERE id=?',
                       (status, reason, str(result_path) if result_sha else None, result_sha, resume_path, resume_sha, attempt['trial_id']))


def verify_reusable(row):
    if not row['result_path'] or file_hash(row['result_path']) != row['result_sha']:
        raise ValueError('Stored completed-result integrity failure')
    data = read_json(row['result_path'])
    objective = json.loads(row['spec'])['config']['metric']
    status = validate_result(data, row['id'], row['last_attempt'], objective, Path(row['result_path']).parent)
    if status not in SUCCESS:
        raise ValueError('Stored successful result identity mismatch')


def eligible(trial, states, trials):
    if 'anchor' in trial['roles']:
        return True, ''
    anchors = [t for t in trials if 'anchor' in t['roles'] and t['method'] == trial['method'] and t['family'] == trial['family']]
    if not anchors:
        return False, 'missing_family_anchor'
    allowed = {'saturated'}
    if trial['anchor_fallback'] == 'allow_censored':
        allowed |= {'right_censored', 'budget_exhausted'}
    if all(states[a['trial_id']] in allowed for a in anchors):
        return True, ''
    return False, 'anchor_not_observed_saturated_or_explicitly_censored'


def stop_attempt(attempt, reason):
    path = Path(attempt['output']) / 'stop.json'
    if not path.exists():
        atomic_json(path, {'reason': reason, 'requested_unix': time.time()})


def report(reg):
    total, methods = reg.costs()
    reported_trials = []
    for row in reg.db.execute('SELECT * FROM trials'):
        item = {key: row[key] for key in ('id', 'state', 'reason', 'last_attempt', 'result_path', 'resume_path')}
        item['evaluated_result'] = None
        item['result_artifacts'] = None
        if row['result_path']:
            if file_hash(row['result_path']) != row['result_sha']:
                raise ValueError('Recorded result changed before reporting')
            result = read_json(row['result_path'])
            objective = json.loads(row['spec'])['config']['metric']
            validate_result(result, row['id'], row['last_attempt'], objective, Path(row['result_path']).parent)
            item['result_artifacts'] = result['artifacts']
            if 'metric' in result:
                item['evaluated_result'] = {'integrity_verified': True, 'status': result['status'],
                                            'metric': result['metric'], 'artifacts': result['artifacts'],
                                            'checkpoint_available': any(a['role'] == 'checkpoint' for a in result['artifacts'])}
        reported_trials.append(item)
    anchor_states = [next(r['state'] for r in reported_trials if r['id'] == t['trial_id'])
                     for t in reg.study['trials'] if 'anchor' in t['roles']]
    value = {'schema_version': VERSION, 'study_hash': reg.study['_hash'], 'mode': reg.study['mode'],
             'physical_device_seconds': total, 'method_attributed_seconds': methods,
             'cost_convention': 'union per explicit physical device; equal share among concurrent workers for method attribution',
             'trials': reported_trials,
             'attempts': [dict(r) for r in reg.db.execute('SELECT * FROM attempts')],
             'all_trials_terminal': all(t['state'] in TERMINAL for t in reported_trials),
             'saturation_calibration_complete': bool(anchor_states) and all(state == 'saturated' for state in anchor_states),
             'target_acceptance_integrity_checked': reg.study['mode'] == 'gpu',
             'gpu_validation_claim': False}
    atomic_json(reg.root / 'report.json', value)
    return value


def run_controller(study_path, resume=False):
    study = load_study(study_path)
    with FileLock(Path(study['output_root']) / '.controller.lock'):
        device_policy(study)
        reg = Registry(study)
        children = {}
        try:
            for attempt in reg.db.execute("SELECT * FROM attempts WHERE state IN ('admitted','running') OR id IN (SELECT last_attempt FROM trials WHERE state='running')").fetchall():
                try:
                    reconcile(reg, dict(attempt))
                except Busy as exc:
                    raise Busy('Live orphan owns its lease; wait for it to finish. No reclaim: ' + attempt['id']) from exc
                except (ValueError, OSError, KeyError) as exc:
                    raise ValueError('Unacknowledged orphan. Run explicit recover only after its recorded process group is quiescent: ' + attempt['id']) from exc
            for row in reg.db.execute('SELECT * FROM trials').fetchall():
                if row['state'] in SUCCESS:
                    verify_reusable(row)
                elif resume and row['state'] in ('interrupted', 'failed', 'budget_exhausted') and row['resume_path']:
                    if file_hash(row['resume_path']) != row['resume_sha']:
                        raise ValueError('Resume manifest changed')
                    verify_resume(row['resume_path'], row['id'])
                    reg.db.execute("UPDATE trials SET state='pending',reason='resume_requested' WHERE id=?", (row['id'],))
            ordered = sorted(study['trials'], key=lambda t: 0 if 'anchor' in t['roles'] else 1 if 'baseline' in t['roles'] else 3 if 'confirmation' in t['roles'] else 2)
            interrupted = False
            while True:
                reg.begin()
                try:
                    reg.settle(); reg.db.commit()
                except BaseException:
                    reg.db.rollback(); raise
                for aid, proc in list(children.items()):
                    if proc.poll() is not None:
                        attempt = dict(reg.db.execute('SELECT * FROM attempts WHERE id=?', (aid,)).fetchone())
                        reconcile(reg, attempt)
                        del children[aid]
                total, methods = reg.costs()
                b = study['budgets']
                shutdown = b['grace_seconds'] + 2*b['terminate_seconds'] + 2*b['poll_seconds']
                active = reg.db.execute("SELECT * FROM attempts WHERE state IN ('admitted','running')").fetchall()
                for a in active:
                    if interrupted or total >= b['total_device_seconds']-shutdown or methods[a['method']] >= b['per_method_device_seconds'][a['method']]-shutdown:
                        stop_attempt(a, 'controller_interrupt' if interrupted else 'study_budget')
                launched = False
                if not interrupted:
                    states = {r['id']: r['state'] for r in reg.db.execute('SELECT id,state FROM trials')}
                    for trial in ordered:
                        if states[trial['trial_id']] != 'pending':
                            continue
                        ok, why = eligible(trial, states, study['trials'])
                        if not ok:
                            reg.db.execute('UPDATE trials SET reason=? WHERE id=?', (why, trial['trial_id']))
                            continue
                        admitted = None
                        for device in study['devices']:
                            for slot in range(device['slots']):
                                # Do not briefly acquire a probe lock on a slot
                                # already promised to our own starting runner.
                                occupied = reg.db.execute("SELECT 1 FROM attempts WHERE device=? AND slot=? AND state IN ('admitted','running')",
                                                          (device['id'], slot)).fetchone()
                                if occupied:
                                    continue
                                if slot_available(study, device['id'], slot):
                                    admitted, _ = reg.admission(trial, device['id'], slot)
                                    if admitted:
                                        break
                            if admitted:
                                break
                        if admitted:
                            out = Path(admitted['output']); out.mkdir(parents=True, exist_ok=True)
                            config = out.parents[1] / 'config.json'
                            if config.exists():
                                if digest(read_json(config)) != trial['trial_id']:
                                    raise ValueError('Config artifact changed')
                            else:
                                atomic_json(config, trial['config'])
                            row = reg.db.execute('SELECT * FROM trials WHERE id=?', (trial['trial_id'],)).fetchone()
                            atomic_json(out / 'control.json', {'schema_version': VERSION, 'trial_id': trial['trial_id'], 'attempt_id': admitted['id'],
                                        'stop_file': str(out / 'stop.json'), 'resume_manifest': row['resume_path'],
                                        'device_id': admitted['device'], 'mode': study['mode']})
                            command = [sys.executable, str(Path(__file__).resolve()), '_worker', str(Path(study_path).resolve()), admitted['id']]
                            log = (out / 'runner.log').open('ab')
                            try:
                                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                        start_new_session=(os.name != 'nt'))
                            finally:
                                log.close()
                            children[admitted['id']] = proc
                            launched = True
                        elif not reg.db.execute('SELECT reason FROM trials WHERE id=?', (trial['trial_id'],)).fetchone()[0]:
                            reg.db.execute('UPDATE trials SET reason=? WHERE id=?', ('device_unavailable_or_unacknowledged', trial['trial_id']))
                if not children:
                    return report(reg)
                try:
                    time.sleep(b['poll_seconds'])
                except KeyboardInterrupt:
                    interrupted = True
        finally:
            # On unexpected cooperative controller errors, request its workers to
            # stop. Do not kill wrappers: they own tree cleanup and acknowledgments.
            for aid, proc in children.items():
                row = reg.db.execute('SELECT * FROM attempts WHERE id=?', (aid,)).fetchone()
                stop_attempt(row, 'controller_interrupt')
            limit = time.monotonic() + study['budgets']['grace_seconds'] + 2*study['budgets']['terminate_seconds'] + 2
            for proc in children.values():
                try:
                    proc.wait(timeout=max(0.01, limit-time.monotonic()))
                except subprocess.TimeoutExpired:
                    pass  # Durable leases block unsafe reuse; report as orphan.
            reg.close()


def linux_group_alive(pgid):
    # /proc distinguishes live members from already-dead zombies. No unrelated
    # process is signaled by this inspection.
    if not Path('/proc').is_dir():
        raise ValueError('This recovery inspection requires Linux /proc')
    for item in Path('/proc').iterdir():
        if not item.name.isdigit():
            continue
        try:
            text = (item / 'stat').read_text()
            fields = text[text.rfind(')')+2:].split()
            if int(fields[2]) == pgid and fields[0] != 'Z':
                return True
        except (OSError, ValueError, IndexError):
            continue
    return False


def windows_job():
    # Assign the runner itself before creating any trainer; child processes then
    # inherit membership without the Popen-to-assignment race. Last handle close
    # kills the job tree if the runner crashes. Windows 10+ nested jobs required.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    class BASIC(ctypes.Structure):
        _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                    ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                    ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                    ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD), ('SchedulingClass', wintypes.DWORD)]
    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in ('ReadOperationCount','WriteOperationCount','OtherOperationCount','ReadTransferCount','WriteTransferCount','OtherTransferCount')]
    class EXT(ctypes.Structure):
        _fields_ = [('BasicLimitInformation', BASIC), ('IoInfo', IO), ('ProcessMemoryLimit', ctypes.c_size_t),
                    ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    handle = kernel.CreateJobObjectW(None, None)
    info = EXT(); info.BasicLimitInformation.LimitFlags = 0x2000
    if not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)) or not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        raise OSError(ctypes.get_last_error(), 'Cannot establish kill-on-close Windows Job')
    # Keep handle alive until process exit; closing it would terminate this runner.
    return handle


def windows_job_members(handle):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    size = 8 + 1024 * ctypes.sizeof(ctypes.c_size_t)
    buf = ctypes.create_string_buffer(size)
    if not kernel.QueryInformationJobObject(handle, 3, buf, size, None):
        raise OSError(ctypes.get_last_error(), 'Cannot enumerate owned Windows Job')
    count = ctypes.c_ulong.from_buffer(buf, 4).value
    values = (ctypes.c_size_t * count).from_buffer(buf, 8)
    return [int(pid) for pid in values if int(pid) != os.getpid()]


def terminate_owned(proc, deadline, job=None):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        end = time.monotonic()+deadline
        while time.monotonic() < end:
            members = windows_job_members(job)
            if not members:
                proc.wait(timeout=deadline)
                return True
            for pid in members:
                # Enumeration is limited to this runner's own Job. A handle pins
                # process identity while termination is requested.
                handle = kernel.OpenProcess(0x1001, False, pid)
                if handle:
                    try:
                        kernel.TerminateProcess(handle, 137)
                    finally:
                        kernel.CloseHandle(handle)
            time.sleep(0.02)
        return not windows_job_members(job)
    pgid = proc.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    end = time.monotonic()+deadline
    while linux_group_alive(pgid) and time.monotonic() < end:
        time.sleep(0.02)
    if linux_group_alive(pgid):
        os.killpg(pgid, signal.SIGKILL)
    proc.wait(timeout=deadline)
    end = time.monotonic()+deadline
    while linux_group_alive(pgid) and time.monotonic() < end:
        time.sleep(0.02)
    return not linux_group_alive(pgid)


def worker_runner(study_path, aid):
    study = load_study(study_path)
    job = windows_job() if os.name == 'nt' else None
    reg = Registry(study)
    attempt = dict(reg.db.execute('SELECT * FROM attempts WHERE id=?', (aid,)).fetchone())
    base, slot_lock, slot_owner = device_paths(study, attempt['device'], attempt['slot'])
    out = Path(attempt['output']); out.mkdir(parents=True, exist_ok=True)
    receipt = {'attempt_id': aid, 'token': attempt['token'], 'tree_terminated': False, 'returncode': None, 'stop_reason': None}
    with FileLock(attempt['lease']) as lease:
        try:
            # Another controller's availability probe may briefly overlap our
            # first acquisition. Wait a bounded second; never evict a real owner.
            with FileLock(slot_lock, timeout=1.0) as slotlease:
                if not clear_owner(slot_owner):
                    raise ValueError('Unacknowledged device owner blocks launch')
                atomic_json(slot_owner, {'attempt_id': aid, 'token': attempt['token'], 'receipt': attempt['receipt']})
                atomic_json(str(slot_owner) + '.initialized', {'schema_version': VERSION})
                reg.start(aid)
                command = study['worker']['argv'] + ['--config', str(out.parents[1] / 'config.json'), '--output', str(out), '--control', str(out / 'control.json')]
                env = os.environ.copy()
                env['STUDY_DEVICE_ID'] = attempt['device']
                if study['mode'] == 'gpu':
                    env['CUDA_VISIBLE_DEVICES'] = attempt['device']
                log = (out / 'worker.log').open('ab')
                try:
                    kwargs = {'start_new_session': True, 'pass_fds': (lease.stream.fileno(), slotlease.stream.fileno())} if os.name != 'nt' else {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
                    proc = subprocess.Popen(command, env=env, cwd=Path(study['_source']).parent, stdout=log, stderr=subprocess.STDOUT, **kwargs)
                finally:
                    log.close()
                atomic_json(out / 'launch.json', {'attempt_id': aid, 'token': attempt['token'], 'pid': proc.pid,
                            'platform': sys.platform, 'process_group': proc.pid if os.name != 'nt' else None})
                started, stop_at = time.monotonic(), None
                b = study['budgets']
                while proc.poll() is None:
                    if time.monotonic()-started >= attempt['max_seconds']:
                        stop_attempt(attempt, 'max_seconds')
                    stop = out / 'stop.json'
                    if stop.exists():
                        if stop_at is None:
                            receipt['stop_reason'] = read_json(stop)['reason']; stop_at = time.monotonic()
                        if time.monotonic()-stop_at >= b['grace_seconds']:
                            break
                    time.sleep(min(b['poll_seconds'], 0.25))
                # Always reconcile the owned tree, including descendants left by
                # a main worker that exited. Windows Job handles crash descendants;
                # normal Windows workers must not detach children (see README).
                clean = terminate_owned(proc, b['terminate_seconds'], job)
                receipt.update(tree_terminated=clean, returncode=proc.returncode,
                               ended_unix=time.time())
                if not clean:
                    raise RuntimeError('Owned tree did not terminate; manual recovery required')
                atomic_json(attempt['receipt'], receipt)
                reg.end(aid)
        except BaseException as exc:
            # Only acknowledge failures before any child was launched. Once a
            # child exists, missing acknowledgment intentionally blocks reuse.
            if 'proc' not in locals():
                receipt.update(tree_terminated=True, returncode=-1, error=str(exc), ended_unix=time.time())
                atomic_json(attempt['receipt'], receipt)
                reg.end(aid)
            raise
        finally:
            reg.close()


def recover(study_path):
    """Acknowledge only recorded Linux groups already proven quiescent.

    Never kills or reclaims a live orphan. Windows abrupt-runner recovery needs
    external Job/process verification; automatic recovery is intentionally closed.
    """
    study = load_study(study_path)
    if os.name == 'nt':
        raise ValueError('Abrupt Windows runner recovery requires external process-tree verification; automatic recovery unsupported')
    with FileLock(Path(study['output_root']) / '.controller.lock'):
        reg = Registry(study)
        try:
            for row in reg.db.execute("SELECT * FROM attempts WHERE state IN ('admitted','running')").fetchall():
                attempt = dict(row)
                _, slotlock, _ = device_paths(study, attempt['device'], attempt['slot'])
                with FileLock(attempt['lease']), FileLock(slotlock):
                    launch = read_json(Path(attempt['output']) / 'launch.json')
                    if launch.get('attempt_id') != attempt['id'] or launch.get('token') != attempt['token'] or launch.get('platform') != 'linux':
                        raise ValueError('Cannot establish durable process-group ownership')
                    if linux_group_alive(launch['process_group']):
                        raise Busy('Live orphan group; recovery refuses to reclaim it')
                    atomic_json(attempt['receipt'], {'attempt_id': attempt['id'], 'token': attempt['token'],
                                'tree_terminated': True, 'returncode': -1, 'stop_reason': 'controller_interrupt',
                                'recovery': 'OS locks acquired and recorded Linux group is quiescent', 'ended_unix': time.time()})
                    reg.end(attempt['id'])
                reconcile(reg, attempt)
            return report(reg)
        finally:
            reg.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('run', 'validate', 'scope', 'report', 'recover'):
        item = sub.add_parser(name); item.add_argument('study')
        if name == 'run':
            item.add_argument('--resume', action='store_true')
    item = sub.add_parser('_worker'); item.add_argument('study'); item.add_argument('attempt_id')
    args = parser.parse_args()
    try:
        if args.command == '_worker':
            worker_runner(args.study, args.attempt_id); return 0
        if args.command == 'scope':
            study = load_study(args.study, acceptance_required=False)
            value = {'scope_sha256': acceptance_scope(study), 'device_ids': sorted(d['id'] for d in study['devices']),
                     'launch_ready': False, 'gpu_verified': False}
        elif args.command == 'validate':
            study = load_study(args.study)
            value = {'valid': True, 'study_hash': study['_hash'], 'trials': len(study['trials']),
                     'mode': study['mode'], 'gpu_verified': False}
        elif args.command == 'run':
            value = run_controller(args.study, args.resume)
        elif args.command == 'recover':
            value = recover(args.study)
        else:
            study = load_study(args.study)
            with FileLock(Path(study['output_root']) / '.controller.lock'):
                reg = Registry(study)
                try:
                    value = report(reg)
                finally:
                    reg.close()
        print(json.dumps(value, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, sqlite3.DatabaseError, Busy, subprocess.TimeoutExpired) as exc:
        print(json.dumps({'status': 'blocked', 'error': str(exc)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
