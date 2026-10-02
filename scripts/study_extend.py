#!/usr/bin/env python3
"""Append a declared stage to a quiescent study without resetting its ledger.

Python 3.11+, standard library. This helper does not select finalists, infer LR
anchors, certify independent seeds, increase budgets, or launch workers.
"""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import time

from study_controller import (
    EXTENSION_JOURNAL, VERSION, Busy, FileLock, Registry, SUCCESS,
    atomic_json, canonical, digest, file_hash, load_study, number, read_json,
    receipt_valid, verify_reusable,
)


def plans(old_path, new_path):
    old_source, new_source = Path(old_path).resolve(), Path(new_path).resolve()
    if old_source == new_source:
        raise ValueError('Keep separate fixed OLD and NEW study files')
    # Old acceptance can be stale because appending trials changes its scope.
    # Its bound study identity is still checked against the existing registry.
    old = load_study(old_source, acceptance_required=False)
    new = load_study(new_source)
    fixed = lambda study: {k: v for k, v in study.items()
                           if not k.startswith('_') and k not in ('trials', 'target_acceptance')}
    if canonical(fixed(old)) != canonical(fixed(new)):
        raise ValueError('Extension must preserve output, worker, devices, budgets and all other study settings')
    count = len(old['trials'])
    if len(new['trials']) <= count or canonical(new['trials'][:count]) != canonical(old['trials']):
        raise ValueError('Extension must only append new trials; every existing trial and its order must remain unchanged')
    return old, new


def expected_record(old, new, reason):
    return {
        'schema_version': VERSION, 'kind': 'append_only_study_extension',
        'old_study_hash': old['_hash'], 'new_study_hash': new['_hash'],
        'old_source': old['_source'], 'new_source': new['_source'],
        'old_source_sha256': file_hash(old['_source']),
        'new_source_sha256': file_hash(new['_source']),
        'added_trial_ids': [t['trial_id'] for t in new['trials'][len(old['trials']):]],
        'reason': reason,
    }


def validate_record(wrapper, expected, methods):
    if not isinstance(wrapper, dict) or set(wrapper) != {'record', 'sha256'}:
        raise ValueError('Invalid study extension record envelope')
    record = wrapper['record']
    if not isinstance(record, dict) or wrapper['sha256'] != digest(record):
        raise ValueError('Study extension record hash mismatch')
    extra = {'created_unix', 'before_costs', 'before_attempt_ids'}
    if set(record) != set(expected) | extra:
        raise ValueError('Study extension record fields changed')
    if canonical({key: record[key] for key in expected}) != canonical(expected):
        raise ValueError('Study extension record differs from the fixed OLD/NEW files or reason')
    number(record['created_unix'], 'extension creation time', True)
    costs = record['before_costs']
    if not isinstance(costs, dict) or set(costs) != {'physical_device_seconds', 'method_attributed_seconds'}:
        raise ValueError('Invalid study extension before-cost snapshot')
    number(costs['physical_device_seconds'], 'extension physical cost')
    shares = costs['method_attributed_seconds']
    if not isinstance(shares, dict) or set(shares) != set(methods):
        raise ValueError('Invalid study extension method cost snapshot')
    for value in shares.values():
        number(value, 'extension method cost')
    ids = record['before_attempt_ids']
    if (not isinstance(ids, list) or any(not isinstance(value, str) or not value for value in ids)
            or ids != sorted(set(ids))):
        raise ValueError('Invalid study extension attempt snapshot')
    return record


def costs(reg):
    physical, methods = reg.costs()
    return {'physical_device_seconds': physical, 'method_attributed_seconds': methods}


def attempt_ids(reg):
    return [row[0] for row in reg.db.execute('SELECT id FROM attempts ORDER BY id')]


def quiescent(reg):
    if reg.db.execute("SELECT 1 FROM attempts WHERE state!='ended' LIMIT 1").fetchone():
        raise ValueError('Study extension requires quiescent attempts; finish or recover active work first')
    if reg.db.execute("SELECT 1 FROM trials WHERE state='running' LIMIT 1").fetchone():
        raise ValueError('Study extension requires reconciled trial states')
    for row in reg.db.execute('SELECT * FROM attempts'):
        with FileLock(row['lease']):
            receipt_valid(row)
    for row in reg.db.execute('SELECT * FROM trials'):
        if row['state'] in SUCCESS:
            verify_reusable(row)


def snapshot_unchanged(reg, record):
    if canonical(costs(reg)) != canonical(record['before_costs']) or attempt_ids(reg) != record['before_attempt_ids']:
        raise ValueError('Accounting or attempts changed during pending study extension')


def write_audit(path, wrapper):
    if path.exists():
        if canonical(read_json(path)) != canonical(wrapper):
            raise ValueError('Refusing to overwrite conflicting immutable study extension record')
    else:
        atomic_json(path, wrapper)


def database_hash(path):
    # URI read-only mode cannot accidentally recreate a deleted registry.
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as db:
        row = db.execute("SELECT value FROM meta WHERE key='hash'").fetchone()
    if row is None:
        raise ValueError('Study registry has no bound hash')
    return row[0]


def commit_extension(reg, new, added):
    """One transaction changes the trial set and its bound study hash together."""
    reg.begin()
    try:
        quiescent(reg)
        for trial in added:
            reg.db.execute('INSERT INTO trials(id,spec,state) VALUES (?,?,?)',
                           (trial['trial_id'], canonical(trial), 'pending'))
        reg.db.execute("UPDATE meta SET value=? WHERE key='hash'", (new['_hash'],))
        reg.study = new
        reg.validate()
        reg.db.commit()
    except BaseException:
        reg.db.rollback()
        raise


def extend(old_path, new_path, reason):
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('A nonempty extension reason is required')
    old, new = plans(old_path, new_path)
    expected = expected_record(old, new, reason)
    root = Path(old['output_root'])
    marker = root / 'study-identity.json'
    registry_path = root / 'registry.sqlite3'
    if not marker.is_file() or not registry_path.is_file():
        raise ValueError('Study extension requires an existing bound registry and identity marker')
    journal_path = root / EXTENSION_JOURNAL
    audit_path = root / 'extensions' / (new['_hash'] + '.json')
    with FileLock(root / '.controller.lock'):
        # Recheck after taking the same lock used by run/report/recover.
        if not marker.is_file() or not registry_path.is_file():
            raise ValueError('Bound study registry or identity marker disappeared')
        wrapper = read_json(journal_path) if journal_path.exists() else None
        if wrapper is not None:
            record = validate_record(wrapper, expected, old['budgets']['per_method_device_seconds'])
            if audit_path.exists() and canonical(read_json(audit_path)) != canonical(wrapper):
                raise ValueError('Pending journal conflicts with immutable extension record')
            bound = database_hash(registry_path)
            if bound not in (old['_hash'], new['_hash']):
                raise ValueError('Pending extension registry hash is neither OLD nor NEW')
            selected = old if bound == old['_hash'] else new
            reg = Registry(selected, extension_recovery=wrapper)
        elif audit_path.exists():
            wrapper = read_json(audit_path)
            record = validate_record(wrapper, expected, old['budgets']['per_method_device_seconds'])
            reg = Registry(new)
            try:
                quiescent(reg)
                current = costs(reg)
                if (not set(record['before_attempt_ids']) <= set(attempt_ids(reg))
                        or current['physical_device_seconds'] < record['before_costs']['physical_device_seconds']
                        or any(current['method_attributed_seconds'][method] < value
                               for method, value in record['before_costs']['method_attributed_seconds'].items())):
                    raise ValueError('Completed extension lost historical attempts or accounting')
                return {'status': 'already_extended', 'study_hash': new['_hash'],
                        'added_trial_ids': expected['added_trial_ids'], 'audit_record': str(audit_path),
                        'costs': current}
            finally:
                reg.close()
        else:
            reg = Registry(old)
            try:
                quiescent(reg)
                record = dict(expected, created_unix=time.time(), before_costs=costs(reg),
                              before_attempt_ids=attempt_ids(reg))
                wrapper = {'record': record, 'sha256': digest(record)}
                atomic_json(journal_path, wrapper)
            except BaseException:
                reg.close()
                raise
        try:
            quiescent(reg)
            snapshot_unchanged(reg, record)
            if reg.study['_hash'] == old['_hash']:
                commit_extension(reg, new, new['trials'][len(old['trials']):])
            snapshot_unchanged(reg, record)
            write_audit(audit_path, wrapper)
            atomic_json(marker, {'schema_version': VERSION, 'study_hash': new['_hash']})
            journal_path.unlink()
            return {'status': 'extended', 'study_hash': new['_hash'],
                    'added_trial_ids': expected['added_trial_ids'], 'audit_record': str(audit_path),
                    'costs': costs(reg)}
        finally:
            reg.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('old_study', type=Path)
    parser.add_argument('new_study', type=Path)
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    try:
        import json
        print(json.dumps(extend(args.old_study, args.new_study, args.reason), indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, sqlite3.DatabaseError, Busy) as error:
        import json
        print(json.dumps({'status': 'blocked', 'error': str(error)}), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
