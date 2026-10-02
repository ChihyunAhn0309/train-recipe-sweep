#!/usr/bin/env python3
"""CPU-only protocol fixture. Never a model trainer or GPU benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from study_controller import atomic_json, read_json


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'output', 'control'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    config, control = read_json(args.config), read_json(args.control)
    if control['mode'] != 'cpu_fixture':
        raise ValueError('fake_worker refuses GPU mode: it is only a CPU protocol fixture')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    fake = config.get('fake', {})
    behavior = fake.get('behavior', 'normal')
    atomic_json(out / 'started.json', {'pid': os.getpid(), 'attempt_id': control['attempt_id'], 'resumed': bool(control['resume_manifest'])})
    if behavior == 'no_result':
        return 0
    if behavior == 'descendant':
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        atomic_json(out / 'descendant.json', {'pid': child.pid})
    duration = fake.get('duration', 0.2)
    deadline = time.monotonic()+duration
    interrupted = behavior == 'interrupt_once' and not control['resume_manifest']
    stopped = False
    while time.monotonic() < deadline:
        if behavior != 'ignore_stop' and Path(control['stop_file']).exists():
            stopped = True; break
        time.sleep(0.01)
    (out / 'checkpoint.json').write_text(json.dumps({'fixture_weights': [1, 2], 'step': 1}), encoding='utf-8')
    (out / 'training-state.json').write_text(json.dumps({'optimizer': 'fixture', 'step': 1}), encoding='utf-8')
    (out / 'metrics.jsonl').write_text(json.dumps({'step': 1, 'fixture_score': 1.0})+'\n', encoding='utf-8')
    entries = []
    for path, role in [('checkpoint.json', 'checkpoint'), ('training-state.json', 'training_state'), ('metrics.jsonl', 'metrics')]:
        data = (out / path).read_bytes()
        entries.append({'path': path, 'role': role, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
    resume = {'schema_version': 1, 'trial_id': control['trial_id'], 'artifacts': entries}
    atomic_json(out / 'resume.json', fake.get('resume_override', resume))
    status = 'interrupted' if interrupted or stopped else fake.get('status', 'saturated')
    result = {'trial_id': control['trial_id'], 'attempt_id': control['attempt_id'], 'status': status,
              'reason': 'CPU fixture only', 'metric': {'name': 'fixture_score', 'direction': 'max', 'value': 1.0},
              'artifacts': entries}
    if behavior == 'bad_hash':
        result['artifacts'][0]['sha256'] = '0'*64
    if behavior == 'wrong_id':
        result['trial_id'] = '0'*64
    if 'metric_override' in fake:
        result['metric'] = fake['metric_override']
    if 'artifact_role_override' in fake:
        result['artifacts'][0]['role'] = fake['artifact_role_override']
    if fake.get('omit_metric'):
        result.pop('metric', None)
    atomic_json(out / 'result.json', fake.get('result_override', result))
    return int(fake.get('exit_code', 0))


if __name__ == '__main__':
    raise SystemExit(main())
