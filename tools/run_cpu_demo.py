"""Run the documented CPU protocol demo in a new, self-contained directory."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[1] / 'scripts'
    for name in ('study_controller.py', 'fake_worker.py', 'example-study.json'):
        shutil.copyfile(source / name, destination / name)
    command = [sys.executable, '-B', str(destination / 'study_controller.py')]
    study = str(destination / 'example-study.json')

    def invoke(*arguments):
        result = subprocess.run(command + list(arguments), check=True,
                                capture_output=True, text=True, timeout=90)
        return json.loads(result.stdout)

    invoke('validate', study)
    first = invoke('run', study)
    second = invoke('run', study, '--resume')
    if len(first['attempts']) != 4 or len(second['attempts']) != 4:
        raise RuntimeError('The demo did not execute exactly four reusable trials')
    if first['physical_device_seconds'] != second['physical_device_seconds']:
        raise RuntimeError('Completed reuse changed the accumulated fixture cost')
    states = [trial['state'] for trial in second['trials']]
    if sorted(states) != ['completed', 'completed', 'saturated', 'saturated']:
        raise RuntimeError('Unexpected synthetic trial states: ' + repr(states))
    print(json.dumps({'kind': 'cpu_fixture', 'actual_gpu_training': False,
                      'trials': len(states), 'states': states,
                      'duplicate_attempts_on_resume': 0,
                      'fixture_device_seconds': second['physical_device_seconds'],
                      'report': str(destination / 'example-run/report.json')}, indent=2))


if __name__ == '__main__':
    main()
