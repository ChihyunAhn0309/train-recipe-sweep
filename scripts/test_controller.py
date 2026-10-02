"""CPU fixtures with actual child processes; no GPU claims."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

from study_controller import Registry, FileLock, Busy, atomic_json, digest, file_hash, load_study

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('CONTROLLER_TEST_WORK', tempfile.gettempdir())).resolve()
WORK.mkdir(parents=True, exist_ok=True)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='controller-test-', dir=WORK)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.study_path = self.root / 'study.json'

    def trial(self, label='anchor', method='full_ft', roles=None, **fake):
        return {'method': method, 'family': 'family', 'roles': roles or ['anchor'], 'max_seconds': 2,
                'config': {'method': method, 'family': 'family', 'seed': label, 'schedule': {'immutable_steps': 1},
                           'metric': {'name': 'fixture_score', 'direction': 'max'},
                           'fake': dict(duration=0.15, **fake)}}

    def study(self, trials=None, slots=1, **budgets):
        study = {'schema_version': 1, 'study_id': 'cpu-protocol-test', 'mode': 'cpu_fixture',
                 'output_root': 'run', 'device_lock_root': 'device-locks',
                 'worker': {'argv': ['{python}', str(HERE / 'fake_worker.py')]},
                 'devices': [{'id': 'cpu:fixture', 'slots': slots}],
                 'budgets': dict(total_device_seconds=60, confirmation_reserve_device_seconds=5,
                                 per_method_device_seconds={'full_ft': 50, 'lora_r16': 50},
                                 poll_seconds=0.03, grace_seconds=0.1, terminate_seconds=0.4, **budgets),
                 'trials': trials or [self.trial()]}
        if slots > 1:
            measurement = {'device_id': 'cpu:fixture', 'slots': slots, 'kind': 'cpu_fixture',
                           'peak_memory_bytes': 1, 'available_memory_bytes': 2,
                           'aggregate_samples_per_second': 1}
            atomic_json(self.root / 'packing.json', measurement)
            study['devices'][0]['packing_evidence'] = {'path': 'packing.json', 'sha256': file_hash(self.root / 'packing.json')}
        atomic_json(self.study_path, study)
        return study

    def command(self, *args):
        return [sys.executable, '-B', str(HERE / 'study_controller.py'), *args, str(self.study_path)]

    def run_cli(self, command='run', resume=False, timeout=15):
        cmd = self.command(command)
        if resume:
            cmd.append('--resume')
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def successful(self, **kwargs):
        result = self.run_cli(**kwargs)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return json.loads(result.stdout)

    def db(self):
        db = sqlite3.connect(self.root / 'run/registry.sqlite3')
        self.addCleanup(db.close)
        return db

    def test_real_worker_result_integrity_and_completed_reuse(self):
        self.study()
        first = self.successful()
        self.assertEqual(first['trials'][0]['state'], 'saturated')
        self.assertGreater(first['physical_device_seconds'], 0)
        again = self.successful()
        self.assertEqual(len(again['attempts']), 1)
        self.assertEqual(first['physical_device_seconds'], again['physical_device_seconds'])

    def test_exit_zero_without_result_and_bad_hash_are_failed(self):
        trials = [self.trial('missing', behavior='no_result'), self.trial('corrupt', behavior='bad_hash')]
        self.study(trials)
        result = self.successful()
        self.assertTrue(all(t['state'] == 'failed' for t in result['trials']))

    def test_duplicate_config_dispatches_only_once(self):
        trial = self.trial()
        self.study([trial, copy.deepcopy(trial)])
        result = self.successful()
        self.assertEqual(len(result['trials']), 1)
        self.assertEqual(len(result['attempts']), 1)
        self.assertEqual(len(result['trials'][0]['id']), 64)

    def test_resume_requires_hashed_complete_state_and_retains_attempts(self):
        self.study([self.trial(behavior='interrupt_once')])
        first = self.successful()
        self.assertEqual(first['trials'][0]['state'], 'interrupted')
        again = self.successful(resume=True)
        self.assertEqual(again['trials'][0]['state'], 'saturated')
        self.assertEqual(len(again['attempts']), 2)
        self.assertGreater(again['physical_device_seconds'], first['physical_device_seconds'])
        started = json.loads((Path(again['attempts'][-1]['output']) / 'started.json').read_text())
        self.assertTrue(started['resumed'])

    def test_anchor_dependency_and_explicit_censored_fallback(self):
        anchor = self.trial('anchor', status='right_censored')
        blocked = self.trial('blocked', roles=['candidate'])
        allowed = self.trial('allowed', roles=['candidate'])
        allowed['anchor_fallback'] = 'allow_censored'
        self.study([blocked, allowed, anchor])
        result = self.successful()
        states = {json.loads(row[0])['config']['seed']: row[1] for row in self.db().execute('SELECT spec,state FROM trials')}
        self.assertEqual(states, {'blocked': 'pending', 'allowed': 'saturated', 'anchor': 'right_censored'})
        self.assertEqual(len(result['attempts']), 2)

    def test_confirmation_reserve_and_method_budget_block_admission(self):
        a = self.trial('anchor')
        confirm = self.trial('confirm', roles=['confirmation'])
        self.study([a, confirm])
        raw = json.loads(self.study_path.read_text())
        raw['budgets']['total_device_seconds'] = 7
        raw['budgets']['confirmation_reserve_device_seconds'] = 5
        atomic_json(self.study_path, raw)
        result = self.successful()
        self.assertEqual(len(result['attempts']), 0)
        self.assertIn('confirmation_reserve', result['trials'][0]['reason'])
        # A separate identity/output checks per-method admission without resuming
        # a changed study into old accounting.
        raw['output_root'] = 'run-method'
        raw['budgets']['total_device_seconds'] = 60
        raw['budgets']['per_method_device_seconds']['full_ft'] = 1
        atomic_json(self.study_path, raw)
        result = self.successful()
        self.assertEqual(len(result['attempts']), 0)
        self.assertEqual(result['trials'][0]['reason'], 'method_budget')

    def test_packed_physical_union_is_not_sum_of_worker_times(self):
        self.study([self.trial('one'), self.trial('two', method='lora_r16')], slots=2)
        result = self.successful()
        self.assertTrue(all(t['state'] == 'saturated' for t in result['trials']))
        attempts = result['attempts']
        summed = sum(a['ended'] - a['started'] for a in attempts)
        union = max(a['ended'] for a in attempts) - min(a['started'] for a in attempts)
        self.assertLess(result['physical_device_seconds'], summed)
        self.assertAlmostEqual(result['physical_device_seconds'], union, delta=0.04)
        self.assertAlmostEqual(sum(result['method_attributed_seconds'].values()), result['physical_device_seconds'], places=5)

    def test_missing_registry_and_deleted_accounting_segment_fail_closed(self):
        self.study()
        self.successful()
        db = self.db()
        db.execute('DELETE FROM segments WHERE id=(SELECT MIN(id) FROM segments)'); db.commit()
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn('accounting', result.stderr)
        db.close()
        registry = self.root / 'run/registry.sqlite3'
        registry.unlink()
        result = self.run_cli()
        self.assertEqual(result.returncode, 2)
        self.assertIn('lost registry', result.stderr)

    def test_corrupt_completed_checkpoint_blocks_reuse(self):
        self.study()
        result = self.successful()
        checkpoint = Path(result['attempts'][0]['output']) / 'checkpoint.json'
        checkpoint.write_bytes(b'corrupted')
        retry = self.run_cli()
        self.assertEqual(retry.returncode, 2)
        self.assertIn('integrity', retry.stderr.lower())

    def test_live_controller_lock_prevents_duplicate_dispatch(self):
        trial = self.trial(); trial['config']['fake']['duration'] = 0.6
        self.study([trial])
        child = subprocess.Popen(self.command('run'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic()+5
            while not list((self.root / 'run').glob('trials/*/attempts/*/started.json')):
                if child.poll() is not None or time.monotonic() > deadline:
                    self.fail('First worker did not start: '+repr(child.communicate(timeout=1)))
                time.sleep(0.02)
            other = self.run_cli()
            self.assertEqual(other.returncode, 2)
            self.assertIn('Lock is held', other.stderr)
            out, err = child.communicate(timeout=10)
            self.assertEqual(child.returncode, 0, err)
            self.assertEqual(len(json.loads(out)['attempts']), 1)
        finally:
            if child.poll() is None:
                child.kill(); child.communicate(timeout=10)

    def test_controller_kill_does_not_reclaim_live_orphan_and_later_recovers(self):
        trial = self.trial(); trial['config']['fake']['duration'] = 0.8
        self.study([trial])
        child = subprocess.Popen(self.command('run'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic()+5
            while not list((self.root / 'run').glob('trials/*/attempts/*/started.json')):
                if child.poll() is not None or time.monotonic() > deadline:
                    self.fail('Worker did not start: '+repr(child.communicate(timeout=1)))
                time.sleep(0.02)
            child.kill(); child.communicate(timeout=5)
            immediate = self.run_cli()
            self.assertEqual(immediate.returncode, 2)
            self.assertIn('Live orphan', immediate.stderr)
            time.sleep(1)
            recovered = self.successful()
            self.assertEqual(recovered['trials'][0]['state'], 'saturated')
            self.assertEqual(len(recovered['attempts']), 1)
        finally:
            if child.poll() is None:
                child.kill(); child.communicate(timeout=5)

    def test_uncooperative_worker_is_stopped_within_bounded_allowance(self):
        trial = self.trial(behavior='ignore_stop'); trial['config']['fake']['duration'] = 30
        trial['max_seconds'] = 0.2
        self.study([trial])
        start = time.monotonic()
        result = self.successful(timeout=10)
        self.assertLess(time.monotonic()-start, 5)
        self.assertEqual(result['trials'][0]['state'], 'budget_exhausted')
        receipt = json.loads(Path(result['attempts'][0]['receipt']).read_text())
        self.assertTrue(receipt['tree_terminated'])

    def test_descendants_are_reconciled_before_termination_ack(self):
        self.study([self.trial(behavior='descendant')])
        result = self.successful()
        self.assertEqual(result['trials'][0]['state'], 'saturated')
        receipt = json.loads(Path(result['attempts'][0]['receipt']).read_text())
        self.assertTrue(receipt['tree_terminated'])
        pid = json.loads((Path(result['attempts'][0]['output'])/'descendant.json').read_text())['pid']
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.restype = wintypes.HANDLE
            handle = kernel.OpenProcess(0x1000, False, pid)
            if handle:
                code = wintypes.DWORD()
                kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
                kernel.GetExitCodeProcess(handle, ctypes.byref(code))
                kernel.CloseHandle.argtypes = [wintypes.HANDLE]; kernel.CloseHandle(handle)
                self.assertNotEqual(code.value, 259)
        else:
            stat = Path('/proc') / str(pid) / 'stat'
            self.assertTrue(not stat.exists() or stat.read_text().split(') ')[1].startswith('Z '))

    def test_two_database_clients_cannot_oversubscribe_reserved_budget(self):
        self.study([self.trial('one'), self.trial('two')], slots=2)
        raw = json.loads(self.study_path.read_text())
        raw['budgets']['total_device_seconds'] = 4
        raw['budgets']['confirmation_reserve_device_seconds'] = 0.1
        atomic_json(self.study_path, raw)
        reg = Registry(load_study(self.study_path)); reg.close()
        code = ('import sys,json;sys.path.insert(0,sys.argv[1]);'
                'from study_controller import Registry,load_study;'
                's=load_study(sys.argv[2]);r=Registry(s);'
                'a,why=r.admission(s["trials"][int(sys.argv[3])],"cpu:fixture",int(sys.argv[3]));'
                'print(json.dumps({"admitted":a is not None,"reason":why}));r.close()')
        processes = [subprocess.Popen([sys.executable, '-B', '-c', code, str(HERE), str(self.study_path), str(i)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(2)]
        outcomes = []
        for proc in processes:
            out, err = proc.communicate(timeout=10)
            self.assertEqual(proc.returncode, 0, err)
            outcomes.append(json.loads(out))
        self.assertEqual(sum(item['admitted'] for item in outcomes), 1)

    def test_restart_cannot_reset_spending_to_admit_more_work(self):
        trial = self.trial(status='interrupted')
        trial['max_seconds'] = 0.6
        trial['config']['fake']['duration'] = 0.45
        self.study([trial])
        raw = json.loads(self.study_path.read_text())
        raw['budgets']['total_device_seconds'] = 3
        raw['budgets']['confirmation_reserve_device_seconds'] = 1
        atomic_json(self.study_path, raw)
        first = self.successful()
        self.assertEqual(first['trials'][0]['state'], 'interrupted')
        second = self.successful(resume=True)
        self.assertEqual(len(second['attempts']), 1)
        self.assertEqual(second['trials'][0]['state'], 'pending')
        self.assertIn('confirmation_reserve', second['trials'][0]['reason'])
        self.assertEqual(first['physical_device_seconds'], second['physical_device_seconds'])

    def test_missing_device_ownership_record_blocks_unsafe_reuse(self):
        self.study()
        self.successful()
        owner = next((self.root / 'device-locks').glob('*/slot-0.owner.json'))
        owner.unlink()
        raw = json.loads(self.study_path.read_text())
        raw['output_root'] = 'second-run'
        atomic_json(self.study_path, raw)
        second = self.successful()
        self.assertEqual(second['attempts'], [])
        self.assertEqual(second['trials'][0]['reason'], 'device_unavailable_or_unacknowledged')

    def test_gpu_mode_rejects_logical_indices_and_unmeasured_packing(self):
        self.study()
        raw = json.loads(self.study_path.read_text())
        raw['mode'] = 'gpu'; raw['devices'][0]['id'] = '0'
        local_code = self.root / 'worker.txt'; local_code.write_text('schema test only')
        raw['worker']['code_artifacts'] = [{'path': 'worker.txt', 'sha256': file_hash(local_code), 'bytes': local_code.stat().st_size}]
        atomic_json(self.study_path, raw)
        bad = self.run_cli('validate')
        self.assertEqual(bad.returncode, 2)
        self.assertIn('physical GPU UUID', bad.stderr)
        raw['devices'][0].update(id='GPU-00000000-0000-0000-0000-000000000001', slots=2)
        atomic_json(self.study_path, raw)
        bad = self.run_cli('validate')
        self.assertEqual(bad.returncode, 2)
        self.assertIn('hashed measurement', bad.stderr)

    def test_nonanchor_baseline_waits_for_own_anchor_on_two_slots(self):
        anchor = self.trial('anchor'); anchor['config']['fake']['duration'] = 0.3
        baseline = self.trial('baseline', roles=['baseline'])
        self.study([baseline, anchor], slots=2)
        result = self.successful()
        records = {a['trial_id']: a for a in result['attempts']}
        self.assertGreaterEqual(records[digest(baseline['config'])]['started'], records[digest(anchor['config'])]['ended'])

    def test_censored_evaluation_exposes_metric_and_checkpoint_without_success_label(self):
        self.study([self.trial(status='right_censored')])
        result = self.successful()
        trial = result['trials'][0]
        self.assertEqual(trial['state'], 'right_censored')
        self.assertEqual(trial['evaluated_result']['status'], 'right_censored')
        self.assertTrue(trial['evaluated_result']['integrity_verified'])
        self.assertTrue(trial['evaluated_result']['checkpoint_available'])
        self.assertEqual(trial['evaluated_result']['metric']['value'], 1.0)

    def test_gpu_acceptance_schema_only_missing_stale_and_corrupt_evidence(self):
        # Deliberately synthetic schema fixture. This test never launches a GPU
        # worker and is not evidence of any real acceptance gate passing.
        from study_controller import GATES, acceptance_scope
        self.study()
        raw = json.loads(self.study_path.read_text())
        raw['mode'] = 'gpu'; raw['devices'][0]['id'] = 'GPU-00000000-0000-0000-0000-000000000001'
        evidence = self.root / 'evidence.txt'; evidence.write_text('SYNTHETIC SCHEMA TEST; NO GPU MEASUREMENT')
        entry = {'path': 'evidence.txt', 'sha256': file_hash(evidence), 'bytes': evidence.stat().st_size}
        raw['worker']['code_artifacts'] = [entry]
        atomic_json(self.study_path, raw)
        self.assertEqual(self.run_cli('validate').returncode, 2)
        scope = acceptance_scope(load_study(self.study_path, acceptance_required=False))
        acceptance = {'schema_version': 1, 'kind': 'gpu_acceptance', 'execution_mode': 'gpu',
                      'scope_sha256': scope, 'device_ids': [raw['devices'][0]['id']],
                      'gates': {name: {'status': 'passed', 'evidence': [entry]} for name in GATES}}
        accept_path = self.root / 'acceptance.json'
        atomic_json(accept_path, acceptance)
        raw['target_acceptance'] = {'path': 'acceptance.json', 'sha256': file_hash(accept_path)}
        atomic_json(self.study_path, raw)
        accepted = self.run_cli('validate')
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertFalse(json.loads(accepted.stdout)['gpu_verified'])
        for change in ('na', 'cpu'):
            invalid = copy.deepcopy(acceptance)
            if change == 'na':
                invalid['gates']['hardware_environment'].update(status='not_applicable', reason='No GPU available')
            else:
                invalid['kind'] = 'cpu_fixture'
            atomic_json(accept_path, invalid)
            altered = copy.deepcopy(raw)
            altered['target_acceptance']['sha256'] = file_hash(accept_path)
            atomic_json(self.study_path, altered)
            self.assertEqual(self.run_cli('validate').returncode, 2)
        atomic_json(accept_path, acceptance)
        atomic_json(self.study_path, raw)
        changed = copy.deepcopy(raw); changed['trials'][0]['config']['seed'] = 'changed'
        atomic_json(self.study_path, changed)
        self.assertIn('scope is stale', self.run_cli('validate').stderr)
        atomic_json(self.study_path, raw)
        evidence.write_text('corrupt')
        self.assertEqual(self.run_cli('validate').returncode, 2)

    def test_faulty_metrics_are_quarantined_other_family_finishes_and_rerun_is_stable(self):
        metrics = [[], None, 'score', True, {},
                   {'name': 'fixture_score', 'direction': 'max'},
                   {'name': 'fixture_score', 'direction': 'max', 'value': True},
                   {'name': 'fixture_score', 'direction': 'max', 'value': []},
                   {'name': 'fixture_score', 'direction': 'max', 'value': 10**400}]
        trials = [self.trial('bad-'+str(i), metric_override=value) for i, value in enumerate(metrics)]
        good = self.trial('good', method='lora_r16')
        good['family'] = good['config']['family'] = 'unrelated'
        trials.append(good)
        for trial in trials:
            trial['config']['fake']['duration'] = 0.01
        self.study(trials, slots=2)
        first = self.successful(timeout=30)
        for trial in first['trials']:
            if trial['id'] == digest(good['config']):
                self.assertEqual(trial['state'], 'saturated')
            else:
                self.assertEqual(trial['state'], 'failed')
                self.assertTrue(trial['reason'].startswith('invalid_result:'), trial['reason'])
                self.assertIsNone(trial['evaluated_result'])
        second = self.successful()
        self.assertEqual(len(second['attempts']), len(first['attempts']))
        self.assertEqual([t['state'] for t in first['trials']], [t['state'] for t in second['trials']])

    def test_partial_outcomes_must_match_objective_before_evaluation_reporting(self):
        states = ['right_censored', 'budget_exhausted', 'interrupted', 'pruned', 'failed']
        trials = [self.trial('bad-'+state, status=state,
                            metric_override={'name': 'unrelated_metric', 'direction': 'min', 'value': 0.01})
                  for state in states]
        good = self.trial('valid-censored', method='lora_r16', status='right_censored')
        trials.append(good)
        for trial in trials:
            trial['config']['fake']['duration'] = 0.01
        self.study(trials, slots=2)
        result = self.successful(timeout=30)
        for trial in result['trials']:
            if trial['id'] == digest(good['config']):
                self.assertEqual(trial['state'], 'right_censored')
                self.assertTrue(trial['evaluated_result']['integrity_verified'])
            else:
                self.assertEqual(trial['state'], 'failed')
                self.assertIn('immutable objective', trial['reason'])
                self.assertIsNone(trial['evaluated_result'])

    def test_malformed_result_and_artifact_role_do_not_poison_controller(self):
        trials = [self.trial('result-'+str(i), result_override=value) for i, value in enumerate([[], None, 'text', True])]
        trials += [self.trial('role-'+str(i), artifact_role_override=value) for i, value in enumerate([[], {}])]
        good = self.trial('good', method='lora_r16'); trials.append(good)
        for trial in trials:
            trial['config']['fake']['duration'] = 0.01
        self.study(trials, slots=2)
        result = self.successful(timeout=30)
        self.assertEqual(sum(t['state'] == 'failed' for t in result['trials']), 6)
        self.assertEqual(sum(t['state'] == 'saturated' for t in result['trials']), 1)
        self.assertTrue(all(t['reason'].startswith('invalid_result:') for t in result['trials'] if t['state'] == 'failed'))
        self.assertEqual(len(self.successful()['attempts']), len(result['attempts']))

    def test_malformed_resume_is_quarantined_and_never_admitted_as_resumable(self):
        trials = [self.trial('resume-'+str(i), resume_override=value)
                  for i, value in enumerate([[], None, 'text', True, {'artifacts': []}])]
        good = self.trial('good', method='lora_r16'); trials.append(good)
        for trial in trials:
            trial['config']['fake']['duration'] = 0.01
        self.study(trials, slots=2)
        result = self.successful(timeout=30)
        failed = [t for t in result['trials'] if t['state'] == 'failed']
        self.assertEqual(len(failed), 5)
        for trial in failed:
            self.assertTrue(trial['reason'].startswith('invalid_resume:'), trial['reason'])
            self.assertIsNone(trial['resume_path'])
            self.assertIsNone(trial['evaluated_result'])
        self.assertEqual(len(self.successful(resume=True)['attempts']), len(result['attempts']))

    def test_nonevaluated_partial_result_has_no_verified_metric_claim(self):
        self.study([self.trial(status='interrupted', omit_metric=True)])
        result = self.successful()
        trial = result['trials'][0]
        self.assertEqual(trial['state'], 'interrupted')
        self.assertIsNone(trial['evaluated_result'])
        self.assertTrue(trial['result_artifacts'])

    def test_deduplication_cannot_merge_confirmation_with_anchor(self):
        anchor = self.trial('same')
        confirmation = copy.deepcopy(anchor); confirmation['roles'] = ['confirmation']
        self.study([anchor, confirmation])
        for command in ('validate', 'run'):
            result = self.run_cli(command)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Deduplicated confirmation', result.stderr)
        self.assertFalse((self.root/'run/registry.sqlite3').exists())

    def test_duplicate_role_names_normalize_without_changing_confirmation_semantics(self):
        anchor = self.trial('anchor'); anchor['roles'] = ['anchor', 'anchor']
        confirmation = self.trial('confirmation', roles=['confirmation', 'confirmation'])
        self.study([anchor, confirmation, copy.deepcopy(confirmation)])
        result = self.successful()
        self.assertEqual(len(result['trials']), 2)
        attempts = {a['trial_id']: a for a in result['attempts']}
        self.assertEqual(attempts[digest(confirmation['config'])]['confirmation'], 1)
        self.assertEqual(attempts[digest(anchor['config'])]['confirmation'], 0)

    def test_transient_slot_probe_cannot_poison_an_admitted_real_worker(self):
        from study_controller import device_paths, device_policy, reconcile
        self.study()
        study = load_study(self.study_path)
        device_policy(study)
        reg = Registry(study)
        self.addCleanup(reg.close)
        trial = study['trials'][0]
        attempt, why = reg.admission(trial, 'cpu:fixture', 0)
        self.assertIsNone(why)
        out = Path(attempt['output']); out.mkdir(parents=True)
        atomic_json(out.parents[1]/'config.json', trial['config'])
        atomic_json(out/'control.json', {'schema_version': 1, 'trial_id': trial['trial_id'],
                    'attempt_id': attempt['id'], 'stop_file': str(out/'stop.json'), 'resume_manifest': None,
                    'device_id': 'cpu:fixture', 'mode': 'cpu_fixture'})
        _, slot, _ = device_paths(study, 'cpu:fixture', 0)
        process = None
        try:
            with FileLock(slot):
                process = subprocess.Popen([sys.executable, '-B', str(HERE/'study_controller.py'),
                                             '_worker', str(self.study_path), attempt['id']],
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                deadline = time.monotonic()+5
                while not Path(attempt['lease']).exists():
                    if process.poll() is not None or time.monotonic() > deadline:
                        self.fail('Runner did not reach lease acquisition')
                    time.sleep(0.005)
                # Deliberately keep the harmless availability probe held while
                # the new runner first tries the slot. It must wait, not fail.
                time.sleep(0.08)
                self.assertIsNone(process.poll())
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, stderr+stdout)
            reconcile(reg, dict(reg.db.execute('SELECT * FROM attempts WHERE id=?', (attempt['id'],)).fetchone()))
            self.assertEqual(reg.db.execute('SELECT state FROM trials').fetchone()[0], 'saturated')
        finally:
            if process is not None and process.poll() is None:
                process.kill(); process.communicate(timeout=5)

    def test_gpu_acceptance_schema_only_controller_code_change_invalidates_scope(self):
        # Synthetic validation-only fixture. No GPU worker is launched.
        from study_controller import GATES, acceptance_scope
        self.study()
        raw = json.loads(self.study_path.read_text())
        raw['mode'] = 'gpu'
        raw['devices'][0]['id'] = 'GPU-00000000-0000-0000-0000-000000000001'
        evidence = self.root/'evidence.txt'
        evidence.write_text('SYNTHETIC SCHEMA TEST; NO GPU MEASUREMENT')
        entry = {'path': 'evidence.txt', 'sha256': file_hash(evidence), 'bytes': evidence.stat().st_size}
        raw['worker']['code_artifacts'] = [entry]
        atomic_json(self.study_path, raw)
        original_scope = acceptance_scope(load_study(self.study_path, acceptance_required=False))
        acceptance = {'schema_version': 1, 'kind': 'gpu_acceptance', 'execution_mode': 'gpu',
                      'scope_sha256': original_scope, 'device_ids': [raw['devices'][0]['id']],
                      'gates': {name: {'status': 'passed', 'evidence': [entry]} for name in GATES}}
        accept_path = self.root/'acceptance.json'; atomic_json(accept_path, acceptance)
        raw['target_acceptance'] = {'path': 'acceptance.json', 'sha256': file_hash(accept_path)}
        atomic_json(self.study_path, raw)
        copied = self.root/'copied_controller.py'
        copied.write_bytes((HERE/'study_controller.py').read_bytes())
        def copied_cli(command):
            return subprocess.run([sys.executable, '-B', str(copied), command, str(self.study_path)],
                                  capture_output=True, text=True, timeout=10)
        same = copied_cli('validate')
        self.assertEqual(same.returncode, 0, same.stderr)
        with copied.open('ab') as stream:
            stream.write(b'\n# Benign code change: must invalidate prior target acceptance.\n')
        changed_scope = copied_cli('scope')
        self.assertEqual(changed_scope.returncode, 0, changed_scope.stderr)
        self.assertNotEqual(json.loads(changed_scope.stdout)['scope_sha256'], original_scope)
        refused = copied_cli('validate')
        self.assertEqual(refused.returncode, 2)
        self.assertIn('scope is stale', refused.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
