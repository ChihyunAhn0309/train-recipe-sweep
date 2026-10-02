"""CPU-only staged study extension and interruption recovery regressions."""
import copy
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import study_extend
from study_controller import (
    EXTENSION_JOURNAL, Busy, FileLock, Registry, atomic_json, digest,
    file_hash, load_study, read_json,
)


HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('CONTROLLER_TEST_WORK', tempfile.gettempdir())).resolve()
WORK.mkdir(parents=True, exist_ok=True)


class StudyExtensionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='study-extension-', dir=WORK)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old_path, self.new_path = self.root / 'stage1.json', self.root / 'stage2.json'
        self.output = self.root / 'run'
        self.journal = self.output / EXTENSION_JOURNAL
        self.reason = 'Validation selected the existing recipe; confirm with a fresh seed'
        anchor = {'method': 'full_ft', 'family': 'family', 'roles': ['anchor', 'baseline'],
                  'max_seconds': 2, 'config': {'method': 'full_ft', 'family': 'family', 'seed': 11,
                  'metric': {'name': 'fixture_score', 'direction': 'max'},
                  'fake': {'duration': 0.04}}}
        self.old_raw = {
            'schema_version': 1, 'study_id': 'staged-cpu-fixture', 'mode': 'cpu_fixture',
            'output_root': 'run', 'device_lock_root': 'locks',
            'worker': {'argv': ['{python}', str(HERE / 'fake_worker.py')]},
            'devices': [{'id': 'cpu:extension', 'slots': 1}],
            'budgets': {'total_device_seconds': 30, 'confirmation_reserve_device_seconds': 3,
                        'per_method_device_seconds': {'full_ft': 30},
                        'poll_seconds': 0.03, 'grace_seconds': 0.1, 'terminate_seconds': 0.4},
            'trials': [anchor],
        }
        self.new_raw = copy.deepcopy(self.old_raw)
        confirmation = copy.deepcopy(anchor)
        confirmation['roles'] = ['confirmation']
        confirmation['config']['seed'] = 29
        self.new_raw['trials'].append(confirmation)
        atomic_json(self.old_path, self.old_raw)
        atomic_json(self.new_path, self.new_raw)

    def run_stage(self, path):
        result = subprocess.run([sys.executable, '-B', str(HERE / 'study_controller.py'),
                                 'run', str(path)], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout)

    def extend(self):
        return study_extend.extend(self.old_path, self.new_path, self.reason)

    def db_snapshot(self):
        with closing(sqlite3.connect(self.output / 'registry.sqlite3')) as db:
            return {table: db.execute('SELECT * FROM ' + table + ' ORDER BY 1').fetchall()
                    for table in ('trials', 'attempts', 'segments', 'meta')}

    def leave_before_commit_journal(self):
        with patch('study_extend.commit_extension', side_effect=OSError('injected before commit')):
            with self.assertRaisesRegex(OSError, 'injected before commit'):
                self.extend()
        self.assertTrue(self.journal.is_file())

    def assert_normal_registry_blocked(self):
        for source in (self.old_path, self.new_path):
            with self.assertRaisesRegex(ValueError, 'Pending study extension'):
                Registry(load_study(source))

    def test_two_real_cpu_stages_reuse_anchor_and_preserve_prefix_accounting(self):
        first = self.run_stage(self.old_path)
        before = self.db_snapshot()
        result = self.extend()
        after = self.db_snapshot()
        self.assertEqual(result['status'], 'extended')
        self.assertEqual(after['attempts'], before['attempts'])
        self.assertEqual(after['segments'], before['segments'])
        old_ids = {row[0] for row in before['trials']}
        self.assertEqual([row for row in after['trials'] if row[0] in old_ids], before['trials'])
        self.assertEqual(result['costs']['physical_device_seconds'], first['physical_device_seconds'])
        self.assertFalse(self.journal.exists())
        audit = read_json(result['audit_record'])
        self.assertEqual(audit['sha256'], digest(audit['record']))
        self.assertEqual(audit['record']['before_attempt_ids'], [first['attempts'][0]['id']])
        second = self.run_stage(self.new_path)
        self.assertTrue(second['all_trials_terminal'])
        self.assertEqual(len(second['attempts']), 2)
        old_attempt = next(a for a in second['attempts'] if a['id'] == first['attempts'][0]['id'])
        self.assertEqual(old_attempt, first['attempts'][0])
        self.assertGreater(second['physical_device_seconds'], first['physical_device_seconds'])
        self.assertAlmostEqual(sum(second['method_attributed_seconds'].values()), second['physical_device_seconds'])
        self.assertEqual({t['state'] for t in second['trials']}, {'saturated'})
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            Registry(load_study(self.old_path))

    def test_completed_retry_is_idempotent_before_and_after_stage_two(self):
        self.run_stage(self.old_path)
        first = self.extend()
        audit_bytes = Path(first['audit_record']).read_bytes()
        before = self.db_snapshot()
        self.assertEqual(self.extend()['status'], 'already_extended')
        self.assertEqual(self.db_snapshot(), before)
        self.run_stage(self.new_path)
        after = self.db_snapshot()
        self.assertEqual(self.extend()['status'], 'already_extended')
        self.assertEqual(self.db_snapshot(), after)
        self.assertEqual(Path(first['audit_record']).read_bytes(), audit_bytes)

    def test_changed_settings_or_old_trial_and_removal_are_rejected_without_mutation(self):
        self.run_stage(self.old_path)
        before = self.db_snapshot()
        variants = []
        for key in ('total_device_seconds', 'confirmation_reserve_device_seconds'):
            value = copy.deepcopy(self.new_raw); value['budgets'][key] += 1; variants.append(value)
        value = copy.deepcopy(self.new_raw); value['worker']['argv'].append('--changed'); variants.append(value)
        value = copy.deepcopy(self.new_raw); value['devices'][0]['id'] = 'cpu:other'; variants.append(value)
        value = copy.deepcopy(self.new_raw); value['trials'][0]['max_seconds'] += 1; variants.append(value)
        value = copy.deepcopy(self.new_raw); value['trials'][0]['config']['seed'] += 1; variants.append(value)
        value = copy.deepcopy(self.new_raw); value['trials'][0]['roles'] = ['anchor']; variants.append(value)
        value = copy.deepcopy(self.new_raw); value['trials'].pop(0); variants.append(value)
        value = copy.deepcopy(self.new_raw); value['trials'].reverse(); variants.append(value)
        value = copy.deepcopy(self.new_raw); value['output_root'] = 'other'; variants.append(value)
        for index, value in enumerate(variants):
            with self.subTest(index=index):
                atomic_json(self.new_path, value)
                with self.assertRaises(ValueError):
                    self.extend()
                self.assertEqual(self.db_snapshot(), before)
                self.assertFalse(self.journal.exists())

    def test_no_new_trials_and_same_file_are_rejected(self):
        self.run_stage(self.old_path)
        atomic_json(self.new_path, self.old_raw)
        with self.assertRaisesRegex(ValueError, 'only append'):
            self.extend()
        with self.assertRaisesRegex(ValueError, 'separate fixed'):
            study_extend.extend(self.old_path, self.old_path, self.reason)

    def test_extension_cannot_initialize_or_recreate_a_registry(self):
        with self.assertRaisesRegex(ValueError, 'existing bound registry'):
            self.extend()
        self.assertFalse(self.output.exists())
        self.run_stage(self.old_path)
        (self.output / 'registry.sqlite3').unlink()
        with self.assertRaisesRegex(ValueError, 'existing bound registry'):
            self.extend()
        self.assertFalse((self.output / 'registry.sqlite3').exists())

    def test_live_controller_lock_blocks_extension(self):
        self.run_stage(self.old_path)
        before = self.db_snapshot()
        with FileLock(self.output / '.controller.lock'):
            with self.assertRaises(Busy):
                self.extend()
        self.assertEqual(self.db_snapshot(), before)
        self.assertFalse(self.journal.exists())

    def test_active_attempt_blocks_extension(self):
        old = load_study(self.old_path)
        reg = Registry(old)
        try:
            admitted, _ = reg.admission(old['trials'][0], 'cpu:extension', 0)
            self.assertIsNotNone(admitted)
        finally:
            reg.close()
        before = self.db_snapshot()
        with self.assertRaisesRegex(ValueError, 'quiescent attempts'):
            self.extend()
        self.assertEqual(self.db_snapshot(), before)
        self.assertFalse(self.journal.exists())

    def test_corrupt_successful_checkpoint_blocks_extension(self):
        first = self.run_stage(self.old_path)
        (Path(first['attempts'][0]['output']) / 'checkpoint.json').write_text('corrupt')
        with self.assertRaisesRegex(ValueError, 'integrity'):
            self.extend()
        self.assertFalse(self.journal.exists())

    def test_journal_before_commit_blocks_normal_use_and_retry_preserves_history(self):
        self.run_stage(self.old_path)
        before = self.db_snapshot()
        self.leave_before_commit_journal()
        self.assertEqual(self.db_snapshot(), before)
        self.assert_normal_registry_blocked()
        self.assertEqual(self.extend()['status'], 'extended')
        self.assertEqual(self.db_snapshot()['attempts'], before['attempts'])
        self.assertEqual(self.db_snapshot()['segments'], before['segments'])

    def test_transaction_failure_rolls_back_trial_insert_and_registry_hash(self):
        self.run_stage(self.old_path)
        before = self.db_snapshot()
        new_hash = load_study(self.new_path)['_hash']
        original = Registry.validate

        def reject_new(reg):
            if reg.study['_hash'] == new_hash:
                raise ValueError('injected transaction validation failure')
            return original(reg)

        with patch.object(Registry, 'validate', reject_new):
            with self.assertRaisesRegex(ValueError, 'injected transaction'):
                self.extend()
        self.assertEqual(self.db_snapshot(), before)
        self.assert_normal_registry_blocked()
        self.assertEqual(self.extend()['status'], 'extended')

    def test_journal_after_database_commit_recovers_old_marker_without_reinserting(self):
        first = self.run_stage(self.old_path)
        with patch('study_extend.write_audit', side_effect=OSError('injected after commit')):
            with self.assertRaisesRegex(OSError, 'injected after commit'):
                self.extend()
        self.assertEqual(read_json(self.output / 'study-identity.json')['study_hash'], load_study(self.old_path)['_hash'])
        self.assertEqual(study_extend.database_hash(self.output / 'registry.sqlite3'), load_study(self.new_path)['_hash'])
        self.assert_normal_registry_blocked()
        before_retry = self.db_snapshot()
        self.assertEqual(self.extend()['status'], 'extended')
        self.assertEqual(self.db_snapshot(), before_retry)
        self.assertEqual(len(self.run_stage(self.new_path)['attempts']), len(first['attempts']) + 1)

    def test_journal_after_marker_write_recovers_idempotently(self):
        self.run_stage(self.old_path)
        original = study_extend.atomic_json

        def interrupt_marker(path, value):
            original(path, value)
            if Path(path).name == 'study-identity.json':
                raise OSError('injected after marker')

        with patch('study_extend.atomic_json', side_effect=interrupt_marker):
            with self.assertRaisesRegex(OSError, 'injected after marker'):
                self.extend()
        self.assertEqual(read_json(self.output / 'study-identity.json')['study_hash'], load_study(self.new_path)['_hash'])
        before = self.db_snapshot()
        self.assert_normal_registry_blocked()
        self.assertEqual(self.extend()['status'], 'extended')
        self.assertEqual(self.db_snapshot(), before)

    def test_corrupt_or_mismatched_pending_journal_fails_closed(self):
        self.run_stage(self.old_path)
        self.leave_before_commit_journal()
        original = read_json(self.journal)
        before = self.db_snapshot()
        for repair_hash in (False, True):
            with self.subTest(repair_hash=repair_hash):
                corrupt = copy.deepcopy(original)
                corrupt['record']['reason'] = 'changed request'
                if repair_hash:
                    corrupt['sha256'] = digest(corrupt['record'])
                atomic_json(self.journal, corrupt)
                with self.assertRaisesRegex(ValueError, 'hash mismatch|fixed OLD/NEW'):
                    self.extend()
                self.assertEqual(self.db_snapshot(), before)
                self.assert_normal_registry_blocked()
        atomic_json(self.journal, original)
        self.assertEqual(self.extend()['status'], 'extended')

    def test_fixed_source_file_change_during_pending_extension_is_rejected(self):
        self.run_stage(self.old_path)
        self.leave_before_commit_journal()
        self.new_path.write_text(self.new_path.read_text() + '\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'fixed OLD/NEW'):
            self.extend()
        self.assert_normal_registry_blocked()

    def test_conflicting_immutable_audit_cannot_be_overwritten(self):
        self.run_stage(self.old_path)
        with patch('study_extend.write_audit', side_effect=OSError('injected after commit')):
            with self.assertRaises(OSError):
                self.extend()
        audit = self.output / 'extensions' / (load_study(self.new_path)['_hash'] + '.json')
        atomic_json(audit, {'unexpected': 'preserve me'})
        before = audit.read_bytes()
        with self.assertRaisesRegex(ValueError, 'conflicts with immutable'):
            self.extend()
        self.assertEqual(audit.read_bytes(), before)
        self.assertTrue(self.journal.exists())

    def test_extension_does_not_reset_spending_to_admit_confirmation(self):
        first = self.run_stage(self.old_path)
        budget = self.old_raw['budgets']
        tail = budget['grace_seconds'] + 2 * budget['terminate_seconds'] + 2 * budget['poll_seconds']
        self.new_raw['trials'][-1]['max_seconds'] = budget['total_device_seconds'] - tail - first['physical_device_seconds'] / 2
        atomic_json(self.new_path, self.new_raw)
        self.extend()
        second = self.run_stage(self.new_path)
        self.assertEqual(second['attempts'], first['attempts'])
        self.assertEqual(second['physical_device_seconds'], first['physical_device_seconds'])
        pending = [t for t in second['trials'] if t['state'] == 'pending']
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]['reason'], 'total_budget_or_confirmation_reserve')

    def test_new_gpu_stage_requires_new_acceptance_even_when_old_verification_is_skipped(self):
        # Schema rejection only; never launch the fake worker in GPU mode.
        for path, raw in ((self.old_path, self.old_raw), (self.new_path, self.new_raw)):
            raw['mode'] = 'gpu'
            raw['devices'][0]['id'] = 'GPU-12345678-1234-1234-1234-123456789abc'
            raw['worker']['code_artifacts'] = [{'path': 'worker-code.txt', 'sha256': '', 'bytes': 4}]
            code = self.root / 'worker-code.txt'
            code.write_bytes(b'code')
            raw['worker']['code_artifacts'][0]['sha256'] = file_hash(code)
            atomic_json(path, raw)
        self.assertEqual(load_study(self.old_path, acceptance_required=False)['mode'], 'gpu')
        with self.assertRaisesRegex(ValueError, 'requires hashed target_acceptance'):
            self.extend()
        self.assertFalse(self.output.exists())

    def test_cli_executes_extension_and_retry(self):
        self.run_stage(self.old_path)
        command = [sys.executable, '-B', str(HERE / 'study_extend.py'), str(self.old_path),
                   str(self.new_path), '--reason', self.reason]
        for expected in ('extended', 'already_extended'):
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads(result.stdout)['status'], expected)


if __name__ == '__main__':
    unittest.main()
