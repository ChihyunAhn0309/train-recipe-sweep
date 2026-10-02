"""Failure-path checks using real files, processes, and output state."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from runtime_guard import BudgetExceeded, LocalStudy, atomic_json, sha256, verify_assets


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def manifest(self):
        path = self.root / 'assets/model/weights.bin'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'correct weights')
        return {'assets': [{'kind': 'model', 'filename': 'weights.bin',
                            'sha256': sha256(path), 'bytes': path.stat().st_size,
                            'local_path': 'historical/invalid/path'}]}

    def test_actual_loader_path_is_verified_after_relocation(self):
        manifest = self.manifest()
        result = verify_assets(manifest, self.root / 'assets')
        self.assertEqual(result['model', 'weights.bin'].read_bytes(), b'correct weights')

    def test_valid_historical_file_cannot_hide_corrupt_loader_file(self):
        manifest = self.manifest()
        historical = self.root / 'historical.bin'
        historical.write_bytes(b'correct weights')
        manifest['assets'][0]['local_path'] = str(historical)
        (self.root / 'assets/model/weights.bin').write_bytes(b'corrupt weights')
        with self.assertRaises(ValueError):
            verify_assets(manifest, self.root / 'assets')

    def test_duplicate_and_traversal_paths_rejected(self):
        manifest = self.manifest()
        duplicate = copy.deepcopy(manifest)
        duplicate['assets'].append(duplicate['assets'][0].copy())
        with self.assertRaises(ValueError):
            verify_assets(duplicate, self.root / 'assets')
        for filename in ('../weights.bin', '/weights.bin', 'C:/weights.bin', '..\\weights.bin'):
            manifest['assets'][0]['filename'] = filename
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                verify_assets(manifest, self.root / 'assets')

    def test_failed_atomic_write_preserves_existing_file(self):
        path = self.root / 'state.json'
        atomic_json(path, {'valid': True})
        before = path.read_bytes()
        with self.assertRaises(ValueError):
            atomic_json(path, {'invalid': float('nan')})
        self.assertEqual(path.read_bytes(), before)

    def test_legacy_results_are_not_overwritten(self):
        (self.root / 'results.json').write_text('old result')
        with self.assertRaisesRegex(ValueError, 'historical'):
            with LocalStudy(self.root, {'config': 1}, 60):
                self.fail('must not enter')
        self.assertEqual((self.root / 'results.json').read_text(), 'old result')

    def test_changed_config_refused_but_exact_resume_accepted(self):
        with LocalStudy(self.root, {'config': 1}, 60):
            pass
        before = (self.root / 'run-identity.json').read_bytes()
        with self.assertRaises(ValueError):
            with LocalStudy(self.root, {'config': 2}, 60):
                self.fail('must not enter')
        with LocalStudy(self.root, {'config': 1}, 60):
            pass
        self.assertEqual((self.root / 'run-identity.json').read_bytes(), before)

    def test_concurrent_process_cannot_enter_same_output(self):
        code = ('from runtime_guard import LocalStudy\n'
                'import sys\n'
                'with LocalStudy(sys.argv[1], {"config": 1}, 60): print("entered")\n')
        with LocalStudy(self.root, {'config': 1}, 60):
            result = subprocess.run([sys.executable, '-c', code, str(self.root)],
                                    cwd=Path(__file__).parent, capture_output=True, text=True, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('entered', result.stdout)

    def test_boolean_and_number_identities_are_not_conflated(self):
        with LocalStudy(self.root, {'config': 1}, 60):
            pass
        for value in (True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                with LocalStudy(self.root, {'config': value}, 60):
                    self.fail('must not enter')

    def test_exception_runs_cleanup_and_releases_lock(self):
        called = []
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with LocalStudy(self.root, {'config': 1}, 60) as run:
                run.cleanups.append(lambda: called.append(True))
                raise RuntimeError('injected')
        self.assertEqual(called, [True])
        with LocalStudy(self.root, {'config': 1}, 60):
            pass
        ledger = json.loads((self.root / 'wall-budget.json').read_text())
        self.assertEqual(ledger['attempts'][0]['status'], 'failed')

    def test_total_wall_budget_is_not_reset_on_restart(self):
        with LocalStudy(self.root, {'config': 1}, 60) as run:
            with patch('runtime_guard.time.monotonic', return_value=run.started + 61):
                with self.assertRaises(BudgetExceeded):
                    run.check()
            run.started -= 61
        with self.assertRaises(BudgetExceeded):
            with LocalStudy(self.root, {'config': 1}, 60):
                self.fail('must not enter')
        with LocalStudy(self.root, {'config': 1}, 120) as run:
            self.assertGreaterEqual(run.spent, 61)

    def test_crash_recovery_charges_reserved_time_conservatively(self):
        with LocalStudy(self.root, {'config': 1}, 60):
            pass
        ledger = json.loads((self.root / 'wall-budget.json').read_text())
        ledger['attempts'][0].update(status='active', started_unix=0, elapsed_seconds=0,
                                     reserved_seconds=60)
        atomic_json(self.root / 'wall-budget.json', ledger)
        with LocalStudy(self.root, {'config': 1}, 120) as run:
            self.assertEqual(run.spent, 60)
            self.assertEqual(run.ledger['attempts'][0]['status'], 'crash_reserved_time_estimate')

    def test_bad_budgets_rejected(self):
        for budget in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                LocalStudy(self.root, {'config': 1}, budget)

    def test_bound_study_cannot_reset_spending_after_ledger_loss(self):
        with LocalStudy(self.root, {'config': 1}, 60):
            pass
        ledger = self.root / 'wall-budget.json'
        ledger.unlink()
        with self.assertRaisesRegex(ValueError, 'lost its wall ledger'):
            with LocalStudy(self.root, {'config': 1}, 60):
                self.fail('Missing accounting must not reset spending')
        self.assertFalse(ledger.exists())

    def test_invalid_ledger_is_preserved_and_lock_is_released(self):
        with LocalStudy(self.root, {'config': 1}, 60):
            pass
        path = self.root / 'wall-budget.json'
        original = path.read_bytes()
        ledger = json.loads(original)
        for key, value in [('elapsed_seconds', -1), ('elapsed_seconds', True),
                           ('elapsed_seconds', float('inf')), ('reserved_seconds', 600),
                           ('cap_seconds', 0), ('status', 'unknown')]:
            broken = copy.deepcopy(ledger)
            broken['attempts'][0][key] = value
            path.write_text(json.dumps(broken), encoding='utf-8')
            before = path.read_bytes()
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                with LocalStudy(self.root, {'config': 1}, 60):
                    self.fail('Invalid accounting accepted')
            self.assertEqual(path.read_bytes(), before)
        path.write_bytes(original)
        with LocalStudy(self.root, {'config': 1}, 60):
            pass

    def test_duplicate_ledger_keys_cannot_erase_history(self):
        with LocalStudy(self.root, {'config': 1}, 60):
            pass
        path = self.root / 'wall-budget.json'
        prior = json.loads(path.read_text())['attempts']
        path.write_text('{"attempts":'+json.dumps(prior)+',"attempts":[]}',encoding='utf-8')
        before = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Duplicate state key'):
            with LocalStudy(self.root, {'config': 1}, 60):
                self.fail('Duplicate keys accepted')
        self.assertEqual(path.read_bytes(), before)

    def test_empty_or_malformed_ledger_does_not_create_new_allowance(self):
        with LocalStudy(self.root, {'config': 1}, 60):
            pass
        path = self.root / 'wall-budget.json'
        for ledger in ({'attempts': []}, {'attempts': {}}, [], {'attempts': [{}]}):
            path.write_text(json.dumps(ledger),encoding='utf-8')
            with self.subTest(ledger=ledger), self.assertRaises(ValueError):
                with LocalStudy(self.root, {'config': 1}, 60):
                    self.fail('Malformed ledger accepted')


if __name__ == '__main__':
    unittest.main()
