import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fast_search import decide, fingerprint, read_json, validate_plan
from sweep_guard import trial_id


class FastSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shared = {'method': 'full_ft', 'seed': 17, 'comparison_context_sha256': 'a' * 64,
                  'metric': {'name': 'accuracy', 'direction': 'max'}}
        self.plan = dict(shared, schema_version=1, resource_unit='epochs', rungs=[2, 4],
                         keep_counts=[1], ranking='rung_metric', tie_margin=0.0,
                         candidates=[{'config': dict(shared, lr=lr, max_resource=4,
                                                     schedule={'kind': 'cosine', 'full_steps': 40}),
                                      'roles': ['baseline'] if i == 0 else ['candidate'],
                                      'prune_after_resource': 2}
                                     for i, lr in enumerate([0.01, 0.02, 0.03, 0.04])])
        self.ids = [fingerprint(c['config']) for c in self.plan['candidates']]

    def ref(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = content.encode()
        path.write_bytes(payload)
        return {'path': name, 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}

    def row(self, index, at, metric, best=None, best_at=None):
        identity = self.ids[index]
        best = metric if best is None else best
        best_at = at if best_at is None else best_at
        return {'trial_id': identity, 'resource': at, 'status': 'ok', 'metric': metric,
                'best_metric': best, 'best_resource': best_at,
                'resume_checkpoint': self.ref(f'{identity}/resume-{at}', f'resume {identity} {at}'),
                'best_checkpoint': self.ref(f'{identity}/best-{best_at}-{best}', f'best {identity} {best_at} {best}'),
                'metrics': self.ref(f'{identity}/metrics-{at}', f'metrics {identity} {at}')}

    def history(self, rows):
        return {'plan_sha256': fingerprint(self.plan), 'observations': rows}

    def decide(self, rows):
        return decide(self.plan, self.history(rows), self.root)

    def first(self):
        return [self.row(i, 2, value) for i, value in enumerate([0.2, 0.4, 0.6, 0.8])]

    def test_initial_actions_have_no_resume_and_full_configs_stay_unchanged(self):
        before = copy.deepcopy(self.plan)
        result = self.decide([])
        self.assertEqual(len(result['actions']), 4)
        self.assertTrue(all(a['from_resource'] == 0 and a['target_resource'] == 2
                            and a['resume_checkpoint'] is None for a in result['actions']))
        self.assertEqual(self.plan, before)

    def test_partial_barrier_only_requests_missing_trials(self):
        result = self.decide(self.first()[:2])
        self.assertEqual({a['trial_id'] for a in result['actions']}, set(self.ids[2:]))
        self.assertEqual(result['pruned'], [])

    def test_promotes_top_candidate_and_protected_baseline_with_exact_resume(self):
        rows = self.first()
        result = self.decide(rows)
        self.assertEqual({a['trial_id'] for a in result['actions']}, {self.ids[0], self.ids[3]})
        for action in result['actions']:
            prior = next(row for row in rows if row['trial_id'] == action['trial_id'])
            self.assertEqual(action['resume_checkpoint'], prior['resume_checkpoint'])
            self.assertEqual((action['from_resource'], action['target_resource']), (2, 4))
        self.assertEqual(len(result['pruned']), 2)

    def test_best_so_far_ranking_preserves_earlier_better_checkpoint(self):
        self.plan['ranking'] = 'best_so_far'
        rows = self.first()
        rows[1] = self.row(1, 2, 0.4, best=0.95, best_at=1)
        result = self.decide(rows)
        self.assertEqual({a['trial_id'] for a in result['actions']}, {self.ids[0], self.ids[1]})

    def test_minimization_direction(self):
        self.plan['metric']['direction'] = 'min'
        for candidate in self.plan['candidates']:
            candidate['config']['metric'] = copy.deepcopy(self.plan['metric'])
        self.ids = [fingerprint(c['config']) for c in self.plan['candidates']]
        result = self.decide(self.first())
        self.assertEqual({a['trial_id'] for a in result['actions']}, {self.ids[0], self.ids[1]})

    def test_slow_candidate_receives_grace_outside_quota(self):
        self.plan['candidates'][1]['prune_after_resource'] = 4
        result = self.decide(self.first())
        self.assertEqual(len(result['actions']), 3)
        self.assertIn(self.ids[1], result['rounds'][0]['guarded'])

    def test_anchor_and_late_control_are_not_competitively_pruned(self):
        self.plan['candidates'][1]['roles'] = ['anchor', 'candidate']
        self.plan['candidates'][2]['roles'] = ['late_control']
        result = self.decide(self.first())
        self.assertEqual(len(result['actions']), 4)

    def test_boundary_ties_are_retained_outside_quota(self):
        self.plan['tie_margin'] = 0.03
        rows = self.first()
        rows[2] = self.row(2, 2, 0.78)
        result = self.decide(rows)
        self.assertEqual({a['trial_id'] for a in result['actions']}, {self.ids[0], self.ids[2], self.ids[3]})

    def test_failed_candidate_is_not_a_numeric_score(self):
        rows = self.first()
        rows[3] = {'trial_id': self.ids[3], 'resource': 2, 'status': 'failed', 'reason': 'numerical_failure'}
        result = self.decide(rows)
        self.assertEqual(len(result['failed']), 1)
        self.assertEqual({a['trial_id'] for a in result['actions']}, {self.ids[0], self.ids[2]})

    def test_failed_protected_control_blocks_promotion(self):
        rows = self.first()
        rows[0] = {'trial_id': self.ids[0], 'resource': 2, 'status': 'failed', 'reason': 'invalid_data'}
        result = self.decide(rows)
        self.assertEqual(result['status'], 'blocked_control_failure')
        self.assertEqual(result['actions'], [])

    def test_failure_without_reason_rejected(self):
        with self.assertRaises(ValueError):
            self.decide([{'trial_id': self.ids[0], 'resource': 2, 'status': 'failed'}])

    def test_final_winner_links_verified_best_checkpoint_not_last_state(self):
        rows = self.first() + [self.row(0, 4, 0.5), self.row(3, 4, 0.7, best=0.8, best_at=2)]
        result = self.decide(rows)
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['winner']['trial_id'], self.ids[3])
        self.assertEqual(result['winner']['best_checkpoint'], rows[3]['best_checkpoint'])
        self.assertFalse(result['fresh_seed_confirmation_complete'])

    def test_best_metric_cannot_regress_across_resume(self):
        rows = self.first() + [self.row(0, 4, 0.5), self.row(3, 4, 0.7)]
        with self.assertRaisesRegex(ValueError, 'regressed'):
            self.decide(rows)

    def test_historical_best_cannot_be_replaced_with_other_bytes(self):
        rows = self.first() + [self.row(0, 4, 0.5), self.row(3, 4, 0.7, best=0.8, best_at=2)]
        rows[-1]['best_checkpoint'] = self.ref('replacement', 'other weights')
        with self.assertRaisesRegex(ValueError, 'Historical best'):
            self.decide(rows)

    def test_pruned_better_checkpoint_remains_visible(self):
        rows = self.first()
        rows[1] = self.row(1, 2, 0.4, best=0.99, best_at=1)
        rows += [self.row(0, 4, 0.5), self.row(3, 4, 0.9)]
        result = self.decide(rows)
        self.assertEqual(result['winner']['trial_id'], self.ids[3])
        self.assertEqual(result['best_observed_any_fidelity']['trial_id'], self.ids[1])
        self.assertFalse(result['best_observed_any_fidelity']['reached_final_fidelity'])
        self.assertTrue(result['better_nonfinal_checkpoint_exists'])

    def test_earlier_good_checkpoint_from_later_failure_is_not_labeled_pruned(self):
        rows = self.first() + [self.row(0, 4, 0.5),
            {'trial_id': self.ids[3], 'resource': 4, 'status': 'failed', 'reason': 'deadline'}]
        result = self.decide(rows)
        self.assertTrue(result['better_nonfinal_checkpoint_exists'])
        self.assertEqual(result['best_observed_any_fidelity']['trial_id'], self.ids[3])
        self.assertNotIn(self.ids[3], [row['trial_id'] for row in result['pruned']])
        self.assertIn(self.ids[3], [row['trial_id'] for row in result['failed']])

    def test_future_observation_cannot_bypass_barrier(self):
        with self.assertRaisesRegex(ValueError, 'barrier'):
            self.decide(self.first()[:2] + [self.row(0, 4, 0.9)])

    def test_pruned_candidate_cannot_reappear_later(self):
        with self.assertRaises(ValueError):
            self.decide(self.first() + [self.row(0, 4, 0.5), self.row(3, 4, 0.9), self.row(1, 4, 0.99)])

    def test_stale_plan_history_rejected(self):
        history = self.history([])
        self.plan['keep_counts'] = [2]
        with self.assertRaisesRegex(ValueError, 'different immutable plan'):
            decide(self.plan, history, self.root)

    def test_duplicate_observation_rejected(self):
        row = self.first()[0]
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.decide([row, row])

    def test_unknown_trial_or_nonrung_resource_rejected(self):
        for key, value in [('trial_id', 'b' * 64), ('resource', 3), ('resource', True)]:
            row = self.first()[0]
            row[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.decide([row])

    def test_nonfinite_or_boolean_metric_rejected(self):
        for value in [True, float('nan'), float('inf')]:
            row = self.first()[0]
            row['metric'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.decide([row])

    def test_corrupt_resume_checkpoint_rejected_before_promotion(self):
        rows = self.first()
        (self.root / rows[3]['resume_checkpoint']['path']).write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'Artifact'):
            self.decide(rows)

    def test_artifact_path_escape_rejected(self):
        for name in ['../outside', 'C:/outside', '/outside', 'x\\outside', './outside']:
            row = self.first()[0]
            row['metrics']['path'] = name
            with self.subTest(path=name), self.assertRaises(ValueError):
                self.decide([row])

    def test_cohort_rank_seed_evaluator_changes_rejected(self):
        for key, value in [('method', 'lora_r16'), ('seed', 18), ('comparison_context_sha256', 'b' * 64),
                           ('metric', {'name': 'test_accuracy', 'direction': 'max'})]:
            plan = copy.deepcopy(self.plan)
            plan['candidates'][1]['config'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_changed_full_horizon_or_missing_schedule_rejected(self):
        for key, value in [('max_resource', 2), ('max_resource', True), ('schedule', {})]:
            plan = copy.deepcopy(self.plan)
            plan['candidates'][1]['config'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(plan)

    def test_duplicate_config_needs_combined_roles(self):
        self.plan['candidates'].append(copy.deepcopy(self.plan['candidates'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate config'):
            validate_plan(self.plan)

    def test_missing_baseline_or_invalid_policy_rejected(self):
        changes = [('rungs', [2, 2]), ('rungs', [4, 2]), ('keep_counts', [0]),
                   ('keep_counts', [1, 2]), ('ranking', 'unregistered'), ('tie_margin', -1)]
        for key, value in changes:
            plan = copy.deepcopy(self.plan)
            plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(plan)
        self.plan['candidates'][0]['roles'] = ['candidate']
        with self.assertRaisesRegex(ValueError, 'baseline'):
            validate_plan(self.plan)

    def test_observation_order_does_not_change_selection(self):
        rows = self.first()
        first, second = self.decide(rows), self.decide(list(reversed(rows)))
        first.pop('history_sha256')
        second.pop('history_sha256')
        self.assertEqual(first, second)

    def test_utf8_config_identity_matches_existing_helper(self):
        config = dict(self.plan['candidates'][0]['config'], labels=['고양이', '개'])
        self.assertEqual(fingerprint(config), trial_id(config))

    def test_duplicate_json_keys_and_nonfinite_json_rejected(self):
        path = self.root / 'input.json'
        for value in ['{"key":1,"key":2}', '{"key":NaN}']:
            path.write_text(value)
            with self.assertRaises(ValueError):
                read_json(path)

    def test_cli_plan_hash_and_decision(self):
        plan_path, history_path = self.root / 'plan.json', self.root / 'history.json'
        plan_path.write_text(json.dumps(self.plan), encoding='utf-8')
        history_path.write_text(json.dumps(self.history([])), encoding='utf-8')
        command = [sys.executable, '-B', str(Path(__file__).with_name('fast_search.py')), str(plan_path)]
        first = subprocess.run(command, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(first.stdout)['plan_sha256'], fingerprint(self.plan))
        second = subprocess.run(command + ['--history', str(history_path), '--artifact-root', str(self.root)],
                                check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(second.stdout)['status'], 'needs_work')

    def test_tiny_real_cpu_training_matches_uninterrupted_reference(self):
        path = Path(__file__).resolve().parents[1] / 'tools/run_fast_demo.py'
        spec = importlib.util.spec_from_file_location('fast_demo_test', path)
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        report = demo.run_demo(self.root / 'demo', compare_exhaustive=True)
        self.assertEqual(report['candidate_epochs_trained'], 90)
        self.assertEqual(report['resumed_segments'], 6)
        self.assertTrue(report['reference_check']['all_finalist_states_equal_uninterrupted'])
        self.assertTrue(report['reference_check']['same_winning_config'])
        self.assertEqual(report['reference_check']['metric_regret'], 0.0)
        self.assertEqual(report['checkpoint_reload_metric'], report['winner']['best_metric'])


if __name__ == '__main__':
    unittest.main()
