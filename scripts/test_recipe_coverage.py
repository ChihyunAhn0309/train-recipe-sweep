"""Exercise omissions and false on/off coverage against real source/config files."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from recipe_coverage import check, inventory


class RecipeCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / 'source.json'
        self.source.write_text(json.dumps({'train': {'multi_scale': True}, 'eval': {'tta': False}}), encoding='utf-8')
        self.methods = ['full_ft', 'lora_r16', 'lora_r32']
        self.ledger = {'schema_version': 1, 'source_sha256': inventory(self.source)['source_sha256'],
                       'methods': self.methods, 'rows': [
            {'factor': 'train_multi_scale', 'source_keys': ['/train/multi_scale'],
             'source_stage': 'checkpoint_generation', 'evidence': ['fixture:config/train/multi_scale'],
             'evidence_status': 'verified', 'ablatable': True,
             'choices': {m: {'disposition': 'sweep', 'reason': 'Transfer is uncertain; compare source and disabled control',
                            'target_pointer': '/train/multi_scale', 'candidates': [True, False]} for m in self.methods}},
            {'factor': 'evaluation_tta', 'source_keys': ['/eval/tta'], 'source_stage': 'source_evaluation',
             'evidence': ['fixture:config/eval/tta'], 'evidence_status': 'verified', 'ablatable': True,
             'choices': {m: {'disposition': 'fixed', 'reason': 'Hold official evaluation constant',
                            'ablation_reason': 'TTA changes evaluation, not the training comparison',
                            'target_pointer': '/eval/tta', 'candidates': [False]} for m in self.methods}}
        ]}
        self.trials = {'trials': [{'method': m, 'config': {'method': m, 'train': {'multi_scale': enabled}, 'eval': {'tta': False}}}
                                  for m in self.methods for enabled in [True, False]]}

    def test_three_methods_both_states_and_fixed_evaluator_are_covered(self):
        result = check(self.source, self.ledger, self.trials)
        self.assertTrue(result['candidate_coverage_verified'])
        self.assertEqual(len(result['ablation_exceptions']), 3)
        self.assertFalse(result['trainer_behavior_verified'])
        self.assertFalse(result['source_recovery_completeness_verified'])

    def test_source_default_missing_from_ledger_fails(self):
        self.source.write_text(json.dumps({'train': {'multi_scale': True, 'mosaic': True}, 'eval': {'tta': False}}), encoding='utf-8')
        self.ledger['source_sha256'] = inventory(self.source)['source_sha256']
        with self.assertRaisesRegex(ValueError, 'Unreviewed.*mosaic'):
            check(self.source, self.ledger, self.trials)

    def test_stale_source_hash_fails(self):
        self.source.write_text(self.source.read_text() + ' ', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'hash changed'):
            check(self.source, self.ledger)

    def test_duplicate_and_invented_source_mappings_fail(self):
        for key in ['/train/multi_scale', '/invented']:
            ledger = copy.deepcopy(self.ledger)
            ledger['rows'][1]['source_keys'] = [key]
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Unknown or multiply mapped'):
                check(self.source, ledger)

    def test_missing_rank_decision_and_missing_rank_trials_fail(self):
        ledger = copy.deepcopy(self.ledger)
        del ledger['rows'][0]['choices']['lora_r32']
        with self.assertRaisesRegex(ValueError, 'all requested method'):
            check(self.source, ledger, self.trials)
        self.trials['trials'] = [t for t in self.trials['trials'] if t['method'] != 'lora_r32']
        with self.assertRaisesRegex(ValueError, 'omit a requested method'):
            check(self.source, self.ledger, self.trials)

    def test_disabled_only_requires_explicit_ablation_reason(self):
        choice = self.ledger['rows'][0]['choices']['full_ft']
        choice.update(disposition='fixed', candidates=[False])
        with self.assertRaisesRegex(ValueError, 'both enabled and disabled'):
            check(self.source, self.ledger)
        choice['ablation_reason'] = 'Explicitly constrained example budget; enabled branch remains untested'
        result = check(self.source, self.ledger)
        self.assertEqual(len(result['ablation_exceptions']), 4)

    def test_declared_on_off_sweep_without_enabled_trial_fails(self):
        self.trials['trials'] = [t for t in self.trials['trials'] if not t['config']['train']['multi_scale']]
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            check(self.source, self.ledger, self.trials)

    def test_undeclared_effective_value_and_missing_pipeline_key_fail(self):
        self.trials['trials'][0]['config']['eval']['tta'] = True
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            check(self.source, self.ledger, self.trials)
        del self.trials['trials'][0]['config']['eval']['tta']
        with self.assertRaisesRegex(ValueError, 'Missing target config pointer'):
            check(self.source, self.ledger, self.trials)

    def test_excluded_factor_must_be_absent_or_match_explicit_inactive_value(self):
        choice = self.ledger['rows'][0]['choices']['full_ft']
        choice.update(disposition='excluded', candidates=[], ablation_reason='Unsupported transform in this example task')
        with self.assertRaisesRegex(ValueError, 'exclusion_checks'):
            check(self.source, self.ledger, self.trials)
        del choice['target_pointer']
        with self.assertRaisesRegex(ValueError, 'require explicit'):
            check(self.source, self.ledger, self.trials)
        choice['exclusion_checks'] = [{'pointer': '/train/multi_scale', 'equals': False}]
        with self.assertRaisesRegex(ValueError, 'Excluded factor remains'):
            check(self.source, self.ledger, self.trials)
        for trial in self.trials['trials']:
            if trial['method'] == 'full_ft':
                trial['config']['train']['multi_scale'] = False
        self.assertTrue(check(self.source, self.ledger, self.trials)['candidate_coverage_verified'])
        choice['exclusion_checks'] = [{'pointer': '/train/multi_scale', 'absent': True}]
        with self.assertRaisesRegex(ValueError, 'Excluded factor remains'):
            check(self.source, self.ledger, self.trials)
        for trial in self.trials['trials']:
            if trial['method'] == 'full_ft':
                del trial['config']['train']['multi_scale']
        self.assertTrue(check(self.source, self.ledger, self.trials)['candidate_coverage_verified'])

    def test_method_envelope_cannot_mislabel_full_ft_as_lora(self):
        for trial in self.trials['trials']:
            trial['config']['method'] = 'full_ft'
        with self.assertRaisesRegex(ValueError, 'matching config.method'):
            check(self.source, self.ledger, self.trials)
        for trial in self.trials['trials']:
            del trial['config']['method']
        with self.assertRaisesRegex(ValueError, 'matching config.method'):
            check(self.source, self.ledger, self.trials)

    def test_boolean_enabled_state_cannot_be_replaced_with_integer(self):
        self.ledger['rows'][0]['choices']['full_ft']['candidates'] = [1, 0]
        with self.assertRaisesRegex(ValueError, 'boolean enabled states'):
            check(self.source, self.ledger)

    def test_plan_only_and_unknown_source_never_claim_measurement(self):
        self.ledger['rows'][0]['evidence_status'] = 'unknown'
        result = check(self.source, self.ledger)
        self.assertFalse(result['candidate_coverage_verified'])
        self.assertEqual(result['nonverified_source_factors'], ['train_multi_scale'])

    def test_conditional_branch_must_be_realized(self):
        choice = self.ledger['rows'][0]['choices']['full_ft']
        choice.update(disposition='conditional', when=[{'pointer': '/eval/tta', 'equals': True}])
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            check(self.source, self.ledger, self.trials)
        choice['when'][0]['equals'] = False
        self.assertTrue(check(self.source, self.ledger, self.trials)['candidate_coverage_verified'])

    def test_nested_settings_and_escaped_source_keys(self):
        self.source.write_text(json.dumps({'train/a~b': {'sizes': [512, 640], 'empty': []}}), encoding='utf-8')
        self.assertEqual(set(inventory(self.source)['leaves']), {'/train~1a~0b/sizes/0', '/train~1a~0b/sizes/1', '/train~1a~0b/empty'})
        for row in self.ledger['rows'][:1]:
            row.update(source_keys=list(inventory(self.source)['leaves']))
            for method, choice in row['choices'].items():
                choice['candidates'] = [{'enabled': True, 'sizes': [512, 640]}, {'enabled': False, 'size': 640}]
        self.ledger['rows'] = self.ledger['rows'][:1]
        self.ledger['source_sha256'] = inventory(self.source)['source_sha256']
        for trial in self.trials['trials']:
            enabled = trial['config']['train']['multi_scale']
            trial['config']['train']['multi_scale'] = {'enabled': True, 'sizes': [512, 640]} if enabled else {'enabled': False, 'size': 640}
        self.assertTrue(check(self.source, self.ledger, self.trials)['candidate_coverage_verified'])

    def test_cli_rejects_duplicate_source_keys(self):
        self.source.write_text('{"multi_scale":true,"multi_scale":false}', encoding='utf-8')
        script = Path(__file__).with_name('recipe_coverage.py')
        result = subprocess.run([sys.executable, '-B', str(script), 'inventory', '--source', str(self.source)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)['status'], 'invalid_recipe_coverage')


if __name__ == '__main__':
    unittest.main()
