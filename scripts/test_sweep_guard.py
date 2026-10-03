"""Behavioral regression tests: false convergence and duplicate identity risks."""

import copy
import math
import unittest

from sweep_guard import parse_json, plateau_decision, trial_id


def policy(**overrides):
    result = {
        "direction": "max", "window": 5, "confirmations": 3,
        "val_patience": 8, "min_step": 200, "warmup_end_step": 300,
        "schedule_guard_step": 500, "eval_every_steps": 100,
        "hard_cap_step": 10000, "train_rel_delta": 0.002,
        "train_noise_rel_max": 0.005, "val_min_delta": 0.001,
        "min_train_progress_rel": 0.02, "loss_scale_floor": 1e-8,
    }
    result.update(overrides)
    return result


def history(count=31):
    return [{"step": i * 100,
             "train_loss": [2.0, 1.5, 1.0, 0.6][i] if i < 4 else 0.5,
             "val_score": [0.1, 0.2, 0.4][i] if i < 3 else 0.8,
             "diagnostics_ok": True, "schedule_clear": True}
            for i in range(count)]


class IdentityTests(unittest.TestCase):
    def test_key_order_is_irrelevant(self):
        self.assertEqual(trial_id({"seed": 7, "optimizer": {"lr": 0.001, "wd": 0.1}}),
                         trial_id({"optimizer": {"wd": 0.1, "lr": 0.001}, "seed": 7}))

    def test_semantic_changes_produce_distinct_full_hashes(self):
        base = {"seed": 7, "rank": 16, "head_lr": 0.001, "split_sha": "a"}
        hashes = {trial_id(base)}
        for key, value in [("seed", 8), ("rank", 32), ("head_lr", 0.003), ("split_sha", "b")]:
            item = dict(base)
            item[key] = value
            hashes.add(trial_id(item))
        self.assertEqual(len(hashes), 5)
        self.assertTrue(all(len(value) == 64 for value in hashes))

    def test_nonfinite_nested_config_is_rejected(self):
        with self.assertRaises(ValueError):
            trial_id({"optimizer": {"lr": float("nan")}})

    def test_empty_config_is_rejected(self):
        with self.assertRaises(ValueError):
            trial_id({})

    def test_duplicate_json_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_json('{"config": {"lr": 1, "lr": 2}}')

    def test_json_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_json('{"lr": NaN}')

    def test_json_exponent_overflow_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_json('{"lr": 1e999}')

    def test_numeric_representation_is_not_silently_rewritten(self):
        self.assertNotEqual(trial_id({"x": 1}), trial_id({"x": 1.0}))


class PlateauTests(unittest.TestCase):
    def test_learned_then_flat_can_saturate(self):
        result = plateau_decision(policy(), history())
        self.assertEqual(result["status"], "saturated")
        self.assertTrue(result["stop"])
        self.assertFalse(result["right_censored"])
        self.assertEqual(result["details"]["best_step"], 300)
        self.assertEqual(len(result["details"]["assessments"]), 3)

    def test_small_lr_that_never_learns_is_not_saturation(self):
        rows = history()
        for row in rows:
            row["train_loss"], row["val_score"] = 2.0, 0.1
        result = plateau_decision(policy(), rows)
        self.assertEqual(result["reason"], "no_verified_learning_progress")
        self.assertFalse(result["stop"])

    def test_learning_requirement_cannot_be_disabled_by_zero_threshold(self):
        rows = history()
        for row in rows:
            row['train_loss'], row['val_score'] = 2.0, 0.1
        with self.assertRaises(ValueError):
            plateau_decision(policy(min_train_progress_rel=0), rows)

    def test_cap_without_learning_is_right_censored(self):
        rows = history()
        for row in rows:
            row["train_loss"] = 2.0
        result = plateau_decision(policy(hard_cap_step=3000), rows)
        self.assertEqual(result["status"], "budget_exhausted")
        self.assertTrue(result["right_censored"])
        self.assertNotEqual(result["status"], "saturated")

    def test_hard_cap_must_be_reachable_on_exact_evaluation_cadence(self):
        for cap in (5, 95):
            with self.subTest(cap=cap), self.assertRaisesRegex(
                    ValueError, "hard_cap_step must be a multiple of eval_every_steps"):
                plateau_decision(policy(
                    hard_cap_step=cap, eval_every_steps=10, min_step=0,
                    warmup_end_step=0, schedule_guard_step=0), history(1))

    def test_observed_plateau_at_cap_is_valid(self):
        self.assertEqual(plateau_decision(policy(hard_cap_step=3000), history())["status"],
                         "saturated")

    def test_observations_after_declared_cap_cannot_prove_saturation(self):
        with self.assertRaises(ValueError):
            plateau_decision(policy(hard_cap_step=2000), history())

    def test_warmup_and_schedule_guard_prevent_stopping(self):
        result = plateau_decision(policy(schedule_guard_step=4000), history())
        self.assertEqual(result["reason"], "minimum_exposure_or_schedule_guard")

    def test_future_schedule_event_prevents_stopping(self):
        rows = history()
        rows[-1]["schedule_clear"] = False
        result = plateau_decision(policy(), rows)
        self.assertEqual(result["reason"], "future_schedule_event_or_grace_pending")

    def test_grace_must_cover_confirmation_history(self):
        rows = history()
        rows[-6]["schedule_clear"] = False
        self.assertEqual(plateau_decision(policy(), rows)["reason"],
                         "guard_not_clear_through_confirmation_windows")

    def test_failed_diagnostics_prevent_stopping(self):
        rows = history()
        rows[-1]["diagnostics_ok"] = False
        self.assertEqual(plateau_decision(policy(), rows)["reason"], "diagnostics_not_passed")

    def test_not_enough_post_guard_history(self):
        result = plateau_decision(policy(), history(12))
        self.assertEqual(result["reason"], "insufficient_post_guard_history")

    def test_flat_accuracy_does_not_override_improving_loss(self):
        rows = history()
        for i, row in enumerate(rows):
            row["train_loss"] = 2.0 - 0.03 * i
            row["train_score"] = 0.9
        result = plateau_decision(policy(train_score_direction="max", train_score_min_delta=0.001), rows)
        self.assertEqual(result["reason"], "train_loss_not_flat")

    def test_worsening_loss_is_not_saturation(self):
        rows = history()
        for i, row in enumerate(rows):
            if i >= 5:
                row["train_loss"] = 0.5 + 0.02 * i
        self.assertNotEqual(plateau_decision(policy(), rows)["status"], "saturated")

    def test_noise_is_not_a_plateau(self):
        rows = history()
        for i, row in enumerate(rows):
            if i >= 5:
                row["train_loss"] = 0.8 + [0.0, 0.2, -0.2][i % 3]
        self.assertEqual(plateau_decision(policy(), rows)["reason"], "train_probe_too_noisy")

    def test_small_validation_gains_accumulate(self):
        rows = history()
        for i, row in enumerate(rows):
            row["val_score"] = 0.5 + i * 0.00025
        result = plateau_decision(policy(), rows)
        self.assertEqual(result["reason"], "validation_still_improving_or_patience_pending")

    def test_isolated_symmetric_loss_spike_is_not_a_plateau(self):
        stop = policy(window=3, confirmations=1, val_patience=3, min_step=0,
                      warmup_end_step=0, schedule_guard_step=0,
                      eval_every_steps=10, hard_cap_step=200,
                      train_rel_delta=0.001, train_noise_rel_max=0.001)
        rows = [{"step": i * 10, "train_loss": loss,
                 "val_score": min(0.2 + 0.1 * i, 0.5),
                 "diagnostics_ok": True, "schedule_clear": True}
                for i, loss in enumerate([2, 1.5, 1, .5, .5, .5, .5, 100, .5])]
        result = plateau_decision(stop, rows)
        self.assertEqual(result["reason"], "train_probe_too_noisy")
        evidence = result["details"]["assessments"][0]
        self.assertEqual(evidence["absolute_mad"], 0)
        self.assertGreater(evidence["absolute_max_residual"], evidence["loss_noise_limit"])

    def test_small_isolated_noise_within_declared_limit_can_saturate(self):
        rows = history()
        rows[-3]["train_loss"] += 0.00001
        self.assertEqual(plateau_decision(policy(), rows)["status"], "saturated")

    def test_finite_losses_cannot_hide_nonfinite_intermediate_slope(self):
        stop = policy(window=3, confirmations=1, val_patience=3, min_step=0,
                      warmup_end_step=0, schedule_guard_step=0,
                      eval_every_steps=10, hard_cap_step=200)
        losses = [1.7e308, 1.5e308, 1.2e308, 1e308, 1e308, 1e308, 1.4e308, 0, 1e308]
        rows = [{"step": i * 10, "train_loss": loss, "val_score": min(i * .1, .5),
                 "diagnostics_ok": True, "schedule_clear": True}
                for i, loss in enumerate(losses)]
        with self.assertRaisesRegex(ValueError, "finite"):
            plateau_decision(stop, rows)

    def test_isolated_train_score_drop_is_not_a_plateau(self):
        stop = policy(window=3, confirmations=1, val_patience=3, min_step=0,
                      warmup_end_step=0, schedule_guard_step=0,
                      eval_every_steps=10, hard_cap_step=200,
                      train_score_direction="max", train_score_min_delta=0.001)
        rows = [{"step": i * 10, "train_loss": loss,
                 "val_score": min(0.2 + 0.1 * i, 0.5),
                 "train_score": 0.1 if i == 7 else 0.8,
                 "diagnostics_ok": True, "schedule_clear": True}
                for i, loss in enumerate([2, 1.5, 1, .5, .5, .5, .5, .5, .5])]
        result = plateau_decision(stop, rows)
        self.assertEqual(result["reason"], "train_score_not_flat")
        evidence = result["details"]["assessments"][0]
        self.assertEqual(evidence["train_score_change"], 0)
        self.assertEqual(evidence["train_score_trend"], 0)
        self.assertGreater(evidence["train_score_max_residual"], 0.6)

    def test_raw_best_is_independent_of_patience_delta(self):
        rows = history()
        rows[-1]["val_score"] = 0.8001
        result = plateau_decision(policy(), rows)
        self.assertEqual(result["status"], "saturated")
        self.assertEqual(result["details"]["best_step"], 3000)

    def test_minimization_direction_selects_lowest_metric(self):
        rows = history()
        for row in rows:
            row["val_score"] = 1.0 - row["val_score"]
        result = plateau_decision(policy(direction="min"), rows)
        self.assertEqual(result["status"], "saturated")
        self.assertAlmostEqual(result["details"]["best_val_score"], 0.2)

    def test_optional_train_metric_improving_prevents_stopping(self):
        rows = history()
        for i, row in enumerate(rows):
            row["train_score"] = 0.3 + i * 0.01
        result = plateau_decision(policy(train_score_direction="max", train_score_min_delta=0.001), rows)
        self.assertEqual(result["reason"], "train_score_not_flat")

    def test_direct_api_nonfinite_is_divergence(self):
        rows = history()
        rows[-1]["train_loss"] = math.inf
        result = plateau_decision(policy(), rows)
        self.assertEqual(result["status"], "diverged")
        self.assertTrue(result["stop"])

    def test_missing_or_duplicate_eval_is_rejected(self):
        for rows in (history()[:5] + history()[6:], history() + [history()[-1]]):
            with self.assertRaises(ValueError):
                plateau_decision(policy(), rows)

    def test_absent_step_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            plateau_decision(policy(), history()[1:])

    def test_missing_metrics_and_bad_booleans_are_rejected(self):
        rows = history()
        del rows[-1]["val_score"]
        with self.assertRaises(ValueError):
            plateau_decision(policy(), rows)
        rows = history()
        rows[-1]["diagnostics_ok"] = "true"
        with self.assertRaises(ValueError):
            plateau_decision(policy(), rows)

    def test_invalid_policy_values_are_rejected(self):
        for patch in ({"train_rel_delta": -1}, {"window": True},
                      {"window": 2}, {"val_min_delta": math.nan},
                      {"direction": "maximize"}, {"loss_scale_floor": 0},
                      {"schedule_guard_step": 10001}, {"unknown_key": 2},
                      {"train_score_direction": "max"}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                plateau_decision(policy(**patch), history())

    def test_inputs_are_not_mutated(self):
        config, rows = policy(), history()
        before = copy.deepcopy((config, rows))
        plateau_decision(config, rows)
        self.assertEqual((config, rows), before)

    def test_near_zero_loss_accepts_predeclared_absolute_tolerance(self):
        rows = history()
        for i, row in enumerate(rows):
            if i >= 5:
                row["train_loss"] = 0.0001 / (i - 3)
        relative_only = plateau_decision(policy(), rows)
        self.assertNotEqual(relative_only["status"], "saturated")
        absolute = plateau_decision(policy(train_abs_delta=0.00001), rows)
        self.assertEqual(absolute["status"], "saturated")

    def test_absolute_tolerance_does_not_mask_material_progress(self):
        rows = history()
        for i, row in enumerate(rows):
            row["train_loss"] = 2.0 - 0.03 * i
        result = plateau_decision(policy(train_abs_delta=0.0001), rows)
        self.assertEqual(result["reason"], "train_loss_not_flat")

    def test_explicit_zero_absolute_tolerance_preserves_decision(self):
        rows = history()
        for i, row in enumerate(rows):
            row["train_loss"] = 2.0 - 0.03 * i
        self.assertEqual(plateau_decision(policy(), rows),
                         plateau_decision(policy(train_abs_delta=0.0, train_noise_abs_max=0.0), rows))

    def test_bad_absolute_thresholds_are_rejected(self):
        for patch in ({"train_abs_delta":-1}, {"train_noise_abs_max":float("nan")},
                      {"train_abs_delta":True}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                plateau_decision(policy(**patch), history())


if __name__ == "__main__":
    unittest.main()
