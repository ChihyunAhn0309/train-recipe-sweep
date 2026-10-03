"""CPU-only adversarial checks for replayed, conditional horizon cross-checks."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from horizon_review import review, validate_plan
from sweep_guard import plateau_decision, trial_id


def policy():
    return {"direction": "max", "window": 3, "confirmations": 1,
            "val_patience": 3, "min_step": 0, "warmup_end_step": 0,
            "schedule_guard_step": 0, "eval_every_steps": 10, "hard_cap_step": 500,
            "train_rel_delta": 0.001, "train_noise_rel_max": 0.001,
            "val_min_delta": 0.001, "min_train_progress_rel": 0.02,
            "loss_scale_floor": 1e-8}


def config(scale):
    return {"seed": 7, "comparison_context_sha256": "a" * 64,
            "family": {"model": "fixture", "method": "full_ft", "rank": None,
                       "optimizer": "adamw", "effective_batch": 8,
                       "split_sha256": "b" * 64, "augmentation": "fixed"},
            "schedule": {"kind": "constant", "hard_cap_step": 500},
            "lr_scale": scale, "lr_ratios": {"backbone": 1, "head": 2},
            "stop_policy": policy()}


def plan(third=True, shared=False):
    configs = [config(0.001), config(0.01), config(0.1)]
    ids = [trial_id(c) for c in configs]
    return {"schema_version": 1, "seed": 7, "comparison_context_sha256": "a" * 64,
            "baseline_source_sha256": "c" * 64, "candidates": configs,
            "anchor_trial_id": ids[0], "baseline_trial_id": ids[0] if shared else ids[1],
            "representative_trial_id": ids[1] if shared else None,
            "third_trial_id": ids[2] if third else None,
            "tolerance_steps": 20, "exposure": {"unit": "samples", "per_step": 8}}


def rows(last=120, delay=0, stalled=False):
    result = []
    for step in range(0, last + 1, 10):
        index = max(0, (step - delay) // 10)
        result.append({"step": step, "exposure": step * 8,
                       "train_loss": 2.0 if stalled else [2, 1.5, 1, 0.5][min(index, 3)],
                       "val_score": 0.1 if stalled else [0.1, 0.2, 0.4, 0.8][min(index, 3)],
                       "diagnostics_ok": True, "schedule_clear": True})
    return result


def observation(p, index, records=None, state="saturated"):
    return {"trial_id": trial_id(p["candidates"][index]), "state": state,
            "records": rows() if records is None else records}


def history(p, *observations):
    return {"plan_sha256": trial_id(p), "observations": list(observations)}


def rebind(p):
    p["anchor_trial_id"] = trial_id(p["candidates"][0])
    p["baseline_trial_id"] = trial_id(p["candidates"][1])
    p["third_trial_id"] = trial_id(p["candidates"][2]) if p["third_trial_id"] else None


class PlanTests(unittest.TestCase):
    def test_plan_only_is_pending_and_bootstrap_is_not_blocked(self):
        result = review(plan())
        self.assertEqual(result["horizon_transfer_status"], "pending")
        self.assertFalse(result["saturation_calibration_complete"])
        self.assertIn("may_run", result["bootstrap"])
        self.assertFalse(result["horizon_transfer"]["conditional_third"]["required_for_diagnosis"])

    def test_duplicate_config_cannot_count_twice(self):
        p = plan()
        p["candidates"].append(copy.deepcopy(p["candidates"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate config"):
            validate_plan(p)

    def test_numeric_representation_alone_is_not_a_distinct_recipe(self):
        p = plan()
        p["candidates"][0]["lr_scale"] = 1
        p["candidates"][1]["lr_scale"] = 1.0
        p["candidates"][2]["lr_scale"] = 2
        rebind(p)
        with self.assertRaisesRegex(ValueError, "not distinct recipes"):
            validate_plan(p)

    def test_shared_roles_require_an_existing_distinct_representative(self):
        for replacement in (None, "d" * 64):
            p = plan(shared=True)
            p["representative_trial_id"] = replacement
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                validate_plan(p)
        p = plan(shared=True)
        p["representative_trial_id"] = p["anchor_trial_id"]
        with self.assertRaises(ValueError):
            validate_plan(p)

    def test_shared_roles_use_two_actual_configs(self):
        p = plan(shared=True)
        result = review(p, history(p, observation(p, 0), observation(p, 1)))
        self.assertEqual(result["status"], "supported")
        self.assertEqual(len(set(result["horizon_transfer"]["primary_trial_ids"])), 2)

    def test_baseline_cannot_replace_actual_minimum_anchor(self):
        p = plan()
        p["anchor_trial_id"], p["baseline_trial_id"] = p["baseline_trial_id"], p["anchor_trial_id"]
        with self.assertRaisesRegex(ValueError, "minimum positive"):
            validate_plan(p)

    def test_new_lower_bound_invalidates_old_anchor(self):
        p = plan()
        p["candidates"].append(config(0.0001))
        with self.assertRaisesRegex(ValueError, "minimum positive"):
            validate_plan(p)

    def test_different_optimizer_batch_rank_augmentation_cannot_borrow(self):
        for key, value in (("optimizer", "sgd"), ("effective_batch", 16),
                           ("rank", 8), ("augmentation", "mixup")):
            p = plan()
            p["candidates"][1]["family"][key] = value
            rebind(p)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "convergence family"):
                validate_plan(p)

    def test_schedule_policy_cap_or_extra_field_changes_are_new_family(self):
        for patch in ("schedule", "policy", "cap", "extra"):
            p = plan()
            candidate = p["candidates"][1]
            if patch == "schedule":
                candidate["schedule"]["kind"] = "cosine"
            elif patch == "policy":
                candidate["stop_policy"]["val_patience"] += 1
            elif patch == "cap":
                candidate["stop_policy"]["hard_cap_step"] += 10
            else:
                candidate["other_scientific_field"] = 1
            rebind(p)
            with self.subTest(patch=patch), self.assertRaisesRegex(ValueError, "convergence family"):
                validate_plan(p)

    def test_seed_or_context_mismatch_rejected(self):
        for key, value in (("seed", 8), ("seed", True),
                           ("comparison_context_sha256", "d" * 64)):
            p = plan()
            p["candidates"][1][key] = value
            rebind(p)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_plan(p)

    def test_invalid_lr_scales_and_group_ratios(self):
        for value in (0, -1, True, float("inf"), float("nan")):
            for key in ("lr_scale", "lr_ratios"):
                p = plan()
                if key == "lr_scale":
                    p["candidates"][0][key] = value
                else:
                    p["candidates"][0][key]["head"] = value
                with self.subTest(value=value, key=key), self.assertRaises(ValueError):
                    validate_plan(p)

    def test_group_lr_ratio_changes_are_not_common_scale_comparisons(self):
        p = plan()
        p["candidates"][1]["lr_ratios"]["head"] = 3
        with self.assertRaisesRegex(ValueError, "convergence family"):
            validate_plan(p)

    def test_finite_positive_factors_with_underflowed_group_lr_are_rejected(self):
        p = plan()
        for index, candidate in enumerate(p["candidates"], 1):
            candidate["lr_scale"] = index * 1e-300
            candidate["lr_ratios"]["head"] = 1e-100
        rebind(p)
        self.assertGreater(p["candidates"][0]["lr_scale"], 0)
        self.assertGreater(p["candidates"][0]["lr_ratios"]["head"], 0)
        with self.assertRaisesRegex(ValueError, "Effective LR for group head"):
            validate_plan(p)

    def test_finite_positive_factors_with_overflowed_group_lr_are_rejected(self):
        p = plan()
        for index, candidate in enumerate(p["candidates"], 1):
            candidate["lr_scale"] = index * 1e300
            candidate["lr_ratios"]["head"] = 1e100
        rebind(p)
        with self.assertRaisesRegex(ValueError, "Effective LR for group head"):
            validate_plan(p)

    def test_tiny_representable_positive_group_lr_remains_valid(self):
        p = plan()
        for index, candidate in enumerate(p["candidates"], 1):
            candidate["lr_scale"] = index * 1e-300
            candidate["lr_ratios"]["head"] = 1e-20
        rebind(p)
        effective_lr = p["candidates"][0]["lr_scale"] * p["candidates"][0]["lr_ratios"]["head"]
        self.assertGreater(effective_lr, 0)
        self.assertLess(effective_lr, sys.float_info.min)
        self.assertEqual(set(validate_plan(p)), {trial_id(c) for c in p["candidates"]})

    def test_distinct_scales_rounding_to_same_effective_vector_are_rejected(self):
        p = plan()
        for candidate, scale in zip(p["candidates"], (1.9999999999999996, 1.9999999999999998, 3)):
            candidate["lr_scale"] = scale
            candidate["lr_ratios"] = {"body": 0.55}
        rebind(p)
        self.assertNotEqual(p["candidates"][0]["lr_scale"], p["candidates"][1]["lr_scale"])
        self.assertEqual(p["candidates"][0]["lr_scale"] * 0.55,
                         p["candidates"][1]["lr_scale"] * 0.55)
        with self.assertRaisesRegex(ValueError, "Duplicate effective LR vector"):
            validate_plan(p)

    def test_alias_in_one_group_does_not_hide_a_distinct_complete_vector(self):
        p = plan()
        for candidate, scale in zip(p["candidates"], (1.9999999999999996, 1.9999999999999998, 3)):
            candidate["lr_scale"] = scale
            candidate["lr_ratios"] = {"body": 0.55, "head": 1}
        rebind(p)
        self.assertEqual(len(validate_plan(p)), 3)

    def test_third_cannot_duplicate_primary(self):
        p = plan()
        p["third_trial_id"] = p["baseline_trial_id"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_plan(p)

    def test_tolerance_must_align_and_cannot_be_negative(self):
        for tolerance in (-1, 3, True):
            p = plan()
            p["tolerance_steps"] = tolerance
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                validate_plan(p)

    def test_unknown_plan_field_rejected(self):
        p = plan()
        p["auto_extend_budget"] = True
        with self.assertRaises(ValueError):
            validate_plan(p)


class ReviewTests(unittest.TestCase):
    def test_two_replayed_plateaus_support_without_third(self):
        p = plan()
        result = review(p, history(p, observation(p, 0), observation(p, 1)))
        self.assertEqual(result["status"], "supported")
        self.assertTrue(result["saturation_calibration_complete"])
        self.assertFalse(result["horizon_transfer"]["conditional_third"]["required_for_diagnosis"])
        self.assertIsNone(result["horizon_transfer"]["conditional_third"]["comparison"])
        self.assertTrue(result["horizon_transfer"]["per_trial_guards_required"])
        self.assertFalse(result["horizon_transfer"]["authorizes_stop_or_budget_change"])

    def test_anchor_alone_calibrates_but_does_not_support_transfer(self):
        p = plan()
        result = review(p, history(p, observation(p, 0)))
        self.assertTrue(result["saturation_calibration_complete"])
        self.assertEqual(result["status"], "pending")

    def test_uses_first_replayed_plateau_not_last_logged_step(self):
        p = plan()
        raw = rows()
        expected = next(raw[i]["step"] for i in range(len(raw))
                        if plateau_decision(policy(), raw[:i + 1])["status"] == "saturated")
        result = review(p, history(p, observation(p, 0), observation(p, 1)))
        self.assertEqual(result["anchor_calibration"]["observed_saturation_step"], expected)
        self.assertLess(expected, raw[-1]["step"])
        self.assertEqual(result["horizon_transfer"]["anchor_review_exposure"], expected * 8)

    def test_claimed_saturated_stalled_anchor_is_inconclusive(self):
        p = plan()
        result = review(p, history(p, observation(p, 0, rows(stalled=True))))
        self.assertFalse(result["saturation_calibration_complete"])
        self.assertEqual(result["status"], "inconclusive")

    def test_capped_pruned_stalled_budget_labels_never_prove_saturation(self):
        p = plan()
        for state in ("capped", "pruned", "stalled", "budget_exhausted", "failed", "interrupted"):
            result = review(p, history(p, observation(p, 0), observation(p, 1, state=state)))
            with self.subTest(state=state):
                self.assertEqual(result["status"], "inconclusive")
                self.assertFalse(result["evidence"][p["baseline_trial_id"]]["verified_saturation"])

    def test_false_saturated_flag_cannot_replace_raw_history(self):
        p = plan()
        result = review(p, history(p, observation(p, 0), observation(p, 1, rows(last=20))))
        self.assertEqual(result["status"], "inconclusive")

    def test_running_prefix_before_anchor_horizon_is_pending(self):
        p = plan()
        result = review(p, history(p, observation(p, 0), observation(p, 1, rows(last=20), "running")))
        self.assertEqual(result["status"], "pending")

    def test_improvement_past_anchor_is_counterexample(self):
        p = plan()
        raw = rows()
        for i, row in enumerate(raw):
            row["train_loss"] = 2 - i * 0.03
        result = review(p, history(p, observation(p, 0), observation(p, 1, raw, "running")))
        self.assertEqual(result["status"], "inconclusive")
        self.assertTrue(result["horizon_transfer"]["conditional_third"]["required_for_diagnosis"])
        self.assertTrue(result["horizon_transfer"]["anchor_horizon_counterexamples"])
        self.assertIsNone(result["horizon_transfer"]["provisional_review_step"])

    def test_later_saturation_within_tolerance_supports_later_review_preserves_anchor_caveat(self):
        p = plan()
        result = review(p, history(p, observation(p, 0), observation(p, 1, rows(delay=10))))
        transfer = result["horizon_transfer"]
        self.assertEqual(result["status"], "supported")
        self.assertEqual(transfer["anchor_review_step"], 80)
        self.assertEqual(transfer["provisional_review_step"], 90)
        self.assertEqual(transfer["provisional_review_exposure"], 720)
        self.assertEqual(transfer["observed_duration_range_steps"], [80, 90])
        self.assertTrue(any(item["kind"] == "later_observed_plateau"
                            for item in transfer["anchor_horizon_counterexamples"]))
        self.assertFalse(transfer["conditional_third"]["required_for_diagnosis"])
        self.assertIsNone(transfer["conditional_third"]["comparison"])

    def test_earlier_saturation_within_tolerance_can_agree(self):
        p = plan()
        result = review(p, history(p, observation(p, 0, rows(delay=10)), observation(p, 1)))
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["horizon_transfer"]["provisional_review_step"], 90)
        self.assertEqual(result["horizon_transfer"]["observed_duration_range_steps"], [80, 90])

    def test_still_learning_within_tolerance_does_not_become_duration_agreement(self):
        p = plan()
        raw = rows(last=90)
        for i, row in enumerate(raw):
            row["train_loss"] = 2 - i * 0.03
        result = review(p, history(p, observation(p, 0), observation(p, 1, raw, "running")))
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["horizon_transfer"]["provisional_review_step"])
        self.assertTrue(result["horizon_transfer"]["anchor_horizon_counterexamples"])

    def test_unresolved_at_anchor_horizon_does_not_pass(self):
        p = plan()
        result = review(p, history(p, observation(p, 0), observation(p, 1, rows(last=80, delay=10), "running")))
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["horizon_transfer"]["provisional_review_step"])

    def test_large_duration_mismatch_is_disagreement(self):
        p = plan()
        result = review(p, history(p, observation(p, 0, rows(last=150, delay=40)), observation(p, 1)))
        self.assertEqual(result["status"], "disagreement")

    def test_third_cannot_majority_vote_away_disagreement(self):
        p = plan()
        result = review(p, history(p, observation(p, 0), observation(p, 1, rows(last=150, delay=40)),
                                   observation(p, 2)))
        self.assertEqual(result["status"], "disagreement")
        self.assertEqual(result["horizon_transfer"]["initial_pair_status"], "disagreement")
        self.assertTrue(result["evidence"][p["third_trial_id"]]["verified_saturation"])
        self.assertIsNone(result["horizon_transfer"]["provisional_review_step"])

    def test_third_cannot_replace_unresolved_primary(self):
        p = plan()
        result = review(p, history(p, observation(p, 0), observation(p, 1, rows(last=20), "pruned"),
                                   observation(p, 2)))
        self.assertEqual(result["status"], "inconclusive")

    def test_unplanned_third_requires_revision_not_synthetic_job(self):
        p = plan(third=False)
        result = review(p, history(p, observation(p, 0), observation(p, 1, rows(last=150, delay=40))))
        third = result["horizon_transfer"]["conditional_third"]
        self.assertTrue(third["revision_needed"])
        self.assertIsNone(third["trial_id"])
        self.assertNotIn("actions", result)

    def test_known_third_counterexample_is_not_hidden_by_agreeing_pair(self):
        p = plan()
        result = review(p, history(p, observation(p, 0), observation(p, 1),
                                   observation(p, 2, rows(last=150, delay=40))))
        self.assertEqual(result["horizon_transfer"]["initial_pair_status"], "supported")
        self.assertEqual(result["status"], "disagreement")

    def test_supplied_later_third_within_tolerance_does_not_force_unnecessary_diagnosis(self):
        p = plan()
        result = review(p, history(p, observation(p, 0), observation(p, 1, rows(delay=10)),
                                   observation(p, 2, rows(delay=20))))
        transfer = result["horizon_transfer"]
        self.assertEqual(result["status"], "supported")
        self.assertEqual(transfer["provisional_review_step"], 100)
        self.assertEqual(transfer["observed_duration_range_steps"], [80, 100])
        self.assertFalse(transfer["conditional_third"]["required_for_diagnosis"])
        self.assertEqual(transfer["conditional_third"]["comparison"]["status"], "supported")
        self.assertTrue(transfer["anchor_horizon_counterexamples"])

    def test_third_must_fit_complete_duration_range_not_just_anchor_distance(self):
        p = plan()
        result = review(p, history(p, observation(p, 0, rows(delay=20)), observation(p, 1),
                                   observation(p, 2, rows(delay=30))))
        self.assertEqual(result["horizon_transfer"]["initial_pair_status"], "supported")
        self.assertEqual(result["horizon_transfer"]["observed_duration_range_steps"], [80, 110])
        self.assertEqual(result["status"], "disagreement")

    def test_supplied_third_still_learning_at_shared_horizon_vetoes_support(self):
        p = plan()
        result = review(p, history(p, observation(p, 0), observation(p, 1, rows(delay=10)),
                                   observation(p, 2, rows(last=90, delay=20), "running")))
        self.assertEqual(result["horizon_transfer"]["initial_pair_status"], "supported")
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["horizon_transfer"]["provisional_review_step"])

    def test_regrowth_after_plateau_and_later_replateau_cannot_erase_counterexample(self):
        p = plan()
        raw = rows(last=250)
        for row in raw:
            if 120 <= row["step"] < 150:
                row["train_loss"] = 0.3
                row["val_score"] = 0.9
            elif row["step"] >= 150:
                row["train_loss"] = 0.2
                row["val_score"] = 0.95
        self.assertEqual(plateau_decision(policy(), raw)["status"], "saturated")
        result = review(p, history(p, observation(p, 0), observation(p, 1, raw)))
        self.assertEqual(result["status"], "disagreement")
        self.assertTrue(result["evidence"][p["baseline_trial_id"]]["revocations"])

    def test_revoked_anchor_is_not_calibrated(self):
        p = plan()
        raw = rows()
        raw[-1]["diagnostics_ok"] = False
        result = review(p, history(p, observation(p, 0, raw), observation(p, 1)))
        self.assertEqual(result["status"], "disagreement")
        self.assertFalse(result["saturation_calibration_complete"])
        self.assertIsNone(result["anchor_calibration"]["observed_saturation_step"])

    def test_guard_noise_no_learning_and_schedule_risk_are_not_support(self):
        for key in ("diagnostics_ok", "schedule_clear", "train_loss"):
            p = plan()
            raw = rows()
            for row in raw:
                row[key] = 2.0 if key == "train_loss" else False
            result = review(p, history(p, observation(p, 0), observation(p, 1, raw, "running")))
            with self.subTest(key=key):
                self.assertEqual(result["status"], "inconclusive")

    def test_hard_cap_without_plateau_is_inconclusive(self):
        p = plan()
        result = review(p, history(p, observation(p, 0, rows(last=500, stalled=True), "running")))
        self.assertEqual(result["status"], "inconclusive")
        self.assertFalse(result["saturation_calibration_complete"])

    def test_stale_plan_hash_and_changed_policy_rejected(self):
        p = plan()
        h = history(p, observation(p, 0))
        p["tolerance_steps"] += 10
        with self.assertRaisesRegex(ValueError, "different immutable plan"):
            review(p, h)

    def test_duplicate_unknown_or_unselected_observations_rejected(self):
        p = plan(third=False)
        for observations in ([observation(p, 0), observation(p, 0)], [observation(p, 2)],
                             [{"trial_id": "d" * 64}]):
            with self.subTest(observations=observations), self.assertRaises(ValueError):
                review(p, history(p, *observations))

    def test_missing_duplicate_off_cadence_or_unaligned_exposure_rejected(self):
        for mutation in ("missing", "duplicate", "step", "exposure", "step_zero", "beyond_cap"):
            p = plan()
            raw = rows()
            if mutation == "missing":
                del raw[2]
            elif mutation == "duplicate":
                raw.insert(2, copy.deepcopy(raw[2]))
            elif mutation == "step":
                raw[2]["step"] += 1
            elif mutation == "exposure":
                raw[2]["exposure"] += 1
            elif mutation == "step_zero":
                del raw[0]
            else:
                raw = rows(last=510)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                review(p, history(p, observation(p, 0, raw)))

    def test_nonfinite_raw_metrics_rejected(self):
        p = plan()
        raw = rows()
        raw[-1]["train_loss"] = float("nan")
        with self.assertRaises(ValueError):
            review(p, history(p, observation(p, 0, raw)))

    def test_deterministic_and_does_not_mutate_inputs(self):
        p = plan()
        h = history(p, observation(p, 0), observation(p, 1))
        before = copy.deepcopy((p, h))
        self.assertEqual(review(p, h), review(p, h))
        self.assertEqual((p, h), before)


class CLITests(unittest.TestCase):
    def run_cli(self, plan_text, history_text=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "plan.json").write_text(plan_text, encoding="utf-8")
            command = [sys.executable, str(Path(__file__).with_name("horizon_review.py")),
                       str(root / "plan.json")]
            if history_text is not None:
                (root / "history.json").write_text(history_text, encoding="utf-8")
                command += ["--history", str(root / "history.json")]
            return subprocess.run(command, text=True, capture_output=True, check=False)

    def test_cli_plan_and_replayed_history(self):
        p = plan()
        pending = self.run_cli(json.dumps(p))
        self.assertEqual(pending.returncode, 0, pending.stderr)
        self.assertEqual(json.loads(pending.stdout)["status"], "pending")
        result = self.run_cli(json.dumps(p), json.dumps(history(p, observation(p, 0), observation(p, 1))))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "supported")

    def test_cli_duplicate_keys_nan_and_overflow_fail_closed(self):
        for content in ('{"schema_version":1,"schema_version":1}', '{"seed":NaN}', '{"seed":1e999}'):
            result = self.run_cli(content)
            with self.subTest(content=content):
                self.assertEqual(result.returncode, 2)
                self.assertEqual(json.loads(result.stderr)["status"], "invalid_input")

    def test_cli_deeply_nested_plan_and_history_return_structured_invalid_input(self):
        deep = '{"nested":' + '[' * 5000 + '0' + ']' * 5000 + '}'
        for result in (self.run_cli(deep), self.run_cli(json.dumps(plan()), deep)):
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stderr)["status"], "invalid_input")
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
