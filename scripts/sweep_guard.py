#!/usr/bin/env python3
"""Standard-library trial fingerprint and conservative plateau decision helper.

No training, network, process control, or checkpoint mutation is performed.
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def parse_json(text):
    result = json.loads(text, object_pairs_hook=_unique_object,
                        parse_constant=_bad_constant)
    _check_json(result)  # A valid exponent such as 1e999 can overflow to Inf.
    return result


def _check_json(value):
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("Non-finite number in experiment")
        return
    if isinstance(value, list):
        for item in value:
            _check_json(item)
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _check_json(item)
        return
    raise ValueError(f"Unsupported JSON type: {type(value).__name__}")


def trial_id(experiment):
    if not isinstance(experiment, dict) or not experiment:
        raise ValueError("Experiment must be a non-empty JSON object")
    _check_json(experiment)
    canonical = json.dumps(experiment, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


REQUIRED = {
    "direction", "window", "confirmations", "val_patience", "min_step",
    "warmup_end_step", "schedule_guard_step", "eval_every_steps", "hard_cap_step",
    "train_rel_delta", "train_noise_rel_max", "val_min_delta",
    "min_train_progress_rel", "loss_scale_floor",
}
SCORE_FIELDS = {"train_score_direction", "train_score_min_delta"}
OPTIONAL = SCORE_FIELDS | {"train_abs_delta", "train_noise_abs_max"}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _integer(value, label, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def validate_policy(policy):
    if not isinstance(policy, dict):
        raise ValueError("Policy must be an object")
    missing = REQUIRED - policy.keys()
    extra = policy.keys() - REQUIRED - OPTIONAL
    if missing or extra:
        raise ValueError(f"Policy missing={sorted(missing)}, unknown={sorted(extra)}")
    if policy["direction"] not in ("min", "max"):
        raise ValueError("direction must be min or max")
    for key in ("window", "confirmations", "val_patience", "eval_every_steps",
                "hard_cap_step"):
        _integer(policy[key], key, 3 if key == "window" else 1)
    if policy["hard_cap_step"] % policy["eval_every_steps"]:
        raise ValueError("hard_cap_step must be a multiple of eval_every_steps")
    for key in ("min_step", "warmup_end_step", "schedule_guard_step"):
        _integer(policy[key], key)
        if policy[key] > policy["hard_cap_step"]:
            raise ValueError(f"{key} exceeds hard_cap_step")
    for key in ("train_rel_delta", "train_noise_rel_max", "val_min_delta",
                "min_train_progress_rel", "loss_scale_floor"):
        value = _number(policy[key], key)
        if value < 0 or (key in ("loss_scale_floor", "min_train_progress_rel") and value == 0):
            raise ValueError(f"Invalid {key}")
    for key in ("train_abs_delta", "train_noise_abs_max"):
        if key in policy and _number(policy[key], key) < 0:
            raise ValueError(f"{key} must be nonnegative")
    if bool(SCORE_FIELDS & policy.keys()) and not SCORE_FIELDS <= policy.keys():
        raise ValueError("Provide both optional train-score policy fields")
    if SCORE_FIELDS <= policy.keys():
        if policy["train_score_direction"] not in ("min", "max"):
            raise ValueError("train_score_direction must be min or max")
        if _number(policy["train_score_min_delta"], "train_score_min_delta") < 0:
            raise ValueError("train_score_min_delta must be nonnegative")


def _slope(values, steps):
    center_x, center_y = statistics.mean(steps), statistics.mean(values)
    denominator = sum((x - center_x) ** 2 for x in steps)
    numerator = sum((x - center_x) * (y - center_y)
                    for x, y in zip(steps, values))
    _number(numerator, "intermediate slope numerator")
    _number(denominator, "intermediate slope denominator")
    if denominator <= 0:
        raise ValueError("Invalid slope denominator")
    return _number(numerator / denominator, "intermediate slope")


def _trend(values, steps):
    """Absolute fitted change over window span, not a confidence interval."""
    return _number(abs(_slope(values, steps)) * (steps[-1] - steps[0]),
                   "intermediate fitted trend")


def _residual_mad(values, steps):
    """Separate linear progress from within-window noise."""
    slope = _slope(values, steps)
    residuals = [value - slope * (step - steps[0])
                 for step, value in zip(steps, values)]
    _check_json(residuals)
    center = statistics.median(residuals)
    return _number(statistics.median(abs(value - center) for value in residuals),
                   "intermediate residual MAD")


def _residual_max(values, steps):
    """Do not let a sparse loss spike disappear inside a median-based noise test."""
    slope = _slope(values, steps)
    residuals = [value - slope * (step - steps[0])
                 for step, value in zip(steps, values)]
    _check_json(residuals)
    center = statistics.median(residuals)
    return _number(max(abs(value - center) for value in residuals),
                   "intermediate maximum residual")


def plateau_decision(policy, records):
    validate_policy(policy)
    if not isinstance(records, list) or not records:
        raise ValueError("History must be a non-empty list")
    optional_score = SCORE_FIELDS <= policy.keys()
    previous_step = None
    nonfinite = False
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record {index} must be an object")
        keys = {"step", "train_loss", "val_score", "diagnostics_ok", "schedule_clear"}
        if optional_score:
            keys.add("train_score")
        if not keys <= record.keys():
            raise ValueError(f"Record {index} missing {sorted(keys - record.keys())}")
        _integer(record["step"], f"record {index} step")
        if record["step"] > policy["hard_cap_step"]:
            raise ValueError("History extends beyond immutable hard_cap_step")
        if index == 0 and record["step"] != 0:
            raise ValueError("First record must be the untrained step-0 evaluation")
        if previous_step is not None:
            if record["step"] - previous_step != policy["eval_every_steps"]:
                raise ValueError("Steps must be unique, ordered, and follow eval cadence")
        previous_step = record["step"]
        for key in ("diagnostics_ok", "schedule_clear"):
            if not isinstance(record[key], bool):
                raise ValueError(f"Record {index} {key} must be boolean")
        for key in ("train_loss", "val_score") + (("train_score",) if optional_score else ()):
            value = record[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Record {index} {key} must be numeric")
            nonfinite = nonfinite or not math.isfinite(value)
    latest = records[-1]
    step = latest["step"]
    if nonfinite:
        return {"status": "diverged", "stop": True, "reason": "nonfinite_metric",
                "step": step, "right_censored": False}

    sign = 1 if policy["direction"] == "max" else -1
    best = max(records, key=lambda row: sign * row["val_score"])
    details = {"best_val_score": best["val_score"], "best_step": best["step"]}

    def decision(reason, saturated=False):
        _check_json(details)
        exhausted = step >= policy["hard_cap_step"] and not saturated
        return {
            "status": "saturated" if saturated else
                      "budget_exhausted" if exhausted else "continue",
            "stop": saturated or exhausted,
            "reason": reason,
            "step": step,
            "right_censored": exhausted,
            "details": details,
        }

    guard = max(policy["min_step"], policy["warmup_end_step"],
                policy["schedule_guard_step"])
    if step < guard:
        return decision("minimum_exposure_or_schedule_guard")
    if not latest["diagnostics_ok"]:
        return decision("diagnostics_not_passed")
    if not latest["schedule_clear"]:
        return decision("future_schedule_event_or_grace_pending")

    window, confirmations = policy["window"], policy["confirmations"]
    eligible = [row for row in records if row["step"] >= guard]
    required = 2 * window + confirmations - 1
    if len(eligible) < required:
        return decision("insufficient_post_guard_history")
    tail = eligible[-required:]
    if not all(row["diagnostics_ok"] and row["schedule_clear"] for row in tail):
        return decision("guard_not_clear_through_confirmation_windows")

    losses = [row["train_loss"] for row in records]
    best_smoothed_loss = min(_number(statistics.median(losses[i:i + window]),
                                    "intermediate smoothed loss")
                             for i in range(len(losses) - window + 1))
    scale0 = max(abs(losses[0]), policy["loss_scale_floor"])
    progress = (losses[0] - best_smoothed_loss) / scale0
    details["observed_train_progress_rel"] = progress
    if progress < policy["min_train_progress_rel"]:
        return decision("no_verified_learning_progress")

    meaningful_best = sign * records[0]["val_score"]
    last_improvement = 0
    for index, row in enumerate(records[1:], 1):
        value = sign * row["val_score"]
        if value > meaningful_best + policy["val_min_delta"]:
            meaningful_best = value
            last_improvement = index
    age = len(records) - 1 - last_improvement
    details["evals_since_meaningful_val_improvement"] = age

    assessments = []
    for offset in range(confirmations):
        end = len(eligible) - offset
        older = eligible[end - 2 * window:end - window]
        newer = eligible[end - window:end]
        old_loss = [row["train_loss"] for row in older]
        new_loss = [row["train_loss"] for row in newer]
        old_center = _number(statistics.median(old_loss), "intermediate old loss median")
        new_center = _number(statistics.median(new_loss), "intermediate new loss median")
        scale = max(abs(old_center), policy["loss_scale_floor"])
        change = _number(abs(old_center - new_center) / scale, "intermediate loss change")
        trend = max(_trend([row["train_loss"] for row in part],
                           [row["step"] for row in part]) for part in (older, newer)) / scale
        noise = max(_residual_mad([row["train_loss"] for row in part],
                                  [row["step"] for row in part])
                    for part in (older, newer)) / scale
        max_noise = max(_residual_max([row["train_loss"] for row in part],
                                      [row["step"] for row in part])
                        for part in (older, newer))
        assessment = {"end_step": newer[-1]["step"], "relative_change": change,
                      "relative_trend": trend, "relative_mad": noise,
                      "absolute_change": change * scale,
                      "absolute_trend": trend * scale,
                      "absolute_mad": noise * scale,
                      "absolute_max_residual": max_noise,
                      "relative_max_residual": max_noise / scale,
                      "loss_flatness_limit": max(policy.get("train_abs_delta", 0.0),
                                                  policy["train_rel_delta"] * scale),
                      "loss_noise_limit": max(policy.get("train_noise_abs_max", 0.0),
                                               policy["train_noise_rel_max"] * scale)}
        if optional_score:
            score_change = abs(statistics.median(row["train_score"] for row in older) -
                               statistics.median(row["train_score"] for row in newer))
            score_trend = max(_trend([row["train_score"] for row in part],
                                     [row["step"] for row in part])
                              for part in (older, newer))
            assessment["train_score_change"] = score_change
            assessment["train_score_trend"] = score_trend
            assessment["train_score_max_residual"] = max(
                _residual_max([row["train_score"] for row in part],
                              [row["step"] for row in part]) for part in (older, newer))
        _check_json(assessment)
        assessments.append(assessment)
    details["assessments"] = assessments
    if any(item["absolute_mad"] > item["loss_noise_limit"] or
           item["absolute_max_residual"] > max(item["loss_noise_limit"],
                                               item["loss_flatness_limit"])
           for item in assessments):
        return decision("train_probe_too_noisy")
    if any(max(item["absolute_change"], item["absolute_trend"]) > item["loss_flatness_limit"]
           for item in assessments):
        return decision("train_loss_not_flat")
    if optional_score and any(max(item["train_score_change"], item["train_score_trend"],
                                 item["train_score_max_residual"]) >
                              policy["train_score_min_delta"] for item in assessments):
        return decision("train_score_not_flat")
    if age < policy["val_patience"]:
        return decision("validation_still_improving_or_patience_pending")
    return decision("confirmed_train_and_validation_plateau", saturated=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fingerprint = sub.add_parser("identity", help="Hash fully resolved experiment JSON")
    fingerprint.add_argument("experiment", type=Path)
    plateau = sub.add_parser("plateau", help="Evaluate policy against complete metric history")
    plateau.add_argument("--policy", type=Path, required=True)
    plateau.add_argument("--metrics", type=Path, required=True,
                         help="JSON array or .jsonl; include initial step-0 evaluation")
    args = parser.parse_args()
    try:
        if args.command == "identity":
            result = {"trial_id": trial_id(parse_json(args.experiment.read_text(encoding="utf-8-sig")))}
        else:
            policy = parse_json(args.policy.read_text(encoding="utf-8-sig"))
            content = args.metrics.read_text(encoding="utf-8-sig")
            records = ([parse_json(line) for line in content.splitlines() if line.strip()]
                       if args.metrics.suffix.lower() == ".jsonl" else parse_json(content))
            result = plateau_decision(policy, records)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ValueError, OSError, TypeError, OverflowError) as error:
        print(json.dumps({"status": "invalid_input", "error": str(error)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
