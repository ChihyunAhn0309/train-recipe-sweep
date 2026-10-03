#!/usr/bin/env python3
"""Replay a pinned, adaptive two/conditional-three horizon review; never launch jobs."""

import argparse
import copy
import json
import math
import sys
from pathlib import Path

from sweep_guard import parse_json, plateau_decision, trial_id, validate_policy


STATES = {"running", "completed", "saturated", "budget_exhausted", "capped",
          "pruned", "stalled", "failed", "interrupted"}
USABLE_STATES = {"running", "completed", "saturated"}
PROGRESS_REASONS = {"train_loss_not_flat", "train_score_not_flat",
                    "validation_still_improving_or_patience_pending"}


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(label + " needs exactly: " + ", ".join(sorted(keys)))


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + " must be an integer >= " + str(minimum))


def _positive(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(label + " must be finite and positive")


def _sha(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(label + " must be a full lowercase SHA-256")


def _family(config):
    # Deliberately conservative: v1 transfers only across the common LR scale.
    projected = copy.deepcopy(config)
    del projected["lr_scale"]
    return trial_id(projected)


def validate_plan(plan):
    """Return configs by immutable ID; all candidates must be one LR-scale family."""
    _keys(plan, {"schema_version", "seed", "comparison_context_sha256",
                 "baseline_source_sha256", "candidates", "anchor_trial_id",
                 "baseline_trial_id", "representative_trial_id", "third_trial_id",
                 "tolerance_steps", "exposure"}, "Plan")
    if type(plan["schema_version"]) is not int or plan["schema_version"] != 1:
        raise ValueError("Expected horizon-review schema_version 1")
    trial_id(plan)  # Reject non-JSON/nonfinite values, including nested values.
    _integer(plan["seed"], "seed")
    _integer(plan["tolerance_steps"], "tolerance_steps")
    for key in ("comparison_context_sha256", "baseline_source_sha256"):
        _sha(plan[key], key)
    exposure = plan["exposure"]
    _keys(exposure, {"unit", "per_step"}, "Exposure")
    if exposure["unit"] not in ("updates", "samples", "tokens"):
        raise ValueError("Exposure unit must be updates, samples or tokens")
    _integer(exposure["per_step"], "exposure per_step", 1)
    if exposure["unit"] == "updates" and exposure["per_step"] != 1:
        raise ValueError("Update exposure must have per_step=1")
    if not isinstance(plan["candidates"], list) or not plan["candidates"]:
        raise ValueError("Candidates must list existing immutable sweep configs")
    configs = {}
    for config in plan["candidates"]:
        identity = trial_id(config)
        required = {"seed", "comparison_context_sha256", "lr_scale", "lr_ratios",
                    "stop_policy", "schedule", "family"}
        if not required <= config.keys():
            raise ValueError("Config missing " + str(sorted(required - config.keys())))
        if (type(config["seed"]) is not int or config["seed"] != plan["seed"]
                or config["comparison_context_sha256"] != plan["comparison_context_sha256"]):
            raise ValueError("Config seed/comparison context differs from plan")
        for key in ("family", "schedule", "lr_ratios"):
            if not isinstance(config[key], dict) or not config[key]:
                raise ValueError("Config needs nonempty " + key + " manifest")
        _positive(config["lr_scale"], "lr_scale")
        for group, ratio in config["lr_ratios"].items():
            _positive(ratio, "LR group ratio")
            label = "Effective LR for group " + group
            try:
                _positive(config["lr_scale"] * ratio, label)
            except OverflowError as error:
                raise ValueError(label + " must be finite and positive") from error
        validate_policy(config["stop_policy"])
        if identity in configs:
            raise ValueError("Duplicate config: one trial cannot count twice")
        configs[identity] = config
    if len({_family(config) for config in configs.values()}) != 1:
        raise ValueError("Different convergence family: only lr_scale may differ")
    if len({config["lr_scale"] for config in configs.values()}) != len(configs):
        raise ValueError("Numerically identical LR scales are not distinct recipes")
    vectors = {tuple((group, config["lr_scale"] * ratio)
                     for group, ratio in sorted(config["lr_ratios"].items()))
               for config in configs.values()}
    if len(vectors) != len(configs):
        raise ValueError("Duplicate effective LR vector: rounded aliases are not distinct recipes")
    for key in ("anchor_trial_id", "baseline_trial_id"):
        if plan[key] not in configs:
            raise ValueError(key + " must name an existing candidate")
    anchor, baseline = plan["anchor_trial_id"], plan["baseline_trial_id"]
    if configs[anchor]["lr_scale"] != min(c["lr_scale"] for c in configs.values()):
        raise ValueError("Anchor is not the actual minimum positive LR scale")
    representative = plan["representative_trial_id"]
    if anchor == baseline:
        if representative not in configs or representative == anchor:
            raise ValueError("Shared anchor/baseline needs another distinct existing representative")
        primary = {anchor, representative}
    else:
        if representative is not None:
            raise ValueError("Distinct anchor/baseline already form the initial pair")
        primary = {anchor, baseline}
    third = plan["third_trial_id"]
    if third is not None and (third not in configs or third in primary):
        raise ValueError("Conditional third must be another distinct existing candidate")
    cadence = configs[anchor]["stop_policy"]["eval_every_steps"]
    if plan["tolerance_steps"] % cadence:
        raise ValueError("Duration tolerance must align with evaluation cadence")
    return configs


def _replay(config, observation, exposure):
    if observation is None:
        return {"state": "not_observed", "verified_saturation": False,
                "first_saturation_step": None, "decisions": [], "revocations": []}
    _keys(observation, {"trial_id", "state", "records"}, "Observation")
    if observation["state"] not in STATES:
        raise ValueError("Unknown observation state")
    records = observation["records"]
    # Validate complete cadence, required fields and immutable cap before replay.
    plateau_decision(config["stop_policy"], records)
    for record in records:
        _integer(record.get("exposure"), "record exposure")
        if record["exposure"] != record["step"] * exposure["per_step"]:
            raise ValueError("History does not match pinned comparable exposure")
    decisions, first, revocations = [], None, []
    for end in range(1, len(records) + 1):
        decision = plateau_decision(config["stop_policy"], records[:end])
        decisions.append({key: decision[key] for key in ("step", "status", "reason")})
        if first is None and decision["status"] == "saturated":
            first = decision["step"]
        elif first is not None and decision["status"] != "saturated":
            revocations.append(decisions[-1])
    return {"state": observation["state"], "first_saturation_step": first,
            "verified_saturation": (first is not None and not revocations
                                    and observation["state"] in USABLE_STATES),
            "latest": decisions[-1], "decisions": decisions, "revocations": revocations,
            "records_sha256": trial_id({"records": records}),
            "stop_policy_sha256": trial_id(config["stop_policy"])}


def _comparison(anchor, other, tolerance, reference_steps=()):
    if anchor["revocations"] or other["revocations"]:
        return "disagreement", "observed_plateau_was_later_revoked"
    horizon = anchor["first_saturation_step"]
    if not anchor["verified_saturation"]:
        waiting = (anchor["state"] in ("not_observed", "running")
                   and anchor.get("latest", {}).get("status") != "budget_exhausted")
        return ("pending" if waiting else "inconclusive"), "anchor_not_calibrated"
    if other["verified_saturation"]:
        steps = [horizon, other["first_saturation_step"], *reference_steps]
        if max(steps) - min(steps) > tolerance:
            return "disagreement", "saturation_durations_outside_prespecified_tolerance"
        return "supported", "distinct_replayed_plateaus_agree_within_tolerance"
    if other["state"] == "not_observed":
        return "pending", "participant_not_observed"
    if other["state"] != "running" or other["latest"]["status"] == "budget_exhausted":
        return "inconclusive", "participant_ended_without_verified_saturation"
    if other["latest"]["step"] >= max([horizon, *reference_steps]):
        return "inconclusive", "participant_unresolved_at_or_after_reference_horizon"
    return "pending", "participant_history_still_before_reference_horizon"


def _anchor_counterexamples(anchor, evidence):
    """Keep shorter-anchor caveats separate from completed duration agreement."""
    horizon = anchor["first_saturation_step"]
    result = []
    if horizon is None:
        return result
    for identity, participant in evidence.items():
        first = participant["first_saturation_step"]
        if first is not None and first > horizon:
            result.append({"trial_id": identity, "anchor_step": horizon, "step": first,
                           "kind": "later_observed_plateau",
                           "verified_saturation": participant["verified_saturation"]})
        for decision in participant["decisions"]:
            if decision["step"] >= horizon and decision["status"] != "saturated":
                result.append({"trial_id": identity, "anchor_step": horizon, **decision,
                               "kind": "nonflat_or_improving_at_or_after_anchor_horizon"
                               if decision["reason"] in PROGRESS_REASONS else
                               "unresolved_at_or_after_anchor_horizon"})
    return result


def review(plan, history=None):
    """Return evidence only. Missing evidence never blocks budgeted bootstrap probes."""
    configs = validate_plan(plan)
    plan_hash = trial_id(plan)
    if history is None:
        history = {"plan_sha256": plan_hash, "observations": []}
    _keys(history, {"plan_sha256", "observations"}, "History")
    if history["plan_sha256"] != plan_hash:
        raise ValueError("History belongs to a different immutable plan")
    history_hash = trial_id(history)
    if not isinstance(history["observations"], list):
        raise ValueError("History observations must be a list")
    anchor_id, baseline_id = plan["anchor_trial_id"], plan["baseline_trial_id"]
    second_id = baseline_id if baseline_id != anchor_id else plan["representative_trial_id"]
    third_id = plan["third_trial_id"]
    selected = {anchor_id, second_id} | ({third_id} if third_id is not None else set())
    observed = {}
    for observation in history["observations"]:
        if not isinstance(observation, dict) or observation.get("trial_id") not in selected:
            raise ValueError("Observation is not a prespecified review participant")
        identity = observation["trial_id"]
        if identity in observed:
            raise ValueError("Duplicate participant observation")
        observed[identity] = observation
    evidence = {identity: _replay(configs[identity], observed.get(identity), plan["exposure"])
                for identity in sorted(selected)}
    anchor = evidence[anchor_id]
    pair_status, pair_reason = _comparison(anchor, evidence[second_id], plan["tolerance_steps"])
    primary_steps = [evidence[identity]["first_saturation_step"]
                     for identity in (anchor_id, second_id)
                     if evidence[identity]["verified_saturation"]]
    trigger = pair_status in ("disagreement", "inconclusive")
    status, reasons = pair_status, [pair_reason]
    # Supplied third evidence is never hidden, even when the initial pair agrees.
    third_comparison = None
    if third_id is not None and (trigger or third_id in observed):
        third_status, third_reason = _comparison(anchor, evidence[third_id],
                                                 plan["tolerance_steps"], primary_steps)
        third_comparison = {"status": third_status, "reason": third_reason}
        if third_status == "disagreement":
            status = "disagreement"
            reasons.append(third_reason)
        elif third_id in observed and third_status != "supported" and status == "supported":
            status = "inconclusive"
            reasons.append("supplied_third_evidence_is_unresolved")
        # A third cannot overwrite disagreement or stand in for missing pair evidence.
    calibrated = anchor["verified_saturation"]
    calibration_status = "complete" if calibrated else (
        "pending" if anchor["state"] in ("not_observed", "running")
        and not anchor["revocations"]
        and anchor.get("latest", {}).get("status") != "budget_exhausted" else "inconclusive")
    horizon = anchor["first_saturation_step"] if calibrated else None
    reviewed_steps = primary_steps + ([evidence[third_id]["first_saturation_step"]]
        if third_id in observed and evidence[third_id]["verified_saturation"] else [])
    provisional = max(reviewed_steps) if status == "supported" else None
    return {
        "schema_version": 1, "plan_sha256": plan_hash, "history_sha256": history_hash,
        "convergence_family_sha256": _family(configs[anchor_id]), "seed": plan["seed"],
        "status": status, "horizon_transfer_status": status,
        "saturation_calibration_complete": calibrated,
        "anchor_calibration": {"status": calibration_status, "trial_id": anchor_id,
                               "observed_saturation_step": horizon},
        "horizon_transfer": {
            "status": status, "reasons": reasons, "primary_trial_ids": [anchor_id, second_id],
            "initial_pair_status": pair_status, "anchor_review_step": horizon,
            "anchor_review_exposure": None if horizon is None else horizon * plan["exposure"]["per_step"],
            "provisional_review_step": provisional,
            "provisional_review_exposure": None if provisional is None else provisional * plan["exposure"]["per_step"],
            "observed_duration_range_steps": ([min(reviewed_steps), max(reviewed_steps)]
                                               if len(reviewed_steps) >= 2 else None),
            "anchor_horizon_counterexamples": _anchor_counterexamples(anchor, evidence),
            "exposure_unit": plan["exposure"]["unit"],
            "conditional_third": {"required_for_diagnosis": trigger, "trial_id": third_id,
                                  "revision_needed": trigger and third_id is None,
                                  "comparison": third_comparison},
            "scope": "empirical_participant_and_seed_evidence_only",
            "per_trial_guards_required": True, "authorizes_stop_or_budget_change": False,
        },
        "bootstrap": "initial_pair_may_run_within_existing_authorized_limits",
        "evidence": evidence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--history", type=Path)
    args = parser.parse_args()
    try:
        plan = parse_json(args.plan.read_text(encoding="utf-8-sig"))
        history = (parse_json(args.history.read_text(encoding="utf-8-sig"))
                   if args.history is not None else None)
        print(json.dumps(review(plan, history), ensure_ascii=False, allow_nan=False, indent=2))
        return 0
    except (ValueError, OSError, TypeError, OverflowError, RecursionError) as error:
        print(json.dumps({"status": "invalid_input", "error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
