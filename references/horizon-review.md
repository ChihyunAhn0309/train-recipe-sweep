# Adaptive horizon cross-check

`scripts/horizon_review.py` is a Python 3.9+ standard-library evidence helper. It reuses the actual minimum-positive-LR sweep anchor and a distinct source-derived baseline in the same convergence family. If the baseline and anchor are the same immutable trial, preselect another existing representative. Both role names on one trial still provide only one trajectory.

Start with these two participants. Preselect an optional third existing candidate before observing their outcomes; consult its evidence only when duration mismatch or unresolved evidence warrants diagnosis. Agreement does not require a third full run. A configured third that has no observations adds no requirement when the pair agrees. Already supplied third evidence is nevertheless inspected: a known counterexample must not be hidden. When diagnosis is needed and no third was preselected, the result requests an explicit plan revision; it never invents a configuration or launches work.

Minimum-LR calibration and horizon transfer are separate claims. The anchor's replayed plateau can complete calibration while transfer remains pending, disagrees, or is inconclusive. The minimum LR is not assumed to converge last. Pair agreement is empirical evidence for the observed participants and seed; it does not certify a universal maximum duration. Every trial retains its own progress, diagnostic, schedule, patience and plateau guards. An improving trial may need more exposure than the anchor within its unchanged schedule, immutable cap and authorized budget.

## Small API and exact boundaries

```python
from horizon_review import validate_plan, review
configs_by_id = validate_plan(plan)
decision = review(plan, history)  # history may be omitted for plan-only pending state
```

```sh
python scripts/horizon_review.py horizon-plan.json
python scripts/horizon_review.py horizon-plan.json --history horizon-history.json
python -B -m unittest discover -s scripts -p test_horizon_review.py -v
```

The CLI writes JSON and exits 0 when it computed a decision; 0 is not evidence of saturation. Invalid input writes `status=invalid_input` to stderr and exits 2. Duplicate JSON keys, nonfinite numbers, excessive JSON nesting, stale plan hashes, invalid cadence and exposure mismatches fail closed. No training packages, network access, GPU, process control, installation, schedule editing or budget extension are involved.

This helper is **not integrated into `study_controller.py`, `fast_search.py`, or a real trainer**. The caller must implement and test that connection. The existing controller's anchor dependency is unchanged; assigning a horizon-review participant does not make a higher-LR trial a controller anchor. Selection and role metadata belong in this external manifest, never in a retroactively edited experimental config. Do not change the old controller roles to bypass its dependency gate. A separately tested scheduler may admit budgeted bootstrap probes for the initial pair; cross-check evidence must not be required before running the very trials needed to create it. Broad transfer assumptions and competitive pruning remain conservative while evidence is pending.

## Pinned plan v1

The plan has exactly these fields:

| Field | Meaning |
|---|---|
| `schema_version` | Integer `1` |
| `seed` | One prespecified nonnegative integer seed shared by participants |
| `comparison_context_sha256` | Full lowercase SHA-256 of the immutable comparison manifest: model/checkpoint, data/splits/probe, evaluator and exposure semantics |
| `baseline_source_sha256` | Full SHA-256 of the source recipe/reconciliation evidence identifying the baseline or source-nearest matched representative |
| `candidates` | Full immutable configs of the existing finite sweep cohort in this convergence family; the helper verifies the actual minimum over this list |
| `anchor_trial_id` | `sweep_guard.trial_id(config)` for the actual minimum positive common LR scale |
| `baseline_trial_id` | ID of the preselected source-derived baseline in this family |
| `representative_trial_id` | A distinct existing candidate when anchor and baseline IDs coincide; otherwise `null` |
| `third_trial_id` | Another distinct existing candidate for conditional diagnosis, or `null` |
| `tolerance_steps` | Prespecified nonnegative duration tolerance in optimizer updates, aligned to evaluation cadence |
| `exposure` | `{ "unit": "updates" / "samples" / "tokens", "per_step": positive_integer }`; `updates` requires `per_step=1` |

Plan selection, numerical tolerances, context, seed and the optional third must be recorded before outcome-based selection. The helper hashes this complete plan; the caller stores and protects its original snapshot. A hash alone does not prove that a plan was prespecified or that the supplied candidate catalog covers the real sweep. Candidate generation, source provenance and minimum-positive-LR coverage are caller audits. A newly introduced lower LR invalidates the old minimum anchor. Independent LR vectors/Pareto minima require separately designed anchor cohorts.

**V1 intentionally supports a narrow, conservative config schema.** Every config already contains `seed`, `comparison_context_sha256`, positive numeric `lr_scale`, nonempty positive `lr_ratios` mapping, complete `stop_policy` consumed by `plateau_decision`, nonempty `schedule`, and a nonempty `family` manifest. Include all other scientific identity fields too. Actual group LRs are the common scale times the fixed group ratios; the trainer must consume those values without hidden LR overrides. The family manifest must resolve model/checkpoint, data/splits, method/rank/targets, optimizer, effective batch, augmentation/loss, trainability, precision, probe and evaluator. Complete scheduler semantics remain in `schedule`; immutable exposure bounds and stopping semantics remain in the config. Merely giving these manifests names does not establish their completeness.

The helper validates both LR factors and every effective `lr_scale * ratio` product as finite and strictly positive. Positive finite factors whose product underflows to zero or overflows are rejected; a tiny product that remains representably positive is allowed. Complete effective LR vectors must also be numerically distinct: different scale values that round to identical group LRs cannot manufacture two recipes. One shared rounded group LR is allowed when another group actually differs. These are Python numeric checks, not evidence that the target optimizer/dtype produces distinct, nonzero parameter updates: the trainer must verify actual group values and learning/update behavior at target precision.

The helper removes **only** `lr_scale` and requires the remainder of every supplied config to hash identically. Thus an optimizer, batch, rank, schedule, ratio, policy/cap or other regime change cannot borrow this evidence, even if the caller reuses a family label. Numerically equal scales such as `1` and `1.0` cannot manufacture two distinct recipes. For another family, make a separate review plan with its own actual minimum-positive-LR anchor. If the source baseline lies outside that family, preselect the source-nearest existing same-family representative and document the deviation in the pinned source/reconciliation evidence; do not import the unrelated baseline's horizon.

Existing configs in another schema require a reviewed adapter/helper extension that preserves their original identities and verifies the equivalent family projection. **Do not insert the required fields into an already trained config, rehash it, and call the result the old trial.** This helper's schema must be used at initial config generation or explicitly adapted before integration. It does not normalize arbitrary framework configs. Unknown config fields are retained and compared; display labels, output paths, roles and review points should never have been part of experimental identity.

All participants therefore have identical evaluation cadence, immutable stop policy and schedule, fixed seed and comparable exposure mapping. V1 uses exact integer samples/tokens per update; variable token exposure, different batches/cadences or interpolated histories require a separately reviewed implementation. Do not convert epochs into apparently comparable steps without the exact sampler/exposure contract. The helper checks the supplied mapping; the trainer must attest actual counts.

## Raw history and replay

A complete immutable snapshot has exactly `plan_sha256` and `observations`. Each observation has exactly `trial_id`, `state`, and `records`. Records are the complete raw history from the untrained step-0 evaluation at the immutable exact cadence, with `step`, `exposure`, `train_loss`, `val_score`, `diagnostics_ok`, `schedule_clear`, and `train_score` when the policy requires it. Extra metric fields are allowed. `exposure` must equal `step * exposure.per_step`. Missing observations are a normal pending state; a supplied history cannot omit its initial row or internal evaluations.

Allowed operational states are `running`, `completed`, `saturated`, `budget_exhausted`, `capped`, `pruned`, `stalled`, `failed`, and `interrupted`. These labels never prove saturation. Each prefix is replayed through the existing `plateau_decision`; the first confirmed plateau supplies the duration estimate. A capped, budget-exhausted, pruned, stalled, failed or interrupted result is not accepted as calibration evidence, even when its metrics look flat. A genuinely observed plateau at the immutable cap can be accepted if the actual termination was a verified plateau, rather than censoring. Preserve the real reason.

All later observations are also replayed. A later non-plateau decision revokes an earlier plateau and remains in `revocations`, even if the final record plateaus again. Do not remove an inconvenient middle interval, crop the history at the first plateau, substitute a claimed saturation step, or overwrite prior snapshots. The caller must verify config/history/checkpoint provenance, append-only lineage and snapshot integrity; the helper cannot detect fabricated metrics or rollback to an otherwise valid older snapshot. It returns hashes of the plan, complete history, metric records, stop policies and config family for auditing. Complete-prefix replay prioritizes transparency over speed and has quadratic work in the number of evaluations; it is intended for ordinary evaluation histories, not per-minibatch telemetry.

## Decision meaning

Outputs include `horizon_transfer_status` (also top-level `status`), `saturation_calibration_complete`, separate `anchor_calibration` and `horizon_transfer` evidence, and per-participant replay details. Duration agreement and safety of the shorter anchor horizon are distinct claims: the helper preserves the latter's counterexamples even when completed durations agree within tolerance.

| Transfer state | Meaning |
|---|---|
| `pending` | Necessary observations are absent or a running participant has not yet reached the comparison point. Budgeted initial-pair probes can create that evidence. |
| `supported` | The distinct initial pair has unrevoked replayed plateaus whose absolute duration difference fits the declared tolerance, with no supplied contradictory third evidence. This is symmetric participant/seed duration evidence only. |
| `disagreement` | Observed saturation durations span more than the declared tolerance, or an observed plateau is later revoked. |
| `inconclusive` | Censoring, stalled learning, guards, noisy/unresolved evidence or insufficient terminal histories prevent the comparison. |

Tolerance describes agreement between **observed, unrevoked plateau durations**; it is not permission to truncate a still-improving candidate. For plateaus at steps 80 and 90 with tolerance 20, either ordering supports a provisional review at step 90. It does not support stopping both at step 80. A running candidate at step 90 that has not demonstrated saturation remains pending/inconclusive, even though its current step lies inside the tolerance. Unresolved guards, learning or censoring cannot be substituted for an observed duration.

`horizon_transfer.provisional_review_step` and `provisional_review_exposure` are provided only when transfer is supported and use the **latest observed plateau among the reviewed participants**. With only the required pair, that is `max(pair durations)`. `observed_duration_range_steps` records the minimum and maximum verified durations when at least two are available. `anchor_review_step` and `anchor_review_exposure` retain the anchor's own calibration estimate; they are not the recommended shared review point.

`horizon_transfer.anchor_horizon_counterexamples` separately retains later observed plateaus and non-plateau/progress observations at or after the original anchor horizon, with participant IDs, steps and reasons. A later within-tolerance plateau can complete duration agreement while this list continues to show why the shorter anchor horizon was insufficient. Subsequent agreement never rewrites that earlier evidence. A revoked plateau or duration outside tolerance remains disagreement. Much earlier saturation outside tolerance is a duration mismatch, not a failed training run.

`horizon_transfer.conditional_third.required_for_diagnosis` is an evidence request after initial-pair disagreement or inconclusive evidence. It is **not a launch command or requirement to complete a third full run**. `revision_needed=true` means no eligible third was preselected. Existing evidence may suffice; any additional observations must fit declared remaining limits. Complete within-tolerance primary plateaus never trigger a third merely because the baseline took longer than the anchor. A third's agreement cannot erase a primary duration mismatch/revocation by majority vote and cannot silently replace an unresolved required primary participant. Use the third to understand heterogeneity and decide whether an explicit family/policy revision is justified; preserve the original result.

Already supplied third evidence is checked against the **entire observed duration range**. For example, primary plateaus at 80/90 and an existing third at 100 fit tolerance 20; transfer remains supported and the provisional review point becomes 100. Primary plateaus at 80/100 and a third at 110 span 30, so tolerance 20 cannot support transfer even though the third is close to the anchor at 100. A revoked, out-of-range or unresolved supplied third vetoes support; a harmless later plateau inside the complete range's tolerance does not. An absent configured third remains unused when the primary pair agrees.

The provisional review point is an operational estimate, not a stop instruction. `authorizes_stop_or_budget_change=false` and `per_trial_guards_required=true` always. Keep individual stopping decisions active, including for new candidates whose own histories contradict transfer. Neither duration agreement nor taking the later observed step proves a universal upper bound for new recipes. Paired fresh-seed confirmation, global convergence, full-search coverage and target-GPU verification are separate requirements not established by this review.

## Incremental compute and bounded fast mode

Count training once by immutable trial ID and exact continuation lineage. Reusing already planned/completed anchor and baseline metrics for this check has **zero additional training cost**; their original training cost remains in the study ledger. Selecting a third that was already going to run to the same endpoint adds zero training solely for review, though changing priority can have opportunity/deadline costs. Do not describe all anchor/baseline work as free.

For example, an anchor and baseline each already scheduled for up to 2 GPU-hours remain the same two trials and the same authorized reservation. Comparing their histories adds only CPU review. If the existing baseline needs an extra 30-minute diagnostic tail on one allocated GPU that would otherwise have stopped, that tail costs 0.5 additional GPU-hours; it must fit both its original schedule/cap and the study's remaining allowance. If a third was planned for a short rung but review needs 20 extra minutes on one GPU, those 20 minutes are real extra work, not free reuse. Use the actual physical-device allocation ledger, including failures/idle allocation and packing, rather than summing overlapping worker wall time as if every worker owned a separate GPU. Preserve shared prefixes exactly once and never restart a trajectory from initialization to obtain the same evidence.

Fast mode should pin a maximum diagnostic exposure/time allowance and finish reserve before launch. One illustrative policy is: reuse the initial pair; if unresolved, inspect the preselected third's existing history; allow at most one bounded continuation chunk per unresolved participant within its unchanged cap/schedule and the remaining diagnostic allowance; otherwise return pending/inconclusive/disagreement with the limitation. The numerical allowance is task-specific and must be affordable; it is not hardcoded here. Never force three full saturation runs or automatically increase a cap, schedule endpoint, total GPU budget or deadline. A fixed-exposure diagnostic search with incomplete calibration must retain `saturation_calibration_complete=false`. The scheduler adapter, not this helper, enforces the chosen limits, atomic deduplication, exact resume, finish reserve and between-evaluation deadlines.

## Runnable CPU fixture inputs

From the repository root, this standard-library fixture uses the test module's explicit synthetic policy and histories to write real CLI inputs. It does not train or claim target evidence:

```sh
python -c "import sys,json; from pathlib import Path; sys.path.insert(0,'scripts'); from test_horizon_review import plan,history,observation; p=plan(); Path('horizon-plan.json').write_text(json.dumps(p),encoding='utf-8'); Path('horizon-history.json').write_text(json.dumps(history(p,observation(p,0),observation(p,1))),encoding='utf-8')"
python scripts/horizon_review.py horizon-plan.json --history horizon-history.json
```

The expected transfer status is `supported`, with a calibrated synthetic anchor and no third required. These numbers are test fixtures, not a task-ready saturation policy. Replace them only through a prespecified, source-backed real study design and a verified trainer adapter.
