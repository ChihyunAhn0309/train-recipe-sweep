# What a recipe comparison can establish

Read when defining baseline comparisons, selecting finalists, or assessing whether a sweep works. Software correctness, source recovery, local performance and search quality need different evidence. CPU execution can include real small-model training; a fixture worker is not such training. Label each executed command accordingly.

## Register the comparison before searching

Keep these separate for every model, task and method/rank:

- **Published source result:** metric/direction/scale, evaluation split, data release, checkpoint and context. A published search range is not the winning configuration. Code defaults do not prove a historical run used them. Pretraining and downstream fine-tuning recipes differ.
- **Source-equivalent reproduction baseline:** relevant weights, data membership, preprocessing, evaluator, optimizer implementation, parameter groups, schedule, exposure and selection protocol are recovered and matched. Declare a tolerance appropriate to nondeterminism. Matching hyperparameters alone does not establish reproduction.
- **Source-derived target baseline:** the usual transfer setting. Preserve compatible source techniques and record every head/data/framework change or other deviation. Historical unknowns remain unknown; their target replacements are proposed. Epsilon/bias correction, schedule, input length, subset and head LR can change an experiment despite matching optimizer/model names.
- **Proposed baseline:** identify the closest applicable evidence when exact model/task/rank sources are unavailable. Do not use another architecture's LoRA score as this model's reference score.

Evaluate the registered target baseline, not just the untrained/reheaded model. The latter diagnoses learning but cannot establish improvement over a fine-tuned baseline. Register baseline trials, deduplicate identical baseline/anchor candidates and follow the declared checkpoint policy. Duration completion alone cannot establish minimum-LR saturation. An identical minimum-LR baseline/anchor trial can satisfy both roles if its observed history actually passes the calibration criteria.

Register a source-linked comparison contract:

| Field | Meaning |
|---|---|
| `baseline_kind` | Reproduction, source-derived target or proposed; unknowns/deviations |
| `comparison_context` | Model/weights/head, train/validation membership and independent groups, tokenizer/transforms, metric/evaluator hashes, exposure/selection policy |
| `selection` | Objective, checkpoint rule, practical effect/tie margin, finalist rule and search budget |
| `seed_policy` | Exploration seeds, reserved fresh confirmation seeds, paired baseline/finalist runs, aggregation and missing-run policy |
| `claim_scope` | Best observed, fresh-seed comparison, source reproduction or search-algorithm evaluation |
| `test_policy` | Untouched test/holdout or official evaluation procedure; unavailable labels remain unavailable |

Hold validation preprocessing/evaluator fixed. Multiple training changes yield a combined delta; use prespecified ablations for causal attribution. Do not subtract scores from different datasets, splits or protocols as if paired.

## Confirmation without recycling selection evidence

Reserve compute for baseline and finalists, including close contenders, on the same **fresh, prespecified seed list**. Match splits/evaluation and state which initialization/sampling streams can actually be paired across methods. Freeze recipe IDs, selection rule/evidence hashes, seed list and aggregation in a selection artifact before confirmation. Only the seed and its stochastic realizations (initialization, data order, dropout and augmentation draws) may change; retuning LR, stopping thresholds or preprocessing policy creates a new exploratory revision.

Default intent is at least three fresh independent seeds per compared recipe when feasible; this is an engineering starting point, not guaranteed statistical power. Include all prespecified runs, report failures/censoring and do not drop poor seeds selectively. Insufficient budget yields best observed candidates with missing evidence, not a confirmed winner. A bounded smoke test need not perform expensive confirmation.

Reuse exploratory runs/costs, but keep selected-seed scores outside the fresh confirmation aggregate. Resume, relabeling or an irrelevant config change does not create an independent seed. The generic controller's `confirmation` role reserves budget and separates config identities; it cannot prove seed independence or frozen selection. The model-specific adapter enforces those policies.

Report per-seed baseline/finalist metrics, paired directional deltas where valid, mean/spread, failures/censoring, exposure and cost. Fix tie-breaking before selection. For improvement, equivalence or noninferiority claims, predeclare an effect margin and suitable uncertainty method. Overlapping point estimates or inconclusive tests do not prove equality. Seed variability and finite-data/group uncertainty differ.

Fresh seeds on reused validation data check seed stability, not untouched-test generalization. Use a prespecified holdout, group/nested CV or official evaluation service when the claim needs it. Confirmation-driven retuning turns that evidence into selection data; reserve fresh evidence and charge its cost. Hidden test labels must not be reconstructed from upstream metadata for evaluation/tuning.

## Stage transitions with durable reuse

Baselines, minimum-LR calibration and exploration share a cumulative budget. New lower bounds or incomparable regimes need new anchors. Censored calibration may support a declared diagnostic search, never completed saturation-calibrated optimization.

Use the append-only extension in [controller protocol](study-controller.md) for finalists, fresh seeds or boundary candidates. Preserve old immutable trials, worker, devices and budget caps; anchors/results/costs remain in the registry. Freeze selection before adding confirmation. Changed GPU scope needs renewed applicable acceptance evidence. Do not overwrite study JSON, delete its registry or reset allowances to simulate extension.

The helper does not enlarge budgets, mutate old caps, change worker code or select finalists. Such operations need an explicitly implemented migration/lineage procedure and scientific/target revalidation. Plans requiring them must supply and test that implementation or declare the unsupported operation. Complete handoffs include stage generation, selection freeze, extension and recovery. A scientific specification may leave implementation pending.

## Report evidence levels separately

Use outcome fields linked to evidence instead of one ambiguous `passed` or `optimal` label:

- `software_checks_passed`: revision, fixture versus real model, platform and behaviors tested.
- `source_recipe_recovery`: recovered, partial or proposed; provenance and unknowns.
- `baseline_evaluated`: valid local baseline on the declared target setting.
- `saturation_calibration_complete`: observed plateau for every required anchor; capped/stalled anchors do not pass.
- `best_observed`: best valid candidate actually evaluated; omitted axes/boundaries.
- `fresh_seed_confirmation_complete`: registered baseline/finalist runs and aggregate, with uncertainty/scope.
- `official_reproduction_verified`: matched source setting and tolerances; never inferred from LR overlap.
- `target_gpu_verified`: actual target evidence, separate from reconstruction or CPU results.

Hashes establish artifact identity, not measurement truth. Do not present an older real-model experiment as validation of later controller/coverage changes. Preserve its original code/hashes; distinguish re-evaluation from new training.

When asked to validate the **search procedure**, compare against the registered baseline and a budget-matched reference search (for example random search) over a declared common domain. Account for profiling, anchors, failures and confirmation. Repeat across appropriate study seeds/tasks before generalizing search quality. This optional benchmark is not required for ordinary fine-tuning. Recipe similarity cannot prove search quality; no finite benchmark proves global optimality.

See [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) for selection bias and [Bergstra and Bengio](https://www.jmlr.org/papers/v13/bergstra12a.html) for a reference HPO baseline. These sources do not prescribe universal seed counts, margins or budgets.
