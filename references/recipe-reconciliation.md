# Reconcile the source recipe before designing a sweep

Use for every model/checkpoint and every requested Full FT or LoRA recipe. A paper summary or a list of common hyperparameters is insufficient. Recover the recipe that actually produced the selected weights, then account explicitly for what transfers to each target method. This is a scientific comparison, not an instruction to copy every pretraining choice unchanged.

## Recover the effective source recipe

Keep checkpoint-generation, later adaptation stages, source downstream Full FT and model-specific LoRA recipes separate. For each, pin the code/config revision and preserve original files, command line, config inheritance/includes, environment and library defaults, trainer callbacks/hooks, data transforms, launcher overrides and available run logs. Resolve precedence to the effective values used for that checkpoint; preserve conflicts and unknowns when logs or historical versions cannot establish them. Current library defaults are not evidence of historical settings.

Inspect the input pipeline and actual implementation, not just optimizer settings or the model card. Record sample/step/epoch units, probability distributions, activation conditions and schedules. Capture:

- Input resize/crop policy, multi-scale/random-resize ranges or discrete sizes, probabilities, draw cadence (sample/batch/step/epoch), aspect-ratio constraints, stride rounding, padding and interpolation.
- Augmentation composition/order, flips, color transforms, mosaic/mixup/cutmix, erasing, label smoothing, probabilities, magnitude and any late-epoch disabling or curriculum.
- GT transforms for boxes, masks, keypoints and ignore regions; filtering/clipping and loss normalization after transforms.
- Dataset mixture/sampling/reweighting, tokenizer/sequence packing/truncation and curriculum where applicable.
- Optimizer/parameter groups, LR scaling and layer decay, schedule/warmup, effective batch/accumulation, exposure, clipping, loss components/matching weights, auxiliary losses, normalization/freeze phases, EMA and precision.
- Evaluation resize/crop, thresholds, matching/NMS/decoding, checkpoint selection, EMA evaluation and test-time augmentation (TTA). **Training multi-scale augmentation and multi-scale evaluation/TTA are separate factors.** Do not change the official validation protocol while comparing training recipes.

Verify these behaviors against a resolved configuration dump and inspected pipeline/callback code. In execution mode log the instantiated pipeline and sample representative transformed inputs/GT. A parser accepting `multi_scale=true` is not evidence that the trainer uses it. Compare observed dimensions/draw frequency with the specified distribution and audit GT alignment. During CPU-only planning, supply these checks in the executable handoff and distinguish checks actually run from deferred GPU profiling.

## Mandatory source-to-target decision ledger

Create a machine-readable ledger plus a readable table. Inventory **every leaf** in each resolved source configuration, including operational/evaluation fields; add code-only behavior to a source-linked normalized recipe manifest. Every source leaf must map to exactly one factor row. Group related leaves when needed, but do not hide missing keys behind “other defaults.” Keep original artifacts and provenance for the normalized manifest: complete JSON coverage alone cannot establish complete source recovery.

For each row record its factor, source keys and stage, exact source evidence, evidence status (`verified`, `derived`, `proposed`, `unknown`), and whether it admits a meaningful enabled/disabled comparison. For **every requested method/rank**, record:

| Field | Meaning |
|---|---|
| disposition | `sweep`, `fixed`, `conditional`, or `excluded` |
| candidates and target pointer | Exact values and where the real trainer's resolved config consumes them |
| reason | Why to inherit, adapt, test, fix or omit this factor for this dataset/task/method |
| ablation reason | If an ablatable factor does not compare both enabled and disabled, explain evidence, task constraints or finite-budget tradeoffs explicitly |
| condition | Any dependency on another concrete config value; preserve branch-specific settings |
| trial coverage | Which compiled trial configs realize each planned candidate; later report completed/failed/not-run coverage separately |

Unknown source values remain unknown even if a proposed target value is reasonable. Unsupported architecture settings, prohibited benchmark changes and unsafe GT transforms are not candidates merely to fill an on/off table. Mark exclusion with a specific reason. Do not exclude an expensive technique just because it was absent from the first local implementation; implement it correctly or expose the missing implementation/budget constraint.

## Decide when to compare enabled and disabled

For an applicable technique with uncertain transfer, plan both the source-enabled setting and a disabled control by default, within the declared finite budget. This includes multi-scale training when present in the checkpoint/source baseline. Also consider evidence-backed techniques absent from the source as proposed additions. Strong task-specific evidence, incompatibility, user constraints or budget feasibility can justify fixing/excluding a technique, but the ledger must make that decision visible for each rank/method. Never claim both states were tested when one was only documented.

Preserve a source-derived baseline with compatible source techniques enabled. Target-required head/label/input changes and chosen deviations must be explicit. Share identical baseline/anchor trials. Use matched initialization/seeds, evaluation protocol and declared exposure for informative controls. Explore interactions (for example multi-scale × batch/LR, mosaic × late disabling, augmentation × LoRA rank), using staged/conditional sweeps rather than an unbounded binary Cartesian product. A one-factor ablation is not proof of the best interaction.

Example for a source detector using multi-scale training (values must come from that model's actual sources):

| Factor | Source | Target search |
|---|---|---|
| Train multi-scale | Enabled; resolved sizes, cadence and GT transform | Source-equivalent enabled policy versus fixed-resolution disabled control; preserve scale settings in configs |
| Multi-scale bounds | Source distribution | Refine within the enabled branch only if evidence/budget warrant it |
| Validation resize | Official deterministic protocol | Fixed across training variants |
| Evaluation TTA | Source evaluator policy | Separate declared evaluation experiment only when benchmark rules permit; never silently enabled |

Profile the enabled branch at representative and largest shapes, including evaluation/save peaks. If memory is tight, first measure viable microbatch/accumulation and packing with unchanged effective-batch semantics, or declare batch as a search axis. Do not silently turn multi-scale off to avoid OOM. A material augmentation/input regime change requires a new minimum-LR anchor or documented evidence that horizon transfer is valid. Scheduled augmentation shutdown is a stopping guard: do not declare saturation before observing its intended effect.

## Mechanical coverage check

The bundled [recipe_coverage.py](../scripts/recipe_coverage.py) inventories JSON source leaves, binds decisions to the source hash, checks every requested method and ablation decision, and optionally checks candidate coverage against actual compiled trial configs. It does not research sources, prove the normalized manifest is complete, execute augmentation, or certify measurements.

```text
python scripts/recipe_coverage.py inventory --source source-recipe.json
python scripts/recipe_coverage.py check --source source-recipe.json --decisions recipe-decisions.json --trials compiled-study.json
```

The decisions JSON uses `schema_version=1`, `source_sha256`, `methods`, and `rows`. A row has `factor`, `source_keys` (JSON pointers from inventory), `source_stage`, `evidence` (nonempty source locations), `evidence_status`, `ablatable` and `choices` keyed by every method. Each choice has `disposition`, `reason` and `candidates`; non-excluded choices also need `target_pointer`. Excluded choices have no candidates, must justify omission, and require nonempty `exclusion_checks`: each entry has `pointer` and either `absent: true` or `equals: <exact inactive value>`. These assertions are checked against every trial for that method. Excluded choices cannot carry unchecked `target_pointer` or `when` fields. Choosing the correct inactive value remains a source/pipeline audit responsibility. Non-excluded candidates are either literal values or complete setting objects. An ablatable candidate is a boolean or an object with a boolean `enabled`. Missing either on or off requires nonempty `ablation_reason`; this explicit exception is not proof the decision is scientifically justified.

Optional `when` is a list of `{pointer, equals}` conditions, all of which must hold in a trial. `--trials` accepts a study object with a `trials` list containing `method` and `config`, and each `config.method` must match its envelope (as in the bundled controller's format). For each non-excluded choice, all eligible trial values must be among its candidates and each candidate must occur in at least one eligible trial. Conditional branches with no realizations fail. Config-to-trainer behavior still needs pipeline smoke checks. Without `--trials`, the result explicitly reports candidate coverage unverified.

Run the check for each source baseline represented by a separate source manifest. Embed original/reconstructed source configs, provenance, ledgers, the checker and compiled coverage in portable plans. Before launch rerun after target resolution. At completion compare planned coverage with valid actually executed configs and report anything untested; failed/pruned runs must not count as completed technique comparisons.
