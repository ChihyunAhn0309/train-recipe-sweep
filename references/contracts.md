# Input, study identity, and deliverables

## Input contract and questions

Resolve from existing context before asking. Bundle missing essentials:

`model IDs/variants + dataset/version/task + train/validation/test mapping + requested methods/ranks + primary metric/direction + target host/allocated GPUs + plan/execute + budget/workspace`.

Ask about consequential missing choices. Continue independent research. A plan can have explicit unresolved fields, but it is not execution-ready until resolved. Never fabricate model IDs, inaccessible GT, credentials, GPU measurements or literature values.

If the user explicitly delegates unspecified choices, use the official task metric and split, a compatible widely used official checkpoint supported by evidence, and the requested method list. If methods too are delegated, propose/use Full FT plus standard LoRA rank 16 and 32 where supported. Explain architecture exceptions. Respect already authorized execution.

For delegated budgets with no existing project default, a **proposed initial cap** is 24 aggregate GPU-hours for the entire study, up to 30 non-confirmation trials per method/rank, at most two boundary-expansion rounds, and at most two OOM retries per layout. GPU-hours include baseline, profiling, anchors, failed attempts and confirmation. These are ceilings, not targets or promises of sufficiency. Use profiling to reserve confirmation budget and ensure all families fit; if even baselines/anchors cannot fit, present the feasible reduced scope or required budget rather than dropping requested families silently. Never interpret “optimal” as unlimited compute.

## Machine-readable study specification

Generate a resolved `study.yaml` (or project-native equivalent) with these sections. Fields are a contract, not a required new framework:

| Section | Required meaning |
|---|---|
| identity | Study ID/version, mode, objective metric/direction, selection/tie policy |
| sources | Evidence ledger and source checkpoint generation config paths |
| model | ID/revision/config/weight hashes, processor, head contract, scratch/partial/pretrained status |
| data | Exact releases, raw/GT manifests, split membership/ontology hashes, audit report, train probe |
| methods | Full FT and each LoRA rank with own baseline, target paths, trainable policy and parameter groups |
| search | Axis disposition, bounds, distributions, conditional rules, seeds, sampling algorithm and seed |
| convergence | Family definition, anchor configs, eval cadence, numerical plateau policy, schedule guards |
| resources | Observed hardware/allocation, budget, profiling matrix, packing plan, retry and retention policy |
| execution | Environment lock, exact entrypoint/args, scheduler/launch/resume and metric contract |
| selection | Frozen finalist IDs/evidence, fresh paired baseline/finalist seeds, aggregation/tie and failed-run policy, validation-only comparison, final test policy |
| limitations | Unknown/proposed values, blocked access, unsearched space and estimates |

Store source-linked baseline configs before resolving framework defaults. Every launch must dump the fully resolved configuration including defaults, actual parameter group LRs and selector expansion. Unknown required values block that launch, not unrelated work.

Include the [scientific comparison contract](scientific-validation.md): baseline kind/deviations, matched context/evaluator hashes, fresh confirmation seeds, selection freeze and evidence-level report fields. Baseline evaluation, saturation calibration, best observed candidate, confirmation and official reproduction are separate states. A source candidate grid or code default is not its winning historical recipe.

Attach the [source recipe reconciliation ledger](recipe-reconciliation.md), resolved source inventories/hashes, evidence locations and per-method on/off decisions. Every source field needs a disposition; source-enabled techniques cannot vanish from the target baseline without a recorded reason. Include compiled candidate coverage and later actual completed/failed/not-run coverage. Run `recipe_coverage.py` after compiling target configs, with pipeline smoke evidence showing that the trainer consumes the declared settings.

Include a `readiness` map with evidence for: `design_resolved`, `assets_verified`, `trainer_available`, `cpu_or_device_smoke_verified`, `target_environment_verified`, `target_hardware_profiled`, and `launch_ready`. A researched plan can be useful while later fields are false. Exact-looking commands aimed at an absent trainer are not execution-ready. Either supply and check the trainer/controller entrypoint or name the missing implementation and its concrete input/output contract. Plan mode does not need to perform GPU training merely to mark a plan complete; it must describe remaining execution prerequisites honestly.

## Trial identity and registry

Compute a SHA-256 of normalized, fully resolved immutable experimental JSON. Include model/code/weights/processor and data/split hashes, head initialization policy, method/rank/targets/trainable policy, loss/augmentation, optimizer/group LRs/batch/precision, entire schedule definition, stopping policy, immutable cap, evaluator, seed and relevant distributed semantics. Normalize units and numeric representations in the producer; integer `1` and float `1.0` are deliberately distinct in the helper. Reject NaN/Inf and duplicate JSON keys. Do not truncate the stored hash.

Exclude display label, output path, job ID, timestamps, role labels and `review_at_step`: these belong to mutable execution metadata, not the experimental JSON. A schedule change, new seed, data revision or changed precision cannot reuse an identity. Trial ID is not a lock: the launcher must atomically claim the ID in SQLite/a transactional store or another reliable job registry before launch; local filesystem locks may not be sufficient across cluster filesystems.

Registry fields: trial ID, immutable config path/hash, family, roles (including anchor/candidate), status, attempt/lineage, seed, job/device allocation, metrics path, best/last checkpoints, best/stop step and equivalent epoch, termination reason, elapsed GPU-hours, peak VRAM, throughput, errors and checkpoint integrity. Deduplicate before dispatch. A valid completed trial is reused; a resumable incomplete one continues; failed attempts are preserved. Reruns justified by new seeds or changed configs get distinct IDs. Do not reuse results across different datasets or head mappings.

## Outputs in plan mode

For a requested executable handoff using only the skill and document, the [portable-plan contract](portable-plan.md) is mandatory: embed a complete executable bundle with scientific specification, conditional target checks and acquisition metadata. Reconstruct and test it in a fresh directory. Record `portable_plan_complete` with evidence; author-side CPU checks cannot set target verification or launch readiness. Missing trainer/controller implementation cannot satisfy a complete executable handoff. For a specification-only request, use `scientific_specification_complete` and mark implementation/target readiness pending; document format or CPU-only authoring does not require an executable bundle. The list below applies within the requested deliverable scope.

1. Human-readable model/checkpoint choice and evidence table; dataset/split audit and unresolved issues.
2. Source, Full FT and per-rank LoRA baselines with value provenance and exact head/adapter mapping.
3. Resolved or explicitly conditional study/configs, parameter ranges and rationale, minimum-LR anchors, numerical stopping policy, budget allocation and GPU profile/packing plan.
4. Exact target-environment commands for download, integrity/data audit, smoke/profile, launch, monitor and resume. Provide real scripts/configs when sufficient details exist; mark untested or blocked commands accurately.
5. Acceptance criteria and estimated cost/VRAM/storage with assumptions and uncertainty. Do not invent a best recipe or measured saturation epoch before experiments.

## Outputs in execute mode

In addition to the plan, deliver verified asset manifests and environment lock, resolved configs and code revision/patch, all trial records and metrics, best and resumable checkpoints, train/validation curves, telemetry summaries, each method/rank's best recipe and reproducible commands. Report baseline delta, seed spread, best/stop epoch, convergence status, rank/target/trainable parameter counts, and cost/memory/throughput tradeoffs. Preserve access restrictions when linking artifacts.

Completion claims must distinguish: files created; static/dry-run checks passed; actual GPU smoke passed; search finished within stated coverage; final confirmations completed. If budget or access limits progress, report achieved results and exact remaining work. Never call budget exhaustion “optimal recipe found” without the declared limited-search qualification.

Every execute invocation, including a transferred plan, must perform the [target acceptance sequence](gpu-acceptance.md) and resolve its measurement-dependent choices before long trials. Previous authorization persists; a fresh host or changed device/framework/data/layout invalidates the affected evidence, not the user's entire request.

## Invocation examples

```text
$train-recipe-sweep
Model: [exact ID and variant]
Data: [version], train=[subsets], validation=[subsets], test=[if available]
Methods: full_ft, lora_r16, lora_r32
GPU: [host, allocated device types/count/VRAM]
Metric: [name and maximize/minimize]
Mode: plan
Budget: [total GPU-hours or time limit]
```

```text
$train-recipe-sweep Execute the existing plan and verified assets on the specified host.
The remaining total budget is 12 GPU-hours. Resolve unspecified recipe details
using evidence and record your choices.
```
