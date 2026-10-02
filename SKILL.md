---
name: train-recipe-sweep
description: Verify models, pretrained checkpoints, datasets and ground truth, then plan or execute evidence-based training-recipe sweeps for Full FT and separately requested LoRA ranks. Use for train recipe optimization, fine-tuning plans, saturation-epoch calibration, or GPU sweep execution; not for ordinary inference or an isolated training-code fix.
---

# Train Recipe Sweep

Build traceable baselines and reproducible sweeps for the user's models, datasets and requested training methods. The outcome is the **best recipe evaluated within the declared search space and budget**. Do not promise a global optimum, an exhaustive literature review, unobserved convergence or unmeasured GPU utilization. Explain decisions in the user's language.

## 1. Resolve the request and execution scope

First consult the conversation, project configuration and model/data documentation. Ask concise, bundled questions about consequential missing information; continue research and design that do not depend on the answers.

- Exact model name, size, variant, official repository or checkpoint ID; dataset version and task.
- Datasets, subsets and splits assigned to train/validation/test. Preserve the user's assignments.
- Requested methods: `full_ft`, `lora_r16`, `lora_r32`, etc. Treat each rank as a separate search.
- Primary metric and direction, actual target host and allocated GPU IDs/count/VRAM, workspace, time/GPU-hour/storage limits.
- `plan` or `execute`. A planning request does not authorize a long training run.

When the user delegates unspecified choices, choose with evidence and record assumptions. The finite default budget in [contracts](references/contracts.md) may apply. Do not invent server access, private-data permission or credentials. Do not request execution approval that the user has already given. Resolve ambiguous model/data names before they change the experiment.

## 2. Verify the model, checkpoint, data and provenance

Read [evidence and data](references/evidence-and-data.md). Compare current primary sources with actual files.

- Connect the original paper, official code/config, model card, public checkpoint and recipe that produced that checkpoint. Compare popularity using observed evidence such as downloads or official adoption; do not guess which checkpoint is most used.
- Distinguish the checkpoint's original training recipe from downstream Full FT. Label unpublished values `unknown`, transformed values `derived`, and new choices `proposed`.
- In execution mode, acquire weights, architecture/config, processor/tokenizer, required raw data and **all public GT-related metadata for the selected dataset release**. Record revisions, hashes, sizes and missing items; reuse verified caches.
- Check label coverage, GT alignment, group/time splits, duplicates, leakage, ontology and evaluability as well as sample counts. Explain missing or inadequate train/validation data with concrete remedies. Never repurpose test data as validation.
- In plan mode, supply the same acquisition/verification commands and completion criteria. Do not mark unacquired large files as verified.

Read [source recipe reconciliation](references/recipe-reconciliation.md) before defining any baseline or search. Recover effective settings from config inheritance, launch overrides, versioned defaults, augmentation code and phase schedules. Produce a source-to-target ledger covering every recovered setting for every method/rank. Include multi-scale training, resize/crop distributions and cadence, GT transforms, augmentation on/off and late-stage changes; distinguish training policy from evaluation/TTA. No source setting may disappear silently. Use the coverage checker and actual pipeline checks described in that reference.

## 3. Define the task head and trainable parameters

Read [model adaptation and LoRA](references/model-adaptation.md).

- With usable pretrained weights, load the compatible backbone and replace/adapt only heads whose output meaning or dimensions differ. **Full FT trains the entire model, including the replacement head**; it is not a head-only linear probe.
- Equal output dimensions do not make different label meanings/orders compatible. Conversely, do not reset a language model's compatible LM head merely because its dataset changed when the tokenizer/vocabulary contract is unchanged.
- If no compatible weights exist, obtain the architecture and official implementation, initialize all parameters, and call this `from_scratch`. Do not describe it as pretrained fine-tuning or automatically substitute LoRA on a frozen random backbone. Inaccessible weights are not evidence that weights do not exist.
- Define `backbone_lr` and `head_lr` separately for Full FT; `adapter_lr` and `head_lr` separately for LoRA. Separate decay/no-decay groups and check coverage and duplicates by parameter identity.
- Research each model's LoRA papers, official implementations and relevant studies. Record exact placement/module paths, rank/alpha/scaling, dropout, trainable heads, optimizer and recipe. Do not copy generic LLM target names into unrelated architectures.
- LoRA normally freezes the backbone and trains adapters plus the new task head. Follow the baseline's freeze policy for existing compatible LM heads. Declare any trainable bias/norm/embedding exceptions. Keep baselines, search spaces, convergence calibration and results separate for every requested rank.

## 4. Establish baselines and a finite search space

Read [search and stopping](references/search-and-stopping.md), then build the study using [contracts](references/contracts.md).

Preserve the evidence chain: checkpoint-generation recipe → task-adapted Full FT baseline → model-specific LoRA baselines. Evaluate baselines on the actual task. Do not claim original-paper reproduction without reproducing the original setting.

Classify every relevant axis as `sweep / fixed / conditional / excluded`, with bounds, rationale, budget and interactions. Cover LR, head LR, effective batch, optimizer, weight decay, scheduler, warmup, training length, regularization/augmentation, and LoRA targets/alpha/dropout. Do not promise infinitely many values or all combinations. Expand a winning boundary within the authorized budget.

For applicable techniques whose transfer is uncertain, compare source-enabled settings against disabled controls within budget, including multi-scale when present. Any decision to fix or omit an ablatable technique needs a method-specific reason; omission from an initial implementation is not a reason to silently drop it. Preserve the compatible source-derived baseline and declare interacting/conditional factors. Reconcile the ledger against compiled trial configs so a listed on/off sweep cannot pass as scheduled when one state is absent. Report planned versus actually completed coverage separately.

## 5. Use the minimum-LR trial as the first convergence search

For each model × data split × method/rank × important scheduler/batch family, register the smallest **positive** LR candidate as a real trial. With multiple LR groups, include the minimum scale for each defined LR ratio; use separate anchors for incomparable small-LR combinations.

- Start with minimum-LR anchors using the baseline's other settings. Their checkpoints, metrics and cost belong to the sweep. Do not repeat the same run from scratch just for calibration.
- An LR too small to begin learning is not saturation. Check learning progress, gradients/updates, completed warmup, pending scheduler changes, training loss and validation trends together.
- Use observed saturation steps/epochs to initialize other trial horizons, not as a mandatory ceiling. Extend improving trials within budget; stop earlier when the declared criteria are satisfied.
- If a cap arrives before convergence, report `budget_exhausted` or `right_censored`, not a saturation epoch. Add anchors when introducing a lower LR.
- Preserve the historical LR trajectory. Changing a cosine/one-cycle horizon that determines previous LR values creates a different trial.

Evaluate convergence at every evaluation. Use task-specific absolute and relative change criteria, including near-zero losses. The standard-library [sweep_guard.py](scripts/sweep_guard.py) helps with identities and conservative plateau decisions. It is not a trainer or GPU scheduler: configure numerical thresholds and external diagnostics before integrating it. Read the [guard guide](references/guard-tool.md).

## 6. Maximize useful throughput on the actual GPUs

Read [execution and reproducibility](references/execution.md). Every execution must complete [GPU target acceptance](references/gpu-acceptance.md), repair failures, and revalidate before long sweeps. This also applies when transferring a CPU-authored plan. Without GPUs, defer measurements; leave explicit decisions to be resolved from real measurements at execution time.

For a document-only handoff, follow [portable plans](references/portable-plan.md). Include required code, configuration, acquisition information and launch/recovery commands. Reconstruct from the document alone in a fresh directory. A document depending on the previous conversation, author-local paths or an absent trainer/controller is not a complete executable handoff. Plan completeness and GPU verification are separate states.

Maximize useful samples/tokens per second and completed trials per GPU-hour on allocated devices while pursuing high utilization and useful VRAM occupancy. Measure peak memory with headroom for variability. Do not use dummy allocations to fill VRAM or silently change batch semantics. Compare independent packing for small trials with validated distributed execution for large trials. Never terminate unrelated work on shared servers.

Execution proceeds through environment pinning, data/parameter checks, small forward/backward and save/reload checks, profiling, baselines/minimum-LR anchors, search, and finalist confirmation. Supply a real trainer and launch/resume commands for the selected framework/version. Generic pseudocode is not execution readiness.

Verify the actual Python interpreter and record readiness separately for design, assets, trainer, and target acceptance. Verify the paths consumed by loaders, bind outputs to immutable trial IDs, and preserve old results. Keep cumulative budgets and checkpoint/metric consistency across restarts. Use [runtime guard](references/guard-tool.md#local-execution-guard) for a single invocation's file/time protection, or the [study controller](references/study-controller.md) for single-host sweeps with one physical GPU per trial. The controller needs a real trainer adapter and target evidence; use an appropriate scheduler for clusters or multidevice trials. Count overlapping packed trials by the union of physical-device allocation intervals.

## 7. Deliver the best recipe and limits for each method

Compare validation improvements over baseline, seed variability, trainable parameter counts, best/stop epochs, termination reasons, GPU time, peak VRAM and throughput. Report Full FT and every requested LoRA rank separately. Select using validation; evaluate test only after selection is complete.

Plans include execution order, conditional spaces, time/storage estimates, recovery, concrete commands and unresolved assumptions. Execution results include winning configurations/checkpoints, reproduction commands, environment lock, complete trial registry, source/data manifests, train/validation curves and unsearched regions. Distinguish file creation, dry-run validation and actual GPU execution.

[Sources](references/sources.md) provide starting references, not a substitute for current model-specific research.
