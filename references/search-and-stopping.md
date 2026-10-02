# Baselines, search design, and saturation

The numerical defaults below are proposed engineering starting points, not literature-proven universal optima. Set and freeze task-specific values before candidate selection; record all changes as study revisions.

## Baselines and coverage

Create separate immutable configs for source checkpoint generation (possibly incomplete), target Full FT baseline, and each model/rank's LoRA baseline. Reconstruct missing source fields only as labeled proposals. Include the source-derived baseline in the sweep, even when it is not the minimum-LR anchor. If two configs resolve identically, share the trial instead of rerunning it.

Require the [source-to-target decision ledger](recipe-reconciliation.md) before compiling trials. For multi-scale and other applicable techniques with uncertain transfer, compare enabled and disabled candidates unless an explicit method-specific justification fixes/excludes one state. Preserve exact distributions, transform order, cadence, GT semantics and schedule transitions. Separate training augmentation from validation/TTA. Validate actual candidate coverage against compiled configs; do not count a proposed or failed branch as a completed ablation. Consider interactions within the finite budget, and invalidate or verify horizon transfer across material augmentation regimes.

For each relevant axis list `sweep`, `fixed`, `conditional`, or `excluded`; numerical bounds/categories, log/linear scale, origin, dependency and why the coverage is adequate. Cover:

- Backbone or adapter LR; independent head LR/multiplier; optional layer-wise decay.
- Optimizer family and relevant betas/momentum/epsilon; weight decay and exclusions.
- Effective batch (samples or non-padding tokens), input resolution/length and exposure.
- Scheduler family, warmup steps, final LR, milestones and training length.
- Task-appropriate augmentation, regularization, dropout, smoothing, imbalance handling, loss and loss weights.
- LoRA target modules/layers, requested fixed rank, alpha/effective scale, dropout, initialization and extra trainable modules.
- Precision, clipping, normalization and EMA policy when they affect the recipe; otherwise fix and record them.

Avoid blindly taking the full Cartesian product. Use a staged, conditional space: broad log-scale LR/group-ratio exploration; optimizer/batch/scheduler interactions; then local refinement and regularization/adapter ablations. Every relevant dimension must have a disposition, not necessarily every combination a run. If asked for a finite exhaustive grid, calculate its actual trial count and GPU cost and honor it within the authorized budget; do not call random search exhaustive.

When evidence provides baseline LR L, a *candidate initial* log grid can be `L * {0.1, 0.3, 1, 3, 10}`, after stability/feasibility checks. A replacement head can initially explore LR ratios `{1, 3, 10}` if supported; these are hypotheses, not mandatory values. Handle zero weight decay separately from a log distribution. Do not choose lower bounds so small that realistic progress cannot occur within budget. Baseline batch/schedule should anchor comparisons before expanding their axes.

Declare finite exploration, extension and confirmation budgets. Allocate usable trials to every requested method/rank; flag when the budget cannot support that comparison. Ensure finalists are not favored simply because one family received far more tuning. Report either matched search budgets or the actual differences. Use objective metric and direction fixed in advance; tie-break within declared uncertainty by compute/memory only if this fits the user's objective.

## Minimum-LR anchors and reuse

Run a minimum **positive** LR anchor before the main search in each convergence family. A family includes model/checkpoint, data split/preprocessing, method/rank, optimizer, effective batch and materially different scheduler. Other major regime changes (augmentation/loss/adapter targets) require a validation of horizon transfer or a new anchor. Do not extrapolate one epoch count across unrelated architectures, datasets or ranks.

For group LRs define a common scale and explicit ratios, then anchor the minimum scale for every ratio to be searched. If independent LR vectors have incomparable minima, use their lower/Pareto-minimal configurations; do not declare the minimum backbone LR to be the minimum of all groups. A tied minimum can use baseline non-LR settings and be recorded as representative. Retain evidence on whether it transfers. New lower LR bounds need new anchors.

Register the anchor with its final config hash and seed, before starting it. Give it role `saturation_anchor` plus `sweep_candidate`. Append evaluations and retain both `last` resumable state and validation-best checkpoints. No calibration-only rerun from the beginning. A separate baseline that is higher LR may run alongside the anchor for sanity checking, but cannot replace minimum-LR calibration.

Find **actual learning progress first**: compare against the untrained/reheaded model and early fixed-train-probe loss, verify gradients, update/weight ratios, labels, loss scale and LR. A flat random predictor is stalled, not converged. If it does not learn within the finite diagnostic allowance, label it stalled/inconclusive, revise the lower bound with explanation, and keep the failed evidence. Do not increase LR inside the same immutable trial.

Record optimizer updates, samples/tokens seen, dataset passes and equivalent epochs. Define an epoch under replacement sampling or mixtures; streaming data may only have step/token budgets. Batch changes make raw epochs a poor comparison on their own.

## Extensible horizons without changing the experiment

Choose a schedule before starting:

- A predefined step-based schedule or horizon-independent policy can continue with its exact scheduler state.
- For total-step-dependent cosine/one-cycle, fix the full horizon upfront. Changing total steps changes earlier intended LRs; use a new trial from the same initialization if that schedule is to be compared.
- A predefined continuation tail or restart is allowed if included in the original config. It is not equivalent to having trained with a longer cosine from the beginning.

Separate `hard_cap` (immutable trial exposure bound), the actual immutable scheduler, `review_at_step` (operational checkpoint where continuation is evaluated), and the study's cumulative resource budget. Moving only the review point within the unchanged cap and schedule preserves trial identity. Continue eligible anchors in chunks without resetting weights/optimizer/RNG or rerunning completed steps. Do not extend beyond the authorized resource budget.

Raising an immutable trial cap creates a new config hash linked to the parent's checkpoint and preserved censored result. Stateful continuation is allowed only if the entire prior training trajectory remains equivalent: unchanged initialization, batches, optimizer, LR path, stopping semantics and data. Count shared prefix execution once in aggregate cost; do not rerun that prefix or count the child as an independent seed. A cosine/one-cycle duration change generally fails this equivalence and requires a fresh trial. A study resource-budget increase is a separate explicit amendment, never an automatic consequence of restarting a process. None of these amendments proves saturation.

Once an anchor plateaus, use its observed step/exposure as an initial horizon estimate; e.g. review other trials around `0.8–1.2 * anchor exposure`, rounded to their eval cadence. These multipliers are proposed heuristics, not evidence of convergence. If testing different fixed schedule lengths, each length is its own trial. Every trial has its own early-stop and extension decision. Improving runs can exceed anchor exposure; converged runs can end earlier. Capped anchors are right-censored and cannot supply an observed saturation epoch.

Declare the fallback before launch if an anchor remains censored: continue its same trajectory within remaining authorized resources, revise the study with a recorded continuation amendment, or complete a bounded diagnostic search under an explicit fixed exposure cap. The last option can return useful partial candidates but does not fulfill saturation-calibrated optimization. Record `saturation_calibration_complete=false`; never use a censored cap as if the minimum-LR saturation requirement passed.

## Explicit stopping policy

Training accuracy alone can saturate while loss, calibration or validation still improves. Use a deterministic fixed train probe in eval mode (without stochastic augmentation) plus actual optimization loss, and validation metrics with the official evaluator. The probe is sampled from train, never from validation. Hold probe composition and metric scaling fixed; do not compare augmented minibatch accuracy across incompatible recipes.

Before launch define `eval_every_steps`, minimum training/exposure, warmup and schedule guards, window length W, plateau confirmation count K, validation patience P, train-loss relative tolerance, maximum accepted within-window noise, and metric-scale absolute validation min_delta. If useful, add train-metric absolute tolerance. Example starting policy: W=5 evaluations, K=3, P=8, train loss change tolerance 0.002 (0.2%). Set min_delta based on practical effect size and measurement resolution/uncertainty; there is no universal value for accuracy, mAP, loss and perplexity. Large noise requires longer/cleaner measurement, not automatic saturation.

At every evaluation, call saturation only if **all** hold:

1. Sufficient exposure, actual prior learning, finite updates, data/gradient diagnostics passed, warmup finished and schedule guard passed.
2. No unobserved planned LR drop/restart, unfreezing phase or curriculum transition that could restart learning. Allow adequate evaluations after the last such event. For cosine, define a minimum fraction/tail region observed. Plateau-responsive schedules must have had their declared adaptation/grace opportunity first.
3. Across K consecutive assessments, compare two adjacent W-evaluation windows of fixed-probe train loss. Absolute median relative change and within-window fitted trend are both below tolerance; large worsening or oscillations fail this criterion. Reject windows with noise above its predeclared cap. Optional train score must also be flat.
4. No cumulative validation improvement greater than min_delta over P evaluations; compare against the last meaningful best, so many small improvements can accumulate. Selection still saves the actual best checkpoint independently of the patience threshold.

These are empirical stopping rules, not a proof of global convergence. Save values supporting the decision. The included guard implements this conservative rule and distinguishes insufficient history, no learning, schedule guard, high noise and budget exhaustion.

Near-zero loss needs particular care: relative change can remain large even when the absolute improvement is practically irrelevant. Predeclare an absolute loss-change tolerance alongside the relative one, using the actual loss reduction/normalization. The helper supports `max(absolute_tolerance, relative_tolerance * window_loss_scale)` and a corresponding optional absolute noise cap. Default omitted absolute tolerances remain zero for compatibility. Changing a stopping tolerance after seeing a run is a study-policy revision; retrospective replay must be labeled counterfactual and cannot rewrite the run's observed stop epoch.

Other end states must remain separate:

- `overfit_stop`: sustained meaningful validation degradation while train improves, after schedule guards and a separately specified patience/uncertainty policy. Select best validation checkpoint; this is not training saturation.
- `pruned`: resource allocation based on an interim objective; not evidence of convergence. If using ASHA/Hyperband, protect anchors and slow low-LR trials with adequate grace and comparable fidelity. Maintain full-budget control trials; do not prune all late improvers.
- `diverged`, `stalled`, `oom`, `invalid_data`, `interrupted`, `budget_exhausted`: report true cause, never relabel as saturation. NaN/Inf are numerical failures; inaccurate/missing logs are invalid history.

## Final confirmation

Select finalists using validation only. Default intent is at least three independent seeds per winning family when feasible; count an already valid winning-seed run toward this total. Reinitialize for new seeds; resuming a run is not a new seed. Under tight budgets report single-seed uncertainty instead of inventing confidence. Only an operational review extension within unchanged immutable bounds continues the identical trial; cap or recipe changes follow the amendment rules above.

Check a subset of early-stopped finalists beyond the stopping point using a prespecified continuation to detect premature stopping if budget permits. If late improvement occurs, revise the stopping policy, record the study change and reassess affected candidates. Keep the original logs. Retraining on train+validation, if requested, is a new final fit with fixed selected recipe and predeclared duration, not another validation-tuned run.

Report per method/rank the baseline delta, best epoch and stop epoch, mean/spread across seeds, trainable parameters, exposure, actual GPU-hours and peak memory. Do not hide failed/pruned trials or unsearched boundaries. Evaluate test only after recipe choice; do not feed test results back into tuning.
