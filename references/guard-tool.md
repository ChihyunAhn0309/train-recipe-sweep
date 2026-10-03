# Trial identity and plateau helper

`scripts/sweep_guard.py` uses Python 3.9+ standard library only. It does not train models, download files, launch jobs, lock trial IDs, save checkpoints, detect data leakage or independently prove gradient correctness. Connect it to a task-specific trainer/controller. Unit tests are in `scripts/test_sweep_guard.py`.

## Identity

```bash
python scripts/sweep_guard.py identity resolved-experiment.json
```

Pass only the immutable, fully resolved experimental config described in [contracts.md](contracts.md). It hashes every supplied key and returns the full SHA-256. Key ordering is normalized; list ordering, types and units are significant. The caller must normalize defaults, units and numeric representations and include semantic hashes. Timestamp/output-path/role metadata must live outside this input. Hashing does not check that required scientific fields are present: the launch audit must do that. Duplicate JSON keys and non-finite numbers are rejected. An atomic registry claim is still required to prevent duplicate jobs.

## Plateau policy

Example policy for a hypothetical accuracy task using scores on a 0–1 scale. These values are **not a ready-made policy for any supplied dataset**; resolve cadence, noise, effect size and schedule guards from that study before launch.

```json
{
  "direction": "max",
  "window": 5,
  "confirmations": 3,
  "val_patience": 8,
  "min_step": 1000,
  "warmup_end_step": 500,
  "schedule_guard_step": 2000,
  "eval_every_steps": 100,
  "hard_cap_step": 20000,
  "train_rel_delta": 0.002,
  "train_noise_rel_max": 0.005,
  "val_min_delta": 0.001,
  "min_train_progress_rel": 0.02,
  "loss_scale_floor": 0.00000001
}
```

Optional fields `train_score_direction` (`min`/`max`) and `train_score_min_delta` must be supplied together; then every row needs `train_score`. The optional score is a flatness check using absolute change, so direction does not change its flatness calculation. The required validation direction controls improvement and checkpoint selection. Unknown policy keys fail to catch misspellings.

Optional `train_abs_delta` and `train_noise_abs_max` default to zero for backward compatibility. For each window, the loss-change/trend threshold is `max(train_abs_delta, train_rel_delta * loss_scale)`; the residual-noise threshold is `max(train_noise_abs_max, train_noise_rel_max * loss_scale)`. A justified absolute tolerance prevents nearly zero loss from making relative changes look permanently large. Specify these in actual loss units before a trial, based on its reduction/normalization. They are not universal constants and must not be tuned after viewing a candidate simply to make it stop. An absolute threshold does not waive progress, validation patience or schedule guards. Decision evidence contains absolute values and the applied thresholds.

Policy values use optimizer updates, not epochs. `schedule_guard_step` is the earliest acceptable point after required LR drops/restarts/unfreezing and their grace; for cosine it encodes the minimum meaningful schedule fraction. `schedule_clear` must remain false while future scheduled changes or grace invalidate plateau inference. Do not simply set it to true because the loss is flat.

## Metric history

At initial step 0 and every exact `eval_every_steps`, append a complete evaluation. Use a fixed deterministic train probe and the same validation evaluator. Each record contains:

```json
{"step": 0, "train_loss": 2.31, "val_score": 0.10, "diagnostics_ok": true, "schedule_clear": false}
```

`diagnostics_ok` comes from external checks of data, trainability, finite gradients/updates and loss behavior. It can be true at initial evaluation after the external smoke test passes; no update is expected at step 0 itself. `schedule_clear` is supplied by the schedule controller. Both flags must be true throughout the final confirmation windows. Do not fabricate them. The helper separately requires a minimum observed relative loss improvement from step 0 to the best W-point median, preventing a completely unchanged model from being called converged by the example policy. Tune this threshold for the task; an already excellent pretrained baseline needs separate interpretation.

```bash
python scripts/sweep_guard.py plateau --policy stop-policy.json --metrics metrics.jsonl
python -m unittest discover -s scripts -p "test_*.py"
```

JSON arrays also work. Missing, duplicate, out-of-order or off-cadence evaluations fail rather than silently shortening patience. `hard_cap_step` must be an integer multiple of `eval_every_steps`; otherwise the terminal cap cannot be observed on this helper's exact cadence and policy validation rejects it before training. Choose and freeze an aligned cap within the authorized exposure bound. Outer-controller time/resource limits still apply independently. A different cadence requires a separate documented policy/controller adaptation. Do not interpolate missing validation values. Extra metric fields are allowed.

The algorithm compares two adjacent W-point windows at K consecutive endpoints, requiring small median change and fitted absolute trend in each window. Median absolute deviation of residuals after removing the fitted linear trend gates excessive noise; it is not a statistical confidence interval. All windows are after the maximum of min_step, warmup_end_step and schedule_guard_step. Validation patience uses cumulative improvement relative to the last meaningful best, while `best_step` tracks the actual raw best independently.

Sparse spikes need an additional check because residual MAD can be zero even with a large central outlier. The guard also reports maximum absolute detrended residual and requires it not to exceed `max(loss_noise_limit, loss_flatness_limit)`. The separate MAD noise limit still applies. The flatness allowance accommodates declared small absolute changes/curvature near zero loss; it does not let a large isolated excursion pass through a zero median. The optional train score also bounds its maximum detrended residual by `train_score_min_delta`. Nonfinite intermediate statistics fail as invalid input even when every raw number was finite; overflowing arithmetic cannot silently certify a plateau. A noisy result needs more clean evidence or diagnosis, not a forced plateau label. Historical results produced by older code retain their original decisions; replays under changed guard code must be labeled as such.

Results:

- `continue`: gives a concrete reason; training controller may continue only within remaining resources.
- `saturated`: empirical plateau criteria passed; save the best validation checkpoint and decision evidence.
- `budget_exhausted`, `right_censored=true`: cap reached without demonstrated saturation. `reason` retains the failed convergence criterion.
- `diverged`: in-process API detected a non-finite metric. JSON NaN/Infinity is invalid JSON here and the CLI instead returns `invalid_input`; either condition must stop normal candidate selection and trigger diagnostics.
- CLI exit 2 / `invalid_input`: malformed input/policy/history. Quarantine it and fix the logging/config error; do not mark the run successful or saturated. Exit 0 only means the helper computed a decision, not that training succeeded.

`min_train_progress_rel` must be strictly positive. Zero would let an unchanged predictor pass the learning requirement. An already excellent pretrained model that cannot show this progress needs an explicitly justified external assessment; do not disable the invariant to call it saturated.

The controller separately implements budget/wall-time limits between evaluations, overfit stopping, stalled-learning diagnostics, pruning, checkpointing and atomic deduplication. It must not wait for the next evaluation to enforce a hard compute limit. This conservative helper intentionally does not classify overfitting, perform statistical significance testing, or guarantee global convergence.

## Local execution guard

`scripts/runtime_guard.py` is an optional standard-library helper for a **single host with a local filesystem**:

- `verify_assets(manifest, root)` validates `root / kind / filename`, the path a loader actually consumes. Each listed entry needs `kind`, relative POSIX `filename`, lowercase `sha256`, and optional byte size. Historical `local_path` fields are ignored. Callers must separately require every loader input, pin repository revisions and prevent external mutation during a run. Hashing a partial manifest does not prove completeness.
- `LocalStudy(output, identity, total_seconds)` holds an OS lock, binds output to an immutable identity, and rejects unrelated or historical nonempty outputs. Include all semantic inputs/code/environment in that identity. Resume only the same identity; use a new directory for changed experiments.
- Call `check()` between bounded work units in training, validation and preflight. An individual kernel or file operation may overrun a wall limit; this is cooperative checking, not a process deadline or GPU scheduler.
- The cumulative wall ledger includes failed attempts and restarts. A caller-provided larger **total** cap is logged on the new attempt and must already be authorized. After a hard crash, reserved time is charged conservatively until recovery, bounded by the old allowance, and labeled an estimate. This helper does not measure physical GPU-hours.
- An already-bound output requires its intact, nonempty accounting ledger. Missing files, duplicate JSON keys, invalid amounts/statuses or inconsistent reservations block restart before changing the records. Restore verified accounting from a backup or scheduler record; never recreate it as zero spending. Preserve the damaged evidence. A crash during first-time output initialization can also require recovery or a genuinely new output with any incurred cost still accounted for. The helper detects structural corruption, not malicious rewriting or rollback to an older otherwise valid ledger; authoritative cluster billing needs external accounting.
- Register cleanup functions on `cleanups` to reconcile a registry before releasing the lock. A hard process kill skips cleanup; a later owner may reclaim only after acquiring the OS lock and checking checkpoint integrity. Missing or corrupt checkpoint state must not be silently resumed.

Use a transactional cluster registry and scheduler leases for multi-host/shared-filesystem execution instead. The local helper is not a substitute for those facilities. Unit tests include a separate-process lock conflict, wrong loader files, historical output protection, cumulative budgets and exception cleanup.
