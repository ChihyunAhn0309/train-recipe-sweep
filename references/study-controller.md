# Single-host study controller reference

This portable Python 3.11+ standard-library reference implements study admission, local process ownership, durable physical-device allocation accounting, and result integrity. It is **not a model trainer, universal scheduler, cluster scheduler, GPU benchmark, or proof that a recipe converged**. `fake_worker.py`, `example-study.json`, and the executed tests are CPU protocol fixtures. The fake worker refuses GPU mode.

Keep `study_controller.py`, `fake_worker.py`, `test_controller.py`, and `example-study.json` together if copying the complete example; the accompanying protocol document may live separately. The controller itself has no import dependency on the fixture or test file. Actual training adapters must implement the protocol below in their own pinned environment.

## Portable use

Copy the scripts/example into a fresh workspace and run there with the explicit interpreter/environment intended for execution. Keep generated study outputs outside the installed skill directory:

```text
python study_controller.py validate example-study.json
python study_controller.py run example-study.json
python study_controller.py report example-study.json
python study_controller.py run example-study.json --resume
python -B -m unittest discover -s . -p test_controller.py -v
```

The example produces only small CPU fixture artifacts. Relative study paths resolve from the supplied JSON file. `{plan_dir}` expands to that directory and `{python}` to the controller's actual interpreter. Worker commands run with that directory as their working directory. There are no author-machine paths, hidden credentials, or network dependencies. Set `CONTROLLER_TEST_WORK` to an appropriate scratch directory to redirect test temporary files.

A complete portable training plan must already contain its real trainer/worker adapter, download/audit/profile code and environment contract, tested as far as the authoring environment permits. After transferring the document, reconstruct it, resolve its data/model downloads and target environment, and generate target evidence before GPU execution. Adapt only what target checks show requires changes, then revalidate. This controller reference and fake worker alone are not a complete model-specific training plan. Existing bound outputs are intentionally path/command bound: relocating old results is an explicit accounting/state migration, not automatic reinterpretation.

## Study JSON v1

`example-study.json` is the complete executable CPU schema example. Required top-level fields are `schema_version=1`, display `study_id`, `mode` (`cpu_fixture` or `gpu`), `output_root`, shared `device_lock_root`, `worker.argv`, `devices`, `budgets`, and `trials`.

Each trial has:

- A nonempty immutable `config`, including matching `method` and `family`, and `metric={name,direction}`. The producer must include all scientific semantics: model/code/environment, weights/processor/data/split/ontology hashes, head mapping, trainable parameters, rank/targets, seeds, optimizer/group LRs, precision, batch, full schedule, evaluator, stopping policy, and immutable exposure cap. The generic controller cannot infer omitted scientific facts.
- `trial_id` may be omitted; it is computed as the **full 64-character SHA-256** of canonical UTF-8 config JSON, matching `sweep_guard.trial_id`. If supplied, it must match. Changing config creates a new ID. Identical entries deduplicate; conflicting execution metadata is rejected.
- `roles`: one or more of `anchor`, `baseline`, `candidate`, or the sole role `confirmation`. Duplicate role names normalize to one value. A confirmation requires a distinct config identity; merging identical config entries may never mix confirmation with another role. This invariant is checked again after deduplication. It does not establish independent seeds, matched recipes, validation-based finalist selection, or a selection barrier: those remain the adapter's obligations under [scientific validation](scientific-validation.md). `method` and `family` define anchor dependencies.
- `max_seconds`: bounded useful worker runtime, plus separately reserved shutdown allowance. This operational wall allowance does not authorize changing an immutable LR schedule/exposure cap.
- Optional `anchor_fallback`: `reject` (default) or `allow_censored`. All non-anchor trials, including higher-LR baselines, wait for all anchors in their own method/family to report `saturated`. Explicit `allow_censored` also admits after anchors report `right_censored` or `budget_exhausted`, with calibration still reported incomplete. A generic `completed` anchor does not prove saturation. The caller must correctly construct minimum-positive-LR anchors; this controller does not infer LR minima.

This reference supports **one device per worker**, with independently packed workers allowed. It does not launch DDP/multidevice jobs. GPU device IDs must be unique full physical `GPU-...` UUIDs; indices and MIG partitions are rejected to avoid aliasing/double counting. All controllers that share a host/device allocation must use the **same local** `device_lock_root`.

The UTF-8 identity correction preserves ASCII-only config hashes. Existing outputs containing non-ASCII configs that were bound by an older controller must not be reinterpreted or reset under the corrected hashes. Preserve the original controller for faithful inspection, or perform an explicit identity/accounting migration; a genuinely new study must retain the historical cost in its overall budget ledger. The append-only stage helper below does not perform that migration.

`slots=1` is the default safe layout. `slots>1` requires `packing_evidence={path,sha256}` pointing to JSON with exact `device_id`, `slots`, `kind` (`gpu_measurement` or `cpu_fixture`), positive `peak_memory_bytes`, `available_memory_bytes`, and `aggregate_samples_per_second`; measured peak must leave memory margin. GPU measurements must come from representative combined workloads, including save/eval peaks. Hashes verify the declared file, not its scientific authenticity. CPU fixture packing evidence cannot be used in GPU mode. A shared immutable device-policy file prevents controllers from silently disagreeing on slot count/evidence.

## Budget meaning and admission

Budgets require `total_device_seconds`, `per_method_device_seconds`, `confirmation_reserve_device_seconds`, `poll_seconds`, `grace_seconds`, and `terminate_seconds`. These are finite positive values except reserve, which can be zero and must be smaller than total. Total and method limits are **the remaining allowance assigned to this controller**, after already-consumed preparation/GPU smoke/profile costs and other study allocations have been deducted. The overall study ledger must separately retain and add those costs. External scheduler idle time while a physical GPU is reserved is also study cost; this controller does not discover or account for scheduler intervals outside its worker allocations. Do not describe the controller's subtotal as all study GPU cost.

Admission is one SQLite `BEGIN IMMEDIATE` transaction. It reserves `max_seconds + grace_seconds + 2*terminate_seconds + 2*poll_seconds` per proposed worker, checks total and method limits plus already reserved active work, and preserves the confirmation reserve for non-confirmation trials. Reservation is conservative: it counts each packed worker's full remaining wall allowance. Actual charges are smaller when packing overlaps; admission deliberately prefers staying within budgets over promising every theoretically feasible packing.

Actual accounting stores nonoverlapping time segments per physical device, including idle time during the controller's recorded allocation. **Two packed workers for one second count as one physical device-second.** Methods split that interval equally among concurrent workers; method shares sum to physical cost. Both conventions are recorded in the report. This is not a claim of fractional hardware isolation or cloud billing equivalence. CPU-fixture values are simulated-device wall seconds, never measured GPU-hours.

The controller checks budgets between polls; each runner independently checks its runtime and stop file, so controller death does not reset its deadline. Stop is cooperative first, then bounded termination of the owned process tree. Polling, OS scheduling, filesystem calls, and final cleanup can overrun an instant deadline; this is not a real-time kill guarantee. Admission includes a shutdown allowance. A failed termination leaves durable ownership unacknowledged and blocks reuse.

Never delete/recreate a registry or change study caps in an existing output to obtain new budget. This reference freezes its study/budgets; amendments require a reviewed new study with prior cost carried forward into the overall ledger and preserved attempt lineage. It does not automatically approve budget increases.

## Worker CLI and control protocol

The controller calls the literal `worker.argv` array followed by:

```text
--config <immutable-config.json> --output <unique-attempt-directory> --control <control.json>
```

`control.json` contains `schema_version`, `trial_id`, `attempt_id`, `stop_file`, `resume_manifest` (absolute path or null), `device_id`, and `mode`. GPU workers also receive `CUDA_VISIBLE_DEVICES` set to the declared physical UUID and `STUDY_DEVICE_ID`. The trainer must use that allocation, perform no hidden independent GPU launch, and not detach descendants or interfere with inherited lock descriptors. All paths are supplied explicitly.

The trainer must poll the stop file between bounded train/eval/save units, atomically commit resumable state, and exit. The file contains `reason` and `requested_unix`. Supported reasons include `max_seconds`, `study_budget`, and `controller_interrupt`. Scientific stopping remains the trainer's job: configure `sweep_guard` thresholds, diagnostics, schedule guards, and complete metric history before launch. A stopped time-capped anchor must not report saturation merely because time expired.

For a successful/evaluated attempt, atomically write `result.json`:

```json
{
  "trial_id": "<64-character config hash>",
  "attempt_id": "<control.attempt_id>",
  "status": "saturated",
  "reason": "confirmed task-specific convergence policy",
  "metric": {"name": "validation_accuracy", "direction": "max", "value": 0.81},
  "artifacts": [
    {"path": "best.bin", "role": "checkpoint", "sha256": "<hash>", "bytes": 123},
    {"path": "metrics.jsonl", "role": "metrics", "sha256": "<hash>", "bytes": 456}
  ]
}
```

Statuses remain distinct: `completed`, `saturated`, `right_censored`, `budget_exhausted`, `interrupted`, `pruned`, `failed`. `completed` means the declared work completed; it does not imply convergence. Only completed/saturated outputs with zero worker exit, verified checkpoint+metrics, matching IDs/objective, and acknowledged process-tree termination qualify for successful reuse. Exit code zero alone fails. Every supplied metric, including censored/interrupted/pruned/failed evaluations, must be an object with a nonempty name, min/max direction and finite non-boolean numeric value matching the immutable objective. Other valid evaluated results retain their status and expose verified metric/artifact information for model-specific comparison; censoring is never silently relabeled as success or saturation.

An incomplete attempt that never reached an evaluation may omit the `metric` key. Its valid artifact list is retained as `result_artifacts`, while `evaluated_result` remains null; an explicit malformed/null metric is an error rather than an evaluation claim. Successful results cannot omit the objective metric. Malformed result/resume top-level values, malformed metrics, objective mismatches and invalid artifact roles quarantine the attempt as `failed` with an `invalid_result` or `invalid_resume` reason. The original files remain for diagnosis; they are not exposed as integrity-verified evaluated results or admitted as corrupt resume state. Other families can continue, and a later ordinary run does not rerun those failed attempts automatically.

Artifacts must use contained relative POSIX paths, unique resolved files, exact byte sizes and SHA-256 values. Symlink/path escapes and missing/corrupt artifacts fail. The controller checks hashes after process-tree termination. It does not prove checkpoint deserialization, tensor correctness, optimizer coverage, evaluator validity, or semantic alignment; those require the trainer's acceptance/smoke tests.

To support incomplete-run continuation, write `resume.json` with `schema_version=1`, matching `trial_id`, and the same artifact-list format containing **checkpoint, training_state, and metrics** roles. `training_state` must include optimizer, scheduler, scaler, sampler/RNG, exposure, patience/best tracking, config hash and backend state as applicable. The trainer must commit a consistent snapshot before its manifest; three unrelated atomic files do not by themselves form a transaction. On `run --resume`, only interrupted/failed/budget-exhausted trials with intact recorded manifests are returned to pending. A new attempt receives the verified previous manifest, preserves all old attempts, and shares the original trial identity. Scientific equivalence of resumed state is the trainer's responsibility.

## SQLite, locks, ownership, and crash recovery

### Append a later search or confirmation stage

Use [study_extend.py](../scripts/study_extend.py) when validation results justify adding boundary candidates, anchors for a lower LR, or fresh-seed confirmations to an existing study. Keep separate fixed `stage1.json` and `stage2.json` files. The new file must preserve every existing resolved trial in its original order and append at least one new configuration. All other settings, including worker, devices, output paths, budgets, existing roles and runtime limits, must remain unchanged. GPU mode may refresh `target_acceptance`, and the new file must pass acceptance for its expanded scope before the extension is applied.

```text
python study_extend.py stage1.json stage2.json --reason "Add the validation-selected recipe with prespecified fresh seeds"
python study_controller.py run stage2.json
```

The helper holds the controller lock, requires an existing bound registry with no active attempts or unacknowledged process trees, and rechecks successful artifacts. It inserts only the added trials as pending, preserving old trial states, attempts, checkpoints, and cumulative physical/method charges. A completed anchor is reused by the next controller run. The original budget remains cumulative; adding a stage does not buy more time. Changing an old config, removing/reordering trials, increasing budgets, or editing worker/device policy is rejected. Do not start a new output merely to reset spending or replay a completed anchor.

Each amendment writes an immutable hashed audit record at `output_root/extensions/<new-study-hash>.json`, with old/new study and source-file hashes, added IDs, reason, prior attempt IDs and before-cost totals. The CLI returns its path. Preserve both input files exactly: even a formatting-only source-file change invalidates a pending amendment retry. The helper does not select winners, certify independent seeds or freeze scientific selection; the study adapter must establish those decisions and compile the next stage before invoking it.

A durable `study-extension.pending.json` journal covers the SQLite transaction and identity-marker update. While it exists, ordinary controller/Registry access refuses to proceed. After an interruption, retry the exact same extension command with the unchanged files and reason; the helper validates whether SQLite committed and completes the audit/marker update without reinserting trials or resetting costs. A completed retry against the same current stage is idempotent. Corrupt or conflicting journal/audit/accounting data fail closed. Preserve that evidence and recover from a consistent backup; do not delete the journal to bypass the checks. Source-file/registry relocation, budget amendments and migration of older identity formats are outside this helper.

The local SQLite registry uses WAL, full synchronization, foreign keys, atomic admission, unique live trial/device-slot claims, persistent attempt records, and a physical-accounting hash chain/count. Missing registry, missing metadata, removed accounting segments, invalid amounts or changed config/results fail closed. Local files are not an anti-tampering database: an adversary who rewrites all state or rolls back an otherwise valid complete snapshot is outside this guarantee. Backup registry/WAL consistently.

The controller holds a study OS lock. Each runner independently holds a worker lease and a physical-device slot OS lock, with durable slot ownership pointing to its matching termination receipt. A released OS lock alone is insufficient to reuse a dirty slot; the prior owner must acknowledge process-tree termination. Missing initialized slot ownership also blocks reuse. This protects against live or unacknowledged orphans across controller restarts and across studies using the same device-lock root.

The controller does not probe slots already reserved for its starting workers. A runner permits a bounded one-second acquisition wait so a harmless availability probe from another controller cannot turn a newly admitted worker into an artificial failure. The wait never releases, steals, or force-unlocks another process's lease; a genuinely occupied slot remains protected.

On Windows 10+, the runner joins a kill-on-close Job **before** spawning its worker; descendants inherit membership, and normal cleanup enumerates/terminates only that Job's child members. On Linux, the worker starts an isolated session/process group and inherits the worker/device lock descriptors. Normal cleanup sends TERM then KILL only to that owned group and verifies no live group members remain via `/proc`. Already-dead zombies are not live GPU owners. Other operating systems are unsupported for execution/recovery.

If a controller dies, a live runner keeps its locks and deadline. A second controller refuses to reclaim that live orphan; wait for the original runner to finish/stop, then rerun. Costs from active allocations remain charged across the gap. A hard-killed runner with no termination receipt remains dirty even if its OS lock becomes free. On Linux, `recover <study.json>` may acknowledge only a durably recorded process group after acquiring both leases and proving that group is already quiescent; it **never kills/reclaims a live orphan**. Unknown launch ownership fails closed. Recovery charges the unclosed allocation until recovery conservatively, so it may exhaust budget rather than silently refund unknown time.

Automatic recovery after abrupt **Windows runner** death is intentionally unsupported because this reference does not independently reopen/verify the dead runner's Job tree. Preserve evidence, verify termination externally, and reconcile ownership/accounting explicitly. Do not delete lock/owner files to bypass this requirement. A crash during initial identity/registry creation may also require explicit recovery. Normal Windows controller-death recovery is exercised by tests because the independent runner remains alive and writes its own acknowledgment.

## Mandatory GPU target acceptance

GPU `validate` and `run` require `worker.code_artifacts` (nonempty `{path,sha256,bytes}` list) covering the actual worker and relevant local imports/config/environment lock, and `target_acceptance={path,sha256}`. Prepare target evidence on the actual GPU machine. First obtain the exact acceptance scope:

```text
python study_controller.py scope gpu-study.json
```

`scope` checks config/code/packing structure and returns `scope_sha256`, physical `device_ids`, and `launch_ready=false`; it does not claim any acceptance gate passed. The scope covers the SHA-256 of the actual executing controller source, resolved worker argv/code hashes, all trial configs/roles/families/runtime bounds, device/packing policy, and budgets, excluding the acceptance document itself to avoid recursion. Updating the controller alone invalidates prior acceptance even if worker/config files are unchanged. The controller source hash is a scope payload value; it is never written into the source file, so no self-referential file hash is required.

The referenced acceptance JSON must contain:

```json
{
  "schema_version": 1,
  "kind": "gpu_acceptance",
  "execution_mode": "gpu",
  "scope_sha256": "<scope command output>",
  "device_ids": ["GPU-<actual full physical UUID>"],
  "gates": {
    "hardware_environment": {"status": "passed", "evidence": [{"path": "hardware.json", "sha256": "<hash>", "bytes": 123}]},
    "assets_and_gt": {"status": "passed", "evidence": [{"path": "asset-audit.json", "sha256": "<hash>", "bytes": 123}]},
    "model_smoke": {"status": "passed", "evidence": [{"path": "model-smoke.json", "sha256": "<hash>", "bytes": 123}]},
    "controller_recovery": {"status": "passed", "evidence": [{"path": "recovery.json", "sha256": "<hash>", "bytes": 123}]},
    "performance_profile": {"status": "passed", "evidence": [{"path": "profile.json", "sha256": "<hash>", "bytes": 123}]},
    "budget_feasibility": {"status": "passed", "evidence": [{"path": "budget.json", "sha256": "<hash>", "bytes": 123}]}
  }
}
```

All six top-level gates must have `status=passed` and nonempty hashed underlying artifacts. An unavailable GPU is a blocked gate, never an N/A pass. A gate may optionally include a `subchecks` list for optional items such as unrequested DDP/LoRA. Each subcheck needs `name`, `status` (`passed` or `not_applicable`), and nonempty hashed `evidence`; N/A additionally requires a specific nonempty `reason`. A subcheck exception cannot waive its top-level gate. Missing/stale scope, changed worker/code/config/budget/device IDs, malformed gates, or changed evidence block validation/launch. CPU acceptance/packing documents cannot satisfy GPU mode.

Hashes establish declared evidence integrity, **not truth or current hardware state**. The skill/operator must review original measurements and re-probe the target at invocation (driver/environment, available allocated UUIDs, asset/GT audit, real forward/backward and save/reload/resume, recovery, representative throughput/peaks, bounded OOM policy, all-method/confirmation budget feasibility). An invented document with hashes is still invented evidence. Acceptance-schema fixture tests exercise rejection logic only; they never assert these real gates passed. Controller reports therefore keep `gpu_validation_claim=false`; an actual validated training report must separately link the real evidence.

## Reports and known boundaries

`report.json` preserves trial/attempt statuses, reasons, valid result/resume locations, integrity-verified evaluated metrics/artifacts, cumulative physical and attributed method costs, and calibration state. Pending trials carry budget/anchor/device blockers. `all_trials_terminal` and `saturation_calibration_complete` are separate; neither says an optimal recipe was found. Finalist selection, seed uncertainty, validation-only ranking and final test policy remain the model-specific study adapter's responsibility.

This reference is bounded to one host, local filesystems, one physical device per worker, cooperative worker protocol, and explicit shared device locks. It does not provide multi-host leases, DDP launch, remote authentication, scheduler/cloud billing, automatic external preflight accounting, anti-tampering security, pretrained/dataset downloads, or a model-specific training implementation. The future plan must carry those implementations/configs/download manifests explicitly before claiming launch readiness.
