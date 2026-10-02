# Target-GPU execution and reproducibility

## Resource contract

Resolve the actual host, allocated device UUIDs/indices, GPU model/count/VRAM, driver/CUDA, interconnect, CPU/RAM/NUMA, storage throughput/free capacity and framework versions. Use existing authorized access. Read allocation and running jobs; do not take an occupied GPU or kill another user's process. Do not rent new infrastructure implicitly.

Resolve the interpreter/environment explicitly before declaring a dependency absent. Inspect project launchers/venvs and available interpreter versions; on Windows, `python`, `py -3.11`, and `py -3.13` may be different installations. Probe imports, versions and accelerator availability using the exact interpreter that will launch the trainer. Record that path in commands and the environment manifest. A CPU-only local installation says nothing about a remote CUDA environment.

Both plan and execution must target these GPUs. If hardware is unknown, ask; if user delegates, choose from an observed authorized allocation. A paper estimate is not a measured performance claim. In plan-only mode include profiling commands, candidate layouts and conditional decisions without starting a long study or bulk downloads. Identify any small probe actually executed and its cost.

## Optimize useful work

The objective is high useful throughput and completed valid trials per GPU-hour, while keeping validation behavior comparable. High utilization and useful VRAM occupancy are means to that objective, not independent scoreboards. Do not reserve dummy memory or busy-loop to report 100%.

Benchmark representative real train and evaluation batches after kernel/compile warmup. Include largest image/sequence shapes, optimizer-state initialization, backward, gradient accumulation, checkpoint/save peaks and validation. Measure several hundred representative steps when cost permits, or enough to distinguish steady-state throughput from warmup. Record benchmark length and variability.

Tune microbatch, accumulation, data-loader workers, pinned memory/prefetch, caching, mixed precision, supported fused kernels/attention, compilation and activation checkpointing as appropriate. Verify numerical behavior after optimizations. Activation checkpointing trades compute for memory; retain it only when it improves feasibility or throughput. Do not introduce quantization into a standard LoRA comparison without labeling a separate method.

`effective_batch = microbatch_per_rank * gradient_accumulation * data_parallel_world_size` for fixed-size samples. Tensor/pipeline parallel ranks are not additional data-parallel replicas. For variable lengths log actual non-padding tokens/update, packing policy and loss normalization. Keep effective batch fixed while profiling microbatch where feasible. Changing it creates a recipe change needing LR/schedule review. Accumulation is not exactly equivalent to a large batch for all normalization/stochastic layers.

Select the fastest stable layout from measured profiles. As an initial engineering allowance, reserve roughly 5–15% VRAM for variance, then adjust from measured peaks and workload shape; this is not a universal target. Report allocated/reserved framework memory separately from device-used memory. A low reserved fraction is fine when the workload is compute-bound and further packing hurts throughput.

Compare independent trials per GPU against DDP/FSDP/ZeRO or other supported sharding for large models. Allocate independent trials across all available GPUs first when that yields more useful study throughput; use multiple GPUs per trial when needed for fit or measured efficiency. Several trials on one GPU are allowed only after combined peaks, CPU/I/O and throughput are measured. Use an allocation lock/lease and account for nested process counts. Do not claim more speedup than measured.

## Observability and recovery

Sample telemetry periodically (for example every 5–10 seconds) and report time-weighted utilization, memory used, power/thermal constraints when available, samples/tokens per second, step/evaluation/save time, data wait, NCCL/communication time and actual GPU-hours. Exclude initialization from steady-state throughput but include it in total study cost. Diagnose I/O/CPU/eval bottlenecks instead of promising fixed utilization.

Count study GPU-hours from the union of allocated time intervals for each unique physical device, including idle time while reserved. Two packed trials sharing one GPU for one hour consume one physical allocated GPU-hour, not two. For per-trial costs either use documented fractional allocations whose sums fit capacity or report shared-device wall time separately; label these conventions. For MIG/partition allocations record partition identity/capacity and report partition-hours separately unless a justified full-device conversion is available. Do not confuse logical job-hours, physical allocation hours and billed cloud cost. CPU demonstrations need a wall/CPU budget even though their GPU-hours are zero.

On OOM, capture the phase/shape and peak, release only the failed job's resources, and apply a bounded retry policy (default at most two layout reductions). Prefer smaller microbatch with adjusted accumulation preserving effective batch, or approved checkpointing/sharding. Resume only from a valid prior checkpoint; record layout changes and whether numerical/data ordering equivalence is maintained. If semantic config changes, create a new trial. Do not silently truncate sequences or reduce image resolution. Persistent infeasibility becomes an explicit failed candidate or revised plan.

On NaN/divergence, stop, preserve diagnostics, inspect loss/labels/precision/LR/clipping and propose a new config. Do not quietly lower LR mid-trial. Save on preemption/interrupt when possible. Bound retries and storage use. Preserve winning, last-resumable, and necessary audit artifacts; follow a declared retention policy for others.

## Reproducibility and readiness

Pin source commit, local patch, packages/lock or image digest, CUDA/backend settings, weights/processor revision, dataset/split/label-map hashes, seed(s), deterministic/nondeterministic kernels, world size and evaluator version. Seed data workers and samplers. DDP must aggregate metrics correctly, avoid padded-sampler duplicates, and normalize loss appropriately. Different batch/distributed layouts can change numerics: record this rather than promising bitwise equivalence.

Before a long run, verify a real batch end-to-end, GT alignment, finite forward/loss/backward, parameter-group coverage and update behavior, official validation aggregation, and model+adapter+head save/reload equivalence. A tiny-batch overfit diagnostic is useful for catching label/loss bugs; it is a separate smoke test, not evidence of validation quality or saturation.

For save/reload parity, explicitly restore evaluation backend settings as well as weights: attention implementation, dtype, TF32, fused kernels, model mode and preprocessing. Library defaults may select a different backend after reload even with the same weights. First compare under matching settings and a declared tolerance. If testing a different inference backend, report that as a separate numerical-compatibility test; do not silently relax tolerance or rerun training to fix an export-only mismatch.

Resume state includes weights/adapters/head, optimizer, scheduler, scaler, global step, epoch/exposure, sampler/data-loader state where supported, RNG states, best metric/checkpoint, patience history and immutable config hash. Exact mid-epoch replay may not be supported for every loader: state the limitation and choose a safe resume boundary. Write checkpoints/registry updates atomically. Checkpoint a failed save as incomplete, never resume it as valid.

Commit the last state, best-checkpoint reference and corresponding metric history consistently. Atomic writes of each file alone do not make the group a transaction: a crash between best and last saves must not pair a future best artifact with an older optimizer/history. Use versioned content-addressed snapshots plus an atomic commit pointer, or keep the recoverable best state inside the committed last snapshot. Reconcile external logs from that committed history and report replayed work. Release/reclassify claimed trials on cooperative failure; recover a stale claim only after establishing the previous owner is no longer active, not merely because a timestamp is old.

Validate the exact file paths the loader will use, not just historical manifest paths. Bind output directories to immutable experiment identity before writing reports, and refuse to overwrite unrelated results. Persist cumulative resource spending across restarts; a fresh process does not reset the study allowance. For local CPU/single-host examples, the optional [runtime guard](guard-tool.md#local-execution-guard) implements file verification, an OS lock and a cumulative wall ledger; cluster GPU allocation remains the project scheduler's responsibility.

The agent must generate or adapt a real trainer in the target project's supported framework, plus download/audit/profile/launch/resume commands. Test argument parsing, config resolution and smoke execution there. Do not claim this generic skill contains a universal runnable trainer. For long work use the existing scheduler/job system or an authorized persistent process, record job IDs and logs, and preserve shutdown/resume commands. Do not create recurring automations unless requested.
