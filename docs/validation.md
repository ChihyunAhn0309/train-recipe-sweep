# Validation scope

This page distinguishes reproducible software checks from target-specific training evidence. The CI badge reports current workflow status; it is not a GPU certification.

## Reproducible checks

The v0.2.0 suite contained **163 unittest cases**; the current suite additionally covers adaptive horizon review and sparse-loss-spike rejection. The exact current count is printed by the command below and by CI. The original suite passed the [initial public CI run](https://github.com/ChihyunAhn0309/train-recipe-sweep/actions/runs/36980198256) on **Ubuntu and Windows with Python 3.11 and 3.13**. The workflow tests the current suite, CPU demos and repository checks in that four-job matrix; the README badge links to current results. A historical run is evidence for its own commit, not later changes.

```sh
python -B -m unittest discover -s scripts -p "test_*.py" -v
python tools/check_repository.py
python tools/run_cpu_demo.py --output ./demo-run
```

| Area | Behavior exercised |
|---|---|
| Trial identity | Full SHA-256, immutable config, duplicate-key/nonfinite rejection, schedule/config changes. |
| Source recipe coverage | Unreviewed source keys, stale hashes, missing rank decisions, unjustified missing on/off controls, unrealized conditional branches and differences from compiled configs. |
| Convergence | Learned versus stagnant trajectories, warmup/schedule guards, near-zero thresholds and censoring. |
| Horizon transfer | Raw-history replay for distinct existing recipes, comparable family/exposure, conditional third evidence, numeric LR identity, plateau revocation, censoring and separate calibration/transfer outcomes. |
| Assets and budgets | Actual loader paths, hash/size checks, interrupted state and accumulated cost persistence. |
| Document transport | Reconstructable content, hashes, portable names, traversal/collision rejection, literal argv. |
| Controller | Atomic claims, anchor gating, confirmations, valid result reuse, concurrent device ownership, physical accounting, bounded stop/resume. |
| Stage extension | Strict append-only plans, retained anchor/results/costs, immutable budget caps, active-work rejection, interruption recovery across SQLite/marker boundaries, idempotence and refreshed GPU acceptance scope. |
| Invalid results | Typed finite metrics, objective agreement, checkpoint/metrics integrity, malformed resume isolation. |
| GPU gate schema | Missing, stale and incompatible evidence rejected; source/config/device scope changes invalidate acceptance. |

The CPU demo launches real subprocesses with synthetic outputs. Its scores and device-seconds are protocol fixtures, not model quality or GPU-hours.

## Adaptive horizon review checks

The [horizon review helper](../references/horizon-review.md) replays raw histories through the plateau guard. Its fixtures test reuse of two existing recipes, coincident baseline/anchor roles, conditional third evidence, immutable identity/family mismatches, comparable exposure, censored/stalled/pruned results and late-learning counterexamples. Prespecified duration agreement uses the later observed plateau as the provisional review point; evidence that the shorter anchor horizon was insufficient remains visible. An agreeing pair does not require a third full training run.

Independent review found sparse loss/optional-score spikes that the existing median-based checks could miss, nonfinite intermediate statistics from finite extreme inputs, and invalid or duplicate effective LR vectors caused by floating-point underflow, overflow or rounding. The corresponding regressions exercise the guard and resolved group-LR validation. Deep malformed JSON produces the documented structured CLI error. These are synthetic behavior checks, not new measured model-training results.

The final independent code recheck passed 633 assertions per Windows interpreter (Python 3.11 and 3.13), including 600 duration-order/tolerance/third-participant combinations. Original counterexamples were rerun after repairs, along with near-zero absolute-tolerance and valid low-noise cases. The rechecked helper SHA-256 was `e0d12af6217b4d37949a9c5bddeff66f4b401d7d4936f52dce07fd7b34806561`; the plateau guard was `f7c15cc9cb247292680324e00ccef6018d631e8bd0236f9210788c5e9c46f2bb`. No actionable issue remained in that reviewed scope. This is independent development review, not third-party certification.

A separate forward-testing session passed 61 independent API scenarios and nine CLI executions, including realistic Full FT/LoRA family planning, identical anchor/baseline reuse, different-optimizer rejection and matched representative selection. The complete 219-case regression suite passed locally on Windows Python 3.11 and 3.13 after repairs, along with both documented CPU demos and the horizon CLI fixture. The public CI run establishes its own exact commit/platform results.

Run the [documented horizon fixture](../references/horizon-review.md#runnable-cpu-fixture-inputs) to inspect an actual CLI decision. It reuses supplied histories and launches no training. A complete model-specific handoff must separately test real scheduler dispatch, deduplication, continuation, budget accounting and target-GPU integration. This helper does not automatically change the generic controller or fast scheduler.

## Fast-mode checks

The opt-in [fast policy](../references/fast-search.md) has a synchronous decision helper and a tiny real CPU classifier demonstration on generated data. The 33 added regressions cover cohort/seed/schedule identity, barriers, protected/grace candidates, ties, min/max ranking, invalid metrics/artifacts, stale/future/pruned history, preservation of best checkpoints and actual state continuation against uninterrupted training.

```sh
python tools/run_fast_demo.py --output ./fast-demo --compare-exhaustive
```

In the local deterministic toy example, ten configurations use 3/9/27-epoch rungs, a protected baseline and 3/1 unprotected survivors. Best-so-far promotion trains 90 candidate-epochs versus 270 for the full reference, selects the same configuration and validation-best checkpoint metric (BCE 0.1760184564419013), and matches both surviving full training states exactly after six resumed segments. Reloading the selected checkpoint reproduces its metric. The reference training is additional validation work, outside the 90-epoch search allocation.

An earlier current-rung-only promotion experiment on the same toy data discarded a better checkpoint and had BCE regret about 0.03236. This motivated explicit alignment of promotion and checkpoint-selection objectives, preservation of best-so-far state, and separate reporting of better nonfinal checkpoints. It is a demonstrated risk, not evidence that best-so-far ranking always wins. The small demo's I/O and decision overhead can dominate training, so reduced exposure does **not** establish wall-clock speedup. No pretrained Full FT, LoRA, source-benchmark reproduction or GPU throughput is tested by this toy example.

The decision helper is not a scheduler or budget enforcer. A complete fast GPU handoff must supply and test selective dispatch/resume, atomic leases, actual accounting, deadline/finish reservations and target acceptance. The ordinary study controller does not automatically implement halving or ASHA.

Two independent agent sessions reviewed the fast-mode instructions and code. Their findings led to rejection of regressing cumulative best metrics, preservation of historical best-checkpoint identity, a separate nonfinal-checkpoint signal, and clearer standalone-anchor/competitive-pruning/bootstrap gates. The final instruction recheck passed 10 CPU protocol probes; the code reviewer passed 19 independent probes on each Windows interpreter and verified fresh-process reload and JSON pause/resume equivalence for all ten toy configurations on Python 3.11 and 3.13. No actionable issue remained in the rechecked scope. These are development observations, not third-party certification or real model-specific GPU validation.

## Independent development review

Separate agent sessions reviewed and forward-tested the implementation during development. Their feedback led to regression fixes for bundle path collisions, Python option-like entrypoints, malformed result isolation, objective mismatches, confirmation-role aliasing, device-probe races and controller-source acceptance scope. The final independent portable checks covered 18 scenarios per interpreter; the controller checks covered 13 per interpreter on Windows Python 3.11 and 3.13. These are additional development observations, not third-party certification. The public regression suite is the reproducible release check; private machine paths and large raw experiment artifacts are not included.

## What remains target-specific

### Final independent review on 2026-10-03

Three separate agent sessions performed a scientific evidence audit, a blind planning request with primary-source research, and an adversarial code review. Their findings led to these changes:

- Fresh baseline/finalist seeds are paired; exploratory winning seeds stay outside the confirmation aggregate. Official reproduction, source-derived baseline comparison, best observed candidates and seed stability have distinct evidence requirements.
- `study_extend.py` implements the search-to-finalist transition while retaining the same registry, anchors, results and cumulative spending. Eighteen regressions cover its operational contract. Another reviewer independently checked 19 scenarios on Python 3.13 Windows, including three stages, two methods packed on a synthetic device, failed-candidate preservation and post-commit recovery.
- UTF-8 canonical trial fingerprints agree across helpers for non-ASCII configurations. ASCII identities remain unchanged; existing non-ASCII bound outputs need explicit migration or preservation under their original controller, not silent reinterpretation.
- The plateau helper rejects a hard cap that cannot be reached on its exact evaluation cadence. Total-horizon-dependent linear decay is explicitly included in schedule identity rules.
- Scientific-specification requests do not accidentally require building an executable trainer. Complete executable handoffs still require implementation and clean-room reconstruction.

These reviews found no remaining actionable issue within the rechecked scope after repairs. Separate agent review is not third-party certification or proof that arbitrary future studies will work. The extension fixture is synthetic, and a `confirmation` role alone does not certify independent seeds or a correctly frozen selection; the model-specific adapter supplies those semantics.

An earlier local real-model CPU demonstration used BERT-Tiny, 512 SST-2 training rows, 872 validation rows, one seed and nine LR trials. A later checkpoint re-evaluation reproduced all nine saved accuracies. Its best observed Full FT/LoRA16/LoRA32 validation scores were 74.31%/71.10%/71.90%; all original trials were capped without observed saturation. That experiment does not establish source-recipe reproduction, independent confirmation, search-algorithm quality, or validation of every later controller change. Large data/weight/checkpoint artifacts are not distributed here.

The source-recipe reconciliation addition received a separate review with **29 independent scenarios per interpreter** on Windows Python 3.11 and 3.13, plus both documented example commands on each. The review found and verified fixes for two false passes: excluded techniques remaining active, and Full FT configs being labeled as LoRA coverage. The checker now validates explicit exclusion assertions and matching method identities. These checks establish declared configuration consistency, not historical source completeness or real trainer behavior.

**Actual GPU/CUDA training has not been validated for this generic controller release.** Neither have model-specific head/adapter correctness, training throughput/VRAM, mixed precision, real model checkpoint/resume equivalence or distributed execution. CPU tests and document reconstruction cannot establish those properties.

The generic skill is a workflow and a set of control helpers, not a universal pretrained-model trainer. Each complete model-specific plan must supply its actual trainer/worker, pinned dependencies, download and GT audit code, objective/evaluator, search definition, profile procedures and consistent checkpoint/resume implementation.

On the target, follow [GPU acceptance](../references/gpu-acceptance.md). All six gates require actual evidence. The bounded diagnostic bootstrap is separate from long-sweep admission, and its cost belongs to the same overall budget. Plan reconstruction, evidence-integrity validation and measured GPU readiness remain separate claims.

## Operational boundaries

- One host, local filesystem, one physical GPU per worker. No DDP or cluster lease implementation.
- Packing is for independent workers and requires measurements of combined load.
- Physical cost is the union of recorded device allocation intervals; external scheduler reservations and preflight costs belong in the overall ledger separately.
- Automatic recovery after abrupt Windows runner death is intentionally unavailable. Unknown ownership fails closed; do not remove lock files to bypass it.
- Hashes establish declared file integrity, not measurement truth or adversarial tamper resistance.
- Budget deadlines include a shutdown allowance but are not hard real-time guarantees.

See the [controller protocol](../references/study-controller.md) for exact supported recovery paths and semantics.
