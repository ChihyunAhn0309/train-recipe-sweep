# Validation scope

This page distinguishes reproducible software checks from target-specific training evidence. The CI badge reports current workflow status; it is not a GPU certification.

## Reproducible checks

The current suite contains **130 unittest cases**: the previous 110 cases plus Unicode identity interoperability, aligned stopping caps, and 18 staged-extension cases. The original suite passed the [initial public CI run](https://github.com/ChihyunAhn0309/train-recipe-sweep/actions/runs/36980198256) on **Ubuntu and Windows with Python 3.11 and 3.13**. The workflow tests the current suite, CPU demo and repository checks in that four-job matrix; the README badge links to current results. A historical run is evidence for its own commit, not later changes.

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
| Assets and budgets | Actual loader paths, hash/size checks, interrupted state and accumulated cost persistence. |
| Document transport | Reconstructable content, hashes, portable names, traversal/collision rejection, literal argv. |
| Controller | Atomic claims, anchor gating, confirmations, valid result reuse, concurrent device ownership, physical accounting, bounded stop/resume. |
| Stage extension | Strict append-only plans, retained anchor/results/costs, immutable budget caps, active-work rejection, interruption recovery across SQLite/marker boundaries, idempotence and refreshed GPU acceptance scope. |
| Invalid results | Typed finite metrics, objective agreement, checkpoint/metrics integrity, malformed resume isolation. |
| GPU gate schema | Missing, stale and incompatible evidence rejected; source/config/device scope changes invalidate acceptance. |

The CPU demo launches real subprocesses with synthetic outputs. Its scores and device-seconds are protocol fixtures, not model quality or GPU-hours.

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
