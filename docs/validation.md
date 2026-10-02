# Validation scope

This page distinguishes reproducible software checks from target-specific training evidence. The CI badge reports current workflow status; it is not a GPU certification.

## Reproducible checks

The current suite contains **110 unittest cases**: the original 95 guard/bundle/controller cases plus 15 source-recipe coverage cases. The original suite passed locally on Windows with Python 3.11 and 3.13, then passed the [initial public CI run](https://github.com/ChihyunAhn0309/train-recipe-sweep/actions/runs/36980198256) on **Ubuntu and Windows with both Python 3.11 and 3.13**, including the documented CPU demo and repository checks. The workflow continues to test the current suite in that four-job matrix on pushes and pull requests; the README badge links to current results.

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
| Invalid results | Typed finite metrics, objective agreement, checkpoint/metrics integrity, malformed resume isolation. |
| GPU gate schema | Missing, stale and incompatible evidence rejected; source/config/device scope changes invalidate acceptance. |

The CPU demo launches real subprocesses with synthetic outputs. Its scores and device-seconds are protocol fixtures, not model quality or GPU-hours.

## Independent development review

Separate agent sessions reviewed and forward-tested the implementation during development. Their feedback led to regression fixes for bundle path collisions, Python option-like entrypoints, malformed result isolation, objective mismatches, confirmation-role aliasing, device-probe races and controller-source acceptance scope. The final independent portable checks covered 18 scenarios per interpreter; the controller checks covered 13 per interpreter on Windows Python 3.11 and 3.13. These are additional development observations, not third-party certification. The public regression suite is the reproducible release check; private machine paths and large raw experiment artifacts are not included.

## What remains target-specific

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
