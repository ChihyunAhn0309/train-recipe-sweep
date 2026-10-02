# A plan document that can be executed on another machine

Use when planning without GPU or whenever the user will move only the plan document and this skill to a GPU environment. Deliver a human-readable plan with a self-contained embedded file bundle, produced and checked by `scripts/plan_bundle.py`. An optional ZIP is a convenience; the document itself must contain the source/config files needed for reconstruction.

## Content contract

The plan must resolve model IDs/revisions, task/head/label semantics, requested methods/ranks and actual module mapping, source/baseline provenance, metrics/direction, dataset releases and intended splits, acquisition hashes and full GT inventory/audit scope, seeds, finite search/axis dispositions, scheduler/early-stop policies, minimum-LR anchors and reuse/censoring rules, fair budgets/reserves, and final confirmation/report policy. Mark verified/derived/proposed/unknown values and unresolved target facts distinctly.

Embed these runnable resources, or exact pinned public sources plus a checked acquisition/patch command that reconstructs them:

- Model/task trainer and real framework configuration; download scripts and revision/hash manifests for weights/tokenizer/raw data/GT; deterministic data/split/group audit.
- Resolved source recipe manifests/original evidence, the [per-method decision ledger](recipe-reconciliation.md), `recipe_coverage.py`, compiled candidate coverage, and augmentation/pipeline smoke checks. Preserve multi-scale distributions/cadence, enabled/disabled controls and justified exclusions in the transferred document.
- Experiment config producer with environment-independent scientific definitions and explicit target-resolution fields. Generate final semantic trial IDs only after target defaults, backend/precision/layout and code/environment hashes are resolved.
- Controller/worker adapter with real atomic dispatch, resume/reuse, budget admission/accounting and report aggregation. A list of responsibilities for an absent controller is not a complete execution package.
- Real device smoke and export/resume checks; profiling and telemetry; finite layout-selection rules; prepare/verify/profile/run/resume commands.
- Dependency contract that can select an appropriate target CUDA build. Do not freeze the author's CPU-only Torch wheel as the GPU requirement. Pin and save the actual target environment after resolving compatible versions and checking numerics.

For single-host, one-device-per-trial execution, the bundled [controller reference](study-controller.md) supplies dispatch/recovery/accounting. Include it with a real model-specific worker adapter and test their integration before packaging the plan. Its fake worker is only a CPU protocol fixture and cannot substitute for that adapter. Large multi-device trials or cluster execution need the project's appropriate scheduler integration. The target's profiling/verification tools must also include their measured GPU costs in the study ledger before assigning the controller its remaining allowance.

Implement the bounded integration-probe bootstrap described in [gpu-acceptance.md](gpu-acceptance.md): target-host component checks admit a small diagnostic scope, its real controlled GPU evidence admits the main-study scope. Do not require the main controller's own GPU pass before any probe can run, or substitute a fabricated passed flag. These probe and long-study scopes and budgets belong in the generated commands/config producer.

All paths must resolve relative to the extracted workspace or explicit invocation inputs. The plan may use a new workspace-local cache, with every required asset reproducibly downloadable. Do not rely on `C:/Users/...`, `/home/author/...`, an unprovided local patch, prior chat context, or a cached raw file not represented by acquisition metadata. Credentials/private assets cannot be embedded or invented; identify the necessary external input and label completeness conditional on it.

## Portable manifest

Create a JSON spec with `schema_version: 1`, nonempty `plan_id`, a concise `purpose`, `scientific_spec` (the resolved contract above), `external_inputs` (empty when none), `target_checks` describing the six checks in [gpu-acceptance.md](gpu-acceptance.md), and `entrypoints`. Required entrypoint names are `prepare`, `verify_target`, `profile`, `run`, `resume`; each is an argv list beginning with `{python}` followed by an embedded script's relative path and its arguments. `{python}` means the verified target interpreter; no shell expansion is required. Entrypoints must actually implement their named operations, not print TODOs or fabricate passed flags.

The packer embeds UTF-8 files as a hashed base64 payload under an exact marker. This prevents script contents from accidentally closing Markdown code fences. It checks transport integrity and safe extraction, not the correctness of a training recipe. The readable plan must explain proposed choices, unresolved measurements, expected outputs, command order and recovery/acceptance criteria before the payload. No weight/data binaries, credentials or machine-specific caches belong inside it.

```text
python scripts/plan_bundle.py pack --spec portable-spec.json --files execution-package --overview plan-text.txt --output training-plan.md
python scripts/plan_bundle.py verify training-plan.md
python scripts/plan_bundle.py extract training-plan.md --output fresh-workspace
```

The extractor never executes embedded code. It validates the complete payload and rejects traversal/symlink escapes, duplicate paths, corrupt files and overwriting an existing destination. Review the reconstructed source before running it. Use `{python}` with a resolved actual interpreter and run commands as argv, not interpolated shell text.

## Verification required before handing off the document

Perform a clean-room reconstruction: move/copy only the document to a new empty directory, use the skill's extractor, then validate command parsing, config resolution, import/dependency declarations and a small CPU/fake-worker integration where applicable. Exclude the author's caches and helper paths from imports; declared downloaded inputs may be supplied through an explicit test input directory. Test hashes, entrypoint file presence and interruption/reuse/accounting contracts. Record exactly which commands actually ran.

For a CPU-only authoring environment, the following are distinct:

- `portable_plan_complete=true`: transport/reconstruction, runnable implementation and complete conditional specification were checked; no hidden local dependencies remain within the declared external-input boundary.
- `target_verified=false`: actual GPU environment, smoke, performance/profile and device-budget acceptance are deferred.
- `launch_ready=false`: turns true only after the receiving agent resolves and passes every relevant target check.

When no usable model/data/trainer exists yet, do not label the document complete. Resolve missing facts, build the implementation and test what is locally possible; retain genuine access/GT/hardware dependencies with exact resolution steps. A complete conditional plan may still fail target acceptance—then the executing skill diagnoses, repairs and rechecks it rather than blindly starting the sweep.

## Receiving invocation

When the user supplies this document on a GPU host, reconstruct it with the installed skill, validate its scientific/source/data assumptions, acquire declared inputs, and execute [gpu-acceptance.md](gpu-acceptance.md). Prefer reusing the tested implementation; regenerate only what the new environment/task requires, with recorded diffs and new identity where semantics change. Resolve runtime branches from measurements, then execute registered anchors and the approved search within budget. The user does not need the author's original conversation or workspace.
