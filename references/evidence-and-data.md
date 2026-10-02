# Evidence, checkpoint selection, and dataset audit

Read before either a plan or a live study. Prefer primary papers, supplements, authors' repositories/configs, dataset releases and official library documentation. Recheck current versions when invoked.

## Identity and literature search

Resolve architecture family, exact variant/size, modality, task, backbone versus task model, positional encoding, normalization, tokenizer/processor, input dimensions, and implementation/revision. Identical model names in two libraries need not imply identical weights or layer semantics.

Search systematically for each model and requested method:

1. Original paper + supplement + official repo + checkpoint release/model card.
2. Exact checkpoint ID + training configuration/recipe/logs, including pretraining datasets and later adaptation stages.
3. Model aliases + task + dataset + fine-tuning/LoRA/PEFT; follow relevant references and linked official implementations.
4. Model-specific adapter placement and training ablations, then same-architecture studies, then generic evidence if necessary.

Record queries, date, sources, relevant findings, inaccessible material and conflicts. Cover relevant retrieved papers and their directly relevant references; do not claim a complete survey of every publication. Stop a research round after these source classes have been checked and an additional search pass adds no material recipe evidence. Reopen research when a missing value affects a consequential decision.

For consequential claims record `claim_id`, URL, accessed date, paper section/table or repo commit/file/key, model/task/dataset context, extracted value, and status:

- `verified`: directly supported by inspected source or measured artifact.
- `derived`: transformed from a verified value, with formula and assumptions.
- `proposed`: a new experiment choice with rationale.
- `unknown`: unrecovered; never silently filled with a plausible default.

Distinguish authors' reported scores from local reproductions. Resolve paper/config conflicts using the artifact demonstrably used for the selected weights; otherwise expose the conflict. An inference config is not proof of its training recipe.

## Selecting and acquiring weights

Compare candidates: exact ID/revision, official status, pretraining datasets, compatibility, task fit, popularity evidence/date/window, recipe availability, license/access, size and checksum. Prefer the widely adopted official compatible checkpoint when supported. Downloads, stars and citations measure different things; mirror downloads are not globally comparable. If popularity cannot be established, say so and choose the strongest defensible candidate. Preserve a user-selected checkpoint unless an actual incompatibility needs resolving.

Search official releases and maintained registries before concluding weights are absent. Distinguish `not_found_after_search`, `gated`, `inaccessible`, `incompatible`, and `available`. A compatible pretrained backbone can support a new task decoder: document partial pretraining rather than calling the whole model scratch.

Gated access, a network error or a missing local cache does not mean pretrained weights do not exist. Preserve the pretrained plan and report the specific access problem; do not silently switch to scratch. Scratch requires genuinely unavailable/incompatible weights after search or an explicit user choice, with the changed compute/data requirements recorded.

Execute mode downloads required shards/index, architecture config, model code revision, tokenizer/processor, vocabulary, normalization metadata and source recipe. Pin downloads by revision; filter files only after inventorying requirements. Verify publisher size/hash when available and record local SHA-256 otherwise. A local hash proves repeatability, not publisher authenticity. Detect partial downloads and unsafe archive paths. Inspect code before executing downloaded scripts or enabling custom remote code. Prefer safe tensor/state-dict loading over unknown executable pickle files.

Reuse verified caches and resumable downloads. Estimate compressed/unpacked/cache/checkpoint disk requirements first. Do not invent credentials or accept licenses on the user's behalf. Report unresolved access steps while completing independent work.

## Dataset and GT manifest

Inventory the exact release, requested subsets/splits and **all public GT-related metadata for that release**, including metadata not consumed by the trainer. Record downloaded/verified, cached/verified, unavailable/gated, or not applicable with reason. Hidden test GT must not be called downloaded or complete.

Define whether the dataset name denotes an upstream original release or a downstream converted/Hub release. A complete inventory of the latter does not prove complete upstream provenance. Follow documented links for annotation/group/split provenance relevant to the task, inventory upstream GT-related files, and mark any unacquired material separately. For example, phrase-level labels in a converted text dataset may omit original sentence/tree grouping needed to audit related-example leakage. State the precise completeness boundary in the manifest and report.

Include when present: raw assets; class IDs/names and ontology; label mapping/version; instance/semantic/panoptic masks and palettes; polygons/boxes and coordinates; keypoint order, visibility and skeleton; tracking IDs; captions/transcripts/token alignment; camera calibration/depth units; subject/session/site/time/domain IDs; official splits; ignore/crowd/void labels; weak/pseudo-label provenance; official evaluator and metric settings; errata and license. Join through stable IDs, record normalization transforms and preserve originals.

When a converted release omits source IDs, retain every compatible origin candidate rather than inventing one unique lineage. Report raw/exact joins separately from normalization, tokenization or spelling compatibility. A join filtered to the expected official split demonstrates compatible support, not independent proof of the converter's original group assignment. Compare alternate official releases and preserve discrepancies. Keep source snippets or lossy review/group matches as candidate evidence. Public upstream annotations do not authorize using a benchmark's hidden test labels for selection; separate provenance audit outputs from training/evaluation inputs.

Audit before training:

- Expected/actual sample and annotation counts, corrupt/missing files, duplicate IDs, orphan GT, unlabeled samples, empty examples/splits.
- Conflicting GT for identical or normalized-identical inputs. Separate these from harmless same-label duplicates; record IDs and disposition rather than silently keeping the first label or majority voting. Verify whether any selected subset includes them.
- Label meaning/order, invalid values, background/ignore conventions, missing classes across datasets; no implicit concatenation of unrelated ontologies.
- Exact and task-appropriate near duplicates, group/subject/time overlap, paired/multi-view duplicates and augmentation-before-split leakage.
- Class/group/domain distributions, rare-label counts, imbalance, independence units, distribution shift and benchmark rules.
- Input/GT geometry, resize/crop or token alignment, loss normalization and evaluator behavior. Visually inspect representative annotations where useful.
- Pretraining/evaluation contamination when provenance permits checking; report unknown contamination instead of asserting absence.

Preserve user splits. If unsuitable, report the observed issue and propose an official, group-stratified or temporal alternative. Do not silently change a scientifically meaningful split. Under delegated discretion choose and save a suitable split policy and membership hashes. Fit learned preprocessing/statistics only on train, except explicitly allowed pretrained artifacts. Validation must be deterministic.

## Train/validation sufficiency

No universal minimum sample count. Report counts per class/group, absent classes, independent groups, GT quality, metric uncertainty and distinguishable effect sizes. For independent binary correctness a rough worst-case 95% interval half-width is `0.98 / sqrt(n)`; clustered data require group-aware intervals. For mAP/IoU/F1 use appropriate bootstrap or repeated group splits where feasible; identify approximations.

A small validation set can overfit through repeated sweep selection. Recommend larger holdout, nested/group CV or untouched confirmation data when warranted. Folds become part of trial identity and budget. Missing validation/usable GT blocks dependent model selection, not independent audit/plan work. Do not fabricate labels or select on test. If train is inadequate, explain feasibility and uncertainty rather than promising a meaningful optimum.
