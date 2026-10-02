# Task heads, Full FT, and model-specific LoRA

## Output contract

Inspect actual modules/outputs, ontology, loss, target encoding and official evaluator. Class count alone is insufficient.

| Task | Adaptation to verify |
|---|---|
| Single-label classification | Label order, output size, background; logits and cross entropy |
| Multi-label classification | Independent outputs, BCE-style loss, train-derived weights, validation-only thresholds |
| Semantic segmentation | Main and auxiliary classifiers, ignore ID, pixel alignment/resolution |
| Detection/instance segmentation | Classifier, background, class-dependent box/mask heads, query/anchor/postprocessing |
| Keypoint/pose | Keypoint order/count, visibility, skeleton, heatmap/coordinate decoder |
| Regression | Output dimension, units, train-derived target scaling and loss |
| Language modeling | Preserve compatible vocabulary projection; inspect tied embeddings; resize only for vocabulary/task changes |
| Retrieval/multimodal/generative | Projection/decoder and task loss may be the output; do not invent a classifier |

Change incompatible outputs and required interfaces, preserving compatible learned components. Matching dimensions with different label meanings require reinitialization or a justified row remap. Include auxiliary outputs.

## Load and prove trainability

1. Instantiate the verified architecture, load compatible tensors and exclude only identified incompatible ones. `strict=False` alone is not a compatibility audit or a universal shape-mismatch solution.
2. Save matched/missing/unexpected/shape-mismatched keys, counts and bytes with reasons. Fail unexplained backbone omissions. Document positional interpolation or channel adaptation.
3. Initialize replacement components reproducibly. Save head spec and seed. Reuse base weights and initial head within each seed across methods when feasible.
4. Full FT unfreezes the complete intended model, including head. Document non-learnable tensors and required exceptions. Head-only warmup, if justified, is an explicit phase, not the final mode.
5. Build optimizer groups by parameter identity after wrapping/injection. Every trainable tensor appears once; frozen tensors stay out. Separate backbone/head or adapter/head LRs and decay policy. Layer-wise LR decay is an explicit choice.
6. Save actual names/counts/shapes and group LRs. Run several updates: intended tensors receive finite gradients and change where mathematically expected; frozen weights remain unchanged. With zero-initialized LoRA B, A can have zero gradient on the first step: do not require all gradients to be nonzero immediately.
7. Specify BatchNorm buffers and dropout/train/eval behavior: frozen parameters do not freeze running statistics. Save/reload must reproduce evaluation outputs within dtype tolerance, including head, label mapping and processors.

Without compatible pretrained weights, initialize and train the full architecture using a scratch baseline. Equally random heads/backbones need not use an arbitrary LR multiplier. Report compute/data implications. Frozen-random-backbone LoRA is not a pretrained LoRA baseline. Offer compatible pretraining or separately budgeted pretrain-then-adapt; run frozen-random LoRA only if explicitly desired as an experiment.

## LoRA evidence and placement

For each model/rank create:

`paper/repo/config -> model/task -> mathematical target -> module paths/types/shapes -> trainable policy -> recipe -> evidence tier`.

Tiers: exact model/task, same architecture/task family, transferable general research, proposed hypothesis. Report missing direct evidence. Read relevant full methods/experiments, supplements and configs: abstracts rarely specify recipes. Research original LoRA and applicable later studies without automatically substituting variants.

Verify q/k/v/o correspondence, fused QKV slices, MLP gates/up/down, convolution layout, transpose/fan-in-out, layer subsets, MoE targets and shared/tied weights as relevant. Expand selectors against actual modules/parameters; save resolved names/counts. Reject zero matches, unintended layers or accidental task-head adaptation. Paths differ across libraries and versions.

Standard LoRA freezes base weights and trains low-rank updates. Record rank, alpha, scaling, dropout, initialization, targets, trainable head and bias/norm/embedding exceptions. With PEFT, ensure the task head is trainable and saved (often `modules_to_save`) and verify using the installed version. [Original LoRA](https://arxiv.org/abs/2106.09685), [PEFT](https://huggingface.co/docs/peft/package_reference/lora).

The trainable-head requirement applies to newly initialized/replaced task heads. If no new head is needed (for example a compatible existing LM projection), follow the evidenced baseline's freeze policy; do not automatically unfreeze a large vocabulary head. If no head parameters are trainable, mark head LR not applicable instead of inventing an optimizer group.

Each requested rank has a separate search. Report alpha and scale: standard `alpha/r` versus rsLoRA `alpha/sqrt(r)`. Constant alpha changes effective scale across ranks; constant scale requires changing alpha. State the comparison policy and tune when warranted. [rsLoRA](https://arxiv.org/abs/2312.03732).

LoRA+ separates A/B learning rates; DoRA uses magnitude/direction decomposition; QLoRA quantizes base weights. Treat these as optional distinct methods with separate labels/evidence/budgets, not silent versions of standard LoRA. [LoRA+](https://arxiv.org/abs/2402.12354), [DoRA](https://arxiv.org/abs/2402.09353).

## Recipe transfer

Capture optimizer/version, group LRs, decay exclusions, effective batch/tokens, input length/resolution, loss/reduction, augmentation, scheduler/warmup in steps, total exposure, precision, clipping, EMA, seed and evaluator. LR scaling with batch is a hypothesis to validate, not an automatic law. Mark unspecified defaults explicitly.

Keep checkpoint-generation, source downstream, target Full FT and target LoRA configs separately with source-linked diffs. A source recipe is a starting point: new task, data volume and head can change appropriate LR and horizon.
