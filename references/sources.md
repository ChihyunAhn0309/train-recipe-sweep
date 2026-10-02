# Primary starting sources

Checked 2026-10-02. Verify the installed version and current official material at use time. These generic sources do not replace model/task-specific research. Recipe ranges, budgets and plateau thresholds in this skill are design proposals, not claims that these publications prove universal settings.

| Source | Relevant basis |
|---|---|
| [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | Original frozen-base low-rank adaptation; inspect paper/official implementation for the selected architecture |
| [Microsoft LoRA](https://github.com/microsoft/LoRA) | Authors' implementation linked from the paper; inspect applicable files/revision when used |
| [PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora) | Targets, additional saved modules, initialization and scaling APIs; version-sensitive |
| [PEFT custom models](https://huggingface.co/docs/peft/developer_guides/custom_models) | Inspect actual architecture and supported module types |
| [rsLoRA](https://arxiv.org/abs/2312.03732) | Rank-dependent scale changes; distinguish from standard LoRA |
| [LoRA+](https://arxiv.org/abs/2402.12354) | Different learning rates for adapter A/B matrices |
| [DoRA](https://arxiv.org/abs/2402.09353) | Magnitude/direction decomposition, a distinct optional adaptation method |
| [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook) | Source-led incremental search, experiment records, training-duration/schedule coupling and validation checkpoint selection |
| [Hyperband](https://arxiv.org/abs/1603.06560) | Optional multi-fidelity budget allocation; pruning is not convergence detection |
| [PyTorch saving and loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | Model and optimizer state for continuing training |
| [PyTorch performance tuning](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html) | Data, kernel, memory and distributed execution tuning |
| [Hub download guide](https://huggingface.co/docs/huggingface_hub/guides/download) | Version-pinned downloads and file selection |

Do not infer an exact checkpoint recipe from these general documents. Every selected checkpoint and dataset needs its own verified sources, immutable identifiers and evidence ledger.
