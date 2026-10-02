# Multi-scale coverage example

This is a **synthetic configuration example**, not a real model recipe or a runnable training study. The sizes and cadence are illustrative; recover actual values from the selected checkpoint's sources. The configurations contain only enough information to demonstrate coverage checking.

From the repository root:

```sh
python scripts/recipe_coverage.py inventory --source examples/recipe-coverage/source.json
python scripts/recipe_coverage.py check --source examples/recipe-coverage/source.json --decisions examples/recipe-coverage/decisions.json --trials examples/recipe-coverage/trials.json
```

The [source](source.json) includes multi-scale training and a fixed evaluator. The [decision ledger](decisions.json) compares training enabled/disabled for Full FT, LoRA rank 16 and LoRA rank 32. Evaluation TTA stays off with an explicit reason. Six [compiled configuration fragments](trials.json) realize the two training candidates for each method.

The checker rejects an unreviewed source setting, a missing method decision, an enabled-only or disabled-only trial list when both were promised, an undeclared config value, or a changed source hash. It reports ablation exceptions and distinguishes plan-only checks from compiled candidate coverage. It cannot verify historical source completeness or whether a trainer actually applies the transforms; see [source recipe reconciliation](../../references/recipe-reconciliation.md).
