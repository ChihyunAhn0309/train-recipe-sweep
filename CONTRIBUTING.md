# Contributing

Use Python 3.11 or newer. The bundled helpers have no third-party runtime dependencies. Keep real model dependencies in model-specific plans rather than the generic controller.

Before submitting a change, run from the repository root:

```sh
python -B -m unittest discover -s scripts -p "test_*.py" -v
python tools/check_repository.py
python tools/run_cpu_demo.py --output ./demo-run
```

Use a fresh demo directory. Include the problem, the resulting behavior and the validation performed in your pull request. Regression tests should exercise observable behavior, especially identities, budgets, concurrent ownership, checkpoint consistency and portability.

Preserve the separation between CPU fixture evidence and actual GPU measurements. Do not claim a target gate passed because a schema or synthetic test passed. For hardware results, state the model/data revisions, device UUID/type, environment, code/config hashes, allocation, measured throughput and memory, costs and unresolved limits. Never commit credentials, private datasets, checkpoints or host-specific paths.

Keep SKILL.md focused on the agent's decisions. Put protocol details in the linked references and retain the user's requested methods, splits, budget and execution scope. Update documentation when changing schemas or recovery guarantees. The controller's source hash participates in acceptance scope, so controller changes require fresh target acceptance.

By contributing, you agree that your contributions are distributed under this repository's MIT license. Third-party material must have compatible terms and attribution.
