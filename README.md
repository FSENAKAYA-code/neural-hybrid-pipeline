# neural-hybrid-pipeline

> Educational portfolio scaffold by Fatma Sena Kaya · **Multimodal learning**

A modular starting point for combining image and tabular features

## Scope and status

This repository is a reproducible learning project scaffold, not a production service. It contains an executable baseline, clear data boundaries, and extension points. Results are meaningful only when run on documented, appropriate data. No dataset, model weight, credential, or private record is included.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/train.py
uv run pytest
uv run ruff check .
```

The demo uses generated toy data only. Replace it with data you are allowed to use, record its source and license, and keep private or raw data out of Git.

## Data and evaluation

- Expected input: Paired, consented image and tabular examples.
- The baseline separates training, validation, and test partitions with a fixed seed. The test partition is held out until final evaluation.
- Fit preprocessing steps on training data only. Use validation results for model selection; report the test result once for the chosen configuration.
- Check the project's `src/hybrid/` code and tests before adapting it to real data.

## Repository layout

```text
data/       empty raw/ and processed/ locations; contents are ignored by Git
models/     empty local model-output location; model weights are ignored
scripts/    runnable demo entry points
src/        importable project code
tests/      small deterministic unit tests
```

## Responsible use

Late-fusion architecture experiments

## Roadmap

1. Add a documented, licensed dataset adapter.
2. Add domain-specific preprocessing and baselines.
3. Tune against validation data and report held-out test metrics with limitations.
4. Add deployment only after privacy, security, and monitoring review.

## Author

Fatma Sena Kaya · [GitHub](https://github.com/FSENAKAYA-code) · [LinkedIn](https://www.linkedin.com/in/sena-kaya-8634ba383/)
