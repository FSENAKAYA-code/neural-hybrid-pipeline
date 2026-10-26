# Branch scope: `development`

Integration plan: keep the project runnable while integrating ingestion, preprocessing, model, metrics and documentation changes.

- [ ] Implement the scoped change in source/tests.
- [ ] Run `python -m pytest -q`.
- [ ] Document data provenance, limitations, and reproducibility.
- [ ] Merge only after reviewing the diff and generated metrics.
