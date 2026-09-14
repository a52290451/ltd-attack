# LTD-Attack — local root cleanup (Phase 2)

Phase 2 removes residual local clutter from the repository root without deleting project information or changing script logic.

## Files moved

- Five tracked local datasets moved to `data/local/historical/`:
  `01_all_features.csv`, `final_vectors_sites.csv`,
  `05_01_daily_embeddings_maestras.csv`, `06_all_site_embeddings_maestras.csv`,
  and `site_dictionary.csv`.
- Seventeen tracked diagnostic figures from `Graficas/` moved to
  `results/historical/diagnostics/Graficas/`.
- Three tracked historical Macro figures from `features_old/` moved to
  `results/historical/macro/features_old/`.
- Four tracked historical DML figures from `deep_metric_learning/` moved to
  `results/historical/dml/deep_metric_learning/`.
- Four untracked local `.pyc` files moved to
  `legacy/diagnostics/bytecode/` while retaining their origin subdirectories.
- Root `.DS_Store` moved to `legacy/diagnostics/metadata/.DS_Store`.
- Untracked ignored `credentials_qwen.json` moved to `../credentials_qwen.json`.

Every file operation is recorded with `migration_phase=2` in
`docs/migration/LOCAL_MIGRATION.csv`. Trackable files used `git mv`; ignored
local files used `mv`.

## Empty directories removed

The following directories were listed, checked for remaining files, and then
removed with `rmdir` only after the check reported no files:

- `concept_drift/`
- `concept_drift_old/`
- `deep_metric_learning/`
- `estado_del_arte/`
- `features/`
- `features_old/`
- `Graficas/`
- `vectores/`
- `vectores_features_v2/`
- `Vectores-24h/`
- their empty `__pycache__/` subdirectories
- the empty root `__pycache__/`

No directory containing an unknown or unclassified file was removed.

## Files deliberately retained outside the root cleanup

`docs/.DS_Store` remains under the already allowed `docs/` tree as ignored
filesystem metadata. Zeus checkpoints, external results, and paths referenced
by historical scripts were not moved. No local checkpoint file was found.

## Security

`git ls-files credentials_qwen.json` returned no tracked path. The file was
never read or printed, remains covered by `credentials*.json`, and was moved
outside the repository. `.gitignore` now also covers `.env.*` and `secrets/`.

## Non-functional scope

No imports, relative paths, experiment logic, datasets, figures, historical
code, or checkpoint contents were modified. The known path follow-up remains
documented in `docs/migration/PENDING_PATHS.md`.
