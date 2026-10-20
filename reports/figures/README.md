# Archived research figures and exact numeric results

This folder contains **curated experimental outputs** suitable for GitHub and for discussing results with the research advisor. The model checkpoints, uncurated experiment files, and repeatedly generated PNGs are kept in the ignored `reports/transformer_experiment/` folder.

## Publish the October 20 directional run

The full-resolution `directional_sensitivity.png` and exact directional CSVs were generated on the researcher's Windows computer. To publish **those original outputs**, open a PowerShell terminal in the cloned project folder and run:

```powershell
git pull origin main
py code/publish_directional_results.py --label 2026-10-20
git add reports/figures/
git commit -m "Archive October 20 directional sensitivity figure and metrics"
git push origin main
```

Review `git status` before committing, particularly if other local files have been modified. Publishing does **not** retrain a model or rerun any experiment. It copies the graph and summary/data/config files to versioned names, validates the expected 32-position outputs, and generates a Markdown report with actual top-response values. If a different publication already exists for the label, the script refuses to overwrite it.

Expected new files:

- `directional_sensitivity_2026-10-20.png` — original full-resolution heatmap + response curves.
- `directional_summary_2026-10-20.csv` — exact averaged results, 32 positions × 2 models.
- `directional_per_repeat_2026-10-20.csv` — the response for each noise draw.
- `directional_model_config_2026-10-20.json` — non-weight training configuration for provenance.
- `directional_results_2026-10-20.md` — auto-generated result table and interpretation.

This setup avoids uploading `transformer.pt`, bulk synthetic data, or machine caches. It also avoids reconstructing precise data from a screenshot. The screenshot-based [first interpretation](../transformer_experiment/directional_findings_2026-10-20.md) is retained separately for context.
