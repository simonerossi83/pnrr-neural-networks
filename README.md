# PNRR project-title classification

Two deep learning models trained on the local OpenPNRR CSVs: a TF–IDF MLP and an embedding-based text CNN.

[Open in Colab](https://colab.research.google.com/github/simonerossi83/pnrr-neural-networks/blob/main/pnrr_complete_workflow.ipynb) · [Live choropleth](https://simonerossi83.github.io/pnrr-neural-networks/) · [Read the executed notebook](https://simonerossi83.github.io/pnrr-neural-networks/notebook.html) · [Download input data](https://github.com/simonerossi83/pnrr-neural-networks/releases/tag/v1.0.0)

## Get the input snapshot

Download `pnrr-input-data.zip` from the [v1.0.0 release](https://github.com/simonerossi83/pnrr-neural-networks/releases/tag/v1.0.0) and extract it into the project root. It supplies `data/progetti.csv`, `data/misure.csv`, `data/temi.csv`, and the optional map input `data/progetti_territori.csv`. For Colab's upload picker, extract the archive locally and select the first three CSVs when prompted. Raw inputs are release assets; they are not included in a Git clone or GitHub's source-code ZIP. Attribution, licence information and file checksums are in [data/README.md](data/README.md) and [data/SHA256SUMS](data/SHA256SUMS).

GitHub Pages serves the existing map and saved notebook HTML. It does not run or retrain Python models. Each push to `main` republishes those saved deliverables.

Start with **[pnrr_complete_workflow.ipynb](pnrr_complete_workflow.ipynb)**, the merged, executed notebook covering an executive summary, the proposal, justified model/data/evaluation choices, SHAP explanations, deployment conversion and demonstration, limitations/fairness/ethics/energy, and a final plain-language stakeholder briefing. The briefing calculates its results from the current run and explains outcomes, uncertainty, responsible uses, and who should take each next step. The opening executive summary describes the saved experiment dated 6 October 2026. The original two notebooks and `REPORT.md` are retained as earlier source material. Labels are inherited programme themes, not independently verified project-purpose annotations.

## Run the self-contained notebook

On **Google Colab**, upload `pnrr_complete_workflow.ipynb`, select a Python CPU runtime, and choose **Runtime → Run all**. The setup cell installs missing dependencies and asks for the supplied `progetti.csv`, `misure.csv`, and `temi.csv` if they are missing. Files uploaded through Colab's Files panel are also detected. The notebook contains all project code, but its three raw input CSVs are still required; uploading the notebook alone does not include the dataset. No repository clone, source scripts, saved models, or previous outputs are needed. Download the executed notebook and any generated outputs before deleting the runtime. The local isolated execution check does not certify a hosted Colab environment.

Copy **only `pnrr_complete_workflow.ipynb` and `data/progetti.csv`, `data/misure.csv`, `data/temi.csv`** to a new folder. Install the Python packages listed in notebook section 2, start its kernel in that folder, then select **Restart Kernel and Run All**. `DATA_DIR` is the only input path to configure. The notebook contains every project function as executable code and does not import `src/`, read other notebooks, invoke external scripts, or require saved models/results.

Every full run prepares the raw data and retrains all four neural configurations. Calculations and plots use the current in-memory objects. A unique `pnrr_outputs/run_.../` folder receives generated CSVs, checkpoints, figures and deployment graphs. Existing outputs are never a prerequisite. Allow about 15–20 minutes on CPU with the default budget; opening saved notebook outputs does not retrain anything.

Deployment is demonstrated directly in notebook section 8: both selected models are exported to `.pt2`, reloaded, checked for numerical parity, and used by the fully defined `classify_exported` function:

```python
classify_exported(['Riqualificazione energetica della scuola primaria comunale'], 'MLP')
```

Only graphs produced by the current run are reloaded. The fitted vectorizer, vocabulary and ordered labels are also saved for later deployment. Predictions remain uncalibrated suggestions for human review. Energy figures are explicit power-assumption scenarios, not measured electricity or emissions.

Sections **7.4–7.5** explain the validation-selected MLP checkpoint using SHAP word-deletion contributions for two held-out cases: agreement and disagreement with inherited labels. The notebook renders signed contribution plots, checks reconstruction of all class scores and agreement with evaluation predictions, and compares two permutation seeds. The empty-title reference, uncalibrated score units, case-selection rule, and masking limitations are explicit. A critical transparency discussion considers how explanations can help reviewers challenge suggestions while also creating false reassurance. SHAP does not validate a project's true purpose or establish fairness. Each run saves `shap_cases.csv`, `shap_word_contributions.csv`, `shap_diagnostics.csv`, `shap_metadata.json`, and `shap_project_*.png` alongside the existing outputs; SHAP 0.49.1 is included in the environment instructions and `requirements.txt`.

The optional authoring utility `src/build_merged_notebook.py` embeds source definitions **at build time**. It is not required to open, execute or share the resulting notebook. Maintainers can regenerate it with:

```sh
.venv/bin/python src/build_merged_notebook.py
```

Add `--execute` to run a full training experiment after regeneration. The original notebooks and the earlier modular scripts remain available separately.

## Explore project locations across Italy

Open **[the interactive regional choropleth](pnrr_outputs/geography/italy_projects.html)** in a browser. It works offline and lets you select each recorded programme theme, inspect regional counts, and compare the selected theme's share of each region's projects. The [report](REPORT.md#geographic-distribution-of-projects) includes an overall map and a set of maps by theme.

For a separate submission, use **`italy_projects.html` (underscore)**. It is a complete HTML document with embedded geography, data, JavaScript, and styles; no CSVs or companion assets are required for viewing in a modern browser with JavaScript enabled. `italy-projects.html` (hyphen) is a widget fragment for embedding. The map is an independent descriptive companion and is not regenerated by running the classification notebook. Whether a submission platform accepts or previews HTML must be checked in that platform.

The map covers all 298,338 raw project IDs: 297,500 have regional locations, 767 are nationwide only, and 71 have no location record. Each project counts once per recorded region; 806 projects span several regions. Nationwide projects are kept separate. Counts describe the local snapshot (date unknown), not live totals, funding, or population-adjusted access. Themes come from programme metadata, not the neural classifiers. Each theme uses its own labelled linear colour scale.

To rebuild the maps and update the report without retraining models:

```sh
.venv/bin/python -m pip install -r requirements-geography.txt
.venv/bin/python src/map_projects.py
```

This companion uses `progetti_territori.csv` in addition to the three classification CSVs, and the bundled official [ISTAT geographic reference](data/geography/README.md). Outputs, exact regional counts, and source/coverage checks are in `pnrr_outputs/geography/`. The classification notebook still needs only its original three input CSVs.

## Reproduce the earlier modular workflow

Python 3.13 was used. Dependencies and exact installed versions are in `requirements.txt`. On a fresh environment:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
mkdir -p artifacts
.venv/bin/python src/prepare.py > artifacts/preparation.log
.venv/bin/python src/train.py > artifacts/training.log 2>&1
.venv/bin/python src/build_deliverables.py
.venv/bin/python -m unittest discover -s tests -v
```

The current workspace's `.venv` reuses the available Anaconda Python packages. Training uses CPU, four threads, seed 42, deterministic PyTorch operations, at most 14 epochs, 24,000 training titles, and 8,000 validation titles. Every retained test title is evaluated. Timings and floating-point results may vary across environments.

The pipeline fits both vocabulary and TF–IDF on training data only. It keeps root measure families, shared CUPs, and duplicate normalized titles in a single partition. `--prepare` forces preparation during training; `--epochs`, `--train-limit`, and `--validation-limit` adjust the compute budget. Re-running replaces the generated experiment artifacts; preserve a copy before starting a new experiment.

## Use a trained model

```sh
.venv/bin/python src/predict.py 'Riqualificazione energetica della scuola primaria comunale' --model MLP
.venv/bin/python src/predict.py 'Realizzazione di un nuovo ospedale di comunità' --model CNN
```

Predictions are exploratory and probabilities are not calibrated. The model always chooses among the eight retained themes, including when a title belongs to an excluded or unknown category.

## Files

| Path | Purpose |
|---|---|
| `pnrr_complete_workflow.ipynb` | Self-contained notebook with all project code and saved execution outputs |
| `src/build_merged_notebook.py` | Rebuild and optionally execute the merged notebook |
| `src/map_projects.py`, `pnrr_outputs/geography/` | Rebuildable regional choropleths, theme filters, counts and geography audit |
| `src/export_models.py`, `src/deploy_predict.py` | Export graphs, check parity, and run portable local inference |
| `pnrr_outputs/` | Outputs generated by the self-contained notebook and isolated-execution verification |
| `artifacts/merged/` | Preserved outputs from the earlier modular version |
| `src/prepare.py` | Cleaning, hierarchy joins, grouping, split selection, source hashes |
| `src/train.py` | Model definitions, four neural runs, baselines, evaluation, plots |
| `src/predict.py` | Inference from validation-selected saved checkpoints |
| `src/build_deliverables.py` | Generate the report and execute the results notebook |
| `tests/test_pipeline.py` | Cleaning, padding invariance, split isolation, sampling checks |
| `artifacts/data_audit.json` | Counts and exclusions at each cleaning stage |
| `artifacts/prepared_projects.csv.gz` | All cleaned titles and partition assignments |
| `artifacts/experiment_split.csv.gz` | Exact sampled training/validation IDs and full test IDs |
| `artifacts/*_history.csv`, `training.log` | Per-epoch metrics and console training logs |
| `artifacts/results.json` | Hyperparameters, environment, selections, metrics, bootstrap intervals |
| `artifacts/*.pt`, `tfidf.pkl`, `vocabulary.json`, `classes.json` | Reloadable neural models and fitted preprocessing |
| `artifacts/classification_reports.csv`, `test_predictions.csv.gz` | Per-class metrics and per-title predictions |
| `artifacts/label_review_sample.csv` | Example inherited labels with empty fields for future human review |
| `artifacts/*.png` | Learning curves, model comparison, confusion matrices |

Only `progetti.csv`, `misure.csv`, and `temi.csv` are needed for this title-classification task. Payment and territory files are not model inputs. No financing, payment, or regional-policy conclusions are made. The local CSV snapshot date is not inferred from file modification times.
