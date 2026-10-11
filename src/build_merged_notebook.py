"""Author a self-contained notebook; source files are used only at build time.

The generated notebook contains ordinary executable definitions, has no project
imports, and reads only input CSVs and deployment files it creates itself.
"""
import argparse
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import textwrap

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'pnrr_complete_workflow.ipynb'

EXECUTIVE_SUMMARY = '''## Executive summary

**Purpose and scope.** This project asks whether short Italian PNRR project titles can help analysts suggest one of eight recorded programme themes. It provides a complete, reproducible research workflow with two neural models, simpler reference models, performance plots, explanations and an export demonstration. The labels come from programme metadata; independent review is needed to establish each project's actual purpose.

**What the saved experiment found.** Starting with 298,338 project records, cleaning produced 166,521 distinct eligible titles. Training used 24,000 titles, validation used 8,000, and evaluation retained all 21,219 test titles from 24 independent programme groups. Keeping related projects together makes this a demanding test of unfamiliar programmes, while removing repeated titles prevents administrative templates from dominating the score.

| Approach | Match with recorded theme | Score giving every theme equal weight (macro-F1) |
|---|---:|---:|
| Simple logistic classifier | 42.1% | 0.381 |
| Neural model using word features (MLP) | 44.4% | 0.389 |
| Neural model using local word patterns (CNN) | 41.8% | 0.288 |

Accuracy is the share of titles matching their recorded theme. Macro-F1 balances missed and incorrect suggestions within each theme and then averages across themes; it is not a percentage of correct answers. The MLP was selected using validation results. Its advantage over logistic regression was only 0.008 in macro-F1, with an exploratory uncertainty interval from −0.032 to +0.042. This does not establish dependable neural improvement. The proposed 0.80 score and 0.03 improvement goals were missed; the original hypothesis involving independently reviewed project purposes remains untested.

**What explains the decision.** Training performance improved much more than performance on unfamiliar titles, indicating overfitting. Category results were uneven: education/research was strongest and digitalisation weakest. Test titles contained more words outside the CNN's training vocabulary than training titles did. Its selected training run took about 5.6 minutes, compared with about 4.3 seconds for the selected MLP; vocabulary shift is a plausible contributor to its lower result, not a proven cause. These observations favor better labels, broader independent evaluation and review of weak categories before a larger model or operational use.

**Practical outcome.** The project is suitable for teaching and supervised exploratory scrutiny. It does not yet justify automated public decisions. Explanations help a reviewer examine a suggestion; successful export checks establish that saved models reproduce predictions. Neither establishes correctness or fairness. A separate choropleth describes recorded project locations and programme themes, and can be shared as `italy_projects.html`; geographic counts do not establish funding fairness or policy impact.

*This overview describes the saved full run completed on 6 October 2026, not the date of the underlying data snapshot. Section 12 recalculates its numerical conclusions from the current run when the notebook is rerun. Detailed methods, plots, caveats and responsibilities follow below.*'''


def definition(path, name):
    source = (ROOT / path).read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name)
    start = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
    return '\n'.join(source.splitlines()[start - 1:node.end_lineno])


def build(execute=False):
    original = nbf.read(ROOT / '01_pnrr_research_proposal.ipynb', as_version=4)
    # Preserve the existing substantive reflection and dynamic interpretation.
    previous = nbf.read(TARGET, as_version=4)
    reflection_index = next(i for i,c in enumerate(previous.cells) if c.cell_type == 'markdown' and c.source.startswith('## 9. Limitations'))
    reflection_end = next((i for i,c in enumerate(previous.cells) if i > reflection_index and c.cell_type == 'markdown' and c.source.startswith('## 11.')), len(previous.cells))
    reflection = copy.deepcopy(previous.cells[reflection_index:reflection_end])
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    cells = []
    def add(title, prose, source=None):
        cells.append(md(title + '\n\n' + prose))
        if source is not None:
            cells.append(code(textwrap.dedent(source).strip()))

    add('# PNRR project-title classification: self-contained workflow', '''This single notebook contains all project code: preparation, MLP/CNN definitions, feature fitting, training, evaluation, SHAP explanations, plots, deployment export and inference. Copy it together with only `data/progetti.csv`, `data/misure.csv`, and `data/temi.csv`. No Python scripts, previous artifacts, other notebooks, or requirements file are needed at runtime.

**Google Colab:** upload this notebook, select a Python CPU runtime, and choose **Runtime → Run all**. Section 2 installs missing packages and asks you to upload `progetti.csv`, `misure.csv`, and `temi.csv` if they are absent. These three raw data files are required; uploading the notebook alone does not include the dataset. No repository or trained artifacts are needed. Use the supplied snapshot to reproduce this experiment.

**Local Jupyter:** use a Python environment with the packages listed in section 2. Start its kernel in the notebook directory and choose **Restart Kernel and Run All**. Each full run prepares and trains from scratch, then creates a new `pnrr_outputs/run_.../` folder. This folder is output, never a prerequisite. All calculations and plots use objects built in memory during the current run; the deployment demonstration reloads only graphs written by that same run. Allow about 15–20 minutes on CPU. In Colab, save the executed notebook and download any outputs you need before deleting the runtime.

**Separate geographic companion:** `italy_projects.html` can be submitted and opened as a standalone offline HTML file. It uses recorded programme themes, not model predictions. It is not generated by this classification notebook and needs no accompanying CSVs when viewed.

**Scope:** exploratory title classification against inherited programme themes. Funding/payment/territorial analysis and independent human annotation in the original proposal remain future work.

| Requirement | Evidence |
|---|---|
| Clear architecture | Executable MLP/CNN class definitions, layer sizes and shape checks |
| Data pipeline | Executable cleaning, hierarchy joins, group splits, training-only TF–IDF/vocabulary, batched loading |
| Deployment | Graph export to `.pt2`, numerical round-trip checks, prediction function and live examples |
| Explainability | Word-level SHAP contributions for held-out MLP examples, reconstruction and seed-sensitivity checks |
| Responsible design | Weak-label limits, fairness, human review, privacy and explicit energy assumptions |''')
    for i, source in enumerate(original.cells):
        cell = copy.deepcopy(source)
        if i == 0:
            cell.source = '## 1. Original research proposal and hypotheses\n\nThe following proposal records the original motivation. The completed scope is exploratory title classification; funding and territorial analysis and independent human annotation were not completed.'
        if i == 5:
            cell.source = cell.source.replace('$Delta = F1_{mathrm{macro,MLP}} - F1_{mathrm{macro,logistic}}$', r'$\Delta = F1_{\mathrm{macro,MLP}} - F1_{\mathrm{macro,logistic}}$').replace('$$H_0: Delta leq 0 qquad H_1: Delta > 0$$', r'$$H_0: \Delta \leq 0 \qquad H_1: \Delta > 0$$')
        if i == 7:
            cell.source = cell.source.replace('## 1.7. Next step', '## 1.7. Original next step and source references')
        cells.append(cell)

    add('## 2. Environment and parameters', '''The saved experiment used Python 3.13 and the exact versions below. On local Jupyter, install them in your environment if needed:

`python -m pip install torch==2.12.0 numpy==2.1.3 pandas==2.2.3 scikit-learn==1.6.1 matplotlib==3.10.0 ipykernel==6.29.5 shap==0.49.1`

On Colab, the next cell keeps its preinstalled scientific packages and installs missing dependencies, including SHAP. Export requires PyTorch 2.6 or newer. This avoids replacing Colab's entire environment with the local Python 3.13 pins; library versions and numerical results may differ and are recorded in each run. A hosted Colab run has not been certified by the local execution check.

`DATA_DIR` is the only input location. Adjust it if the three CSVs are elsewhere. Colab accepts files uploaded to its runtime root or prompts for missing CSVs. The default budget matches the complete experiment: 24,000 training titles, 8,000 validation titles, up to 14 epochs, and the entire held-out test set. A new output directory prevents accidental mixing with a previous run. See the official [Colab file-loading example](https://colab.research.google.com/notebooks/io.ipynb) and [runtime/file sharing guidance](https://research.google.com/colaboratory/faq.html).''', '''import importlib.util
from importlib.metadata import PackageNotFoundError, version
import shlex

try:
    IS_COLAB = importlib.util.find_spec('google.colab') is not None
except (ModuleNotFoundError, ValueError):
    IS_COLAB = False

package_requirements = {
    'torch': 'torch>=2.6,<3', 'numpy': 'numpy', 'pandas': 'pandas',
    'scikit-learn': 'scikit-learn', 'matplotlib': 'matplotlib',
    'shap': 'shap==0.49.1',
}
missing_packages = []
for package, requirement in package_requirements.items():
    try:
        installed_version = version(package)
    except PackageNotFoundError:
        missing_packages.append(requirement)
        continue
    if package == 'torch' and tuple(map(int, installed_version.split('.')[:2])) < (2, 6):
        missing_packages.append(requirement)
if missing_packages:
    if not IS_COLAB:
        raise ImportError('Install required packages first: python -m pip install ' + ' '.join(missing_packages))
    get_ipython().run_line_magic('pip', 'install -q ' + ' '.join(shlex.quote(p) for p in missing_packages))
    importlib.invalidate_caches()
print('Runtime:', 'Google Colab' if IS_COLAB else 'local Jupyter', '| dependency check complete')''')
    cells.append(code('''from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import copy, hashlib, io, json, os, pickle, platform, random, re, shutil, time, unicodedata

DATA_DIR = Path('data').resolve()
required_csvs = ['progetti.csv', 'misure.csv', 'temi.csv']
if IS_COLAB:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Colab's Files panel uploads to the runtime root by default.
    for name in required_csvs:
        root_file = Path.cwd() / name
        if not (DATA_DIR / name).is_file() and root_file.is_file():
            shutil.copy2(root_file, DATA_DIR / name)
    missing_csvs = [name for name in required_csvs if not (DATA_DIR / name).is_file()]
    if missing_csvs:
        from google.colab import files
        print('Upload the supplied CSV snapshot. Missing:', ', '.join(missing_csvs))
        uploaded_files = files.upload()
        for name in missing_csvs:
            if name in uploaded_files:
                (DATA_DIR / name).write_bytes(uploaded_files[name])
        del uploaded_files
for name in required_csvs:
    if not (DATA_DIR / name).is_file():
        raise FileNotFoundError(f'Missing input CSV: {DATA_DIR / name}. Upload all three CSVs or set DATA_DIR to their folder.')
OUT = Path.cwd() / 'pnrr_outputs' / datetime.now(timezone.utc).strftime('run_%Y%m%dT%H%M%S_%fZ')
OUT.mkdir(parents=True, exist_ok=False)
os.environ.setdefault('MPLCONFIGDIR', str(OUT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from IPython.display import display, Image, Markdown

SEED = 42
EPOCHS, TRAIN_LIMIT, VALIDATION_LIMIT = 14, 24000, 8000
assert EPOCHS >= 1
pd.set_option('display.max_colwidth', 100)
print('Input CSV directory:', DATA_DIR)
print('New output directory:', OUT)
'''))
    add('### 2.1 Reproducible CPU settings', 'Fixed seeds and deterministic PyTorch operations support reproduction within the recorded environment. Four CPU threads bound parallel work.', definition('src/train.py', 'seed_all') + '\n\nseed_all()')
    add('## 3. Load and preprocess the data', '''The functions below are defined here and executed here. Cleaning keeps Italian letters, strips numeric identifiers and normalizes whitespace. Ancestor traversal detects hierarchy cycles.''', '\n\n'.join(definition('src/prepare.py', n) for n in ['clean_title','ancestor']))
    prepare = definition('src/prepare.py', 'prepare')
    prepare = prepare.replace("def prepare(out=ROOT / 'artifacts', seed=42):", 'def prepare(data_dir, out, seed=42):')
    prepare = prepare.replace("ROOT / 'data' / f'{name}.csv'", "data_dir / f'{name}.csv'")
    for name in ['progetti','misure','temi']:
        prepare = prepare.replace(f"ROOT / 'data/{name}.csv'", f"data_dir / '{name}.csv'")
    prepare = prepare.replace('    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)', "    print(f'Prepared {len(p):,} unique titles across {len(labels)} inherited themes.', flush=True)")
    add('### 3.1 Validate joins and construct isolated groups', '''Read only required project columns and validate many-to-one joins. Exclude unmatched, short, conflicting and unsupported examples; collapse duplicate normalized titles. Connected root measures, CUPs and duplicate titles remain in one split. Split search uses labels and group sizes, never model performance. This is a weak-label classification dataset, not a human-labelled gold standard.''', prepare)
    add('### 3.2 Execute preparation from the raw CSVs', 'This cell always runs preparation; no prepared dataset or audit is loaded from an earlier run.', '''prepared, audit = prepare(DATA_DIR, OUT, seed=SEED)
df = prepared.copy()
display(pd.Series({k: audit[k] for k in ['raw_projects', 'unmatched_measure_rows', 'titles_under_three_tokens', 'conflicting_title_rows_removed', 'duplicate_clean_titles_removed', 'retained_unique_titles']}))
display(pd.DataFrame(audit['class_support_before_filter']))
print(clean_title('M1C1I1.04.05 SERVIZI DIGITALI PER LA COMUNITÀ E71F22003570006'))''')
    add('## 4. Training-only features and batched loading', 'Sample within the existing training and validation partitions, retaining every test title. Sparse TF–IDF becomes dense only for the current batch; the CNN uses padded/truncated token ID arrays.', '\n\n'.join(definition('src/train.py', n) for n in ['sample_partition','encode','batch_tensor','len_or_shape']))
    main = definition('src/train.py', 'main')
    def body(start, end):
        part = textwrap.dedent(main[main.index('    ' + start):main.index('    ' + end)])
        return part.replace('args.train_limit','TRAIN_LIMIT').replace('args.validation_limit','VALIDATION_LIMIT').replace('args.epochs','EPOCHS')
    partitions = body('classes =', 'tfidf =')
    partitions = partitions.replace("pd.concat(partitions.values()).drop(columns=['titolo', 'text']).to_csv(OUT / 'experiment_split.csv.gz', index=False)", "sampled = pd.concat(partitions.values()).drop(columns=['titolo', 'text'])\nsampled.to_csv(OUT / 'experiment_split.csv.gz', index=False)")
    add('### 4.1 Sample the partitions', 'Class-stratified sampling does not move a project to another split.', partitions)
    add('### 4.2 Fit TF–IDF and vocabulary on training titles', 'Validation and test titles are transformed without refitting. The saved vectorizer and vocabulary are outputs for later deployment; the notebook continues using the in-memory objects.', body('tfidf =', 'diagnostics ='))
    add('### 4.3 Record coverage and class weights', 'Unknown-token rates, class counts and independent-group counts expose distribution shift. Square-root inverse-frequency class weights reduce dominance by common categories.', body('diagnostics =', '# Baseline tuning') + "\ndisplay(pd.DataFrame({s: d['class_counts'] for s, d in diagnostics.items()}))\ndisplay(pd.Series(results['environment'], name='Training environment'))")
    add('## 5. Explicit neural network architectures', '''`B` denotes batch size; `F` is the fitted TF–IDF feature count (at most 6,000), `V` the vocabulary (at most 12,000), and `C` the retained theme count (8 for these data).

**MLP:** float32 `[B,F]` → Linear(F,128) → ReLU → Dropout → Linear(128,64) → ReLU → Dropout → Linear(64,C) logits. The class below is the actual executable definition.''', definition('src/train.py','TitleMLP'))
    add('### 5.1 Text CNN', '''int64 `[B,80]` → Embedding(V,64) → parallel Conv1d(64,64,k=3/5) with same padding → ReLU → masked global max-pooling → concatenate `[B,128]` → Dropout → Linear(128,64) → ReLU → Dropout → Linear(64,C) logits. Masking prevents padded positions from winning the pool. Cross-entropy uses logits; softmax is applied during inference.''', definition('src/train.py','TitleCNN'))
    add('### 5.2 Verify layer shapes and padding behaviour', 'Instantiate both models, display their real structures, and check forward-pass dimensions before training.', '''architecture_examples = {'MLP': TitleMLP(len(tfidf.vocabulary_), len(classes)),
                         'CNN': TitleCNN(len(vocab), len(classes))}
for kind, model in architecture_examples.items():
    model.eval()
    inputs = batch_tensor(x_mlp['train'] if kind == 'MLP' else x_cnn['train'], np.arange(2))
    with torch.inference_mode():
        assert model(inputs).shape == (2, len(classes))
    print(kind, model, 'Parameters:', sum(p.numel() for p in model.parameters()))
with torch.inference_mode():
    model = architecture_examples['CNN']
    torch.testing.assert_close(model(torch.tensor([[2, 3, 4, 0, 0]])),
                               model(torch.tensor([[2, 3, 4, 0, 0, 0, 0, 0]])), rtol=1e-6, atol=1e-6)
assert clean_title('M1C1I1.04.05 CITTADINANZA DIGITALE – comunità E71F22003570006 2025') == 'cittadinanza digitale comunità'
print('Model shapes, padding invariance and identifier cleaning passed.')''')
    add('## 6. Training, validation and model selection', 'Batched inference and metrics are defined below. Macro-F1 gives every retained class equal weight; accuracy weights titles.', '\n\n'.join(definition('src/train.py', n) for n in ['predict','scores']))
    fit = definition('src/train.py','fit_neural').replace('    return model, result', '    return model, result, pd.DataFrame(history)')
    add('### 6.1 Define the training loop', 'AdamW, weighted cross-entropy, gradient clipping, learning-rate scheduling and early stopping use validation macro-F1. The best checkpoint is restored. Histories stay in memory and are also written as output CSVs.', fit)
    add('### 6.2 Tune the logistic regression baseline', 'Choose C using validation data only. The majority baseline is defined from training class counts.', body('# Baseline tuning', 'chosen_models =') + "\ndisplay(pd.DataFrame(results['baseline_tuning']))")
    neural = body('chosen_models =', '# Freeze all selections')
    neural = neural.replace('chosen_models = {}','chosen_models, histories = {}, {}').replace('model, result = fit_neural(', 'model, result, history = fit_neural(').replace("        results['runs'][name] = result", "        results['runs'][name] = result\n        histories[name] = history")
    add('### 6.3 Train the four neural configurations', '''Compare plain models (dropout 0, weight decay 0) with regularized models (dropout 0.4, weight decay 0.0001). This tests their combined effect. Training progress is printed each epoch; all histories remain available in memory. Model choices are made before viewing test predictions.''', neural)
    add('### 6.4 Paired group bootstrap', 'Resample whole test groups jointly for both classifiers. The few independent groups make these intervals exploratory; an absent class receives F1=0 in a replicate.', definition('src/train.py','group_bootstrap'))
    evaluation = body('# Freeze all selections','plots(results,')
    evaluation = evaluation.replace("pd.concat(reports).to_csv(OUT / 'classification_reports.csv', index=False)", "report = pd.concat(reports)\nreport.to_csv(OUT / 'classification_reports.csv', index=False)")
    evaluation += "\nr = results\nbest_neural = max(r['selected'], key=lambda k: r['runs'][r['selected'][k]]['validation']['macro_f1'])\ndisplay(pd.DataFrame(r['test']).T.round(4))\nprint('Validation-selected architecture:', best_neural)"
    add('## 7. Held-out evaluation and verification', 'Freeze selected checkpoints first. Evaluate the complete test partition once, retaining per-title predictions and per-class metrics as output files. All objects used below were created in this kernel.', evaluation)
    add('### 7.1 Check split boundaries and recompute reported metrics', 'These executable assertions check the current data, not a saved claim that checks passed.', '''for key in ['progetto_id', 'cup', 'misura_id', 'family', 'group', 'text']:
    assert prepared.groupby(key).split.nunique().max() <= 1, key
assert prepared.text.nunique() == len(prepared)
assert prepared.groupby(['label', 'split']).group.nunique().min() >= 2
joined = sampled.merge(prepared[['progetto_id', 'split']], on='progetto_id', validate='one_to_one')
assert (joined.split_x == joined.split_y).all()
assert set(sampled.loc[sampled.split == 'test', 'progetto_id']) == set(prepared.loc[prepared.split == 'test', 'progetto_id'])
assert set(predictions.progetto_id) == set(prepared.loc[prepared.split == 'test', 'progetto_id'])
assert not predictions.progetto_id.duplicated().any()
for kind in r['test']:
    actual = f1_score(predictions.label, predictions[kind + '_prediction'], labels=classes, average='macro', zero_division=0)
    assert np.isclose(actual, r['test'][kind]['macro_f1'])
    assert np.isclose(accuracy_score(predictions.label, predictions[kind + '_prediction']), r['test'][kind]['accuracy'])
print('Passed: identity/group isolation, complete test membership and recomputed metrics.')''')
    plot_source = definition('src/train.py','plots').replace('def plots(results, classes, ytest, probabilities):', 'def plots(results, classes, ytest, probabilities, histories):')
    plot_source = plot_source.replace("h = pd.read_csv(OUT / f'{kind}_{variant}_history.csv')", "h = histories[f'{kind}_{variant}']")
    for name in ['training_curves','model_comparison','confusion_matrices']:
        plot_source = plot_source.replace(f"fig.savefig(OUT / '{name}.png'); plt.close(fig)", f"fig.savefig(OUT / '{name}.png')\n    buffer = io.BytesIO()\n    fig.savefig(buffer, format='png')\n    display(Image(data=buffer.getvalue()))\n    plt.close(fig)")
    add('### 7.2 Plot the current run', '''Editable plotting code uses the in-memory histories and predictions. Train metrics use the fixed monitoring subset with dropout off; plotted losses are unweighted evaluation losses. Optimization loss in the histories is weighted training loss.''', plot_source + "\n\nplots(r, classes, y['test'], probabilities, histories)")
    add('### 7.3 Inspect class-level performance and confident errors', 'Aggregate accuracy conceals large category differences. These are errors against inherited labels, not verified human annotations.', '''runs = pd.DataFrame([{'run': name, 'validation_macro_f1': v['validation']['macro_f1'],
                      'best_epoch': v['best_epoch'], 'epochs_run': v['epochs_run'],
                      'seconds': v['seconds'], 'parameters': v['parameters']}
                     for name, v in r['runs'].items()])
display(runs.round(4))
display(report[report.category.isin(classes)].pivot(index='category', columns='model', values='f1-score').round(3))
for kind in ['MLP', 'CNN']:
    print(kind, '— three confident errors against inherited labels')
    errors = predictions[predictions[kind + '_prediction'] != predictions.label]
    display(errors.sort_values(kind + '_confidence', ascending=False)[['titolo', 'label', kind + '_prediction', kind + '_confidence']].head(3))
display(r['bootstrap_mlp_minus_logistic'])
display(r['bootstrap_cnn_minus_mlp'])''')
    add('### 7.4 Explain individual MLP predictions with SHAP', r'''**Question:** which words contribute to the MLP's score for a particular theme? We explain the MLP checkpoint selected by validation macro-F1 in section 6; this does not explain the CNN or change either model. SHAP (SHapley Additive exPlanations) allocates a prediction's difference from a reference among input features, averaging their marginal contributions across feature coalitions.

Here a feature is a **word occurrence in the cleaned title**, identified by its position. A binary mask includes or deletes each occurrence; retained words keep their order. Each masked title passes through the already fitted TF–IDF vectorizer and MLP, so explanations include preprocessing effects, including changed bigrams and TF–IDF normalization. We use the model-agnostic [PermutationExplainer](https://shap.readthedocs.io/en/latest/generated/shap.PermutationExplainer.html) with an [Independent masker](https://shap.readthedocs.io/en/latest/generated/shap.maskers.Independent.html) whose single background row is all zeros. This defines a word-deletion game with an **empty-title reference**, not an average training title or a population prior.

For theme $c$ and cleaned title $x$, the identity link gives

$$p_c(x) = p_c(\text{empty title}) + \sum_{j=1}^{m}\phi_{j,c}.$$

Positive contributions increase that theme's softmax score relative to the empty title; negative contributions decrease it. Values multiplied by 100 are **percentage points**, not relative percent changes, causal effects, or calibrated probabilities of correctness. The code checks this reconstruction for all eight outputs. It uses 20 forward/reverse permutation cycles per example and repeats with another seed to assess approximation sensitivity. Additivity alone does not establish that the individual estimates have converged.

For a bounded, reproducible demonstration, take the lowest project ID among test titles of 3–30 cleaned words in each of two strata: agreement and disagreement with the inherited theme. This deliberately selected pair is a local diagnostic, not a representative sample or a global feature-importance estimate. Labels are used only to select the diagnostic cases after training and model selection; no parameters are changed.''', '''SHAP_CYCLES, SHAP_MAX_WORDS = 20, 30
shap_model_kind = 'MLP'
shap_selected_run = r['selected'][shap_model_kind]

def shap_text_scores(texts):
    # Empty/short strings are permitted only for the explanatory deletion game.
    # The deployment function still rejects invalid user inputs.
    cleaned = [clean_title(str(text)) for text in texts]
    features = tfidf.transform(cleaned)
    return predict(chosen_models[shap_model_kind], features, batch_size=128)

shap_candidates = predictions.merge(partitions['test'][['progetto_id', 'text']],
                                    on='progetto_id', validate='one_to_one')
shap_candidates = shap_candidates[shap_candidates.text.str.split().str.len().between(3, SHAP_MAX_WORDS)]
shap_candidates = shap_candidates.sort_values('progetto_id', kind='stable')
shap_agreement = shap_candidates[shap_model_kind + '_prediction'] == shap_candidates.label
shap_cases = pd.concat([
    shap_candidates.loc[shap_agreement].head(1).assign(case='Label agreement'),
    shap_candidates.loc[~shap_agreement].head(1).assign(case='Label disagreement')
]).reset_index(drop=True)
assert len(shap_cases), 'No eligible held-out title for the SHAP demonstration'
print('Explained checkpoint:', shap_selected_run, '| SHAP version:', shap.__version__)
display(shap_cases[['case', 'progetto_id', 'text', 'label', 'MLP_prediction', 'MLP_confidence']])
if len(shap_cases) < 2:
    print('Only one diagnostic stratum is available in this run.')''')
    add('#### 7.4.1 Compute contributions and verify the explanation', '''The first seed supplies the displayed contributions. A second seed provides a sensitivity check, not a confidence interval. Prediction scores must agree with the held-out evaluation before interpreting either explanation. All per-word/per-class values, case IDs, settings, and diagnostics are saved in this run's output folder.''', '''def explain_word_deletions(text, seed):
    words = text.split()
    feature_names = [f'{position + 1}: {word}' for position, word in enumerate(words)]
    def score_masks(masks):
        masked_texts = [' '.join(word for word, present in zip(words, mask) if present)
                        for mask in masks]
        return shap_text_scores(masked_texts)
    masker = shap.maskers.Independent(np.zeros((1, len(words)), dtype=np.float32))
    explainer = shap.PermutationExplainer(score_masks, masker, feature_names=feature_names,
                                        seed=seed, output_names=classes)
    return explainer(np.ones((1, len(words)), dtype=np.float32),
                     max_evals=SHAP_CYCLES * (2 * len(words) + 1),
                     batch_size=128, silent=True)[0]

shap_started = time.perf_counter()
shap_explanations, shap_diagnostics, shap_rows = [], [], []
for case in shap_cases.itertuples(index=False):
    explanation = explain_word_deletions(case.text, SEED)
    repeated = explain_word_deletions(case.text, SEED + 1)
    full_scores = shap_text_scores([case.text])[0]
    empty_scores = shap_text_scores([''])[0]
    assert explanation.values.shape == (len(case.text.split()), len(classes))
    assert np.isfinite(explanation.values).all()
    for estimate in [explanation, repeated]:
        np.testing.assert_allclose(estimate.base_values, empty_scores, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(estimate.base_values + estimate.values.sum(axis=0),
                                   full_scores, atol=1e-6, rtol=1e-5)
    predicted_class = int(full_scores.argmax())
    assert classes[predicted_class] == case.MLP_prediction
    np.testing.assert_allclose(full_scores[predicted_class], case.MLP_confidence,
                               atol=1e-6, rtol=1e-5)
    values = explanation.values[:, predicted_class]
    repeat_values = repeated.values[:, predicted_class]
    top_k = min(3, len(values))
    top = set(np.argsort(np.abs(values))[-top_k:])
    repeat_top = set(np.argsort(np.abs(repeat_values))[-top_k:])
    shap_diagnostics.append({
        'case': case.case, 'progetto_id': int(case.progetto_id),
        'predicted_theme': classes[predicted_class], 'inherited_theme': case.label,
        'base_score': float(explanation.base_values[predicted_class]),
        'sum_contributions': float(values.sum()),
        'predicted_score': float(full_scores[predicted_class]),
        'max_reconstruction_error_all_classes': float(np.max(np.abs(
            explanation.base_values + explanation.values.sum(axis=0) - full_scores))),
        'max_seed_change_pp_predicted_class': float(100 * np.max(np.abs(values - repeat_values))),
        'top_3_word_positions_shared_across_seeds': len(top & repeat_top)})
    shap_explanations.append(explanation)
    for position, word in enumerate(case.text.split()):
        for class_index, theme in enumerate(classes):
            shap_rows.append({'progetto_id': int(case.progetto_id), 'case': case.case,
                              'word_position': position + 1, 'word': word, 'theme': theme,
                              'shap_value': float(explanation.values[position, class_index]),
                              'repeat_seed_value': float(repeated.values[position, class_index]),
                              'base_score': float(explanation.base_values[class_index]),
                              'predicted_score': float(full_scores[class_index])})
shap_diagnostics = pd.DataFrame(shap_diagnostics)
shap_metadata = {'shap_version': shap.__version__, 'model': shap_model_kind,
                 'selected_run': shap_selected_run, 'seeds': [SEED, SEED + 1],
                 'permutation_cycles_per_seed': SHAP_CYCLES, 'max_words': SHAP_MAX_WORDS,
                 'selection': 'Lowest project ID in each held-out label-agreement stratum',
                 'features': 'Cleaned word occurrences, deleted when masked',
                 'reference': 'Empty title (all word-inclusion features zero)',
                 'output': 'Uncalibrated softmax score, identity link',
                 'seconds': time.perf_counter() - shap_started,
                 'all_class_additivity_passed': True, 'prediction_parity_passed': True}
shap_cases.to_csv(OUT / 'shap_cases.csv', index=False)
pd.DataFrame(shap_rows).to_csv(OUT / 'shap_word_contributions.csv', index=False)
shap_diagnostics.to_csv(OUT / 'shap_diagnostics.csv', index=False)
(OUT / 'shap_metadata.json').write_text(json.dumps(shap_metadata, indent=2, ensure_ascii=False))
display(shap_diagnostics.round(6))
print('Passed: held-out prediction parity and all-class additivity for both seeds.')''')
    add('#### 7.4.2 Read the local explanations', '''Each chart explains the **predicted theme**, including for the disagreement case. Blue positive bars raise its score; orange negative bars lower it. Word positions distinguish repeated words. The 12 largest absolute contributions are shown individually; any remaining contributions are summed into one labelled bar so nothing is silently discarded. Both charts use a common horizontal scale. The full title and all contributions remain available above and in the exported CSVs.''', '''shap_plot_data = []
for case, explanation in zip(shap_cases.itertuples(index=False), shap_explanations):
    class_index = classes.index(case.MLP_prediction)
    values_pp = 100 * explanation.values[:, class_index]
    ranked = np.argsort(-np.abs(values_pp), kind='stable')
    shown = ranked[:12]
    labels = [f'{position + 1}: {case.text.split()[position]}' for position in shown]
    plotted = list(values_pp[shown])
    if len(ranked) > 12:
        labels.append(f'Other {len(ranked) - 12} word occurrences (sum)')
        plotted.append(float(values_pp[ranked[12:]].sum()))
    shap_plot_data.append((case, explanation, class_index, values_pp, labels, np.array(plotted)))
shap_axis_limit = max(1.0, max(np.max(np.abs(item[-1])) for item in shap_plot_data) * 1.25)

for case, explanation, class_index, values_pp, labels, plotted in shap_plot_data:
    fig, ax = plt.subplots(figsize=(10, max(4.0, len(labels) * .36 + 1.8)), layout='constrained')
    ax.barh(np.arange(len(plotted)), plotted,
            color=['#2563a6' if value >= 0 else '#c46628' for value in plotted])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color='#555555', linewidth=.8)
    ax.set_xlim(-shap_axis_limit, shap_axis_limit)
    ax.set_xlabel('SHAP contribution to predicted theme score (percentage points)')
    ax.set_title(f'{case.case} | MLP | project {case.progetto_id}\\n'
                 f'Predicted: {case.MLP_prediction} | Inherited: {case.label}', fontsize=11, pad=12)
    ax.spines[['top', 'right']].set_visible(False)
    for row_index, value in enumerate(plotted):
        ax.text(value + (shap_axis_limit * .02 if value >= 0 else -shap_axis_limit * .02),
                row_index, f'{value:+.2f}', va='center', ha='left' if value >= 0 else 'right', fontsize=9)
    fig.savefig(OUT / f'shap_project_{case.progetto_id}.png', dpi=160)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=160)
    display(Image(data=buffer.getvalue()))
    plt.close(fig)
    diagnostic = shap_diagnostics.loc[shap_diagnostics.progetto_id == case.progetto_id].iloc[0]
    strongest = int(np.argmax(np.abs(values_pp)))
    direction = 'raises' if values_pp[strongest] >= 0 else 'lowers'
    display(Markdown(
        f"**Reading this case.** Word {strongest + 1}, **{case.text.split()[strongest]}**, "
        f"has the largest absolute contribution and {direction} the score for "
        f"**{case.MLP_prediction}** by **{abs(values_pp[strongest]):.2f} percentage points** "
        "in this deletion game. "
        f"The empty-title score is {100 * diagnostic.base_score:.2f}%; adding "
        f"{100 * diagnostic.sum_contributions:+.2f} points gives "
        f"{100 * diagnostic.predicted_score:.2f}%. "
        f"Changing the permutation seed changes an individual contribution by at most "
        f"{diagnostic.max_seed_change_pp_predicted_class:.2f} points for this theme; "
        f"{int(diagnostic.top_3_word_positions_shared_across_seeds)}/3 leading word positions overlap. "
        "This checks sampling sensitivity for these settings only, not robustness to another reference or dataset."))''')
    add('### 7.5 Real-world implication: transparency without false reassurance', '''Imagine this classifier helping an analyst route PNRR projects to thematic reviewers or build a public dashboard. SHAP can make a suggested category contestable: the analyst can inspect which title words support it, compare the recorded programme theme, and flag reliance on an institution name or administrative phrase. The disagreement example above shows why explanations should accompany difficult cases as well as successful ones. A disagreement is not automatically a model error: the inherited label may describe the programme while the title describes a different activity.

**The central risk is false reassurance.** A fluent-looking explanation can make an uncertain classification seem justified. This model learns agreement with weak programme labels; SHAP explains that learned scoring behaviour, not a project's actual purpose, eligibility, public value, or entitlement to funding. A large positive word contribution therefore cannot justify overwriting the source category. Conversely, a small attribution does not prove a word is irrelevant in every context, or that geographic and organizational proxies are absent. Two examples cannot establish population-wide transparency or fairness.

**The explanation itself embeds choices.** Deleting words can create unnatural titles and new adjacent bigrams, and TF–IDF normalization changes when the text changes. The empty title is outside the training population. These attributions answer a specific word-deletion question; a different reference, token grouping, output class, or score scale can allocate importance differently. The seed check measures only finite-permutation variation, while the reconstruction check establishes arithmetic consistency. Neither validates the masking assumption or turns the contributions into causal effects. Softmax also couples the classes, so a contribution to one class's score can reflect changes in competing class scores. This section explains the MLP only.

**Practical consequence.** A review interface should show the original and cleaned title, inherited theme, suggested theme, uncalibrated score, model version, and explanation settings together, and allow the reviewer to retain or challenge the suggestion. Before operational use, test whether explanations help reviewers detect mistakes across themes, title styles and organizations, using independently reviewed labels; record corrections and check whether confident-looking plots instead increase acceptance of errors. Use wider stratified explanation audits and alternative masking choices to investigate shortcuts. The defensible role of SHAP here is to support scrutiny and correction, with a human retaining responsibility for decisions—not to certify correctness or fairness.''')
    add('## 8. Export and reload deployment graphs', '''The function below converts the models trained in this notebook to PyTorch `.pt2` graphs. It reloads only files it has just written, then checks logits and predicted classes at batch sizes 1, 2, the parity sample size, and 512. Preprocessing, labels and metadata are saved for future use, but this notebook never depends on a previous export.

The [PyTorch export documentation](https://docs.pytorch.org/docs/stable/export.html) describes this format. This is a local CPU demonstration requiring Python/PyTorch and fitted text preprocessing; it is not a hosted service or a hardware-compiled executable.''', '''def export_current_models(models, vectorizer, vocabulary, prepared_frame, results, destination):
    destination.mkdir(parents=True, exist_ok=True)
    texts = prepared_frame.groupby(['split', 'label'], sort=True).head(4).text.tolist()
    texts += ['riqualificazione energetica scuola comunità', 'realizzazione nuovo ospedale comunità',
              'servizi digitali comunità ' * 40]
    manifest = {'format': 'torch.export ExportedProgram (.pt2)', 'torch_version': str(torch.__version__),
                'classes': results['classes'], 'sequence_length': results['sequence_length'],
                'max_batch_size': 512, 'models': {}}
    loaded = {}
    for kind, model in models.items():
        model.eval()
        features = (torch.from_numpy(vectorizer.transform(texts).toarray()) if kind == 'MLP'
                    else torch.from_numpy(encode(texts, vocabulary, results['sequence_length'])))
        batch = torch.export.Dim('batch', min=1, max=512)
        graph = torch.export.export(model, (features[:2],), dynamic_shapes={'x': {0: batch}})
        path = destination / f'{kind.lower()}.pt2'
        torch.export.save(graph, path)
        restored = torch.export.load(path).module()
        max_error = 0.0
        sizes = [1, 2, len(texts), 512]
        with torch.inference_mode():
            for size in sizes:
                x = features[torch.arange(size) % len(features)]
                original, deployed = model(x), restored(x)
                torch.testing.assert_close(original, deployed, atol=1e-5, rtol=1e-5)
                assert torch.equal(original.argmax(1), deployed.argmax(1))
                max_error = max(max_error, (original - deployed).abs().max().item())
        loaded[kind] = restored
        manifest['models'][kind] = {'file': path.name, 'selected_run': results['selected'][kind],
                                    'bytes': path.stat().st_size, 'max_absolute_logit_error': max_error,
                                    'parity_batch_sizes': sizes, 'distinct_parity_titles': len(texts)}
    with (destination / 'tfidf.pkl').open('wb') as handle:
        pickle.dump(vectorizer, handle)
    (destination / 'vocabulary.json').write_text(json.dumps(vocabulary, ensure_ascii=False))
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return loaded, manifest

loaded_deployed, manifest = export_current_models(chosen_models, tfidf, vocab, prepared, r, OUT / 'deployment')
display(pd.DataFrame(manifest['models']).T)
print('Graph round-trip and predicted-class parity passed.')''')
    add('### 8.1 Define inference using the reloaded graphs', '''All cleaning, encoding and prediction logic is executable below. No training class definitions are needed by the reloaded graph. Titles shorter than three cleaned words, empty/oversized batches, and inputs without any known features/tokens are rejected for review. Partial vocabulary shift and confidently wrong predictions remain possible; this is not calibrated abstention.''', '''def classify_exported(titles, kind='MLP'):
    if kind not in loaded_deployed:
        raise ValueError('Model must be MLP or CNN')
    if not titles or len(titles) > 512:
        raise ValueError('Supply between 1 and 512 titles')
    cleaned = [clean_title(title) for title in titles]
    if any(len(title.split()) < 3 for title in cleaned):
        raise ValueError('Each title needs at least three meaningful words after cleaning')
    if kind == 'MLP':
        features = tfidf.transform(cleaned)
        if np.any(np.asarray(features.getnnz(axis=1)) == 0):
            raise ValueError('No known TF-IDF features; human review required')
        x = torch.from_numpy(features.toarray().astype(np.float32))
    else:
        features = encode(cleaned, vocab, r['sequence_length'])
        if np.any(~(features > 1).any(axis=1)):
            raise ValueError('No known tokens; human review required')
        x = torch.from_numpy(features)
    with torch.inference_mode():
        probabilities = loaded_deployed[kind](x).softmax(1).numpy()
    return [{'title': title, 'model': kind,
             'predicted_theme': classes[int(p.argmax())],
             'uncalibrated_probability': float(p.max())}
            for title, p in zip(titles, probabilities)]''')
    add('### 8.2 Demonstrate deployment and check invalid inputs', 'These predictions come from the serialized-and-reloaded graphs. They are exploratory suggestions for human review, never funding decisions.', '''examples = ['Riqualificazione energetica della scuola primaria comunale',
            'Realizzazione di un nuovo ospedale di comunità']
display(pd.DataFrame([row for kind in ['MLP', 'CNN'] for row in classify_exported(examples, kind)]))
for kind in ['MLP', 'CNN']:
    for invalid in [[], ['1234'], ['qzxqzx vjkvjk zzzxxqq'], [examples[0]] * 513]:
        try:
            classify_exported(invalid, kind)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Invalid input was accepted by {kind}')
print('Both deployment demonstrations and all invalid-input checks passed.')''')
    for cell in reflection:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
        cell.source = cell.source.replace('Saved preprocessing/checkpoints avoid repeating training merely to inspect results.', 'Opening this executed notebook permits inspecting saved outputs without new computation; Run All deliberately retrains from raw data. Each rerun incurs another training cost.')
        cells.append(cell)
    add('## 11. Execution record', 'All required inputs were the three raw CSVs. The following record identifies this run and its generated outputs; it is never read as a prerequisite.', '''validation = {'executed_utc': datetime.now(timezone.utc).isoformat(),
              'self_contained': True, 'input_files': audit['source_sha256'],
              'training_rows': len(partitions['train']), 'validation_rows': len(partitions['validation']),
              'test_rows': len(partitions['test']), 'neural_runs': list(r['runs']),
              'export_parity_passed': True, 'data_and_metric_assertions_passed': True,
              'shap_additivity_passed': shap_metadata['all_class_additivity_passed'],
              'shap_prediction_parity_passed': shap_metadata['prediction_parity_passed'],
              'shap_version': shap.__version__, 'shap_cases': len(shap_cases),
              'output_directory': str(OUT)}
(OUT / 'validation.json').write_text(json.dumps(validation, indent=2, ensure_ascii=False))
display(pd.Series(validation))
print('Completed from raw CSVs using only the code defined in this notebook.')''')
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb.metadata.language_info = {'name': 'python'}
    # Narrative authoring uses this helper only while building the notebook.
    # Its final stakeholder function is embedded as executable notebook code.
    narrative_spec = importlib.util.spec_from_file_location('pnrr_narrative', ROOT / 'src/notebook_narrative.py')
    narrative = importlib.util.module_from_spec(narrative_spec)
    narrative_spec.loader.exec_module(narrative)
    narrative.enrich_notebook(nb)
    intro = nb.cells[0].source.split('\n\n', 1)
    nb.cells[0].source = intro[0]
    nb.cells[1:1] = [md(EXECUTIVE_SUMMARY), md('## Reading and running the notebook\n\n' + intro[1])]
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'cell {i}', 'exec')
    nbf.validate(nb)
    nbf.write(nb, TARGET)
    if execute:
        from jupyter_client import KernelManager
        manager = KernelManager(kernel_name='python3')
        # Use the interpreter running this builder, including its installed SHAP.
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher',
                                    '-f', '{connection_file}']
        NotebookClient(nb, km=manager, timeout=3600,
                       resources={'metadata': {'path': str(ROOT)}}).execute()
        nbf.write(nb, TARGET)
    print(f'Wrote {TARGET.name}: {len(nb.cells)} cells, {sum(c.cell_type == "code" for c in nb.cells)} executable code cells.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    build(parser.parse_args().execute)
