"""Build an evidence-linked report and execute a notebook from saved real results."""
import inspect
import json
from pathlib import Path
import sys

import nbformat as nbf
from nbclient import NotebookClient
import pandas as pd

from prepare import clean_title
from train import OUT, ROOT, TitleCNN, TitleMLP, fit_neural, plots


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v).replace('|', '/') for v in row) + ' |' for row in rows])


def build():
    r = json.loads((OUT / 'results.json').read_text())
    a = json.loads((OUT / 'data_audit.json').read_text())
    prepared = pd.read_csv(OUT / 'prepared_projects.csv.gz')
    experiment = pd.read_csv(OUT / 'experiment_split.csv.gz')
    predictions = pd.read_csv(OUT / 'test_predictions.csv.gz')
    classes = r['classes']
    # Re-render plots from persisted predictions; no additional model evaluation.
    import numpy as np
    ytest = predictions.label.map(dict(zip(classes, range(len(classes))))).to_numpy()
    hard_probs = {name: np.eye(len(classes))[predictions[name + '_prediction'].map(dict(zip(classes, range(len(classes)))))]
                  for name in ['MLP', 'CNN']}
    plots(r, classes, ytest, hard_probs)
    review = prepared.groupby(['split', 'label'], group_keys=False).sample(n=2, random_state=42)
    review = review[['progetto_id', 'split', 'titolo', 'label', 'misura_id']].copy()
    review['human_reviewed_label'] = ''
    review['review_notes'] = ''
    review.to_csv(OUT / 'label_review_sample.csv', index=False)
    metrics = table(['Model', 'Test accuracy', 'Test macro-F1'],
                    [[name, f"{v['accuracy']:.4f}", f"{v['macro_f1']:.4f}"] for name, v in r['test'].items()])
    ablations = table(['Run', 'Best val. F1', 'Best / run epochs', 'Final val. F1', 'Time (s)', 'LR reductions'],
                     [[name, f"{v['validation']['macro_f1']:.4f}", f"{v['best_epoch']} / {v['epochs_run']}",
                       f"{v['last_epoch_validation_f1']:.4f}", f"{v['seconds']:.1f}", v['lr_reductions']]
                      for name, v in r['runs'].items()])
    split_table = table(['Partition', 'Available unique titles', 'Used titles', 'Used groups', 'CNN unknown-token rate'],
                       [[s, f"{(prepared.split == s).sum():,}", f"{d['rows']:,}", d['groups'], f"{100*d['oov_token_rate']:.1f}%"]
                        for s, d in r['diagnostics'].items()])
    clean_table = table(['Stage', 'Rows removed'], [
        ['Unmatched measure join', a['unmatched_measure_rows']],
        ['Fewer than three cleaned tokens', a['titles_under_three_tokens']],
        ['Normalized titles assigned conflicting root themes', a['conflicting_title_rows_removed']],
        ['Repeated normalized titles (one retained)', a['duplicate_clean_titles_removed']],
        ['Classes with insufficient support', sum(x['titles'] for x in a['excluded_classes'])]])
    class_table = table(['Category', 'Train', 'Validation', 'Test'],
                        [[c] + [r['diagnostics'][s]['class_counts'][c] for s in ['train', 'validation', 'test']] for c in classes])
    improvements = {kind: r['runs'][kind + '_regularized']['validation']['macro_f1'] -
                    r['runs'][kind + '_plain']['validation']['macro_f1'] for kind in ['mlp', 'cnn']}
    selected_rows = {kind: r['runs'][name] for kind, name in r['selected'].items()}
    architecture_table = table(['Model', 'Representation and structure', 'Parameters', 'Selected run'], [
        ['MLP', f"{r['tfidf_features']:,} word unigram/bigram TF–IDF → Dense 128 → Dense 64 → 8 logits",
         f"{selected_rows['MLP']['parameters']:,}", r['selected']['MLP']],
        ['CNN', f"{r['vocab_size']:,}-token vocabulary → 64D embeddings → parallel Conv1D (64 filters, widths 3/5) → masked max pooling → Dense 64 → 8 logits",
         f"{selected_rows['CNN']['parameters']:,}", r['selected']['CNN']]])
    winner = max(r['test'], key=lambda n: r['test'][n]['macro_f1'])
    best_neural = max(['MLP', 'CNN'], key=lambda n: r['test'][n]['macro_f1'])
    b = r['bootstrap_mlp_minus_logistic']
    c = r['bootstrap_cnn_minus_mlp']
    reports = pd.read_csv(OUT / 'classification_reports.csv')
    category_rows = []
    for category in classes:
        row = [category]
        for name in ['Logistic', 'MLP', 'CNN']:
            row.append(f"{reports.loc[(reports.category == category) & (reports.model == name), 'f1-score'].iloc[0]:.3f}")
        category_rows.append(row)
    class_metrics = table(['Category', 'Logistic F1', 'MLP F1', 'CNN F1'], category_rows)
    best_classes = reports[(reports.model == best_neural) & reports.category.isin(classes)].sort_values('f1-score', ascending=False)
    conclusion = (f"{winner} achieved the highest test macro-F1 ({r['test'][winner]['macro_f1']:.3f}). "
                  f"The stronger neural model was {best_neural} ({r['test'][best_neural]['macro_f1']:.3f}). "
                  'These values measure agreement with inherited programme themes on unseen measure groups.')
    interval_note = ('Both reported intervals span zero, so this experiment does not establish a clear model advantage.'
                     if b['ci95'][0] <= 0 <= b['ci95'][1] and c['ci95'][0] <= 0 <= c['ci95'][1]
                     else 'Interpret interval bounds alongside the limitations of this single split and weak labels.')
    report = f'''# Comparing two neural classifiers for PNRR project titles

{conclusion}

## Task and evidence

The local dataset contains **{a['raw_projects']:,} projects**. The task is to predict one of eight broad themes from the Italian **project title only**, extending the existing proposal with actual experiments. Each project is joined to `misure.csv`, then its theme is traced through `temi.csv` to the root category. These are **weak labels inherited from programme metadata**, not independently reviewed purpose annotations. No target label was guessed for an unmatched join. No separate payment, location, financing, organisation, measure-description, or measure-identifier column is supplied to the models. Titles themselves can still contain organisation names or natural-language programme references.

The original proposal and input CSVs are preserved. Source SHA-256 hashes, environment versions, and seed 42 are saved in `artifacts/`. All results below are measured, not illustrative.

## Preprocessing and cleaning

Unicode is normalized, text lowercased, embedded mission/component/investment codes and digit-containing tokens removed, punctuation replaced with spaces, and whitespace collapsed. Accented Italian letters remain; stopwords remain so the CNN can learn local phrasing. Numeric removal reduces identifier shortcuts but can remove meaningful quantities. No missing titles or duplicate project IDs were found.

{clean_table}

Conflicting-label title strings ({a['conflicting_title_strings']:,}) are excluded; identical remaining normalized titles are collapsed. Justice (6 titles) and public administration (54) do not meet the predefined support requirement of 500 unique titles and six independent groups. **{a['retained_unique_titles']:,} unique titles** remain. This changes the estimand from all project rows to eligible unique-title examples; it must not be presented as accuracy on the entire PNRR portfolio.

## Split design and compute budget

Root measure families are joined transitively when titles or CUPs connect them. Whole connected groups go to one partition. Per-category group assignments target 60/20/20, require at least two independent groups and 20 titles per category in each partition, and ensure training is the largest partition for each category. The assignment search sees group sizes and labels only. Assertions verify no overlap of normalized titles, CUPs, measures, families, or connected groups. This is an unseen-programme evaluation, not a random-title split. Exact duplicates are controlled; lexical near-duplicates are not exhaustively detected.

{split_table}

Training and validation are stratified samples within the already separate partitions to bound CPU cost; all test titles are retained. Small groups can disappear from a sampled partition: the sampled validation set contains only one group each for culture/tourism and social inclusion, so tuning evidence for these categories is especially fragile. The complete test set still has at least two groups per class. TF–IDF, vocabulary, and class weights are fitted on the sampled training set alone. Cleaning and label-consistency checks use the dataset's metadata before splitting, not learned text statistics. The manifest records every selected ID.

{class_table}

Perfect stratification is impossible: one connected digitalisation group holds 37,143 of 38,096 eligible digitalisation titles. It stays in training. The resulting distribution shift and sparse held-out support are limitations, not reasons to split that family across partitions.

## Model structure and training

{architecture_table}

The MLP uses ReLU hidden layers; it learns nonlinear combinations of weighted words and bigrams but largely discards longer word order. The CNN learns embeddings from scratch, keeps local order within 3/5-token windows, pools the strongest pattern at valid token positions, and uses a ReLU hidden head. Titles are padded/truncated to 80 tokens; the saved diagnostics show the actual truncation rate. `<PAD>` has a fixed zero embedding, `<UNK>` handles unseen words, and padding positions are masked before pooling. These compact architectures suit short titles and local CPU resources; no pretrained language model or external training data is used.

Both use AdamW, initial learning rate 0.001, batch size 256, gradient clipping at norm 1, and at most 14 epochs. Cross-entropy weights are proportional to the square root of inverse training class frequency and normalized to mean one; this softens majority dominance without full inverse-frequency weighting. All neural configurations share the split, initialization seed, minibatch-order seed, optimizer, and class weights. Macro-F1 selects checkpoints because accuracy alone can hide weak minority-class predictions.

The majority baseline predicts the most frequent **training** class, digitalisation. Its small test prevalence makes this baseline particularly weak under the observed distribution shift. TF–IDF logistic regression uses balanced class weights and selects C from 0.5 and 2.0 on validation macro-F1 (selected C={r['logistic_selected_C']}). The baseline's weighting differs from the neural weighting, so this is a practical pipeline comparison, not a perfectly isolated architecture experiment. Selections were frozen in `frozen_selection.json` before computing test predictions.

## Regularization, tuning, and learning curves

{ablations}

**Dropout and weight decay:** the regularized runs use dropout 0.40 at both hidden/head sites and AdamW weight decay 0.0001. Plain runs set both to zero. Their validation macro-F1 differences are **{improvements['mlp']:+.4f} for MLP** and **{improvements['cnn']:+.4f} for CNN**. This is a paired *combined* ablation; it cannot attribute the difference to dropout or weight decay individually, and one seed does not establish a general advantage.

**Early stopping:** training stops after three epochs without a macro-F1 improvement greater than 0.0001, then restores the best checkpoint. Compare the best and final-epoch columns to see what restoration recovered. Patience was fixed in advance; there is no no-stopping control, so we do not claim a causal performance gain from early stopping alone.

**Learning-rate scheduling:** ReduceLROnPlateau halves the rate after two unimproved epochs. The logs show when this happened; scheduling is common to both ablation arms and is not separately isolated. Gradient clipping is an additional stability measure. Batch normalization was not added; dropout, decay, and early stopping already provide multiple regularization strategies.

![Loss, macro-F1, and accuracy over epochs](artifacts/training_curves.png)

Training curves evaluate a fixed stratified 3,000-title training subset with dropout disabled; validation curves use all 8,000 selected validation titles. Both plotted losses are unweighted cross-entropy in evaluation mode. The separately logged `optimization_loss` is weighted loss during minibatch training and is not directly comparable. Logs include epoch time and learning rate. The gap between training and validation performance exposes overfitting and measure shift even when training loss falls.

## Held-out comparison

{metrics}

![Held-out accuracy and macro-F1](artifacts/model_comparison.png)

{class_metrics}

![Neural-model confusion matrices](artifacts/confusion_matrices.png)

## Brief evaluation and design trade-offs

{conclusion} For the stronger neural model, **{best_classes.iloc[0]['category']}** was the strongest class (F1 {best_classes.iloc[0]['f1-score']:.3f}) and **{best_classes.iloc[-1]['category']}** the weakest (F1 {best_classes.iloc[-1]['f1-score']:.3f}). Aggregate accuracy alone would conceal these differences.

What worked: inherited hierarchy labels make a reproducible supervised task possible; strict grouping prevents obvious programme memorization from inflating scores; compact networks train with available CPU resources; validation-driven regularization and checkpoint selection give a measurable comparison. The MLP is efficient because TF–IDF supplies informative lexical features immediately. The CNN can discover local phrases through learned filters, but learning its vocabulary representations from scratch costs more and is sensitive to novel language. Compare measured run times above rather than assuming deeper means better.

What did not work well: fitting training examples did not produce equally strong validation performance. New measures introduce vocabulary and purpose shifts: the CNN test unknown-token rate is {100*r['diagnostics']['test']['oov_token_rate']:.1f}%, versus {100*r['diagnostics']['train']['oov_token_rate']:.1f}% in training. A programme theme can also differ from the literal subject of a title, creating weak-label ambiguity. For example, inspected titles include photovoltaic installations under “Impresa e lavoro” and hotel digitalisation under “Cultura e turismo”. These examples illustrate the distinction; they are not a human annotation study. The proposal's macro-F1 goal of 0.80 was {'met' if r['test'][best_neural]['macro_f1'] >= .8 else 'not met by either neural model'}. {'The neural models did not outperform the logistic baseline on test macro-F1, so their added complexity is not justified by this experiment.' if winner == 'Logistic' else 'The observed neural advantage on this split should be checked across seeds and independently reviewed labels before preferring the more complex pipeline.'}

The paired group bootstrap (1,000 replicates, {b['independent_test_groups']} test groups) gives MLP minus logistic macro-F1 **{b['point_difference']:+.4f}**, with exploratory 95% interval **[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]** and one-sided 95% lower bound **{b['one_sided_95_lower']:+.4f}**. CNN minus MLP is **{c['point_difference']:+.4f}**, interval **[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}]**. Whole groups are resampled jointly for both models; a missing class has F1=0 in that replicate. {interval_note} Very few groups support some categories, so these intervals are unstable exploratory summaries, not definitive inference. The original hypothesis about independently reviewed project purposes is **not tested**, because no such gold-standard test set exists here.

The main next steps are human review of inherited labels, more independent measure groups per class, and repetition across split/initialization seeds. `label_review_sample.csv` provides examples with blank review fields; it is not claimed to be annotated. Deployment would also need probability calibration and an abstention option for unknown themes. No conclusion about funding fairness, regional performance, or future snapshots follows from these classification scores.

## Reproducibility and checks

See [README.md](README.md) for commands and [the executed notebook](02_pnrr_models_and_evaluation.ipynb) for code and outputs. Checkpoints, fitted vectorizers, source hashes, full split manifests, per-epoch CSVs, per-title test predictions, per-class metrics, and environment versions are retained. Unit checks cover cleaning, padding invariance, identity/group isolation, and preservation of the complete test partition. The original input files were not modified.
'''
    geography_section = ROOT / 'pnrr_outputs/geography/report_section.md'
    if geography_section.exists():
        report = report.rstrip() + '\n\n' + geography_section.read_text()
    (ROOT / 'REPORT.md').write_text(report)

    nb = nbf.v4.new_notebook()
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb['metadata']['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    nb['metadata']['language_info'] = {'name': 'python', 'version': r['environment']['python']}
    nb.cells = [
        md('# PNRR project-title classification: MLP versus CNN\n\nExecuted analysis of real training runs on the workspace data. '+conclusion+
           '\n\nThis notebook loads saved experiment outputs so opening or executing it does not silently retrain or change test selections. Complete implementations are in `src/prepare.py` and `src/train.py`. The original proposal is preserved.'),
        code("from pathlib import Path\nimport json, sys, inspect\nimport pandas as pd\nfrom IPython.display import display, Image, Markdown, Code\nROOT = Path.cwd()\nassert (ROOT / 'data/progetti.csv').exists(), 'Run from the project root'\nsys.path.insert(0, str(ROOT / 'src'))\nfrom prepare import clean_title, prepare\nfrom train import TitleMLP, TitleCNN, fit_neural\nA = ROOT / 'artifacts'\nr = json.loads((A / 'results.json').read_text())\naudit = json.loads((A / 'data_audit.json').read_text())\ndisplay(pd.Series(r['environment'], name='Training environment'))"),
        md('## 1. Dataset, inherited labels, and cleaning\n\nJoin projects → measures → root themes. Use titles only; remove label-revealing identifiers. Conflicting titles, unmatched measures, and unsupported classes are excluded. Exact title duplicates are collapsed. Labels remain provisional programme categories.'),
        code("display(pd.Series({k: audit[k] for k in ['raw_projects', 'unmatched_measure_rows', 'titles_under_three_tokens', 'conflicting_title_rows_removed', 'duplicate_clean_titles_removed', 'retained_unique_titles']}))\ndisplay(pd.DataFrame(audit['class_support_before_filter']))\ndisplay(Code(inspect.getsource(clean_title), language='python'))\nprint(clean_title('M1C1I1.04.05 SERVIZI DIGITALI PER LA COMUNITÀ E71F22003570006'))"),
        md('## 2. Isolated groups and train-only feature fitting\n\nConnected root measures, CUPs, and exact normalized titles cannot cross partitions. Split selection uses labels/group sizes, never model scores. Training/validation are sampled within those partitions; the test partition is complete. Digitalisation is dominated by one large group, so exact stratification is infeasible.'),
        code("display(pd.DataFrame({s: d['class_counts'] for s, d in r['diagnostics'].items()}))\ndisplay(pd.DataFrame([{\"split\": s, **{k: v for k, v in d.items() if k != 'class_counts'}} for s, d in r['diagnostics'].items()]))\nprint('Verified isolation:', audit['leakage_assertions_passed'])\ndisplay(Code(inspect.getsource(prepare), language='python'))"),
        md('## 3. Two different deep learning architectures\n\n**MLP:** TF–IDF unigrams/bigrams, two nonlinear hidden layers.\n\n**CNN:** learned token embeddings, parallel convolutions over ordered tokens, masked global pooling, and a nonlinear dense head. ReLU and 8 output logits are used in both.'),
        code("display(Code(inspect.getsource(TitleMLP), language='python'))\ndisplay(Code(inspect.getsource(TitleCNN), language='python'))\nprint('Selected models:', r['selected'])\nprint(TitleMLP(r['tfidf_features'], len(r['classes'])))\nprint(TitleCNN(r['vocab_size'], len(r['classes'])))"),
        md('## 4. Training, regularization, and tuning\n\nAll runs use AdamW, class-weighted cross-entropy, batch size 256, gradient clipping, a learning-rate scheduler, and early stopping on validation macro-F1. Compare no dropout/decay with dropout 0.4 plus decay 0.0001. This tests their combined effect. Restore the best checkpoint and freeze choices before testing. Logistic regression is also tuned on validation only.'),
        code("display(Code(inspect.getsource(fit_neural), language='python'))\nruns = pd.DataFrame([{\"run\": name, 'best_val_f1': v['validation']['macro_f1'], 'best_epoch': v['best_epoch'], 'epochs_run': v['epochs_run'], 'seconds': v['seconds'], 'parameters': v['parameters'], 'dropout': v['dropout'], 'weight_decay': v['weight_decay']} for name, v in r['runs'].items()])\ndisplay(runs)\ndisplay(pd.DataFrame(r['baseline_tuning']))"),
        code("# Reproduction command, intentionally not run when browsing saved results:\n# !{sys.executable} src/train.py --prepare --epochs 14 --train-limit 24000 --validation-limit 8000\n# Then rebuild this notebook and the report:\n# !{sys.executable} src/build_deliverables.py"),
        md('## 5. Training process over time\n\nTraining metrics are evaluated on the same 3,000-title monitoring subset with dropout off. Validation uses 8,000 titles. Plotted losses are unweighted evaluation loss; optimization loss in the CSVs is weighted training loss. The regularization ablation and train–validation gap are visible below.'),
        code("display(Image(filename=str(A / 'training_curves.png')))\nfor name in r['runs']:\n    print(name)\n    display(pd.read_csv(A / f'{name}_history.csv').round(4))"),
        md('## 6. Held-out test results and class-level errors\n\nAll test rows are evaluated only after selection. Macro-F1 weighs categories equally; accuracy weighs titles. Scores estimate inherited-label agreement for unseen measure groups.'),
        code("display(pd.DataFrame(r['test']).T.round(4))\ndisplay(Image(filename=str(A / 'model_comparison.png')))\ndisplay(Image(filename=str(A / 'confusion_matrices.png')))\nreport = pd.read_csv(A / 'classification_reports.csv')\ndisplay(report[report.category.isin(r['classes'])].pivot(index='category', columns='model', values='f1-score').round(3))"),
        code("predictions = pd.read_csv(A / 'test_predictions.csv.gz')\npd.set_option('display.max_colwidth', 140)\nfor model in ['MLP', 'CNN']:\n    print(model, '— three confident errors against inherited labels')\n    errors = predictions[predictions[model+'_prediction'] != predictions.label]\n    display(errors.sort_values(model+'_confidence', ascending=False)[['titolo', 'label', model+'_prediction', model+'_confidence']].head(3))\nprint('Paired cluster-bootstrap summaries:')\ndisplay(r['bootstrap_mlp_minus_logistic'])\ndisplay(r['bootstrap_cnn_minus_mlp'])"),
        md('## 7. Written evaluation\n\n' + report.split('## Brief evaluation and design trade-offs\n\n')[1].split('## Reproducibility and checks')[0]),
        md('## 8. Reload a trained model\n\nUse the saved preprocessing and the selected checkpoint; never refit vocabulary at inference. These are uncalibrated exploratory predictions.'),
        code("from predict import classify\ndisplay(classify('Riqualificazione energetica della scuola primaria comunale', 'MLP'))\ndisplay(classify('Realizzazione di un nuovo ospedale di comunità', 'CNN'))"),
        md('## Deliverables and validation\n\nSee `REPORT.md` and `README.md`. Saved artifacts include complete preprocessing audits, source hashes, manifests, four neural checkpoints, baseline model, per-epoch logs, predictions, plots, and per-class metrics. Run `python -m unittest discover -s tests -v` for cleaning, padding, and split-integrity checks. The label-review CSV is an unannotated template, not gold-standard validation.')
    ]
    path = ROOT / '02_pnrr_models_and_evaluation.ipynb'
    nbf.write(nb, path)
    client = NotebookClient(nb, timeout=180, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}})
    client.execute()
    nbf.write(nb, path)
    errors = [output for cell in nb.cells if cell.cell_type == 'code' for output in cell.get('outputs', []) if output.output_type == 'error']
    if errors:
        raise RuntimeError(errors)
    print(f'Wrote {path.name} with {sum(c.cell_type == "code" for c in nb.cells)} executed code cells; no errors.')


if __name__ == '__main__':
    build()
