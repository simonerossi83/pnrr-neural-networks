"""Authoring helpers that enrich the standalone PNRR notebook with interpretation.

This module is used only while building the notebook. All runtime narrative code is
embedded into the notebook and depends on objects produced by the current run.
"""
from textwrap import dedent

import nbformat


ADDITIONS = {
    '## 3. Load and preprocess the data': '''
**Why these inputs and cleaning rules?** The three CSVs supply project titles and the
measure/theme hierarchy needed for the completed classification task. Financial,
payment and territorial fields are excluded because they do not answer the title
classification question and would widen the experiment. Unicode normalization and
retaining Italian accented letters reduce spelling variation without translating
away the source language. Removing administrative codes, mixed letter/digit tokens
and punctuation reduces identifier shortcuts; it also loses potentially meaningful
numbers, so this is a documented trade-off rather than complete anonymization.
Repeated spaces are collapsed to make duplicate detection consistent. The hierarchy
cycle check stops invalid ancestry from silently producing incorrect labels.
''',
    '### 3.1 Validate joins and construct isolated groups': '''
**Why exclude, deduplicate and group?** Many-to-one join validation prevents a
hierarchy join from multiplying projects. Missing titles/measure matches and titles
under three cleaned words cannot supply the intended inputs and targets. Identical
cleaned titles with conflicting themes are removed because this title-only model
cannot resolve their contradictory targets. Removing repeated titles limits the
influence of administrative templates: the experiment estimates performance over
**unique titles**, rather than weighting every repeated project equally.

The ancestor theme is a feasible **weak label**, not independently verified project
purpose. A project can genuinely combine education, health or inclusion activities.
At least 500 unique titles and six independent groups per retained theme are
practical support rules, not statistically proven adequacy thresholds. Six groups
allow at least two groups in each split; excluding smaller themes reduces coverage
and requires an explicit unknown/review route.

Connected measure families, CUP identifiers and duplicate titles are kept together
to prevent related examples from crossing split boundaries. This tests unfamiliar
programme groups rather than memorization of a familiar template. The approximate
60/20/20 allocation balances training data with two held-out sets; uneven group
sizes prevent exact proportions. Searching label/group-size balance, without model
scores, avoids selecting a favorable test result. Group isolation takes priority
over matching class proportions. The code stops on mixed-theme components rather
than silently claiming a valid grouped split. This follows the rationale for
[group-aware cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data): dependent observations require dependent splits.
''',
    '## 4. Training-only features and batched loading': '''
**Why a bounded experiment?** The 24,000 training and 8,000 validation title limits
make CPU execution feasible during the project. Class-stratified sampling preserves
proportions within the already isolated partitions; it does not make all themes
uniform or repair distribution shift. Retaining the complete test partition avoids
a small convenience test sample. Sparse storage and converting only a current batch
to dense TF–IDF reduce memory pressure; CNN token arrays have a fixed size. These
are resource choices, not evidence that more training data would add no value.
''',
    '### 4.2 Fit TF–IDF and vocabulary on training titles': '''
**Why these representations?** TF–IDF retains distinctive Italian purpose words;
unigrams capture individual terms and bigrams capture short phrases. Sublinear
term frequency limits repeated-word dominance, while `min_df=2` removes one-off
features. The 6,000-feature cap bounds MLP size. The CNN's at-most-12,000-token
vocabulary and minimum frequency of two likewise bound its learned embedding
matrix; a separate unknown token makes unseen language visible. Padding uses a
separate zero ID, and the 80-word cutoff bounds computation. The diagnostics report
actual truncation so this choice can be revisited if future titles are longer.
These settings are practical starting choices, not tuned optima. Fitting both
representations only on training titles prevents validation/test language from
influencing the learned feature space. Saving fitted preprocessing preserves the
same transformations at inference.
''',
    '### 4.3 Record coverage and class weights': '''
**Why weights and diagnostics?** Square-root inverse-frequency weights reduce the
training influence of common themes while tempering the stronger amplification of
rare, potentially noisy labels under full inverse-frequency weighting. Normalizing
weights to a mean of one keeps their scale readable. This does not guarantee equal
errors across categories. Counts and independent groups describe different kinds
of support; many similar titles cannot replace independent programmes. Unknown
word rates expose vocabulary shift, and truncation rates expose lost tail text.
A fixed, class-stratified 3,000-title training monitoring subset bounds repeated
measurement cost; it estimates training behavior rather than reporting every
training title each epoch. The full validation subset guides selection.
''',
    '## 5. Explicit neural network architectures': '''
**Why an MLP first?** The inputs are short titles with sparse purpose terms, so a
compact nonlinear network can test whether combinations of TF–IDF terms improve
on a linear classifier. Two hidden layers of 128 and 64 units limit capacity and
CPU cost; ReLU supplies the nonlinearity. These sizes are feasible design settings,
not benchmark-proven best values for PNRR. Dropout is varied during the bounded
comparison below. A simpler classifier remains an essential reference because a
neural network is useful only if its quality or practical benefits justify it.
''',
    '### 5.1 Text CNN': '''
**Why compare a CNN?** Learned 64-dimensional word embeddings and three-/five-word
convolution windows test whether local ordering and short phrases provide useful
information beyond TF–IDF. Parallel windows handle more than one phrase length;
max pooling detects a relevant phrase anywhere in a variable-length title. Padding
masking prevents artificial padded positions from winning the pool. This is
informed by [Kim's text CNN](https://arxiv.org/abs/1408.5882), but these embeddings
are learned from scratch and the paper's results do not validate this dataset.
The chosen dimensions bound compute; they were not exhaustively tuned. Eight
logits correspond to mutually exclusive retained programme themes. Cross-entropy
consumes logits directly, and softmax is reserved for scores at evaluation/inference.

| Alternative | Decision and reason |
|---|---|
| Majority prediction | Retained as a minimal no-language reference, fixed from training counts. |
| TF–IDF logistic regression | Retained as a cheap, interpretable linear text benchmark using the same titles/splits. |
| RNN/LSTM | Not implemented: sequential processing adds cost, while short titles make local phrases a reasonable first ordering test. Its quality is unknown here. |
| Pretrained Italian/multilingual transformer | Deferred: potentially useful semantic transfer, especially under vocabulary shift, but additional weights, dependencies and compute widen the CPU experiment. No inferiority claim is made. |
| Larger MLP/CNN or long-document architecture | Deferred until reliable labels and current errors justify more capacity; extra complexity cannot resolve an ambiguous target by itself. |

The MLP/CNN comparison balances conventional text modeling with feasibility; it is
not a claim to state-of-the-art predictive performance.
''',
    '### 5.2 Verify layer shapes and padding behaviour': '''
**What these checks establish.** Shape checks catch wiring errors before training;
adding only right-padding must not alter a CNN prediction. The cleaning assertion
confirms one intended identifier-removal example. These are implementation checks,
not evidence of semantic accuracy, privacy protection or robustness to every title.
''',
    '## 6. Training, validation and model selection': '''
**Why these measures?** Macro-F1 averages the precision/recall balance of each
retained theme equally, making weak small-theme performance harder to hide.
Accuracy answers how often a title receives its inherited label and can be dominated
by prevalent themes. Unweighted log-loss assesses the whole score distribution and
penalizes confident mistakes, so a model can improve class predictions while
worsening probability performance. None measures verified project purpose without
independent annotation. The validation set guides configuration/checkpoint
selection; the test set assesses frozen choices.
''',
    '### 6.1 Define the training loop': '''
**Why this optimization policy?** Weighted cross-entropy fits one inherited theme
per title. AdamW uses adaptive updates with separate weight decay, while clipping
gradient norm at one bounds occasional large updates. An initial learning rate of
0.001 and batch size of 256 are feasible starting settings, not demonstrated optima.
`ReduceLROnPlateau` halves the rate after validation macro-F1 stagnation (patience
one), permitting smaller updates without spending a broad learning-rate search.
The 14-epoch ceiling bounds compute; stopping after three unimproved epochs and
restoring the best validation checkpoint limit degradation from continued fitting.
The 0.0001 improvement tolerance avoids treating negligible score movement as a
fresh improvement. Seeded shuffling makes the order reproducible within the recorded
environment. Weighted optimization loss and unweighted evaluation loss answer
different questions and should not be treated as directly comparable values.
''',
    '### 6.2 Tune the logistic regression baseline': '''
**Why a fair simple reference?** Logistic regression uses the same fitted TF–IDF
features and partitions, so the comparison tests the value of nonlinear modeling
without adding different input information. Validation selects between `C=0.5`
and `C=2.0` (different regularization strengths). These two settings are a bounded
search, not full optimization. Balanced class weights support the minority-theme
objective, but use full inverse-frequency weighting rather than the neural models'
square-root weighting; therefore the losses are not identical. The iteration cap
bounds fitting effort. The majority category is fixed from training data, since
choosing the test-majority category would use held-out labels improperly.
''',
    '### 6.3 Train the four neural configurations': '''
**Why only four runs?** For each architecture, the plain and regularized variants
compare no dropout/decay with dropout 0.4 plus weight decay 0.0001 under the same
seed and budget. This examines their **combined** effect and cannot isolate which
component caused a difference. Validation macro-F1 selects each architecture's
variant and then the leading neural architecture. A single seed and two variants
are a systematic but narrow comparison: observed gains are not proof of a global
optimum or stable superiority. The learning curves and final briefing explain
whether continued training brought diminishing returns or overfitting.
''',
    '### 6.4 Paired group bootstrap': '''
**Why resample groups together?** Related titles share programme wording, so a
row-level bootstrap would overstate independent information. Jointly resampling
whole groups preserves the pairing between classifiers on the same observations.
The 1,000 seeded resamples bound computation and provide an exploratory uncertainty
summary; more titles do not remove the limitation of few groups. Keeping all eight
categories in every replicate, with absent-category F1 set to zero, avoids quietly
changing the definition of macro-F1, but makes the interval sensitive to scarce
classes. An interval containing zero does not support a positive difference; it
also does not prove equivalence. These weak-label intervals cannot test the
proposal's hypothesis about independently reviewed purposes.
''',
    '## 7. Held-out evaluation and verification': '''
**Why freeze and retain outputs?** Writing selected configurations before test
prediction makes the selection sequence inspectable. Complete test predictions,
per-class reports and group-paired intervals reveal failures concealed by one
average. A hard majority prediction is not a useful probability model, so its
log-loss is omitted rather than compared with softmax distributions. The result
briefing is calculated from current-run objects so numerical conclusions follow
any rerun rather than remaining fixed in prose.
''',
    '### 7.1 Check split boundaries and recompute reported metrics': '''
**What passing means.** The assertions verify observed identity/group isolation,
complete held-out membership and arithmetic agreement with saved prediction rows.
They reduce implementation-error risk; they do not establish representative source
coverage, independent label validity or deployment fairness.
''',
    '### 7.2 Plot the current run': '''
**Why these three visual views?** Learning curves show the cost and benefit of
additional epochs; falling training loss with rising validation loss warns of
fitting patterns that do not transfer. The model bars compare a title-weighted
measure with an equal-category measure. Row-normalized confusion matrices display
where each source theme is routed, even if it has few test titles; a large diagonal
entry is recall, not proof of precision. Fixed 0–1 performance scales, numerical
annotations and separate training/validation lines support comparison. Curves
motivate early stopping/restoration; class errors motivate label review and a
supervised use boundary. Numerical interpretation is provided in the final briefing.
''',
    '### 7.3 Inspect class-level performance and confident errors': '''
**Why inspect confident disagreements?** Class-level precision, recall and F1
identify themes where a good overall average could still mislead stakeholders.
Sorting disagreements by score deliberately finds warning examples, not a
representative error sample. A university-named health project can activate
education language, and an inclusion project can describe a school activity.
These cases can reflect a shortcut, ambiguous wording or disagreement between
programme and activity labels; human review is needed to distinguish them.
Near-one scores on disagreements show why softmax must not be interpreted as a
verified probability of being correct. The bootstrap interpretation and current
class disparities are stated in the final briefing.
''',
    '### 7.4 Explain individual MLP predictions with SHAP': '''
**Why this method rather than another explanation?** A model-agnostic word-deletion
SHAP game explains the **whole fitted TF–IDF/MLP pipeline** in words a reviewer can
inspect, including changes in bigrams and normalization. The
[SHAP framework](https://arxiv.org/abs/1705.07874) supplies the additive contribution
logic, not evidence that these weak labels or masking choices are correct.

| Explanation option | Use and trade-off in this task |
|---|---|
| Logistic coefficients | Cheap, readable global associations, but explain the linear baseline rather than the nonlinear MLP or an individual full-pipeline prediction. |
| Permutation SHAP (chosen) | Readable local word contributions with an explicit reference/reconstruction; repeated model calls cost more and deletion can create unnatural titles. |
| [Integrated gradients](https://proceedings.mlr.press/v70/sundararajan17a.html) | Appropriate for differentiable neural representations and potentially cheaper repeated explanation; discrete cleaning/TF–IDF complicates end-to-end word attribution, and a reference/path still needs justification. |
| [LIME](https://arxiv.org/abs/1602.04938) | A local surrogate can be readable, but its sampling/neighborhood and surrogate fidelity add choices; it does not implement this same additive allocation game. |

For reviewers, readable evidence for challenging a suggestion is more useful than
an unexplained hidden-unit signal. For affected communities, a contestable source
category and correction route matter more than a persuasive plot. Only two
purposefully selected MLP examples are explained: this cannot establish global
importance, population fairness or CNN interpretability.
''',
    '#### 7.4.1 Compute contributions and verify the explanation': '''
**Why bound and repeat the calculation?** Three to 30 words and 20 permutation
cycles limit explanation cost. Selecting the lowest ID in agreement/disagreement
strata makes case selection reproducible without optimizing the model. Repeating
with a second seed checks finite-sampling sensitivity; arithmetic reconstruction
and prediction parity check that the explanation matches the evaluated model.
None validates the empty-title reference, estimates causal effects or certifies
convergence. Saved case IDs/settings/contributions make a reviewer able to inspect
the explanation choices rather than only the rendered bars.
''',
    '## 8. Export and reload deployment graphs': '''
**Why export with parity checks?** A saved graph plus the fitted preprocessing,
vocabulary, label order and manifest packages the current model consistently for
later use. Reloading freshly written graphs avoids relying on earlier artifacts.
Batch sizes 1, 2, a diverse parity sample and the 512 limit test practical batch
shape behavior; logits and classes must match the in-memory model within stated
numerical tolerances. Passing establishes fidelity for these sampled inputs, not
semantic quality or readiness for a public service. Serialized preprocessing should
be loaded only from trusted project outputs.
''',
    '### 8.1 Define inference using the reloaded graphs': '''
**Why these input rules?** Reusing cleaning/encoding prevents training–inference
mismatch. The one-to-512 batch limit matches the exported graph's shape contract.
Rejecting empty/short titles and inputs with no known features prevents obvious
unsupported suggestions. Partial vocabulary shift, ambiguous titles and confidently
wrong results remain possible, so these checks are not a calibrated abstention
policy. Explicitly naming the output `uncalibrated_probability` makes the score's
limitation visible to downstream users.
''',
    '### 8.2 Demonstrate deployment and check invalid inputs': '''
**How to read the examples.** These are demonstrations without independently
reviewed correct labels. Agreement on a hospital title shows consistent model
behavior; disagreement on the school/energy title illustrates multiple plausible
activities and why a suggestion cannot replace a recorded category. A high score
expresses the model's preference among its eight themes, not certainty about
actual purpose. Invalid-input checks show that the implemented interface enforces
its contract; they do not guarantee rejection of every unfamiliar or risky input.
''',
}


RESEARCH_CONNECTIONS = '''
### 1.8 Completed research question and connected design rationale

The completed question is narrower than the original proposal: **do nonlinear
combinations of title terms (MLP), or learned local phrases (CNN), improve agreement
with inherited programme themes over TF–IDF logistic regression on unseen connected
programme groups?** The input is a cleaned Italian title, the outcome is one
retained ancestor theme, and the comparison is held-out macro-F1. The anticipated
relationship is that local purpose terms may transfer across programmes, while
programme templates and institution names may fail to transfer. This motivates
training-only features, group isolation, a linear baseline and per-theme errors.
The original human-reviewed-purpose hypothesis is retained as a future hypothesis;
no independent annotation was completed in this notebook.

Three distinct sources contribute complementary concepts rather than added model
complexity: [OpenPNRR's metadata](https://openpnrr.openpolis.io/metadati/) explains
what the programme hierarchy and fields mean, constraining the target and excluding
a measure description from independent text inputs;
[group-aware evaluation guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
explains why repeated programme examples must stay together;
[Kim's text-CNN research](https://arxiv.org/abs/1408.5882) motivates a bounded test
of local phrase order alongside the TF–IDF MLP. The
[SHAP framework](https://arxiv.org/abs/1705.07874) adds a contestable word-level
interpretation of the chosen MLP rather than a new performance objective.

Their contextual connection is the strategy: a **simple title model under a harder
programme-held-out test, with explicit weak-label uncertainty and human-review
explanations**. Novelty lies in fitting the evaluation and intended use to PNRR's
administrative hierarchy; it is not a claim to invent a new architecture. These
sources justify concepts and conventions, not the optimality of this notebook's
hyperparameters or the truth of its inherited labels.
'''


STAKEHOLDER_INTRO = '''
## 12. What this means for stakeholders

This briefing is for citizens, public administrators, project organizations and
reviewers. It explains the completed experiment in plain language and connects
its evidence to practical decisions. **This is an exploratory aid for examining
project categories.** Recorded categories remain authoritative source information;
independent review is needed to assess a project's actual purpose.

The following briefing recalculates its numbers from this run. “Accuracy” is the
share of titles matching their recorded programme theme. “Macro-F1” gives each
theme equal importance and balances missed projects with incorrect assignments;
it runs from zero to one and is **not a percentage of titles classified correctly**.
Both assess agreement with inherited labels, which can themselves be ambiguous.
'''


EXECUTIVE_DECISION_NOTES = '''**Why the target was missed, and how to read 0.389.** The 0.80 target was aspirational, not an established benchmark for this title-only task. Inherited labels, missing context and unfamiliar programmes are plausible constraints; the observed learning curves show overfitting. Digitalisation F1 was 0.008 and infrastructure F1 0.086. If these two scores stayed fixed, even perfect performance on the other six themes would give only 0.762 macro-F1. This illustrates why the weak themes must improve, without proving that 0.80 is unattainable. The 0.389 result is useful exploratory evidence of category clues, with education/research reaching 0.815, but modest overall performance.

**Why retain MLP rather than claim it wins.** Validation macro-F1 was 0.441 for MLP, 0.395 for CNN and 0.382 for logistic regression. MLP was the leading neural candidate before test evaluation, fitting the assignment's neural-network requirement. Its test advantage over logistic regression then narrowed to 0.0084 and was uncertain. Logistic regression remains a practical reference candidate because of its simplicity and lower test log-loss (1.488 versus 1.652). Retaining MLP for research, explanations and export is justified; preferring it for deployment on this evidence is not.'''


TECHNICAL_RESULTS_CODE = dedent(r'''
def technical_result_brief(r, audit, report, histories, shap_diagnostics):
    """Explain current-run results using only already computed notebook objects."""
    selected = max(r['selected'], key=lambda k: r['runs'][r['selected'][k]]['validation']['macro_f1'])
    scores_now = r['test'][selected]
    logistic_now = r['test']['Logistic']
    mlp_delta = r['bootstrap_mlp_minus_logistic']
    cnn_delta = r['bootstrap_cnn_minus_mlp']
    class_rows = report[(report.model == selected) & report.category.isin(r['classes'])].copy()
    strongest = class_rows.loc[class_rows['f1-score'].idxmax()]
    weakest = class_rows.loc[class_rows['f1-score'].idxmin()]
    weakest_two = class_rows.nsmallest(2, 'f1-score')
    fixed_weak_class_ceiling = (len(class_rows) - 2 + weakest_two['f1-score'].sum()) / len(class_rows)
    weak_pair = ' and '.join(f"{row['category']} ({row['f1-score']:.3f})" for _, row in weakest_two.iterrows())
    selected_validation = r['runs'][r['selected'][selected]]['validation']['macro_f1']
    logistic_validation = next(item['validation']['macro_f1'] for item in r['baseline_tuning']
                               if item['C'] == r['logistic_selected_C'])
    selected_history = histories[r['selected'][selected]]
    mlp_class = report[(report.model == 'MLP') & report.category.isin(r['classes'])].set_index('category')['f1-score']
    logistic_class = report[(report.model == 'Logistic') & report.category.isin(r['classes'])].set_index('category')['f1-score']
    class_differences = mlp_class - logistic_class
    improved_classes = int((class_differences > 0).sum())
    worse_classes = int((class_differences < 0).sum())
    train_counts = r['diagnostics']['train']['class_counts']
    test_counts = r['diagnostics']['test']['class_counts']
    majority_theme = max(train_counts, key=train_counts.get)
    majority_share = test_counts.get(majority_theme, 0) / r['diagnostics']['test']['rows']
    interval = mlp_delta['ci95']
    gain = mlp_delta['point_difference']
    selected_gain = scores_now['macro_f1'] - logistic_now['macro_f1']
    target_status = 'met' if scores_now['macro_f1'] >= .80 else 'not met'
    gain_status = 'met' if selected_gain >= .03 else 'not met'
    interval_reading = ('does not support a reliable positive difference' if interval[0] <= 0 <= interval[1]
                        else 'suggests a positive difference under these resampling assumptions' if interval[0] > 0
                        else 'suggests a negative difference under these resampling assumptions')
    lower_reading = ('above zero' if mlp_delta['one_sided_95_lower'] > 0 else 'not above zero')
    excluded = ', '.join(item['label'] for item in audit['excluded_classes']) or 'none in this run'
    optimization_rows, curve_readings = [], []
    for name, run in r['runs'].items():
        h = histories[name]
        optimization_rows.append(
            f"| {name.replace('_', ' ')} | {run['validation']['macro_f1']:.3f} | "
            f"{run['best_epoch']} / {run['epochs_run']} | {run['lr_reductions']} | {run['seconds']:.2f} |")
        if h.iloc[-1]['train_loss'] < h.iloc[0]['train_loss'] and h.iloc[-1]['val_loss'] > h.iloc[0]['val_loss']:
            curve_readings.append(name.replace('_', ' '))
    neural_seconds = sum(run['seconds'] for run in r['runs'].values())
    mlp_seconds = sum(run['seconds'] for name, run in r['runs'].items() if name.startswith('mlp_'))
    cnn_seconds = sum(run['seconds'] for name, run in r['runs'].items() if name.startswith('cnn_'))
    cost_ratio = cnn_seconds / mlp_seconds if mlp_seconds > 0 else float('nan')
    mlp_plain = r['runs']['mlp_plain']['validation']['macro_f1']
    mlp_reg = r['runs']['mlp_regularized']['validation']['macro_f1']
    cnn_plain = r['runs']['cnn_plain']['validation']['macro_f1']
    cnn_reg = r['runs']['cnn_regularized']['validation']['macro_f1']
    logloss_gap = r['test']['MLP']['loss'] - logistic_now['loss']
    loss_reading = ('worse' if logloss_gap > 0 else 'better' if logloss_gap < 0 else 'equal')
    shap_seed = float(shap_diagnostics.max_seed_change_pp_predicted_class.max())
    shap_shared = int(shap_diagnostics.top_3_word_positions_shared_across_seeds.min())
    shap_error = float(shap_diagnostics.max_reconstruction_error_all_classes.max())
    curve_text = (', '.join(curve_readings) +
                  ' fit their training titles better while their validation loss rose. This warns that continued fitting can strengthen patterns that fail on unfamiliar programmes.'
                  if curve_readings else
                  'The curves should be read together: improvements on training titles alone do not establish transfer to unfamiliar programmes.')
    text = [
        f"**What was completed.** From {audit['raw_projects']:,} source project rows, cleaning retained "
        f"{audit['retained_unique_titles']:,} unique titles across {len(r['classes'])} themes. "
        f"The experiment used {r['diagnostics']['train']['rows']:,} training titles, "
        f"{r['diagnostics']['validation']['rows']:,} validation titles and "
        f"{r['diagnostics']['test']['rows']:,} test titles. Related programme groups stayed together. "
        "Training/validation were sampled to keep CPU work feasible; the full held-out title set was evaluated. "
        f"Excluded themes: **{excluded}**. Exclusion reflects insufficient independent support, not lesser public importance. "
        "Removing duplicates reduces template memorization but changes the population to unique titles. "
        "Financial allocation, recorded payments, territorial fairness and independently reviewed purposes were not evaluated by this model.",
        f"**How well it worked.** Validation selected **{selected}**; its held-out accuracy was "
        f"**{scores_now['accuracy']:.1%}** and its equal-theme score was **{scores_now['macro_f1']:.3f}**. "
        f"Logistic regression achieved {logistic_now['accuracy']:.1%} accuracy and {logistic_now['macro_f1']:.3f} macro-F1. "
        f"The proposed 0.80 equal-theme target was **{target_status}**, and the selected neural model's "
        f"+0.03 improvement goal was **{gain_status}** (observed difference {selected_gain:+.3f}). "
        "These goals were chosen for the project, not independently established deployment thresholds.",
        "**Why the 0.80 target is demanding.** It was an aspirational proposal goal, not a literature-backed "
        "benchmark shown to be attainable from these titles. Macro-F1 gives each theme equal weight: "
        f"the two weakest {selected} themes were {weak_pair}. If those two scores stayed fixed, even perfect "
        f"F1 on every other theme would give only {fixed_weak_class_ceiling:.3f} overall. "
        "This is an arithmetic illustration, not an estimated performance ceiling for future models. "
        "Short titles omit context, and inherited programme labels can differ from the activity described; "
        "these are plausible constraints, not measured label-error rates or proof that 0.80 is impossible. "
        "Group isolation deliberately tests unfamiliar programmes rather than repeated templates. "
        f"By the final monitored epoch, {selected} training macro-F1 was "
        f"{selected_history.iloc[-1]['train_macro_f1']:.3f}, versus "
        f"{selected_history.iloc[-1]['val_macro_f1']:.3f} on validation; test evaluation uses the best validation checkpoint. "
        "This observed gap supports overfitting/generalization difficulty, so simply adding epochs is not a demonstrated remedy. "
        "The bounded training sample and narrow tuning are additional constraints whose individual effects were not isolated.",
        f"**What {scores_now['macro_f1']:.3f} does and does not mean.** It is not the percentage of correct titles. "
        f"It shows some transferable category recognition, particularly {strongest['category']} "
        f"(F1 {strongest['f1-score']:.3f}), under the programme-held-out test. This is useful exploratory evidence "
        "for a neural-network learning project, but modest overall predictive quality rather than a strong deployment result. "
        "The training-majority reference collapses under the changed class mix, so beating it cannot establish useful quality. "
        f"The meaningful text baseline already achieves {logistic_now['macro_f1']:.3f}; "
        "there is no task-independent threshold that makes this result good, and weak-theme failures remain consequential.",
        f"**Why retain {selected}, and why not declare it better than logistic regression?** "
        "The selection rule was validation macro-F1, before examining test outcomes. "
        f"The selected neural candidate scored {selected_validation:.3f} on validation, versus "
        f"{logistic_validation:.3f} for the tuned logistic reference. The assignment requires a neural network, "
        "so carrying this candidate through SHAP and graph-export demonstrations is justified as a research choice. "
        f"On test data its advantage shrank to {selected_gain:+.4f}. MLP improved on logistic regression in "
        f"{improved_classes} themes and worsened in {worse_classes}; the average hides different error costs. "
        "Selection therefore does not establish a deployment preference. Logistic regression has fewer moving parts "
        "and readable coefficients; its measured tuning cost and test log-loss are reported below. "
        "Keep it as the practical reference candidate until replicated neural gains, acceptable theme-level errors "
        "and a validated review policy justify additional complexity. Neither model is approved for automated public decisions.",
        f"**What uncertainty allows us to say.** MLP minus logistic regression was {gain:+.4f} macro-F1. "
        f"Its exploratory 95% interval was [{interval[0]:+.4f}, {interval[1]:+.4f}], which {interval_reading}. "
        f"The one-sided 95% lower bound was {mlp_delta['one_sided_95_lower']:+.4f}, {lower_reading}. "
        f"CNN minus MLP was {cnn_delta['point_difference']:+.4f}, with interval "
        f"[{cnn_delta['ci95'][0]:+.4f}, {cnn_delta['ci95'][1]:+.4f}]. "
        f"Only {mlp_delta['independent_test_groups']} independent test groups support these intervals; "
        "missing categories in resamples and one split/seed limit certainty. "
        "The original hypothesis about independently reviewed purposes remains untested; "
        "an inconclusive interval does not prove that models are equivalent.",
        f"**Where it struggles.** {selected}'s strongest theme was **{strongest['category']}** "
        f"(F1 {strongest['f1-score']:.3f}, {int(strongest['support']):,} test titles) and weakest was "
        f"**{weakest['category']}** (F1 {weakest['f1-score']:.3f}, {int(weakest['support']):,} titles). "
        "A successful large theme cannot justify automatic treatment of every theme. "
        "The confusion matrices show where recorded themes are redirected; confident disagreement examples "
        "suggest possible reliance on institutional wording or ambiguity between overlapping activities. "
        "Those cases need targeted explanation and domain review. "
        "Some disagreements may expose ambiguous programme labels rather than an incorrect description of actual activity.",
        f"**Why the simplest reference can look unusually poor.** The training-majority theme was "
        f"**{majority_theme}**, but it represented only {majority_share:.1%} of held-out titles. "
        f"Always choosing it therefore gave {r['test']['Majority']['accuracy']:.1%} test accuracy. "
        "This reflects the changed mix of programmes under group isolation; it is not evidence that every "
        "simple classifier is unhelpful. The tuned logistic model is the more informative language benchmark. "
        f"Unknown CNN words rose from {r['diagnostics']['train']['oov_token_rate']:.1%} in training to "
        f"{r['diagnostics']['test']['oov_token_rate']:.1%} in test titles. This is a warning of changed language, "
        "not proof of a single cause of model failure.",
        f"**Scores need caution.** MLP's probability-error measure was {r['test']['MLP']['loss']:.3f}, "
        f"versus {logistic_now['loss']:.3f} for logistic regression (lower is better), so MLP was {loss_reading} "
        "on this measure. Better theme assignments can coexist with poorer scores on confident errors. "
        "A displayed 90% model score is not evidence that 90 of 100 similar suggestions will be correct. "
        "Calibration and an independently validated review/abstention policy are needed before operational use.",
        "**What training choices achieved.** The table shows the selected checkpoint rather than assuming the last "
        "epoch was best. Both the rate reductions and early stopping followed validation behavior.\n\n"
        "| Configuration | Validation equal-theme score | Best epoch / epochs run | Rate reductions | Timed seconds |\n"
        "|---|---:|---:|---:|---:|\n" + '\n'.join(optimization_rows) + '\n\n' +
        f"Combined dropout/weight decay changed MLP validation macro-F1 from {mlp_plain:.3f} to {mlp_reg:.3f}, "
        f"and CNN from {cnn_plain:.3f} to {cnn_reg:.3f}; the selected variants were "
        f"{r['selected']['MLP'].replace('_', ' ')} and {r['selected']['CNN'].replace('_', ' ')}. "
        "This does not isolate dropout from weight decay or establish an optimal setting. " + curve_text +
        f" CNN's two fits took {cnn_seconds/60:.2f} minutes versus {mlp_seconds:.2f} seconds for the two MLP fits "
        f"(about {cost_ratio:.1f} times as much timed work). "
        f"All neural fits/monitoring took {neural_seconds/60:.2f} minutes; "
        f"logistic tuning took {r['logistic_seconds_including_tuning']:.2f} seconds. "
        "These timings depend on hardware and do not include every notebook operation. "
        "The energy table assumes power levels; electricity and emissions were not measured. "
        "Added complexity should be justified by quality, not by model size alone.",
        f"**What explanations establish.** The {len(shap_diagnostics)} selected word explanations "
        "help a reviewer challenge a suggestion by seeing which title words raised or lowered its score. "
        f"The largest second-seed attribution change was {shap_seed:.2f} percentage points; at least "
        f"{shap_shared}/3 leading word positions overlapped across seeds. "
        f"The largest reconstruction error was {shap_error:.2g}. "
        "These narrow checks concern calculation consistency and sampling variation, not causal truth or fairness. "
        "Deleting words and using an empty-title reference affect the answer; explanations can also create false reassurance. "
    ]
    return '\n\n'.join(text)


display(Markdown(technical_result_brief(r, audit, report, histories, shap_diagnostics)))
''').strip()


TECHNICAL_RESULTS_INTRO = '''
### 7.6 Current-run results and decision interpretation

The following interpretation is computed from the same histories, held-out
predictions and explanation diagnostics as the figures. It connects each outcome
to the model-selection and intended-use decisions; no extra training is performed.
'''


STAKEHOLDER_CODE = dedent(r'''
def stakeholder_brief(r, audit, report, validation, histories, shap_diagnostics, manifest):
    """Summarize this run for readers making public-service and review decisions."""
    selected = max(r['selected'], key=lambda k: r['runs'][r['selected'][k]]['validation']['macro_f1'])
    outcome = r['test'][selected]
    baseline = r['test']['Logistic']
    delta = r['bootstrap_mlp_minus_logistic']
    interval = delta['ci95']
    class_rows = report[(report.model == selected) & report.category.isin(r['classes'])]
    strongest = class_rows.loc[class_rows['f1-score'].idxmax()]
    weakest = class_rows.loc[class_rows['f1-score'].idxmin()]
    excluded = ', '.join(item['label'] for item in audit['excluded_classes']) or 'none'
    target = 'met' if outcome['macro_f1'] >= .8 else 'not met'
    gain = outcome['macro_f1'] - baseline['macro_f1']
    gain_target = 'met' if gain >= .03 else 'not met'
    selected_validation = r['runs'][r['selected'][selected]]['validation']['macro_f1']
    logistic_validation = next(item['validation']['macro_f1'] for item in r['baseline_tuning']
                               if item['C'] == r['logistic_selected_C'])
    uncertainty = ('The uncertainty range crosses zero, so a reliable advantage has not been established.'
                   if interval[0] <= 0 <= interval[1] else
                   'The uncertainty range favors MLP under this limited test, but needs confirmation.'
                   if interval[0] > 0 else
                   'The uncertainty range favors the simpler model under this limited test.')
    train_counts = r['diagnostics']['train']['class_counts']
    majority = max(train_counts, key=train_counts.get)
    majority_share = r['diagnostics']['test']['class_counts'].get(majority, 0) / r['diagnostics']['test']['rows']
    mlp_seconds = sum(v['seconds'] for k, v in r['runs'].items() if k.startswith('mlp_'))
    cnn_seconds = sum(v['seconds'] for k, v in r['runs'].items() if k.startswith('cnn_'))
    cnn_quality = ('produced a lower balanced score' if r['test']['CNN']['macro_f1'] < r['test']['MLP']['macro_f1']
                   else 'produced a higher balanced score' if r['test']['CNN']['macro_f1'] > r['test']['MLP']['macro_f1']
                   else 'produced the same balanced score')
    check_keys = ['export_parity_passed', 'data_and_metric_assertions_passed',
                  'shap_additivity_passed', 'shap_prediction_parity_passed']
    check_result = 'passed' if all(validation.get(k, False) for k in check_keys) else 'need further review'
    paragraphs = [
        f"**Purpose and coverage.** This project tests whether short Italian project titles can help reviewers "
        f"organize PNRR records. It retained {audit['retained_unique_titles']:,} unique titles from "
        f"{audit['raw_projects']:,} project rows, and tested {r['diagnostics']['test']['rows']:,} titles from "
        f"{delta['independent_test_groups']} separate programme groups. Repeated titles were collapsed to avoid "
        f"learning the same template repeatedly. Themes excluded for insufficient independent examples: **{excluded}**. "
        "The result covers these titles and themes, not all public investments.",
        f"**Main finding.** The validation comparison selected **{selected}**. It matched the recorded theme on "
        f"**{outcome['accuracy']:.1%}** of test titles, versus **{baseline['accuracy']:.1%}** for the simpler "
        f"logistic model. Its balanced, equal-theme score was **{outcome['macro_f1']:.3f}**, versus "
        f"{baseline['macro_f1']:.3f}. The project goals of a 0.80 balanced score and a +0.03 neural improvement "
        f"were **{target}** and **{gain_target}**, respectively. These are project goals, not approved standards "
        "for deciding funding or eligibility.",
        "**Why the goal was ambitious.** The proposed 0.80 score was an aspiration, not an established standard "
        "for this dataset. Short titles can omit a project's purpose, and the recorded programme theme can "
        "describe something broader than its activity. Testing separate programmes also exposes unfamiliar "
        "wording. The learning curves show much stronger results on training titles than on new programmes. "
        "These observations and plausible constraints explain the difficulty; they do not prove that 0.80 is impossible. "
        "Better labels and independent evaluation should come before assuming more training will solve it.",
        f"**Is {outcome['macro_f1']:.3f} good?** It is useful early evidence that titles contain category clues, "
        f"especially for {strongest['category']}, but modest overall performance. It is not the share of correct "
        "titles and does not indicate dependable service quality. A successful learning project can still "
        "produce a model that needs substantial improvement.",
        f"**Why the neural model remains in the project.** Before the final test, {selected}'s score on the "
        f"separate validation examples was {selected_validation:.3f}, versus {logistic_validation:.3f} for logistic "
        "regression. That supports selecting it for this neural-network experiment and its explanation/export "
        "demonstrations. It does not show that the small final-test difference warrants deployment. The simpler "
        "logistic model remains a practical reference candidate; any extra neural complexity should earn a "
        "repeatable benefit. Neither model is ready to make public decisions independently.",
        f"**What remains uncertain.** MLP's balanced-score difference from logistic regression was "
        f"{delta['point_difference']:+.3f}; the estimated uncertainty range was "
        f"[{interval[0]:+.3f}, {interval[1]:+.3f}]. {uncertainty} "
        "Only one programme split and one starting seed were tested. Recorded programme themes are provisional "
        "labels: a school-related project may belong to an inclusion programme. No independent reviewers checked "
        "the actual purposes, so the original research hypothesis remains untested.",
        f"**Who needs particular attention.** The strongest theme was **{strongest['category']}**, "
        f"with balanced class score {strongest['f1-score']:.3f}; the weakest was **{weakest['category']}**, "
        f"with {weakest['f1-score']:.3f}. Performance on one theme cannot justify automation across all themes. "
        "Confident disagreements involving universities suggest possible reliance on institutional wording; "
        "these cases need targeted explanation and review. "
        f"The most common training theme, {majority}, was only {majority_share:.1%} of test titles; "
        "this changed mix explains the unusually weak always-pick-one-theme reference. The simpler text "
        "classifier is the more useful comparison.",
        f"**Cost and transparency.** The CNN's two training comparisons took {cnn_seconds/60:.2f} minutes, "
        f"versus {mlp_seconds:.2f} seconds for the MLP comparisons, and {cnn_quality}. "
        "More computation should earn a practical quality benefit. Electricity and emissions were not measured. "
        f"The {len(shap_diagnostics)} word explanations help reviewers question suggestions; a convincing "
        "explanation cannot establish correctness. Implementation checks " + check_result +
        ", including checks that saving/reloading preserves sampled predictions. Working software is not "
        "evidence of safe independent decision-making, and a high model score is not a verified chance of being right.",
        "**Responsibilities before a pilot.** These are proposed next steps, not completed approvals.\n\n"
        "| Owner | Action | Reason |\n"
        "|---|---|---|\n"
        "| Data steward and domain reviewers | Check a sample across themes, territories and organizations; keep unknowns and corrections visible. | Inherited labels and gaps in coverage can misrepresent activities. |\n"
        "| Analyst and named deployment owner | Validate a new snapshot and a human-review rule; monitor errors and suspend unsupported uses. | Unfamiliar wording and weak themes can cause confident mistakes. |\n"
        "| Publisher and service owner | Give citizens and project organizations readable source records and a correction/appeal channel. | Incorrect categories can hide work or distort public perceptions. |\n"
        "| Publisher and governance/privacy lead | Review names/addresses, minimize title logging, verify attribution/licence, and document accountability. | Public source data can still expose people; cleaning is not anonymization. |\n\n"
        "Proceed only to a supervised category-routing pilot after independent label review and agreed review/stop "
        "rules. Keep a human responsible for each decision and preserve source categories. Never use predictions "
        "alone to award or deny funding, rank communities or allege misconduct. Review adds work, but protects "
        "affected communities while allowing administrators to explore useful assistance.",
        "**Reading the map and deciding what comes next.** `italy_projects.html` describes recorded projects and "
        "programme themes, not these predictions; it is not regenerated here. Counts are not funding amounts "
        "or payments, and geographic patterns do not establish fairness, misconduct or policy impact. The map "
        "needs its own coverage/attribution checks. The notebook and self-contained HTML may accompany each "
        "other, subject to the submission platform's HTML policy. Prioritize trustworthy labels, weak-theme "
        "review and accountable oversight before a larger model."
    ]
    return '\n\n'.join(paragraphs)


display(Markdown(stakeholder_brief(r, audit, report, validation, histories, shap_diagnostics, manifest)))
''').strip()



def enrich_notebook(nb):
    """Add rationale and a dynamic stakeholder briefing without altering model code."""
    marker = '<!-- PNRR narrative enrichment -->'
    for cell in nb.cells:
        if cell.cell_type != 'markdown':
            continue
        header = cell.source.splitlines()[0] if cell.source else ''
        if header in ADDITIONS and marker not in cell.source:
            cell.source += '\n\n' + marker + '\n\n' + dedent(ADDITIONS[header]).strip()
    if not any('### 1.8 Completed research question' in c.source for c in nb.cells if c.cell_type == 'markdown'):
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == 'markdown' and cell.source.startswith('## 1.7.'):
                nb.cells.insert(index + 1, nbformat.v4.new_markdown_cell(dedent(RESEARCH_CONNECTIONS).strip()))
                break
    if not any(c.source.startswith('### 7.6 Current-run results and decision interpretation')
               for c in nb.cells if c.cell_type == 'markdown'):
        for index, cell in enumerate(nb.cells):
            if cell.cell_type == 'markdown' and cell.source.startswith('## 8. Export'):
                nb.cells[index:index] = [
                    nbformat.v4.new_markdown_cell(dedent(TECHNICAL_RESULTS_INTRO).strip()),
                    nbformat.v4.new_code_cell(TECHNICAL_RESULTS_CODE),
                ]
                break
    if not any(c.source.startswith('## 12. What this means for stakeholders')
               for c in nb.cells if c.cell_type == 'markdown'):
        nb.cells.extend([
            nbformat.v4.new_markdown_cell(dedent(STAKEHOLDER_INTRO).strip()),
            nbformat.v4.new_code_cell(STAKEHOLDER_CODE),
        ])
    return nb
