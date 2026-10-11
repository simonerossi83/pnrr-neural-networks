# Notebook and submission verification

Reviewed on 6 October 2026. **Share with caveats:** the notebook covers all six core project steps for exploratory classification against inherited programme themes. It does **not yet fully satisfy every advanced rubric criterion**. **Review completeness: partial** for hosted Colab execution and interactive HTML rendering; every supplied rubric criterion was inspected. Execution coverage is recorded separately from academic completeness below. No training, labels, model-selection rules, or metrics were changed during this review.

The notebook and builder were updated with Colab dependency checks, raw-CSV upload handling, and explicit separate-HTML submission instructions. These changes do not supply independent human labels or strengthen the missing literature justification.

**Later narrative revision, 6 October 2026:** at the user's request, the notebook now includes an executive summary, a final plain-language stakeholder briefing, literature-connected design rationale, architecture/explanation alternatives, and computed interpretation of outcomes and trade-offs. The rubric assessment and proposed improvements below record the audit **before that narrative revision**; they should not be read as a fresh grading of the revised prose. The empirical limitations (weak labels, one split/seed, limited independent groups, uncertain improvement, unknown snapshot date and unverified hosted/interactive rendering) remain. All 31 original executable cells and five saved figures are preserved. Both new interpretation cells executed against the verified run, bringing the current notebook to 33 code cells; two targeted standalone/authoring tests passed. Full training was not repeated for these narrative additions. Verification is recorded in `pnrr_outputs/verification/narrative_verification.json`.

## Requirement-by-requirement assessment

| Requirement | Notebook evidence | Assessment |
|---|---|---|
| Step 1: problem, importance, suitable data/model, success | §§1.1–1.6 | Met. The original proposal is retained as historical motivation; unfinished funding analysis and annotation are disclosed. |
| Step 2: relevant, feasible dataset and preprocessing | §§1.3, 1.7, 3–4 | Met with limitations. OpenPNRR source links, source hashes, cleaning, joins, exclusions and sampling are documented. Snapshot date and population representativeness are not established. |
| Step 3: appropriate neural architecture and justification | §5 | Architecture implementation met: explicit TF–IDF MLP and embedding CNN, layer dimensions and shape/padding checks. Justification is less developed than the advanced criterion requires. |
| Step 4: train, monitor, adjust, evaluate | §§6–7 | Met. Four neural configurations, a tuned logistic baseline, validation selection, early stopping, complete held-out test, per-class metrics and paired group bootstrap. |
| Step 5: interpretable performance visuals | §§7.2–7.4 | Met. Learning curves, comparison plot, annotated confusion matrices, confident disagreements and SHAP plots. More interpretation of the learning curves and class failures would strengthen the submission. |
| Step 6: bias, privacy/environment, benefits and harms | §§7.5, 9 | Met. Weak labels, class disparities, privacy, energy scenarios, misuse and human-review limits are explicit. |
| Theory-grounded hypothesis with defined variables and evaluation implications | §§1.5–1.6, 9 | Partial. Δ macro-F1, null/alternative and grouped uncertainty rule are clear, but domain theory/prior-research grounding is missing. The hypothesis involving independently reviewed purposes remains untested. |
| Detailed empirical strategy, constraints, mitigation, rationale/citations | §§3–7, 9 | Mostly met operationally. Leakage controls, preprocessing, training and evaluation are detailed; literature support for key choices remains limited. |
| Clear integration of at least three distinct sources into a justified strategy | Source links in §§1.7, 7.4, 8 | Not demonstrated at the requested level. OpenPNRR, SHAP and PyTorch references support separate components; their joint contribution to the modeling strategy is not articulated. |
| Deep architecture/data alignment, contemporary components, efficiency, benchmark support and rejected alternatives | §§4–5, 9 | Partial. Compact CPU models fit short titles and a bounded budget. Benchmark references and reasoned consideration of recurrent/pretrained transformer alternatives are absent. A larger model is not required to fix this writing gap. |
| Systematic optimization with observed diminishing returns/overfitting/compute trade-offs | §§6, 7.2–7.3, 9 | Mechanisms met; interpretation partial. Scheduling, clipping, dropout, decay, early stopping and validation tuning are implemented. The narrow two-arm combined regularization comparison and one seed should be discussed alongside observed overfitting and compute costs. |
| Ethical and non-ethical risks mapped to behavior, use cases and mitigation | §§7.5, 8.1, 9 | Strongly met. Vocabulary shift, confident errors, invalid inputs, misuse and automation bias inform human review and deployment restrictions. Calibration/abstention remains unvalidated and is disclosed. |
| At least two annotated plot types, interpreted failures/trade-offs/uncertainty informing decisions | §§7.2–7.5, 10 | Mostly met. SHAP interpretation informs review-interface decisions. Explicit discussion of learning-curve overfitting and confused categories would strengthen the evaluation decisions. |
| Interpretability method justified for architecture/stakeholders, with comparison of techniques | §§7.4–7.5 | Partial. Preprocessing-aware word-deletion SHAP, reference choice, score units, additivity, seed sensitivity and stakeholder limits are well explained. Comparison with gradient attribution, LIME or logistic coefficients is missing. |
| Multiple stakeholder perspectives and sociotechnical trade-offs connected to decisions | §§1.2, 7.5, 9 | Mostly met. Citizens, analysts and smaller organizations are considered. Explicit deployer/regulator accountability, correction routes and conflicts between transparency, efficiency and automation bias would deepen the reflection. |

## Prioritized improvements before claiming full rubric compliance

1. **Completed research question (§§1.5, 9).** Keep the original proposal, but state a completed exploratory hypothesis about agreement with inherited programme themes. Do not treat this as a test of true project purpose without independent annotation. **Proposed:** add this scope-aligned question and its literature grounding.
2. **Integrated rationale (§§1.7, 4–5).** Listing three links is insufficient for the integration criterion. **Proposed:** explain how domain label structure, short-text modeling research, and grouped/weak-label evaluation guidance jointly determine inputs, architecture and evaluation.
3. **Architecture and optimization (§§5–7).** Model layers are clearer than the reasons for selecting them. **Proposed:** justify MLP lexical efficiency and the CNN local-order hypothesis; explain why recurrent/pretrained models were deferred; interpret validation gains, overfitting, restored best epochs and extra CNN compute. Dropout and weight decay change together, so their individual effects cannot be separated.
4. **Interpretability and stakeholders (§§7.4–7.5, 9).** The SHAP discussion is thoughtful but does not compare methods or fully expose stakeholder conflicts. **Proposed:** compare preprocessing-aware SHAP with gradient attribution, LIME and baseline coefficients; connect reviewer efficiency, affected-project contestability, public transparency and deployer/regulator accountability to concrete use restrictions.
5. **Pipeline comparison (§§6–7).** Logistic regression uses balanced class weights; the neural runs use square-root inverse-frequency weights. **Proposed:** clarify that the experiment compares full pipelines and does not isolate architecture as the cause of performance differences.

## Verified performance interpretation

The saved run reviewed before the setup changes had 298,338 raw projects, 166,521 retained unique titles and eight inherited themes. It used 24,000 training titles, 8,000 validation titles and all 21,219 test titles. The selected model was the regularized MLP.

| Model | Test macro-F1 | Test accuracy |
|---|---:|---:|
| Logistic regression | 0.3805 | 0.4214 |
| Selected MLP | 0.3890 | 0.4435 |
| Selected CNN | 0.2878 | 0.4179 |

These metrics were independently recomputed from all saved per-title predictions. The MLP–logistic difference was +0.00844, with exploratory 95% paired group-bootstrap interval [−0.03234, +0.04221]. The interval includes zero; superiority is not established. The proposed 0.80 macro-F1 goal was not achieved, which does not violate the core requirement to demonstrate the training/evaluation process.

The plots show strong overfitting: training F1 approaches 1 while validation F1 stays around 0.4 and validation loss rises. Combined regularization increased MLP validation F1 by 0.0161 and reduced CNN validation F1 by 0.0131. In that saved run, the selected MLP trained in 5.79 seconds and selected CNN in 423.42 seconds; timings are hardware-specific and exclude other workflow stages. Education/research MLP F1 was 0.815 versus 0.008 for digitalisation. CNN unknown-token rates rose from about 3.4% in training to 15.7% in test. These observations support limited use, stronger group/label audits and avoiding extra CNN compute without validated benefit.

## Standalone and Colab status

The notebook contains every project function. It does not import project source modules or require old models/results. It requires the three raw input files: `progetti.csv`, `misure.csv`, `temi.csv` (the project file alone is about 111 MB).

For Colab:

1. Upload `pnrr_complete_workflow.ipynb` and select a Python CPU runtime.
2. Choose **Runtime → Run all**. Setup installs missing packages and prompts for the three supplied CSVs; files already uploaded to the runtime root are also detected. Alternatively, set `DATA_DIR` to an accessible directory holding all three files.
3. Allow a full training run; save/download the executed notebook and desired generated outputs before deleting the runtime.

Package handling preserves Colab's installed scientific stack. Missing SHAP is installed as 0.49.1; an already installed SHAP version is retained. PyTorch must provide the export APIs documented in [PyTorch 2.6](https://docs.pytorch.org/docs/2.6/export.html). Numerical results can differ across environments. Colab itself recommends including required library/file-loading cells because runtime files and libraries are not shared with the notebook. [Colab FAQ](https://research.google.com/colaboratory/faq.html), [official file-loading example](https://colab.research.google.com/notebooks/io.ipynb).

Nine mocked setup checks passed, covering installed-package retention, missing SHAP, shell quoting of version constraints, root CSV uploads, upload-picker data, partial uploads, and local missing-input errors. These checks are reproducible with `.venv/bin/python pnrr_outputs/verification/colab_setup_checks.py`. All 13 existing project tests passed. A real hosted Colab session and a fresh package installation were not executed; local execution must not be described as hosted-Colab certification.

The current full isolated execution result is recorded in `pnrr_outputs/standalone_verification.json`: **31 executed code cells, zero notebook execution errors**, default 24,000/8,000 training/validation budget, all 21,219 held-out test titles, all four neural runs and both exported-model parity checks. Execution completed on 6 October 2026 at 17:40 Europe/Rome. It started from only the notebook and the three raw CSVs, without repository modules or prior outputs. Pip requirement quoting was finalized after the full run began; the updated bootstrap was executed separately locally and its Colab branches verified by the nine setup checks. The modeling/data/evaluation code was unchanged. The historical record dated 1 October had only 27 code cells and could not verify later SHAP additions.

Fresh prediction metrics exactly matched the reviewed saved metrics. All five newly generated notebook plot images were byte-identical to the five visually inspected images. A local nbconvert template-path problem affected only the supporting HTML export after notebook execution had succeeded; pointing the exporter to the installed Anaconda templates recovered that export. The executed notebook, fresh run outputs and updated verification record are saved in the workspace.

## Separate choropleth submission

Submit **`pnrr_outputs/geography/italy_projects.html` (underscore)**. It is a complete HTML document with embedded SVG boundaries, data, CSS and JavaScript. Viewing requires no CSVs, other local files, CDN libraries or data fetches. The ISTAT link is optional. Use a modern JavaScript-enabled browser. `italy-projects.html` (hyphen) is an embedding fragment and is not the standalone submission file.

The map is independent of the classification run. Its themes are recorded programme metadata, not neural predictions. All 20 region geometries are valid. Independent source recomputation matched every exported theme/region series, geographic audit and embedded payload: 298,338 raw project IDs; 297,500 region-mapped, 767 nationwide-only, 71 with no location; 300,652 project-region memberships because multi-region projects count once in each recorded region. The standalone map can be viewed without the notebook and is not regenerated by running it.

The exported static overall map was visually inspected. Notebook learning curves, confusion matrices and SHAP figures were also inspected. **Interactive HTML browser rendering and control behavior remain unverified:** browser security rejected local `file://` access. No workaround was used. Whether the university submission platform accepts HTML or preserves its scripts was not checked; the technical standalone file should be downloaded and opened in a browser if the platform preview does not run it.

## Review coverage

The tables below use scoped component counts, not a percentage of overall correctness. Zero observed defects does not mean every presentation state was verified. Academic gaps are tracked in the rubric table above; they are not silently treated as arithmetic errors.

Notebook/submission quality:

| Category | Observed defects | Assessment |
|---|---|---|
| Usefulness/completeness | 8 / 15 | All six core steps present; eight advanced criteria require further support or explanation to claim full compliance. |
| Analytical clarity | 4 / 6 | Scope and risks are clear; hypothesis grounding, integrated sources, architecture alternatives and optimization interpretation need expansion. |
| Visual/interaction consistency | 0 / 11 | Five notebook figures, two static geography figures, the HTML figure and three HTML controls inventoried; several figures inspected. Interactive HTML states were not rendered because local browser access was blocked. |

Analytical correctness and robustness:

| Category | Observed defects | Assessment |
|---|---|---|
| Source authority/confidence | 0 / 3 | OpenPNRR, SHAP and PyTorch controlling references checked. Local snapshot date and independently reviewed label validity remain unknown. |
| Value accuracy | 0 / 8 | Four classification model metric rows and four geographic coverage/series components reconciled against inspectable data. |
| Within-chart agreement | 0 / 8 | Five notebook figures, two static geography figures and the HTML figure inventoried. Plotted methods and several rendered figures checked; HTML rendering unverified. |
| Complete source details | N/A | No dashboard source-detail modals; notebook provenance and geographic embedded/source data assessed elsewhere. |
| Cross-artifact consistency | 0 / 3 | Notebook/builder setup, README instructions and map payload/source data compared. Historical artifacts are explicitly excluded from the current-run contract. |
| Data-quality controls | 0 / 3 | Classification joins/cleaning, grouping/train-only features and geographic deduplication/coverage checked. Weak labels and representativeness remain disclosed constraints. |
| Conclusion support | 0 / 4 | Scope-limited results, uncertainty, local explanation limits and descriptive geography are supported; no causal or verified-purpose claim is established. |
