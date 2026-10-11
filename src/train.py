"""Train two neural text classifiers, validation ablations, and simple baselines."""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import pickle
import platform
import random
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from sklearn.model_selection import train_test_split
import torch
from torch import nn

from prepare import prepare

OUT = ROOT / 'artifacts'
SEED = 42


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)


class TitleMLP(nn.Module):
    """Two hidden layers learn nonlinear combinations of TF-IDF features."""
    def __init__(self, n_features, n_classes, dropout=.4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(64, n_classes))

    def forward(self, x):
        return self.net(x)


class TitleCNN(nn.Module):
    """Learn embeddings and local 3/5-token patterns; masked global max pooling."""
    def __init__(self, vocab_size, n_classes, dropout=.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(64, 64, k, padding=k // 2) for k in (3, 5)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, 64), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x):
        valid = (x != 0).unsqueeze(1)
        emb = self.embedding(x).transpose(1, 2)
        pooled = [torch.relu(conv(emb)).masked_fill(~valid, -1e4).amax(dim=2) for conv in self.convs]
        return self.head(torch.cat(pooled, dim=1))


def encode(texts, vocab, length=80):
    result = np.zeros((len(texts), length), dtype=np.int64)
    for i, text in enumerate(texts):
        ids = [vocab.get(w, 1) for w in text.split()[:length]]
        result[i, :len(ids)] = ids
    return result


def batch_tensor(x, idx):
    data = x[idx]
    if hasattr(data, 'toarray'):
        data = data.toarray()
    return torch.from_numpy(data)


@torch.inference_mode()
def predict(model, x, batch_size=512):
    model.eval()
    return np.concatenate([model(batch_tensor(x, np.arange(i, min(i + batch_size, len_or_shape(x)))))
                           .softmax(dim=1).numpy() for i in range(0, len_or_shape(x), batch_size)])


def len_or_shape(x):
    return x.shape[0]


def scores(y, probs):
    pred = probs.argmax(axis=1)
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, average='macro', labels=np.arange(probs.shape[1]), zero_division=0)),
            'loss': float(log_loss(y, probs, labels=np.arange(probs.shape[1])))}


def fit_neural(name, model, xtrain, ytrain, xval, yval, weights, cfg, monitor_idx):
    seed_all()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=1)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    ytensor = torch.tensor(ytrain, dtype=torch.long)
    rng = np.random.default_rng(SEED)
    history, best, best_epoch, stale = [], -1., 0, 0
    start = time.perf_counter()
    for epoch in range(1, cfg['max_epochs'] + 1):
        epoch_start = time.perf_counter()
        model.train()
        order = rng.permutation(len(ytrain))
        optimization_loss = 0.
        lr = optimizer.param_groups[0]['lr']
        for offset in range(0, len(order), cfg['batch_size']):
            idx = order[offset:offset + cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_tensor(xtrain, idx))
            loss = criterion(logits, ytensor[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            optimization_loss += loss.item() * len(idx)
        train_metrics = scores(ytrain[monitor_idx], predict(model, xtrain[monitor_idx]))
        val_metrics = scores(yval, predict(model, xval))
        row = {'epoch': epoch, 'lr': lr, 'optimization_loss': optimization_loss / len(ytrain),
               **{'train_' + k: v for k, v in train_metrics.items()},
               **{'val_' + k: v for k, v in val_metrics.items()},
               'seconds': time.perf_counter() - epoch_start}
        history.append(row)
        pd.DataFrame(history).to_csv(OUT / f'{name}_history.csv', index=False)
        print(f"{name:18} epoch={epoch:02} train_loss={row['train_loss']:.4f} "
              f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
              f"val_F1={row['val_macro_f1']:.4f} lr={lr:.5f} seconds={row['seconds']:.1f}", flush=True)
        if val_metrics['macro_f1'] > best + 1e-4:
            best, best_epoch, stale = val_metrics['macro_f1'], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        scheduler.step(val_metrics['macro_f1'])
        if stale >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    torch.save({'state_dict': best_state, 'config': cfg, 'best_epoch': best_epoch}, OUT / f'{name}.pt')
    result = {'name': name, **cfg, 'best_epoch': best_epoch, 'epochs_run': len(history),
              'seconds': time.perf_counter() - start,
              'parameters': sum(p.numel() for p in model.parameters()),
              'validation': scores(yval, predict(model, xval)),
              'last_epoch_validation_f1': history[-1]['val_macro_f1'],
              'lr_reductions': sum(history[i]['lr'] < history[i - 1]['lr'] for i in range(1, len(history)))}
    return model, result


def sample_partition(frame, limit):
    if limit is None or len(frame) <= limit:
        return frame.copy()
    chosen, _ = train_test_split(np.arange(len(frame)), train_size=limit, stratify=frame.label, random_state=SEED)
    return frame.iloc[chosen].copy()


def group_bootstrap(y, pred_a, pred_b, groups, classes, n_boot=1000):
    """Paired cluster bootstrap: cache per-group confusion matrices for efficiency."""
    matrices = []
    for group in np.unique(groups):
        idx = groups == group
        matrices.append([confusion_matrix(y[idx], pred[idx], labels=np.arange(classes)) for pred in (pred_a, pred_b)])
    matrices = np.array(matrices)
    rng = np.random.default_rng(SEED)
    deltas = []
    for _ in range(n_boot):
        cm = matrices[rng.integers(len(matrices), size=len(matrices))].sum(axis=0)
        f1 = 2 * np.diagonal(cm, axis1=1, axis2=2) / np.maximum(cm.sum(axis=1) + cm.sum(axis=2), 1)
        deltas.append(float(f1[0].mean() - f1[1].mean()))
    return {'point_difference': float(f1_score(y, pred_a, average='macro') - f1_score(y, pred_b, average='macro')),
            'ci95': np.quantile(deltas, [.025, .975]).tolist(),
            'one_sided_95_lower': float(np.quantile(deltas, .05)),
            'replicates': n_boot, 'independent_test_groups': len(matrices),
            'caveat': 'Few groups per class; absent classes get F1=0 in a replicate. Interval is exploratory.'}


def plots(results, classes, ytest, probabilities):
    plt.rcParams.update({'figure.dpi': 140, 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for row, kind in enumerate(['mlp', 'cnn']):
        for variant, style in [('plain', '--'), ('regularized', '-')]:
            h = pd.read_csv(OUT / f'{kind}_{variant}_history.csv')
            for split, color in [('train', '#26749c'), ('val', '#d47732')]:
                axes[row, 0].plot(h.epoch, h[f'{split}_loss'], style, color=color, label=f'{variant}: {split}')
                axes[row, 1].plot(h.epoch, h[f'{split}_macro_f1'], style, color=color, label=f'{variant}: {split}')
                axes[row, 2].plot(h.epoch, h[f'{split}_accuracy'], style, color=color, label=f'{variant}: {split}')
        axes[row, 0].set(title=f'{kind.upper()} — cross-entropy', xlabel='Epoch', ylabel='Unweighted loss')
        axes[row, 1].set(title=f'{kind.upper()} — macro-F1', xlabel='Epoch', ylabel='Macro-F1', ylim=(0, 1))
        axes[row, 2].set(title=f'{kind.upper()} — accuracy', xlabel='Epoch', ylabel='Accuracy', ylim=(0, 1))
        for ax in axes[row]:
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
    fig.savefig(OUT / 'training_curves.png'); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, metric in zip(axes, ['accuracy', 'macro_f1']):
        names = list(results['test'])
        values = [results['test'][n][metric] for n in names]
        ax.bar(names, values, color=['#a1aab3', '#638095', '#228c9c', '#d47732'])
        ax.set(ylim=(0, 1), ylabel=metric.replace('_', ' ').title(), title='Held-out measure groups')
        for i, value in enumerate(values):
            ax.text(i, value + .025, f'{value:.3f}', ha='center')
    fig.savefig(OUT / 'model_comparison.png'); plt.close(fig)
    short = ['Culture', 'Digital', 'Business', 'Inclusion', 'Infrastructure', 'Health', 'Education', 'Ecology']
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for ax, name in zip(axes, ['MLP', 'CNN']):
        cm = confusion_matrix(ytest, probabilities[name].argmax(1), labels=np.arange(len(classes)), normalize='true')
        ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=short, yticklabels=short,
               xlabel='Predicted category', ylabel='Inherited category', title=f'{name}: row-normalized confusion')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, f'{cm[i,j]:.2f}', ha='center', va='center', fontsize=8,
                        color='white' if cm[i,j] > .5 else 'black')
    fig.savefig(OUT / 'confusion_matrices.png'); plt.close(fig)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=OUT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--epochs', type=int, default=14)
    parser.add_argument('--train-limit', type=int, default=24000)
    parser.add_argument('--validation-limit', type=int, default=8000)
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    OUT.mkdir(exist_ok=True, parents=True)
    seed_all()
    if args.prepare or not (OUT / 'prepared_projects.csv.gz').exists():
        prepare(out=OUT)
    df = pd.read_csv(OUT / 'prepared_projects.csv.gz')
    classes = sorted(df.label.unique())
    label_ids = {c: i for i, c in enumerate(classes)}
    partitions = {s: sample_partition(df[df.split == s], limit) for s, limit in
                  [('train', args.train_limit), ('validation', args.validation_limit), ('test', None)]}
    pd.concat(partitions.values()).drop(columns=['titolo', 'text']).to_csv(OUT / 'experiment_split.csv.gz', index=False)
    y = {s: f.label.map(label_ids).to_numpy() for s, f in partitions.items()}
    texts = {s: f.text.tolist() for s, f in partitions.items()}
    tfidf = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=2, sublinear_tf=True, dtype=np.float32)
    x_mlp = {'train': tfidf.fit_transform(texts['train'])}
    x_mlp.update({s: tfidf.transform(texts[s]) for s in ['validation', 'test']})
    word_counts = Counter(w for text in texts['train'] for w in text.split())
    vocab = {'<PAD>': 0, '<UNK>': 1}
    vocab.update({w: i + 2 for i, (w, count) in enumerate(word_counts.most_common(11998)) if count >= 2})
    length = 80
    x_cnn = {s: encode(texts[s], vocab, length) for s in texts}
    with open(OUT / 'tfidf.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    (OUT / 'vocabulary.json').write_text(json.dumps(vocab, ensure_ascii=False))
    (OUT / 'classes.json').write_text(json.dumps(classes, ensure_ascii=False, indent=2))
    diagnostics = {}
    for s, ts in texts.items():
        words = [w for text in ts for w in text.split()]
        diagnostics[s] = {'rows': len(ts), 'groups': int(partitions[s].group.nunique()),
                          'class_counts': partitions[s].label.value_counts().to_dict(),
                          'oov_token_rate': sum(w not in vocab for w in words) / len(words),
                          'truncated_title_rate': float(np.mean([len(t.split()) > length for t in ts]))}
    counts = np.bincount(y['train'], minlength=len(classes))
    weights = np.sqrt(len(y['train']) / (len(classes) * counts)).astype(np.float32)
    weights /= weights.mean()
    monitor_idx, _ = train_test_split(np.arange(len(y['train'])), train_size=min(3000, len(y['train']) - 1),
                                    stratify=y['train'], random_state=SEED)
    results = {'seed': SEED, 'environment': {'python': platform.python_version(), 'torch': str(torch.__version__),
               'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
               'device': 'cpu', 'threads': 4, 'platform': platform.platform()},
               'classes': classes, 'diagnostics': diagnostics, 'sequence_length': length,
               'vocab_size': len(vocab), 'tfidf_features': len(tfidf.vocabulary_),
               'class_loss_weights': weights.tolist(), 'training_monitor_rows': len(monitor_idx), 'runs': {},
               'selected': {}, 'baseline_tuning': [], 'test': {}}
    print('EXPERIMENT', json.dumps(diagnostics, ensure_ascii=False), flush=True)
    # Baseline tuning uses validation only.
    start = time.perf_counter()
    baselines = []
    for c in [0.5, 2.0]:
        clf = LogisticRegression(C=c, class_weight='balanced', max_iter=400, solver='lbfgs', random_state=SEED)
        clf.fit(x_mlp['train'], y['train'])
        result = {'C': c, 'validation': scores(y['validation'], clf.predict_proba(x_mlp['validation']))}
        results['baseline_tuning'].append(result)
        baselines.append(clf)
    best_i = max(range(len(baselines)), key=lambda i: results['baseline_tuning'][i]['validation']['macro_f1'])
    logistic = baselines[best_i]
    results['logistic_seconds_including_tuning'] = time.perf_counter() - start
    results['logistic_selected_C'] = results['baseline_tuning'][best_i]['C']
    with open(OUT / 'logistic.pkl', 'wb') as f:
        pickle.dump(logistic, f)
    chosen_models = {}
    for kind, features in [('mlp', x_mlp), ('cnn', x_cnn)]:
        candidates = {}
        for variant, dropout, decay in [('plain', 0., 0.), ('regularized', .4, 1e-4)]:
            seed_all()
            model = (TitleMLP(features['train'].shape[1], len(classes), dropout) if kind == 'mlp'
                     else TitleCNN(len(vocab), len(classes), dropout))
            name = f'{kind}_{variant}'
            cfg = {'kind': kind, 'dropout': dropout, 'weight_decay': decay, 'lr': .001,
                   'batch_size': 256, 'max_epochs': args.epochs, 'patience': 3}
            model, result = fit_neural(name, model, features['train'], y['train'], features['validation'],
                                       y['validation'], weights, cfg, monitor_idx)
            results['runs'][name] = result
            candidates[name] = model
            (OUT / 'results_in_progress.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
        selected = max(candidates, key=lambda n: results['runs'][n]['validation']['macro_f1'])
        results['selected'][kind.upper()] = selected
        chosen_models[kind.upper()] = candidates[selected]
    # Freeze all selections before looking at held-out test predictions.
    (OUT / 'frozen_selection.json').write_text(json.dumps({'neural': results['selected'],
                   'logistic_C': results['logistic_selected_C']}, indent=2))
    majority = counts.argmax()
    majority_probs = np.zeros((len(y['test']), len(classes)))
    majority_probs[:, majority] = 1
    probabilities = {'Majority': majority_probs, 'Logistic': logistic.predict_proba(x_mlp['test']),
                     'MLP': predict(chosen_models['MLP'], x_mlp['test']),
                     'CNN': predict(chosen_models['CNN'], x_cnn['test'])}
    predictions = partitions['test'][['progetto_id', 'group', 'titolo', 'label']].copy()
    reports = []
    for name, probs in probabilities.items():
        result = scores(y['test'], probs)
        if name == 'Majority':
            result.pop('loss')  # Hard-class baseline: probabilistic loss is not meaningful.
        results['test'][name] = result
        pred = probs.argmax(axis=1)
        predictions[name + '_prediction'] = np.array(classes)[pred]
        predictions[name + '_confidence'] = probs.max(axis=1)
        report = classification_report(y['test'], pred, labels=np.arange(len(classes)), target_names=classes,
                                       output_dict=True, zero_division=0)
        reports.append(pd.DataFrame(report).T.assign(model=name).rename_axis('category').reset_index())
    pd.concat(reports).to_csv(OUT / 'classification_reports.csv', index=False)
    predictions.to_csv(OUT / 'test_predictions.csv.gz', index=False)
    results['bootstrap_mlp_minus_logistic'] = group_bootstrap(y['test'], probabilities['MLP'].argmax(1),
                      probabilities['Logistic'].argmax(1), partitions['test'].group.to_numpy(), len(classes))
    results['bootstrap_cnn_minus_mlp'] = group_bootstrap(y['test'], probabilities['CNN'].argmax(1),
                      probabilities['MLP'].argmax(1), partitions['test'].group.to_numpy(), len(classes))
    (OUT / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (OUT / 'results_in_progress.json').unlink(missing_ok=True)
    plots(results, classes, y['test'], probabilities)
    print('FINAL RESULTS', json.dumps(results['test'], indent=2), flush=True)


if __name__ == '__main__':
    main()
