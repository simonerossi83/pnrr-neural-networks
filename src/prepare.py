"""Auditable title cleaning, inherited labels, and leakage-aware group splits."""
from pathlib import Path
import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def clean_title(value):
    text = unicodedata.normalize('NFKC', str(value)).lower()
    # Strip embedded mission/component/investment codes before punctuation removal.
    text = re.sub(r'\bm\s*\d+\s*c\s*\d+(?:\s*i\s*[\d.]+)?', ' ', text)
    # CUPs, local identifiers, dates, and numeric references are not language features.
    text = re.sub(r'\b\w*\d\w*\b', ' ', text)
    text = re.sub(r'[^a-zàèéìòóù\s]', ' ', text)
    return ' '.join(text.split())


def ancestor(key, parents):
    seen = set()
    while key in parents and pd.notna(parents[key]):
        if key in seen:
            raise ValueError('Cycle in source hierarchy')
        seen.add(key)
        key = int(parents[key])
    return int(key)


def prepare(out=ROOT / 'artifacts', seed=42):
    out.mkdir(exist_ok=True, parents=True)
    audit = {'seed': seed, 'source_sha256': {},
             'minimum_class_titles': 500, 'minimum_class_groups': 6}
    for name in ['progetti', 'misure', 'temi']:
        path = ROOT / 'data' / f'{name}.csv'
        audit['source_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    p = pd.read_csv(ROOT / 'data/progetti.csv', usecols=['progetto_id', 'titolo', 'cup', 'misura_id'])
    m = pd.read_csv(ROOT / 'data/misure.csv')
    t = pd.read_csv(ROOT / 'data/temi.csv')
    audit['raw_projects'] = len(p)
    audit['missing_titles'] = int(p.titolo.isna().sum())
    audit['duplicate_project_ids'] = int(p.progetto_id.duplicated().sum())
    if p.progetto_id.duplicated().any():
        raise ValueError('Resolve duplicate project IDs before proceeding')
    p = p.merge(m[['id', 'tema']], left_on='misura_id', right_on='id', how='left', validate='many_to_one')
    audit['unmatched_measure_ids'] = sorted(p.loc[p.tema.isna(), 'misura_id'].unique().tolist())
    audit['unmatched_measure_rows'] = int(p.tema.isna().sum())
    p = p.dropna(subset=['tema', 'titolo']).copy()
    theme_parents = t.set_index('id').parent_id.to_dict()
    measure_parents = m.set_index('id').madre_id.to_dict()
    names = t.set_index('id').descrizione.to_dict()
    p['label'] = p.tema.map(lambda x: names[ancestor(int(x), theme_parents)])
    p['family'] = p.misura_id.map(lambda x: ancestor(int(x), measure_parents))
    p['text'] = p.titolo.map(clean_title)
    short = p.text.str.split().str.len() < 3
    audit['titles_under_three_tokens'] = int(short.sum())
    p = p.loc[~short].copy()
    conflict = p.groupby('text').label.nunique()
    conflict = set(conflict[conflict > 1].index)
    audit['conflicting_title_strings'] = len(conflict)
    audit['conflicting_title_rows_removed'] = int(p.text.isin(conflict).sum())
    p = p.loc[~p.text.isin(conflict)].copy()
    # Union measure families when any remaining title or CUP links them.
    parents = {int(x): int(x) for x in p.family.unique()}

    def find(x):
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for col in ['text', 'cup']:
        first = {}
        for key, family in zip(p[col], p.family):
            if pd.isna(key) or not str(key).strip():
                continue
            if key in first:
                union(first[key], int(family))
            else:
                first[key] = int(family)
    p['group'] = p.family.map(find)
    audit['duplicate_clean_titles_removed'] = int(p.text.duplicated().sum())
    p = p.drop_duplicates('text').copy()
    support = p.groupby('label').agg(titles=('text', 'size'), independent_groups=('group', 'nunique'))
    audit['class_support_before_filter'] = support.reset_index().to_dict('records')
    eligible = support.index[(support.titles >= 500) & (support.independent_groups >= 6)]
    audit['excluded_classes'] = support.loc[~support.index.isin(eligible)].reset_index().to_dict('records')
    p = p[p.label.isin(eligible)].copy()
    if len(eligible) < 2:
        raise ValueError('Insufficient independent class groups for a defensible split')
    # Search labels/group sizes only, never model performance, for a feasible 60/20/20 split.
    labels = sorted(eligible)
    table = pd.crosstab(p.group, p.label).reindex(columns=labels, fill_value=0)
    counts = table.to_numpy()
    target = np.array([.6, .2, .2])
    rng = np.random.default_rng(seed)
    if np.any((counts > 0).sum(axis=1) != 1):
        raise ValueError('Mixed-label components require a joint group-split optimizer')
    assignment = np.zeros(len(table), dtype=int)
    class_errors = []
    for column in range(len(labels)):
        indices = np.flatnonzero(counts[:, column])
        n_holdout = max(2, round(len(indices) * .2))
        best = None
        for _ in range(5000):
            order = rng.permutation(indices)
            proposed = np.zeros(len(table), dtype=int)
            proposed[order[:n_holdout]] = 1
            proposed[order[n_holdout:2*n_holdout]] = 2
            parts = np.array([counts[indices[proposed[indices] == s], column].sum() for s in range(3)])
            if parts.min() < 20 or parts[0] < parts[1:].max():
                continue
            error = float(np.mean((parts / parts.sum() - target) ** 2))
            if best is None or error < best[0]:
                best = (error, proposed[indices].copy())
        if best is None:
            raise ValueError(f'No feasible split for {labels[column]}')
        assignment[indices] = best[1]
        class_errors.append(best[0])
    mapping = dict(zip(table.index, np.array(['train', 'validation', 'test'])[assignment]))
    p['split'] = p.group.map(mapping)
    audit['split_balance_error'] = float(np.mean(class_errors))
    audit['retained_unique_titles'] = len(p)
    audit['classes'] = labels
    audit['split_counts'] = pd.crosstab(p.label, p.split).to_dict()
    audit['split_group_counts'] = p.groupby(['split', 'label']).group.nunique().unstack().to_dict('index')
    for col in ['group', 'family', 'misura_id', 'cup', 'text']:
        if p.groupby(col).split.nunique().max() > 1:
            raise AssertionError(f'Split leakage: {col}')
    audit['leakage_assertions_passed'] = ['group', 'family', 'misura_id', 'cup', 'text']
    # Complete cleaned split manifest is preserved, even if training is subsampled later.
    cols = ['progetto_id', 'cup', 'misura_id', 'family', 'group', 'titolo', 'text', 'label', 'split']
    p[cols].to_csv(out / 'prepared_projects.csv.gz', index=False)
    (out / 'data_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return p, audit


if __name__ == '__main__':
    prepare()
