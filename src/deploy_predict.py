"""Portable CPU inference: copy this file with the exported deployment bundle."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch


def load_preprocessor(bundle):
    # Import only the bundled title cleaner, never the training implementation.
    import importlib.util
    spec = importlib.util.spec_from_file_location('bundle_preprocessing', bundle / 'preprocessing.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.clean_title


def classify_batch(titles, kind='MLP', bundle=None):
    bundle = Path(bundle) if bundle is not None else Path(__file__).resolve().parent
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if kind not in manifest['models']:
        raise ValueError('Model must be MLP or CNN')
    if not titles or len(titles) > manifest['max_batch_size']:
        raise ValueError('Supply between 1 and 512 titles per batch')
    cleaner = load_preprocessor(bundle)
    cleaned = [cleaner(t) for t in titles]
    if any(len(t.split()) < 3 for t in cleaned):
        raise ValueError('Each title needs at least three meaningful words after cleaning')
    if kind == 'MLP':
        # Load only the trusted, locally generated bundle; pickle is not a safe exchange format.
        with (bundle / 'tfidf.pkl').open('rb') as handle:
            vectorizer = pickle.load(handle)
        features = vectorizer.transform(cleaned)
        if np.any(np.asarray(features.getnnz(axis=1)) == 0):
            raise ValueError('A title has no known TF-IDF features; human review is required')
        x = torch.from_numpy(features.toarray().astype(np.float32))
    else:
        vocab = json.loads((bundle / 'vocabulary.json').read_text())
        features = np.zeros((len(titles), manifest['sequence_length']), dtype=np.int64)
        for row, title in enumerate(cleaned):
            ids = [vocab.get(w, 1) for w in title.split()[:manifest['sequence_length']]]
            if not any(i > 1 for i in ids):
                raise ValueError('A title has no known tokens; human review is required')
            features[row, :len(ids)] = ids
        x = torch.from_numpy(features)
    model = torch.export.load(bundle / manifest['models'][kind]['file']).module()
    with torch.inference_mode():
        probabilities = model(x).softmax(dim=1).numpy()
    classes = manifest['classes']
    return [
        {'title': title, 'cleaned_title': clean, 'model': kind,
         'top_categories': [{'category': classes[i], 'probability': float(p[i])}
                            for i in np.argsort(p)[::-1][:3]],
         'note': 'Uncalibrated inherited-theme prediction; human review required.'}
        for title, clean, p in zip(titles, cleaned, probabilities)
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('title', nargs='+')
    parser.add_argument('--model', choices=['MLP', 'CNN'], default='MLP')
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(classify_batch(args.title, args.model, args.bundle), ensure_ascii=False, indent=2))
