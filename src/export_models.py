"""Export selected neural models and verify serialized graph parity on real titles."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import pickle
import shutil

import numpy as np
import pandas as pd
import torch

from prepare import clean_title
from train import TitleCNN, TitleMLP, encode


def export_models(artifacts, bundle=None):
    artifacts = Path(artifacts)
    bundle = Path(bundle) if bundle is not None else artifacts / 'deployment'
    bundle.mkdir(parents=True, exist_ok=True)
    results = json.loads((artifacts / 'results.json').read_text())
    with (artifacts / 'tfidf.pkl').open('rb') as handle:
        vectorizer = pickle.load(handle)
    vocab = json.loads((artifacts / 'vocabulary.json').read_text())
    prepared = pd.read_csv(artifacts / 'prepared_projects.csv.gz', usecols=['text', 'split', 'label'])
    # Include each class in each partition, plus longer titles, without tuning on these outputs.
    checks = prepared.groupby(['split', 'label'], sort=True).head(4)
    texts = checks.text.tolist()
    texts += ['riqualificazione energetica scuola comunità', 'realizzazione nuovo ospedale comunità',
              'servizi digitali comunità ' * 40]
    manifest = {'format': 'torch.export ExportedProgram (.pt2)', 'torch_version': str(torch.__version__),
                'classes': results['classes'], 'sequence_length': results['sequence_length'],
                'max_batch_size': 512, 'models': {},
                'preprocessing': 'Bundled cleaner and train-fitted TF-IDF/vocabulary; no refitting.',
                'limitations': 'CPU Python/PyTorch demonstration, not a hosted service or portable hardware binary. '
                               'Uncalibrated closed-set predictions; not suitable for funding decisions.'}
    for kind in ['MLP', 'CNN']:
        selected = results['selected'][kind]
        checkpoint = torch.load(artifacts / f'{selected}.pt', map_location='cpu', weights_only=True)
        if kind == 'MLP':
            model = TitleMLP(results['tfidf_features'], len(results['classes']), checkpoint['config']['dropout'])
            features = torch.from_numpy(vectorizer.transform(texts).toarray())
        else:
            model = TitleCNN(len(vocab), len(results['classes']), checkpoint['config']['dropout'])
            features = torch.from_numpy(encode(texts, vocab, results['sequence_length']))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        batch = torch.export.Dim('batch', min=1, max=512)
        exported = torch.export.export(model, (features[:2],), dynamic_shapes={'x': {0: batch}})
        path = bundle / f'{kind.lower()}.pt2'
        torch.export.save(exported, path)
        restored = torch.export.load(path).module()
        max_error = 0.0
        with torch.inference_mode():
            for size in [1, 2, len(texts), 512]:
                x = features[torch.arange(size) % len(features)]
                original, deployed = model(x), restored(x)
                torch.testing.assert_close(original, deployed, atol=1e-5, rtol=1e-5)
                assert torch.equal(original.argmax(1), deployed.argmax(1))
                max_error = max(max_error, (original - deployed).abs().max().item())
        manifest['models'][kind] = {'file': path.name, 'checkpoint': selected,
            'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'max_absolute_logit_error': max_error, 'parity_batch_sizes': [1, 2, len(texts), 512],
            'distinct_parity_titles': len(texts)}
    for name in ['tfidf.pkl', 'vocabulary.json']:
        shutil.copy2(artifacts / name, bundle / name)
    (bundle / 'preprocessing.py').write_text('import re\nimport unicodedata\n\n' + inspect.getsource(clean_title))
    shutil.copy2(Path(__file__).with_name('deploy_predict.py'), bundle / 'predict.py')
    shutil.copy2(Path(__file__).resolve().parents[1] / 'requirements.txt', bundle / 'requirements.txt')
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/merged'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(export_models(args.artifacts), indent=2, ensure_ascii=False))
