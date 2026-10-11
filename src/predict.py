"""Classify a new Italian project title using a saved, validation-selected model."""
import argparse
import json
import pickle

import numpy as np
import torch

from prepare import clean_title
from train import OUT, TitleCNN, TitleMLP, encode, predict


def classify(title, kind='MLP'):
    cleaned = clean_title(title)
    if len(cleaned.split()) < 3:
        raise ValueError('Provide at least three meaningful words after cleaning.')
    results = json.loads((OUT / 'results.json').read_text())
    classes = results['classes']
    selected = results['selected'][kind]
    checkpoint = torch.load(OUT / f'{selected}.pt', map_location='cpu', weights_only=True)
    dropout = checkpoint['config']['dropout']
    if kind == 'MLP':
        # Only deserialize the locally generated, trusted vectorizer.
        with open(OUT / 'tfidf.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        x = vectorizer.transform([cleaned])
        model = TitleMLP(len(vectorizer.vocabulary_), len(classes), dropout)
    else:
        vocab = json.loads((OUT / 'vocabulary.json').read_text())
        x = encode([cleaned], vocab, results['sequence_length'])
        model = TitleCNN(len(vocab), len(classes), dropout)
    model.load_state_dict(checkpoint['state_dict'])
    probabilities = predict(model, x)[0]
    order = np.argsort(probabilities)[::-1][:3]
    return {'title': title, 'cleaned_title': cleaned, 'model': selected,
            'top_categories': [{'category': classes[i], 'probability': float(probabilities[i])} for i in order],
            'note': 'Uncalibrated model probabilities; predicts inherited programme themes, not verified project purpose.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('title')
    parser.add_argument('--model', choices=['MLP', 'CNN'], default='MLP')
    args = parser.parse_args()
    print(json.dumps(classify(args.title, args.model), indent=2, ensure_ascii=False))
