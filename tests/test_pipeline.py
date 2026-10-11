"""Check the consequential invariants: cleaning, padding, and data isolation."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import pandas as pd
import torch
from prepare import clean_title
from train import TitleCNN


class PipelineTests(unittest.TestCase):
    def test_codes_removed_but_italian_words_retained(self):
        title = 'M1C1I1.04.05 CITTADINANZA DIGITALE – comunità E71F22003570006 2025'
        self.assertEqual(clean_title(title), 'cittadinanza digitale comunità')

    def test_cleaning_is_idempotent(self):
        title = 'SCUOLA*Ristrutturazione, accessibilità e città!'
        self.assertEqual(clean_title(clean_title(title)), clean_title(title))

    def test_cnn_predictions_ignore_extra_padding(self):
        torch.manual_seed(42)
        model = TitleCNN(12, 8, .4).eval()
        with torch.no_grad():
            short = model(torch.tensor([[2, 3, 4, 0, 0]]))
            long = model(torch.tensor([[2, 3, 4, 0, 0, 0, 0, 0, 0]]))
        torch.testing.assert_close(short, long, atol=1e-6, rtol=1e-6)

    def test_saved_split_has_no_identity_or_family_leakage(self):
        frame = pd.read_csv(ROOT / 'artifacts/prepared_projects.csv.gz')
        for column in ['text', 'progetto_id', 'cup', 'misura_id', 'family', 'group']:
            with self.subTest(column=column):
                self.assertLessEqual(frame.groupby(column).split.nunique().max(), 1)
        counts = frame.groupby(['label', 'split']).group.nunique()
        self.assertGreaterEqual(counts.min(), 2)
        self.assertEqual(frame.text.nunique(), len(frame))
        for _, part in frame.groupby('split'):
            self.assertEqual(set(part.label), set(frame.label))

    def test_sampling_preserves_splits_and_all_test_rows(self):
        source = pd.read_csv(ROOT / 'artifacts/prepared_projects.csv.gz', usecols=['progetto_id', 'split'])
        sampled = pd.read_csv(ROOT / 'artifacts/experiment_split.csv.gz', usecols=['progetto_id', 'split'])
        joined = sampled.merge(source, on='progetto_id', validate='one_to_one')
        self.assertTrue((joined.split_x == joined.split_y).all())
        self.assertEqual(set(sampled.loc[sampled.split == 'test', 'progetto_id']),
                         set(source.loc[source.split == 'test', 'progetto_id']))


if __name__ == '__main__':
    unittest.main()
