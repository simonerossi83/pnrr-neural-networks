"""Verify graph round-trips, independent bundle execution, and input rejection."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import torch
from deploy_predict import classify_batch
from export_models import export_models
from predict import classify


class DeploymentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.bundle = Path(cls.temporary.name) / 'bundle'
        # Earlier trusted checkpoints keep this test independent of retraining.
        cls.manifest = export_models(ROOT / 'artifacts', cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exported_graphs_match_checkpoint_logits_across_batches(self):
        for kind, details in self.manifest['models'].items():
            with self.subTest(model=kind):
                self.assertLessEqual(details['max_absolute_logit_error'], 1e-5)
                self.assertIn(1, details['parity_batch_sizes'])
                self.assertIn(512, details['parity_batch_sizes'])
                self.assertGreater(details['distinct_parity_titles'], 24)

    def test_copied_bundle_cli_works_outside_project_and_matches_predictions(self):
        moved = Path(self.temporary.name) / 'copied-deployment'
        shutil.copytree(self.bundle, moved)
        title = 'Riqualificazione energetica della scuola primaria comunale'
        for kind in ['MLP', 'CNN']:
            with self.subTest(model=kind):
                result = subprocess.run([sys.executable, str(moved / 'predict.py'), '--model', kind, title],
                                        cwd=self.temporary.name, capture_output=True, text=True, check=True)
                deployed = json.loads(result.stdout)[0]['top_categories']
                original = classify(title, kind)['top_categories']
                self.assertEqual([r['category'] for r in deployed], [r['category'] for r in original])
                for a, b in zip(deployed, original):
                    self.assertAlmostEqual(a['probability'], b['probability'], places=5)

    def test_invalid_or_unknown_inputs_are_rejected(self):
        for kind in ['MLP', 'CNN']:
            for titles in [[], ['1234'], ['qzxqzx vjkvjk zzzxxqq'], ['scuola primaria comunale'] * 513]:
                with self.subTest(model=kind, rows=len(titles)):
                    with self.assertRaises(ValueError):
                        classify_batch(titles, kind, self.bundle)


if __name__ == '__main__':
    unittest.main()
