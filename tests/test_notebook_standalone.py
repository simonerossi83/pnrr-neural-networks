"""Guard the portable notebook contract and repeatable authoring workflow."""
import ast
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import shutil
import tempfile
import unittest

import nbformat

ROOT = Path(__file__).resolve().parents[1]


class StandaloneNotebookTests(unittest.TestCase):
    def test_project_code_is_defined_in_executable_cells_without_local_imports(self):
        nb = nbformat.read(ROOT / 'pnrr_complete_workflow.ipynb', as_version=4)
        nbformat.validate(nb)
        definitions = set()
        for cell in nb.cells:
            if cell.cell_type != 'code':
                continue
            tree = ast.parse(cell.source)
            definitions.update(n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn((node.module or '').split('.')[0],
                                     {'prepare', 'train', 'predict', 'export_models', 'deploy_predict'})
            for forbidden in ['sys.path', '__file__', 'inspect.getsource', 'subprocess', 'src/', 'artifacts/']:
                self.assertNotIn(forbidden, cell.source)
        self.assertTrue({'clean_title', 'prepare', 'TitleMLP', 'TitleCNN', 'fit_neural',
                         'plots', 'export_current_models', 'classify_exported'} <= definitions)

    def test_authoring_twice_does_not_duplicate_sections_or_restore_dependencies(self):
        spec = importlib.util.spec_from_file_location('notebook_builder', ROOT / 'src/build_merged_notebook.py')
        builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(builder)
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'standalone.ipynb'
            shutil.copy2(ROOT / 'pnrr_complete_workflow.ipynb', target)
            builder.TARGET = target
            with redirect_stdout(io.StringIO()):
                builder.build()
                first = nbformat.read(target, as_version=4)
                builder.build()
                second = nbformat.read(target, as_version=4)
            self.assertEqual([c.source for c in first.cells], [c.source for c in second.cells])
            self.assertEqual(sum(c.source.startswith('## 11.') for c in second.cells), 1)


if __name__ == '__main__':
    unittest.main()
