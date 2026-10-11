import ast
from contextlib import contextmanager
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

BUILDER = Path(__file__).resolve().parents[2] / 'src/build_merged_notebook.py'
constants = [n.value for n in ast.walk(ast.parse(BUILDER.read_text())) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
DEPENDENCIES = next(s for s in constants if s.startswith('import importlib.util'))
DATA_SETUP = next(s for s in constants if s.startswith('from pathlib import Path\nfrom collections import Counter'))
DATA_PRELUDE = DATA_SETUP[:DATA_SETUP.index('OUT =')]
REQUIRED = ['progetti.csv', 'misure.csv', 'temi.csv']

@contextmanager
def isolated_directory():
    old = Path.cwd()
    with tempfile.TemporaryDirectory(prefix='pnrr-colab-setup-audit-') as location:
        os.chdir(location)
        try:
            yield Path(location)
        finally:
            os.chdir(old)

class Shell:
    def __init__(self):
        self.calls = []
    def run_line_magic(self, name, argument):
        self.calls.append((name, argument))

class ColabSetupAudit(unittest.TestCase):
    def dependencies(self, colab, missing=(), torch_version='2.11.0+cpu'):
        shell = Shell()
        def metadata(package):
            if package in missing:
                raise importlib.metadata.PackageNotFoundError(package)
            return torch_version if package == 'torch' else '1.0.0'
        state = {'get_ipython': lambda: shell}
        with patch('importlib.util.find_spec', return_value=object() if colab else None), patch('importlib.metadata.version', side_effect=metadata):
            exec(DEPENDENCIES, state)
        return state, shell

    def test_colab_retains_preinstalled_packages(self):
        state, shell = self.dependencies(True)
        self.assertTrue(state['IS_COLAB'])
        self.assertEqual(shell.calls, [])

    def test_missing_shap_installed_in_colab(self):
        _, shell = self.dependencies(True, missing=['shap'])
        self.assertEqual(shell.calls, [('pip', 'install -q shap==0.49.1')])

    def test_old_torch_installs_one_shell_quoted_requirement(self):
        _, shell = self.dependencies(True, torch_version='2.5.1')
        name, argument = shell.calls[0]
        lexer = shlex.shlex(argument, posix=True, punctuation_chars='<>')
        lexer.whitespace_split = True
        self.assertEqual(name, 'pip')
        self.assertEqual(list(lexer), ['install', '-q', 'torch>=2.6,<3'])

    def test_missing_local_dependency_has_clear_error(self):
        with self.assertRaisesRegex(ImportError, 'python -m pip install shap==0.49.1'):
            self.dependencies(False, missing=['shap'])

    def test_colab_files_panel_root_uploads_are_found(self):
        with isolated_directory() as root:
            for name in REQUIRED:
                (root / name).write_bytes(name.encode())
            state = {'IS_COLAB': True}
            exec(DATA_PRELUDE, state)
            for name in REQUIRED:
                self.assertEqual((root / 'data' / name).read_bytes(), name.encode())

    def test_missing_colab_files_use_upload_picker(self):
        module = ModuleType('google.colab')
        files = ModuleType('google.colab.files')
        files.upload = lambda: {name: name.encode() for name in REQUIRED}
        module.files = files
        with isolated_directory() as root, patch.dict(sys.modules, {'google.colab': module, 'google.colab.files': files}):
            state = {'IS_COLAB': True}
            exec(DATA_PRELUDE, state)
            self.assertTrue(all((root / 'data' / name).is_file() for name in REQUIRED))

    def test_partial_upload_reports_remaining_missing_file(self):
        module = ModuleType('google.colab')
        files = ModuleType('google.colab.files')
        files.upload = lambda: {'progetti.csv': b'example'}
        module.files = files
        with isolated_directory(), patch.dict(sys.modules, {'google.colab': module, 'google.colab.files': files}):
            with self.assertRaisesRegex(FileNotFoundError, 'misure.csv'):
                exec(DATA_PRELUDE, {'IS_COLAB': True})

    def test_local_three_csv_directory_is_used(self):
        with isolated_directory() as root:
            (root / 'data').mkdir()
            for name in REQUIRED:
                (root / 'data' / name).write_bytes(name.encode())
            state = {'IS_COLAB': False}
            exec(DATA_PRELUDE, state)
            self.assertEqual(state['DATA_DIR'], (root / 'data').resolve())
            self.assertEqual(sorted(p.name for p in (root / 'data').iterdir()), sorted(REQUIRED))

    def test_local_missing_data_has_clear_error(self):
        with isolated_directory():
            with self.assertRaisesRegex(FileNotFoundError, 'Upload all three CSVs or set DATA_DIR'):
                exec(DATA_PRELUDE, {'IS_COLAB': False})

if __name__ == '__main__':
    unittest.main(verbosity=2)
