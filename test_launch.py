import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import sys

import launch

SPEC=importlib.util.spec_from_file_location('runtime_manager',Path(__file__).parent/'setup/runtime_manager.py')
runtime=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(runtime)


class LaunchTests(unittest.TestCase):
    def test_pulled_update_restarts_os_bootstrap_before_environment_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            updater=SimpleNamespace(update_repository=lambda root:dict(status='updated'))
            with patch.object(runtime,'ROOT',root),patch.object(sys,'argv',['runtime_manager.py','--uv','/uv','--app','manual','--smoke-test']),patch.dict(sys.modules,updater=updater),patch.dict(os.environ,FLY_UPDATE_DONE=''),patch.object(runtime,'ensure_environment') as ensure,patch.object(runtime.subprocess,'call',return_value=0) as call:
                self.assertEqual(runtime.main(),0)
            ensure.assert_not_called()
            self.assertIn('--smoke-test',call.call_args.args[0])
            self.assertEqual(call.call_args.kwargs['env']['FLY_UPDATE_DONE'],'1')

    def tree(self,folder):
        root=Path(folder).resolve();setup=root/'setup';setup.mkdir()
        (setup/'runtime-lock.json').write_text(json.dumps(dict(python_version='3.12.10')))
        (setup/'requirements.in').write_text('numpy==1.26.4\n')
        (setup/'requirements.lock').write_text('numpy==1.26.4 --hash=sha256:example\n')
        base=root/'.runtime/python/cpython-3.12.10'
        for directory,filename in (('tcl8.6','init.tcl'),('tk8.6','tk.tcl')):
            path=base/'lib'/directory;path.mkdir(parents=True);(path/filename).write_text('test')
        return root,base

    def test_conda_python_and_library_paths_are_removed(self):
        source=dict(CONDA_PREFIX='/conda',CONDA_SHLVL='2',MAMBA_ROOT_PREFIX='/mamba',PYTHONPATH='/other',
                    PYTHONHOME='/conda',VIRTUAL_ENV='/old',TCL_LIBRARY='/conda/tcl',TK_LIBRARY='/conda/tk',
                    UV_PYTHON='/conda/python',PATH='/conda/bin',HOME='/home/test')
        env=runtime.clean_environment(Path('/project'),source)
        for key in ('CONDA_PREFIX','CONDA_SHLVL','MAMBA_ROOT_PREFIX','PYTHONPATH','PYTHONHOME','TCL_LIBRARY','TK_LIBRARY','VIRTUAL_ENV','UV_PYTHON'):
            self.assertNotIn(key,env)
        self.assertNotIn('/conda',env['PATH']);self.assertEqual(source['CONDA_PREFIX'],'/conda')
        self.assertEqual(env['HOME'],'/home/test')

    def test_missing_packages_are_synced_with_hashes_in_dedicated_venv(self):
        with tempfile.TemporaryDirectory() as folder:
            root,base=self.tree(folder)
            ready=dict(python='3.12.10',conda=False,base_prefix=str(base),missing=[])
            with patch.object(runtime.platform,'python_version',return_value='3.12.10'),patch.object(runtime.sys,'base_prefix',str(base)),patch.object(runtime,'probe',side_effect=[dict(ready,missing=['numpy']),ready]),patch.object(runtime.subprocess,'run') as run:
                python,env,_=runtime.ensure_environment(Path('/uv'),root)
            self.assertTrue(python.is_relative_to(root/'.runtime/venvs'))
            self.assertEqual(env['VIRTUAL_ENV'],str(python.parent.parent))
            self.assertEqual(env['TCL_LIBRARY'],str(base/'lib/tcl8.6'))
            calls=[call.args[0] for call in run.call_args_list]
            self.assertEqual(calls[0][1],'venv')
            self.assertIn('--require-hashes',calls[1]);self.assertEqual(calls[1][1:3],['pip','sync'])
            self.assertEqual(calls[2][1:3],['pip','check'])

    def test_wrong_python_is_rejected_without_changing_old_venv_or_learning(self):
        with tempfile.TemporaryDirectory() as folder:
            root,base=self.tree(folder)
            old=root/'.venv';old.mkdir();(old/'keep').write_text('original')
            learning=root/'fly_activity/learning';learning.mkdir(parents=True);(learning/'current.json').write_text('original')
            with patch.object(runtime.platform,'python_version',return_value='3.14.0'),patch.object(runtime.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'3.12.10'):runtime.ensure_environment('/uv',root)
                run.assert_not_called()
            self.assertEqual((old/'keep').read_text(),'original')
            self.assertEqual((learning/'current.json').read_text(),'original')

    def test_external_or_conda_interpreter_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            root,_=self.tree(folder)
            with patch.object(runtime.platform,'python_version',return_value='3.12.10'),patch.object(runtime.sys,'base_prefix',str(root/'conda')),patch.object(runtime.subprocess,'run') as run:
                with self.assertRaisesRegex(RuntimeError,'Conda'):runtime.ensure_environment('/uv',root)
                run.assert_not_called()

    def test_platform_launchers_delegate_without_using_shell_profiles(self):
        mac=launch.command('manual',['--smoke-test'],root='/project',platform='darwin')
        self.assertEqual(mac[:2],['/bin/zsh','-f']);self.assertIn('--app',mac)
        win=launch.command('manual',root='/project',platform='win32')
        self.assertIn('-NoProfile',win);self.assertIn('Bypass',win)
        with self.assertRaises(ValueError):launch.command('invalid',platform='darwin')

    def test_lock_contains_exact_versions_and_hashes_for_all_requirements(self):
        expected=runtime.expected_versions()
        text=(Path(__file__).parent/'setup/requirements.lock').read_text()
        for package,version in expected.items():self.assertIn(package.lower()+'=='+version,text.lower())
        self.assertIn('--hash=sha256:',text)


if __name__=='__main__':unittest.main()
