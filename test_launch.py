import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import launch


class LaunchTests(unittest.TestCase):
    def tree(self,folder):
        root=Path(folder);(root/'fly_activity').mkdir();(root/'fly_activity/requirements.txt').write_text('numpy>=1.24\n')
        return root

    def test_ready_environment_does_not_install(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.tree(folder)
            with patch('launch.probe',return_value=dict(supported=True,tk=True,missing=[])),patch('launch.subprocess.run') as run:
                self.assertEqual(launch.ensure_environment(root,python='/python'),Path('/python'))
                run.assert_not_called()

    def test_missing_requirements_install_only_inside_project_venv(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.tree(folder)
            with patch('launch.probe',side_effect=[dict(supported=True,tk=True,missing=['numpy>=1.24']),dict(supported=True,tk=True,missing=[])]),patch('launch.subprocess.run') as run:
                selected=launch.ensure_environment(root,python='/python')
                calls=[call.args[0] for call in run.call_args_list]
                self.assertEqual(calls[0],['/python','-m','venv',str(root/'.venv')])
                self.assertEqual(calls[1][0],str(selected))
                self.assertEqual(calls[1][-2:],['-r',str(root/'fly_activity/requirements.txt')])
                self.assertEqual(selected.parent.parent,root/'.venv')

    def test_missing_tk_and_failed_install_do_not_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root=self.tree(folder)
            with patch('launch.probe',return_value=dict(supported=True,tk=False,missing=[])),patch('launch.subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError,'Tkinter'):launch.ensure_environment(root,python='/python')
                run.assert_not_called()
            with patch('launch.probe',return_value=dict(supported=True,tk=True,missing=['numpy>=1.24'])),patch('launch.subprocess.run',side_effect=subprocess.CalledProcessError(1,'venv')):
                with self.assertRaises(subprocess.CalledProcessError):launch.ensure_environment(root,python='/python')


if __name__=='__main__':unittest.main()
