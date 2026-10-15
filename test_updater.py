import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC=importlib.util.spec_from_file_location('updater',Path(__file__).parent/'setup/updater.py')
updater=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(updater)
GIT=shutil.which('git')


@unittest.skipUnless(GIT,'Git is required for local repository integration tests')
class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.folder=Path(self.temporary.name)
        # Isolate Git identity, defaults, hooks and credentials from the developer's setup.
        self.env={k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
        config=self.folder/'gitconfig';config.write_text('[user]\n name = Test\n email = test@example.com\n')
        self.env.update(GIT_CONFIG_GLOBAL=str(config),GIT_CONFIG_NOSYSTEM='1')
        self.environment=patch.dict(os.environ,self.env,clear=True);self.environment.start();self.addCleanup(self.environment.stop)
        self.origin=self.folder/'origin.git';self.seed=self.folder/'seed';self.local=self.folder/'app'
        self.git(self.folder,'init','--bare',str(self.origin))
        self.git(self.folder,'init','-b','main',str(self.seed))
        (self.seed/'app.py').write_text('old')
        learning=self.seed/'fly_activity/learning';learning.mkdir(parents=True)
        (learning/'examples.sqlite3').write_bytes(b'original teachers')
        self.git(self.seed,'add','.');self.git(self.seed,'commit','-m','initial')
        self.git(self.seed,'remote','add','origin',str(self.origin))
        self.git(self.seed,'push','-u','origin','main')
        self.git(self.folder,'clone','-b','main',str(self.origin),str(self.local))

    def git(self,root,*args):
        return subprocess.run([GIT,'-C',str(root),*args],check=True,capture_output=True,text=True,env=self.env).stdout.strip()

    def remote_update(self,path='app.py',content='new'):
        (self.seed/path).write_text(content)
        self.git(self.seed,'add',path);self.git(self.seed,'commit','-m','remote change');self.git(self.seed,'push')

    def update(self,root=None):return updater.update_repository(root or self.local,git=GIT)

    def test_updates_code_while_preserving_local_teachers_and_models(self):
        learning=self.local/'fly_activity/learning'
        (learning/'examples.sqlite3').write_bytes(b'new local teachers')
        (learning/'model_local.json').write_text('local model')
        self.remote_update()
        result=self.update()
        self.assertEqual(result['status'],'updated')
        self.assertEqual((self.local/'app.py').read_text(),'new')
        self.assertEqual((learning/'examples.sqlite3').read_bytes(),b'new local teachers')
        self.assertEqual((learning/'model_local.json').read_text(),'local model')
        saved=json.loads((self.local/'.runtime/update_status.json').read_text())
        self.assertEqual(saved['current_head'],self.git(self.seed,'rev-parse','HEAD'))
        self.assertEqual(self.update()['status'],'current')

    def test_conflict_does_not_stash_or_overwrite_even_with_autostash_configured(self):
        self.git(self.local,'config','pull.rebase','true')
        self.git(self.local,'config','rebase.autoStash','true')
        self.git(self.local,'config','merge.autoStash','true')
        (self.local/'app.py').write_text('local edit')
        before=self.git(self.local,'rev-parse','HEAD');self.remote_update()
        self.assertEqual(self.update()['status'],'failed')
        self.assertEqual((self.local/'app.py').read_text(),'local edit')
        self.assertEqual(self.git(self.local,'rev-parse','HEAD'),before)
        self.assertEqual(self.git(self.local,'stash','list'),'')

    def test_learning_conflict_retains_local_data(self):
        path='fly_activity/learning/examples.sqlite3'
        (self.local/path).write_bytes(b'local teachers')
        self.remote_update(path,'remote teachers')
        self.assertEqual(self.update()['status'],'failed')
        self.assertEqual((self.local/path).read_bytes(),b'local teachers')

    def test_divergent_history_is_not_merged(self):
        (self.local/'local.txt').write_text('keep')
        self.git(self.local,'add','local.txt');self.git(self.local,'commit','-m','local')
        before=self.git(self.local,'rev-parse','HEAD');self.remote_update()
        self.assertEqual(self.update()['status'],'failed')
        self.assertEqual(self.git(self.local,'rev-parse','HEAD'),before)
        self.assertEqual((self.local/'app.py').read_text(),'old')

    def test_unavailable_remote_does_not_block_existing_version(self):
        self.git(self.local,'remote','set-url','origin',str(self.folder/'absent.git'))
        self.assertEqual(self.update()['status'],'failed')
        self.assertEqual((self.local/'app.py').read_text(),'old')

    def test_zip_inside_another_repository_does_not_update_parent(self):
        zip_root=self.local/'zip';zip_root.mkdir();self.remote_update()
        self.assertEqual(self.update(zip_root)['status'],'skipped')
        self.assertEqual((self.local/'app.py').read_text(),'old')

    def test_missing_git_and_missing_upstream_are_skipped(self):
        with patch.object(updater,'find_git',return_value=None):
            self.assertEqual(updater.update_repository(self.local)['status'],'skipped')
        self.git(self.local,'branch','--unset-upstream')
        self.assertEqual(self.update()['status'],'skipped')

    def test_detached_head_is_skipped(self):
        self.git(self.local,'checkout','--detach')
        self.assertEqual(self.update()['status'],'skipped')

    def test_timeout_falls_back_to_current_version(self):
        with patch.object(updater.subprocess,'run',side_effect=subprocess.TimeoutExpired('git',30)):
            self.assertEqual(self.update()['status'],'failed')
        self.assertEqual((self.local/'app.py').read_text(),'old')


if __name__=='__main__':unittest.main()
