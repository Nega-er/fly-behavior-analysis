import json
import os
from contextlib import closing
from pathlib import Path
import sqlite3
import sys
import unittest
from unittest.mock import patch
import subprocess

sys.path.insert(0,str(Path(__file__).parent/'setup'))
from data_sync import synchronize_repository,shared_path
from test_updater import UpdateTests,GIT


@unittest.skipUnless(GIT,'Git required')
class DataSyncTests(unittest.TestCase):
    git=UpdateTests.git
    remote_update=UpdateTests.remote_update

    def setUp(self):
        UpdateTests.setUp(self)
        db=self.seed/'fly_activity/learning/examples.sqlite3';db.unlink()
        with closing(sqlite3.connect(db)) as connection,connection:connection.execute('CREATE TABLE teacher(value TEXT)')
        (db.parent/'current.json').write_text('{"version":1}')
        (db.parent/'models').mkdir();(db.parent/'models/model_000001.json').write_text('{"model":1}')
        for name in ('.gitignore','.gitattributes'):
            (self.seed/name).write_bytes((Path(__file__).parent/name).read_bytes())
        folder=self.seed/'fly_activity/outputs/session';folder.mkdir(parents=True)
        (folder/'session.json').write_text('{"labels":{}}')
        self.git(self.seed,'add','.');self.git(self.seed,'commit','-m','data baseline');self.git(self.seed,'push')
        self.git(self.local,'pull','--ff-only')

    def sync(self):return synchronize_repository(self.local,git=GIT)

    def result(self,name='summary.csv',content='value\n1\n'):
        path=self.local/'fly_activity/outputs/local'/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(content);return path

    def test_button_fetches_and_pulls_before_adding_committing_and_pushing(self):
        self.result();self.remote_update();commands=[];stages=[];execute=subprocess.run
        def record(args,**options):
            if '-C' in args:commands.append(args[args.index('-C')+2])
            return execute(args,**options)
        with patch('data_sync.subprocess.run',side_effect=record):
            result=synchronize_repository(self.local,git=GIT,pull_first=True,progress=lambda stage,detail:stages.append(stage))
        self.assertTrue(result['uploaded'],result);self.assertTrue(result['restart_required'])
        for earlier,later in zip(('fetch','pull','add','commit'),('pull','add','commit','push')):
            self.assertLess(commands.index(earlier),commands.index(later),commands)
        self.assertEqual((self.local/'app.py').read_text(),'new')
        self.assertEqual(stages[:2],['fetch','pull']);self.assertEqual(stages[-1],'push')

    def test_button_pull_conflict_retains_uncommitted_annotations_without_pushing(self):
        relative='fly_activity/outputs/session/session.json'
        (self.local/relative).write_text('{"labels":{"1:0":"groom_head"}}')
        before=(self.local/relative).read_bytes();head=self.git(self.local,'rev-parse','HEAD')
        (self.seed/relative).write_text('{"labels":{"1:0":"stopped"}}')
        self.git(self.seed,'add',relative);self.git(self.seed,'commit','-m','remote annotation');self.git(self.seed,'push')
        result=synchronize_repository(self.local,git=GIT,pull_first=True)
        self.assertEqual(result['status'],'failed');self.assertFalse(result['uploaded'])
        self.assertEqual((self.local/relative).read_bytes(),before)
        self.assertEqual(self.git(self.local,'rev-parse','HEAD'),head)
        self.assertEqual(self.git(self.local,'ls-files','-u'),'')

    def test_button_committed_conflicts_keep_both_annotation_histories(self):
        self.git(self.local,'config','pull.ff','only')
        relative='fly_activity/outputs/session/session.json'
        for root,value in ((self.local,'local'),(self.seed,'remote')):
            (root/relative).write_text(json.dumps(dict(labels=value)))
            self.git(root,'add',relative);self.git(root,'commit','-m',value)
        self.git(self.seed,'push');remote=self.git(self.seed,'rev-parse','HEAD')
        result=synchronize_repository(self.local,git=GIT,pull_first=True)
        self.assertTrue(result['uploaded'],result)
        self.assertEqual(json.loads((self.local/relative).read_text())['labels'],'local')
        saved=self.local/'fly_activity/outputs/shared_conflicts'/remote/relative
        self.assertEqual(json.loads(saved.read_text())['labels'],'remote')

    def test_button_cli_reports_results_and_respects_the_shared_runtime_lock(self):
        from runtime_manager import setup_lock
        command=[sys.executable,str(Path(__file__).parent/'setup/sync_now.py'),'--root',str(self.local)]
        def execute():
            run=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'),timeout=30)
            self.assertEqual(run.returncode,0,run.stderr)
            return json.loads(next(line[len('FLY_SYNC_RESULT '):] for line in run.stdout.splitlines() if line.startswith('FLY_SYNC_RESULT ')))
        self.result();self.assertTrue(execute()['uploaded'])
        with setup_lock(self.local/'.runtime/setup.lock'):
            result=execute();self.assertEqual(result['status'],'failed')
            self.assertIn('別のアプリ',result['detail'])

    def test_data_graphs_and_learning_are_uploaded_without_video_background_or_source(self):
        for name in ('summary.csv','session.json','report.png','manual_grooming_periods.png','background.png','video.mp4','.env'):
            self.result(name)
        (self.local/'app.py').write_text('unfinished source')
        db=self.local/'fly_activity/learning/examples.sqlite3'
        with closing(sqlite3.connect(db)) as connection,connection:connection.execute("INSERT INTO teacher VALUES ('local')")
        before=db.read_bytes()
        result=self.sync();self.assertTrue(result['uploaded'],result)
        files=self.git(self.local,'ls-tree','-r','--name-only','origin/main').splitlines()
        for name in ('summary.csv','session.json','report.png','manual_grooming_periods.png'):self.assertIn('fly_activity/outputs/local/'+name,files)
        for name in ('background.png','video.mp4','.env'):self.assertNotIn('fly_activity/outputs/local/'+name,files)
        self.assertEqual(self.git(self.local,'show','origin/main:app.py'),'old')
        self.assertEqual((self.local/'app.py').read_text(),'unfinished source')
        self.assertEqual(db.read_bytes(),before)

    def test_local_data_and_incoming_code_are_merged_and_pushed(self):
        self.result();self.remote_update()
        result=self.sync();self.assertEqual(result['status'],'updated',result)
        self.assertEqual((self.local/'app.py').read_text(),'new')
        self.assertIn('fly_activity/outputs/local/summary.csv',self.git(self.local,'ls-tree','-r','--name-only','origin/main'))

    def test_conflicting_histories_and_sessions_retain_both_versions(self):
        for root,value in ((self.local,'local'),(self.seed,'remote')):
            with closing(sqlite3.connect(root/'fly_activity/learning/examples.sqlite3')) as db,db:db.execute('INSERT INTO teacher VALUES (?)',(value,))
            (root/'fly_activity/learning/models/model_000001.json').write_text(json.dumps(dict(model=value)))
            (root/'fly_activity/outputs/session/session.json').write_text(json.dumps(dict(labels=value)))
        self.git(self.seed,'add','.');self.git(self.seed,'commit','-m','other PC');self.git(self.seed,'push')
        remote=self.git(self.seed,'rev-parse','HEAD')
        result=self.sync();self.assertTrue(result['uploaded'],result)
        with closing(sqlite3.connect(self.local/'fly_activity/learning/examples.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT value FROM teacher').fetchall(),[('local',)])
        archive=self.local/'fly_activity/learning/imports'/remote
        with closing(sqlite3.connect(archive/'examples.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT value FROM teacher').fetchall(),[('remote',)])
        self.assertEqual(json.loads((archive/'models/model_000001.json').read_text())['model'],'remote')
        self.assertEqual(json.loads((self.local/'fly_activity/outputs/session/session.json').read_text())['labels'],'local')
        saved=self.local/'fly_activity/outputs/shared_conflicts'/remote/'fly_activity/outputs/session/session.json'
        self.assertEqual(json.loads(saved.read_text())['labels'],'remote')

    def test_offline_commit_is_retained_and_sent_next_time(self):
        self.result();self.git(self.local,'remote','set-url','origin',str(self.folder/'offline.git'))
        self.assertEqual(self.sync()['status'],'failed')
        self.assertIn('fly_activity/outputs/local/summary.csv',self.git(self.local,'ls-tree','-r','--name-only','HEAD'))
        self.git(self.local,'remote','set-url','origin',str(self.origin))
        self.assertTrue(self.sync()['uploaded'])
        before=self.git(self.local,'rev-parse','HEAD');self.sync()
        self.assertEqual(self.git(self.local,'rev-parse','HEAD'),before)

    def test_source_conflict_aborts_merge_and_retains_data_commit(self):
        (self.local/'app.py').write_text('local committed source')
        self.git(self.local,'add','app.py');self.git(self.local,'commit','-m','user source')
        self.result();self.remote_update()
        result=self.sync();self.assertEqual(result['status'],'failed',result)
        self.assertEqual((self.local/'app.py').read_text(),'local committed source')
        self.assertIn('fly_activity/outputs/local/summary.csv',self.git(self.local,'ls-tree','-r','--name-only','HEAD'))
        self.assertEqual(self.git(self.local,'ls-files','-u'),'')

    def test_staged_user_work_is_untouched(self):
        (self.local/'app.py').write_text('staged source');self.git(self.local,'add','app.py')
        self.result();before=self.git(self.local,'diff','--cached')
        self.assertEqual(self.sync()['status'],'skipped')
        self.assertEqual(self.git(self.local,'diff','--cached'),before)

    def test_sharing_scope(self):
        for path in ('fly_activity/outputs/a/report.png','fly_activity/outputs/a/grooming_report.png','fly_activity/outputs/a/session.json',
                     'fly_activity/outputs/a/manual_grooming_periods.png',
                     'fly_activity/results_old/activity.csv','fly_activity/learning/models/model_000123.json'):
            self.assertTrue(shared_path(path),path)
        for path in ('fly_activity/app.py','fly_activity/outputs/a/background.png',
                     'fly_activity/outputs/a/movie.mp4','fly_activity/outputs/a/appearance_features.npz','.env'):
            self.assertFalse(shared_path(path),path)


if __name__=='__main__':unittest.main()
