import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT=r'''
import sys,time,tempfile,threading
from pathlib import Path
from unittest.mock import patch,Mock
sys.path.insert(0,sys.argv[1])
import tkinter as tk
from test_manual_behavior import fixture
from manual_behavior_app import ManualApp
from app import App

folder=Path(tempfile.mkdtemp(prefix='sync_ui_'));session=fixture(folder)
root=tk.Tk();manual=ManualApp(root);root.update();manual.attach(session);root.update()
errors=[];root.report_callback_exception=lambda kind,value,trace:errors.append(str(value))
manual.second.set('1');manual.select_interval();before=session.path.read_bytes()
gate=threading.Event();calls=[]
def slow_sync(repository,report):
    calls.append(repository);report('git fetch');gate.wait(10)
    return dict(status='current',uploaded=True,detail='同期完了',restart_required=False)
def wait(sync):
    deadline=time.monotonic()+10
    while sync.active and time.monotonic()<deadline:
        root.after(20,root.quit);root.mainloop()
    root.update();assert not sync.active
with patch('repository_sync.run_sync',side_effect=slow_sync):
    manual.sync.button.invoke();root.after(50,root.quit);root.mainloop()
    assert manual.busy and not manual.playing and manual.sync.button.instate(['disabled']),(manual.busy,manual.playing,manual.sync.active,manual.sync.text.get(),errors)
    manual.record('groom_head');assert session.path.read_bytes()==before
    manual.sync.start();assert len(calls)<=1
    gate.set();wait(manual.sync)
assert len(calls)==1 and not manual.busy and manual.second.get()=='1'
assert manual.slot_combo.instate(['readonly','!disabled'])
assert '同期完了' in manual.sync.text.get()
with patch('repository_sync.run_sync',return_value=dict(status='failed',uploaded=False,detail='通信失敗')):
    manual.sync.button.invoke();wait(manual.sync)
assert not manual.busy and manual.sync.button.instate(['!disabled'])
assert '通信失敗' in manual.sync.text.get()

top=tk.Toplevel(root)
with patch('app.active_model_path',return_value=None),patch('app.STORE',folder/'unused_learning'):
    automatic=App(top)
root.update()
automatic.running=True
with patch('repository_sync.run_sync') as worker:
    automatic.sync.start();assert not worker.called
automatic.running=False;automatic.evaluation_process=Mock();automatic.evaluation_process.poll.return_value=None
with patch('repository_sync.run_sync') as worker:
    automatic.sync.start();assert not worker.called
automatic.evaluation_process=None
with patch('repository_sync.run_sync',return_value=dict(status='current',uploaded=True,detail='送信完了')),patch.object(automatic,'update_model_status'):
    automatic.sync.button.invoke();wait(automatic.sync)
assert not automatic.running and '送信完了' in automatic.sync.text.get()
automatic.close()

with patch('repository_sync.run_sync',return_value=dict(status='updated',uploaded=True,detail='更新取得',restart_required=True)):
    manual.sync.button.invoke();wait(manual.sync)
assert manual.busy and manual.sync.restart_button.winfo_ismapped()
manual.record('groom_body');assert session.path.read_bytes()==before
assert manual.sync.button.instate(['disabled'])
with patch('repository_sync.subprocess.Popen') as launch:
    manual.sync.restart_button.invoke()
    args=launch.call_args.args[0]
    assert args[-3:]==['manual','--session',str(session.path)],args
    assert launch.call_args.kwargs['env']['FLY_UPDATE_DONE']=='1'
assert not errors,errors
print('Repository sync GUI passed: both apps, progress, busy guards, failures and restart with session.')
'''


class RepositorySyncGuiTests(unittest.TestCase):
    def test_both_apps_update_without_blocking_or_losing_manual_labels(self):
        root=Path(__file__).resolve().parent.parent;env=None
        if Path(sys.prefix).resolve().is_relative_to(root/'.runtime'):
            spec=importlib.util.spec_from_file_location('runtime_manager',root/'setup/runtime_manager.py')
            runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
            env=runtime.configure_tk(runtime.clean_environment(root),sys.base_prefix)
        run=subprocess.run([sys.executable,'-c',SCRIPT,str(root/'fly_activity')],env=env,capture_output=True,text=True,timeout=45)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.assertIn('Repository sync GUI passed:',run.stdout)


if __name__=='__main__':unittest.main()
