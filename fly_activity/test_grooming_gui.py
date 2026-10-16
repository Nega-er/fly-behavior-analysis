import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

SCRIPT=r'''
import sys,tempfile,time,json
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import tkinter as tk
from test_grooming import fixture
from manual_behavior import ManualSession
from manual_behavior_app import ManualApp
from app import App
root_path=Path(tempfile.mkdtemp(prefix='grooming_ui_'))
video,tracks,_=fixture(root_path)
session=ManualSession.create(video,tracks,root_path/'manual')
session.set_label(1,0,'groom_body');session.set_label(1,1,'stopped')
root=tk.Tk();app=ManualApp(root);errors=[]
root.report_callback_exception=lambda kind,value,trace:errors.append(str(value))
root.update();app.attach(session);root.update()
before=session.path.read_bytes()
with patch('manual_behavior_app.HERE',root_path):
    app.grooming_comparison()
    deadline=time.monotonic()+30
    while app.busy and time.monotonic()<deadline:root.update();time.sleep(.01)
assert not app.busy and hasattr(app,'grooming_window'),errors
view=app.grooming_window;root.update()
assert len(view.table.get_children())==2
assert session.path.read_bytes()==before and app.session is session and app.cap.isOpened()
assert len(list((root_path/'learning/grooming/examples').glob('*.json')))==2
view.play();root.update();assert view.players[-1].cap.isOpened()
view.close()
analysis_root=tk.Toplevel(root)
with patch('app.active_model_path',return_value=None),patch('app.STORE',root_path/'unused_learning'):
    automatic=App(analysis_root)
root.update()
assert automatic.config()[0]['grooming_enabled'] is False
automatic.grooming_enabled.set(True);assert automatic.config()[0]['grooming_enabled'] is True
automatic.results.data=dict(folder=root_path,metadata=dict(direction_video=str(video)),group=dict(excluded_slots=[]))
with patch('app.filedialog.askopenfilename',return_value=''),patch('app.HERE',root_path):
    automatic.grooming_comparison()
    deadline=time.monotonic()+30
    while automatic.running and time.monotonic()<deadline:root.update();time.sleep(.01)
assert not automatic.running and hasattr(automatic,'grooming_window'),errors
root.update();assert automatic.grooming_window.metadata['manual_session'] is None
automatic.grooming_window.close();analysis_root.destroy()
assert not errors,errors
app.close()
print('Grooming UI passed: manual comparison, diagnostic movie, immutable labels and automatic prototype.')
'''


class GroomingGuiTests(unittest.TestCase):
    def test_manual_and_automatic_comparison_workflows(self):
        root=Path(__file__).resolve().parent.parent;env=None
        if Path(sys.prefix).resolve().is_relative_to(root/'.runtime'):
            spec=importlib.util.spec_from_file_location('runtime_manager',root/'setup/runtime_manager.py')
            runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
            env=runtime.configure_tk(runtime.clean_environment(root),sys.base_prefix)
        run=subprocess.run([sys.executable,'-c',SCRIPT,str(root/'fly_activity')],env=env,capture_output=True,text=True,timeout=90)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.assertIn('Grooming UI passed:',run.stdout)


if __name__=='__main__':unittest.main()
