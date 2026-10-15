import subprocess
import sys
import importlib.util
from pathlib import Path
import unittest

SCRIPT = r"""
import sys,time,tempfile,faulthandler
faulthandler.enable()
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,sys.argv[1])
import tkinter as tk
from tkinter import ttk
from test_manual_behavior import fixture
from manual_behavior import ManualSession
from manual_behavior_app import ManualApp
from evaluate import read_csv
folder=Path(tempfile.mkdtemp(prefix='manual_gui_'));session=fixture(folder)
root=tk.Tk();app=ManualApp(root);errors=[]
root.report_callback_exception=lambda kind,value,trace:errors.append(str(value))
root.update();app.attach(session);root.update()
assert list(app.slot_combo.cget('values'))==['1','2']
assert app.playing and app.job is not None
def widgets(parent):
    for child in parent.winfo_children():
        yield child
        yield from widgets(child)
app.advance.set(False)
flying_button=next(widget for widget in widgets(root) if isinstance(widget,ttk.Button) and widget.cget('text')=='飛んでいる 6')
flying_button.invoke();root.update()
assert session.label(1,0)=='flying' and app.behavior.get()=='飛んでいる'
assert ManualSession.load(session.path).label(1,0)=='flying'
assert app.playing and app.second.get()=='0'
app.record('');app.advance.set(True)
deadline=time.monotonic()+2.2
while time.monotonic()<deadline:root.update();time.sleep(.01)
assert app.second.get()=='0' and not session.label(1,0)
assert app.playing and 0<=app.current_frame<10
with patch.object(session,'set_label',side_effect=OSError('disk full')),patch('manual_behavior_app.messagebox.showerror'):
    app.record('forward')
assert app.second.get()=='0' and app.playing
app.behavior.set('グルーミング頭');app.behavior_combo.event_generate('<<ComboboxSelected>>');root.update()
assert session.label(1,0)=='groom_head' and app.second.get()=='1'
assert app.playing and 10<=app.current_frame<20
app.record('groom_body');assert app.second.get()=='2'
app.record('stopped');assert session.label(1,2)=='stopped'
assert not app.playing and app.job is None
app.slot.set('2');app.second.set('0');app.slot_combo.event_generate('<<ComboboxSelected>>');root.update()
app.record('forward');app.record('unknown');app.record('backward')
assert [session.label(2,k) for k in range(3)]==['forward','unknown','backward']
app.export();root.update();assert (session.path.parent/'manual_behavior.png').exists()
assert len(read_csv(session.path.parent/'manual_behavior.csv'))==6
app.attach(ManualSession.load(session.path));root.update();assert app.session.label(1,0)=='groom_head'
# The native file panel pumps Tk events too. No playback callback may decode or
# replace an image until the panel has closed, including a cancel/reopen cycle.
def choose_session(**options):
    assert options['parent'] is root
    assert app.dialog_active and not app.playing and app.job is None
    frame=app.current_frame
    deadline=time.monotonic()+.12
    while time.monotonic()<deadline:root.update();time.sleep(.01)
    app.tick()
    assert app.current_frame==frame and app.job is None
    return str(session.path)
with patch('manual_behavior_app.filedialog.askopenfilename',side_effect=choose_session):
    for _ in range(12):
        app.open_session()
        deadline=time.monotonic()+5
        while app.busy and time.monotonic()<deadline:root.update();time.sleep(.01)
        assert not app.busy and app.playing and app.job is not None
        assert app.session.label(1,0)=='groom_head'
with patch('manual_behavior_app.filedialog.askopenfilename',return_value=''):
    app.open_session()
assert app.playing and app.job is not None and not app.dialog_active
previous=app.session;previous_cap=app.cap
invalid=folder/'invalid.json';invalid.write_text('{}')
with patch('manual_behavior_app.filedialog.askopenfilename',return_value=str(invalid)),patch('manual_behavior_app.messagebox.showerror') as error:
    app.open_session()
    deadline=time.monotonic()+5
    while app.busy and time.monotonic()<deadline:root.update();time.sleep(.01)
    error.assert_called_once()
assert app.session is previous and app.cap is previous_cap and app.cap.isOpened()
app.toggle()
def unavailable_capture(path):
    assert not app.playing and app.job is None
    from unittest.mock import Mock
    return Mock(isOpened=lambda:False)
with patch('manual_behavior_app.cv2.VideoCapture',side_effect=unavailable_capture):
    try:app.attach(previous)
    except ValueError:pass
    else:raise AssertionError('Unreadable video was accepted')
assert app.session is previous and app.cap is previous_cap and app.cap.isOpened()
app.tabs.select(0);app.second.set('1');app.select_interval()
assert app.playing
app.advance.set(False);app.record('groom_head')
assert app.second.get()=='1' and app.playing
app.record('');assert app.second.get()=='1' and app.playing
app.advance.set(True)
for _ in range(8):root.update();time.sleep(.02)
app.pause();assert app.current_frame>=10
assert app.job is None
app.toggle();assert app.playing and app.job is not None
app.pause()
app.video.set(str(session.video));app.polarity.set('dark');app.count.set('2')
with patch('manual_behavior_app.HERE',folder):
    app.detect()
    deadline=time.monotonic()+20
    while app.busy and time.monotonic()<deadline:root.update();time.sleep(.01)
assert not app.busy,'Detection did not finish'
assert len(app.session.slots)==2
assert app.session.path!=session.path
assert (app.session.tracks_path.parent/'tracking_preview.mp4').exists()
assert not errors,errors
app.record('forward');assert ManualSession.load(app.session.path).label(1,0)=='forward'
print('GUI passed: dropdown labels, auto advance, all behaviors, loop playback, save/reload, CSV/graph export and new detection.')
print('Preview folder:',folder)
app.close()
"""


class ManualGuiTests(unittest.TestCase):
    def test_manual_workflow_and_new_video_detection(self):
        root=Path(__file__).resolve().parent.parent
        env=None
        if Path(sys.prefix).resolve().is_relative_to(root/'.runtime'):
            spec=importlib.util.spec_from_file_location('runtime_manager',root/'setup/runtime_manager.py')
            runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
            env=runtime.configure_tk(runtime.clean_environment(root),sys.base_prefix)
        run=subprocess.run([sys.executable,'-c',SCRIPT,str(Path(__file__).resolve().parent)],
                           capture_output=True,text=True,timeout=60,env=env)
        self.assertEqual(run.returncode,0,run.stdout+run.stderr)
        self.assertIn('GUI passed:',run.stdout)


if __name__=='__main__':unittest.main()
