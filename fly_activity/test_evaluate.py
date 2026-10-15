import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from automatic_head import sha256
from evaluate import read_csv,review_clip_plan,save_csv


HERE=Path(__file__).parent


class ReviewClipTests(unittest.TestCase):
    def test_failed_csv_write_keeps_previous_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"human_learning.csv"
            save_csv(path,[dict(human_label="stopped")],["human_label"])
            before=path.read_bytes()
            with self.assertRaises(ValueError):
                save_csv(path,[dict(human_label="forward",unexpected=1)],["human_label"])
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(list(Path(directory).iterdir()),[path])

    def test_each_choice_saves_and_advances_with_learning_on_and_off(self):
        script='''
import sys
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
from tkinter import ttk,messagebox
from evaluate import review,read_csv
folder=Path(sys.argv[1]);learn=sys.argv[2]=="on"
root=tk.Tk();errors=[];clicked=[]
def widgets(parent):
    for child in parent.winfo_children():
        yield child
        yield from widgets(child)
def exception(kind,value,trace):
    errors.append(str(value));root.destroy()
root.report_callback_exception=exception
choices=["前進 F","後退 B","停止 S","横方向等 L","Unknown U"]
def click():
    index=len(clicked)
    prompt=[w.cget("text") for w in widgets(root) if isinstance(w,ttk.Label) and str(w.cget("text")).startswith("評価 ")]
    assert any(f"評価 {index+1}/5 " in text for text in prompt),prompt
    button=next(w for w in widgets(root) if isinstance(w,ttk.Button) and w.cget("text")==choices[index])
    clicked.append(choices[index]);button.invoke()
    if len(clicked)<5:root.after(80,click)
root.after(100,click);root.after(5000,root.destroy)
with patch("tkinter.Tk",return_value=root),patch("tkinter.messagebox.showinfo"),patch("tkinter.messagebox.showerror") as show_error,patch("cumulative_learning.LearningStore") as store,patch("cumulative_learning.TeacherExtractor") as extractor,patch("automatic_head.load_model",return_value={}):
    store.return_value.update.return_value=dict(status="behavior_learned",version=123,learned=True)
    extractor.return_value.extract.return_value={}
    review(folder,learn=learn,sampling="learning")
    assert not show_error.called
    assert store.return_value.update.call_count==(5 if learn else 0)
assert not errors,errors
assert len(clicked)==5,clicked
rows=read_csv(folder/"human_learning.csv")
assert [r["human_label"] for r in rows]==["forward","backward","stopped","lateral","unknown"]
assert all(r["selection_reason"]=="test" for r in rows)
assert all(r["learned_model_version"]==("123" if learn else "") for r in rows)
'''
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);video=folder/"flies.avi"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),30,(320,200))
            self.assertTrue(writer.isOpened())
            for frame in range(30):
                image=np.full((200,320,3),200,np.uint8)
                cv2.ellipse(image,(50+frame*4,100),(10,5),0,0,360,(20,20,20),-1)
                writer.write(image)
            writer.release()
            (folder/"metadata.json").write_text(json.dumps(dict(source=str(video),head_mode="automatic_appearance")))
            rows=[dict(frame=k,slot=1,time_s=k/30,segment=1,x_smooth_px=50+4*k,y_smooth_px=100,locomotion_direction="unknown") for k in range(30)]
            save_csv(folder/"activity.csv",rows,list(rows[0]))
            for enabled in ("on","off"):
                samples=[dict(sample_id=k+1,frame=10+k,slot=1,time_s=(10+k)/30,prediction="unknown",human_label="",selection_reason="test") for k in range(5)]
                save_csv(folder/"human_learning.csv",samples,list(samples[0]))
                (folder/"learning_sampling.json").write_text(json.dumps(dict(activity_sha256=sha256(folder/"activity.csv"),sampling_mode="learning")))
                run=subprocess.run([sys.executable,"-c",script,str(folder),enabled],cwd=HERE,
                                   capture_output=True,text=True,timeout=15)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)

    def rows(self):
        return {(k,1):dict(x_smooth_px=str(100+40*k),y_smooth_px="200",segment="1")
                for k in range(11)}

    def plan(self,rows,width=640,height=480):
        return review_clip_plan(rows,5,1,30,width,height,10)

    def test_fast_target_stays_inside_fixed_crop(self):
        plan=self.plan(self.rows())
        x0,y0,x1,y1=plan["bounds"]
        self.assertEqual((plan["start"],plan["end"]),(0,10))
        for x,y in plan["points"].values():
            self.assertTrue(x0<=x<x1 and y0<=y<y1)
            self.assertGreaterEqual(x-x0,17)
            self.assertGreaterEqual(x1-x,17)

    def test_missing_center_never_falls_back_to_whole_video(self):
        for value in ("","nan","inf","-1","700"):
            with self.subTest(value=value):
                rows=self.rows();rows[5,1]["x_smooth_px"]=value
                self.assertIsNone(self.plan(rows))

    def test_missing_frame_and_reidentified_segment_limit_clip(self):
        rows=self.rows();rows[3,1]["x_smooth_px"]="";rows[7,1]["segment"]="2"
        plan=self.plan(rows)
        self.assertEqual((plan["start"],plan["end"]),(4,6))
        self.assertEqual(set(plan["points"]),{4,5,6})

    def test_missing_row_limits_clip(self):
        rows=self.rows();del rows[4,1]
        self.assertEqual(self.plan(rows)["start"],5)
        del rows[5,1]
        self.assertIsNone(self.plan(rows))

    def test_small_video_and_edge_target_are_in_bounds(self):
        for x,y in ((0,0),(119,79)):
            rows={(k,1):dict(x_smooth_px=x,y_smooth_px=y,segment="1") for k in range(11)}
            self.assertEqual(self.plan(rows,120,80)["bounds"],(0,0,120,80))

    def test_review_gui_with_visible_and_missing_target(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);video=folder/"flies.avi"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),30,(320,200))
            self.assertTrue(writer.isOpened())
            for k in range(20):
                image=np.full((200,320,3),200,np.uint8)
                cv2.ellipse(image,(50+8*k,100),(10,5),0,0,360,(20,20,20),-1)
                writer.write(image)
            writer.release()
            (folder/"metadata.json").write_text(json.dumps(dict(source=str(video),head_mode="automatic_appearance")),encoding="utf-8")
            for mode,missing,recover in (("random",False,False),("random",True,False),("learning",False,False),("learning",True,False),("learning",False,True)):
                with self.subTest(mode=mode,missing=missing,recover=recover):
                    rows=[dict(frame=k,slot=1,time_s=k/30,segment=1,
                               x_smooth_px="" if missing and k==10 else 50+8*k,
                               y_smooth_px="" if missing and k==10 else 100,
                               locomotion_direction="unknown") for k in range(20)]
                    save_csv(folder/"activity.csv",rows,list(rows[0]))
                    samples=[dict(sample_id=1,frame=10,slot=1,time_s=10/30,prediction="unknown",human_label="")]
                    save_csv(folder/("human_learning.csv" if mode=="learning" else "human_evaluation.csv"),samples,list(samples[0]))
                    if recover:
                        save_csv(folder/"human_learning.csv.recovered",samples,list(samples[0]))
                        save_csv(folder/"human_learning.csv",[],list(samples[0]))
                    (folder/("learning_sampling.json" if mode=="learning" else "evaluation_sampling.json")).write_text(json.dumps(dict(activity_sha256=sha256(folder/"activity.csv"),sampling_mode=mode)),encoding="utf-8")
                    run=subprocess.run([sys.executable,str(HERE/"evaluate.py"),"--results",str(folder),"--smoke-test","--no-learn","--sampling",mode],
                                       capture_output=True,text=True,encoding="utf-8",env=dict(os.environ,PYTHONUTF8="1"),timeout=30)
                    self.assertEqual(run.returncode,0,run.stderr)
                    report=json.loads((folder/("learning_metrics.json" if mode=="learning" else "evaluation_metrics.json")).read_text(encoding="utf-8"))
                    self.assertEqual(report["completed_samples"],0)
                    if recover:self.assertEqual(len(read_csv(folder/"human_learning.csv")),1)


if __name__=="__main__":unittest.main()
