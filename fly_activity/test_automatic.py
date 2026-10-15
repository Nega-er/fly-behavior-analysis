import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from analyze_flies import DEFAULTS
from automatic_head import appearance_features,load_model,predict_probability
from direction import resolve_headings
from evaluate import metrics,make_samples
from video_setup import estimate_population,roi_mask

HERE=Path(__file__).parent


class AutomaticTests(unittest.TestCase):
    def test_head_model_axis_sign_invariance(self):
        points=np.array([(x,y) for x in range(-10,11) for y in range(-2,3) if not (x>5 and abs(y)>1)],float)
        features=appearance_features(points,np.ones(len(points)),[1.,0.])
        reverse=appearance_features(points,np.ones(len(points)),[-1.,0.])
        np.testing.assert_allclose(features,-reverse,atol=1e-12)
        model=load_model(HERE/"head_model.json")
        self.assertAlmostEqual(predict_probability(features,model)+predict_probability(reverse,model),1.)

    def test_automatic_head_without_manual_input(self):
        rows=[dict(frame=k,slot=1,segment=1,x_px=k,y_px=0,status="observed",axis_dx=1.,axis_dy=0.,
                   axis_ratio=3.,head_probability_axis_positive=.9) for k in range(5)]
        head,sources=resolve_headings(rows,{},DEFAULTS)
        np.testing.assert_allclose(head,np.tile([1.,0.],(5,1)))
        self.assertTrue(all(s=="automatic_appearance" for s in sources))

    def test_uncertain_auto_head_not_guessed_from_motion(self):
        rows=[dict(frame=k,slot=1,segment=1,x_px=k,y_px=0,status="observed",axis_dx=1.,axis_dy=0.,
                   axis_ratio=3.,head_probability_axis_positive=.5) for k in range(5)]
        head,sources=resolve_headings(rows,{},DEFAULTS)
        self.assertTrue(np.all(np.isnan(head)))

    def test_population_mode_and_no_detection(self):
        self.assertEqual(estimate_population([2,3,3,3,4])["estimated_flies"],3)
        with self.assertRaises(ValueError):estimate_population([0,0])

    def test_roi_is_video_specific(self):
        c=dict(DEFAULTS,roi_rect_px=[10,20,30,40])
        mask=roi_mask((100,120),c)
        self.assertEqual(np.count_nonzero(mask),1200)
        c["roi_rect_px"]=[110,20,30,40]
        with self.assertRaises(ValueError):roi_mask((100,120),c)

    def test_unknown_predictions_are_not_hidden_from_accuracy(self):
        r=[dict(human_label="forward",prediction="forward"),
           dict(human_label="forward",prediction="unknown"),
           dict(human_label="backward",prediction="forward"),
           dict(human_label="unscorable",prediction="forward")]
        report=metrics(r)
        self.assertEqual(report["scorable_samples"],3)
        self.assertAlmostEqual(report["accuracy_including_unknown_as_error"],1/3)
        self.assertAlmostEqual(report["prediction_coverage"],2/3)
        self.assertAlmostEqual(report["classes"]["forward"]["recall"],.5)
        self.assertIsNone(metrics([])["accuracy_including_unknown_as_error"])

    def test_sampling_deterministic_and_training_excluded(self):
        result=HERE/"results_automatic"
        if not result.exists():self.skipTest("Real-video automatic run not available")
        first,provenance=make_samples(result,80,8);second,_=make_samples(result,80,8)
        self.assertEqual(first,second)
        metadata=json.loads((result/"metadata.json").read_text())
        for r in first:
            self.assertTrue(all(abs(int(r["frame"])-k)>12 for k in metadata["training_frame_indices"]))
            self.assertEqual(r["human_label"],"")

    def test_arbitrary_dark_video_and_auto_population_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);video=p/"three_flies.avi"
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"MJPG"),30,(320,200))
            self.assertTrue(writer.isOpened())
            for k in range(18):
                image=np.full((200,320,3),220,np.uint8)
                for y in [40,100,160]:cv2.ellipse(image,(35+4*k,y),(10,5),0,0,360,(20,20,20),-1)
                writer.write(image)
            writer.release()
            background=p/"background.png";cv2.imwrite(str(background),np.full((200,320),220,np.uint8))
            config=dict(DEFAULTS,foreground_polarity="dark",background_path=str(background),
                        max_speed_px_s=1200,min_area_px=30,max_area_px=500)
            (p/"config.json").write_text(json.dumps(config))
            run=subprocess.run([sys.executable,str(HERE/"analyze_flies.py"),"--video",str(video),
                                "--config",str(p/"config.json"),"--flies","auto","--out",str(p/"results")],
                               capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            used=json.loads((p/"results"/"config_used.json").read_text())
            self.assertEqual(used["expected_flies"],3)
            metadata=json.loads((p/"results"/"metadata.json").read_text())
            self.assertFalse(metadata["same_video_as_head_training"])
            self.assertEqual(metadata["head_mode"],"automatic_appearance")
            self.assertTrue((p/"results"/"direction_preview.mp4").exists())
            statistics=json.loads((p/"results"/"period_statistics.json").read_text(encoding="utf-8"))
            self.assertEqual(statistics["status"],"insufficient")
            self.assertEqual(statistics["total_slots"],3)
            with (p/"results"/"tracks.csv").open(encoding="utf-8-sig") as f:
                self.assertEqual(len(list(csv.DictReader(f))),18*3)
            run=subprocess.run([sys.executable,str(HERE/"analyze_flies.py"),"--video",str(video),
                                "--config",str(p/"config.json"),"--flies","2","--out",str(p/"manual_results")],
                               capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            used=json.loads((p/"manual_results"/"config_used.json").read_text())
            self.assertEqual(used["expected_flies"],2)


if __name__=="__main__":unittest.main()
