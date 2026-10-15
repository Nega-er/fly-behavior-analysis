import json
from pathlib import Path
import tempfile
import unittest

from evaluate import make_samples,metrics,save_csv,write_report
from cumulative_learning import TeacherExtractor


class LearningSamplingTests(unittest.TestCase):
    def prepare(self,folder):
        (folder/"metadata.json").write_text(json.dumps(dict(head_mode="automatic_appearance",training_frame_indices=[30])))
        rows=[]
        for frame in range(200):
            for slot,label,confidence in ((1,"unknown",.9),(2,"backward",.9),(3,"forward",.55),(4,"stopped",.9),(5,"unknown",.5)):
                rows.append(dict(frame=frame,slot=slot,time_s=frame/30,locomotion_direction=label,
                                 automatic_head_confidence=confidence,x_smooth_px="" if slot==5 else slot*40,
                                 y_smooth_px="" if slot==5 else 100,valid_interval="0" if slot==5 else "1"))
        save_csv(folder/"activity.csv",rows,list(rows[0]))

    def test_priority_diversity_spacing_and_training_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);self.prepare(folder)
            samples,provenance=make_samples(folder,20,7,"learning")
            self.assertEqual((samples,provenance),make_samples(folder,20,7,"learning"))
            self.assertEqual(len(samples),20)
            for reason in ("unknown","backward","low_head_confidence","random"):
                self.assertEqual(provenance["selection_counts"][reason],5)
            for sample in samples:
                self.assertNotEqual(int(sample["slot"]),5)
                self.assertGreater(abs(int(sample["frame"])-30),12)
                for other in samples:
                    if sample!=other and sample["slot"]==other["slot"]:
                        self.assertGreaterEqual(abs(float(sample["time_s"])-float(other["time_s"])),.6-1e-9)
            random,info=make_samples(folder,9999,7)
            self.assertTrue(any(int(sample["slot"])==5 for sample in random))
            self.assertEqual(info["sampling_mode"],"random")

    def test_sparse_candidates_do_not_repeat_or_cluster(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);self.prepare(folder)
            samples,_=make_samples(folder,9999,7,"learning")
            self.assertLess(len(samples),9999)
            self.assertEqual(len(samples),len({(r["frame"],r["slot"]) for r in samples}))

    def test_learning_report_does_not_replace_random_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            rows=[dict(human_label="forward",prediction="forward")]
            write_report(folder,rows,dict(sampling_mode="random"))
            before={p.name:p.read_bytes() for p in folder.iterdir()}
            report=write_report(folder,rows+[dict(human_label="unknown",prediction="unknown")],dict(sampling_mode="learning"))
            self.assertFalse(report["representative_accuracy"])
            self.assertEqual(report["human_unknown_samples"],1)
            for name,content in before.items():self.assertEqual((folder/name).read_bytes(),content)
            self.assertIn("動画全体の精度を表す値ではありません",(folder/"learning_report.md").read_text())

    def test_human_unknown_is_not_a_behavior_or_correct_prediction(self):
        report=metrics([dict(human_label="unknown",prediction="unknown"),
                        dict(human_label="unscorable",prediction="forward"),
                        dict(human_label="forward",prediction="unknown")])
        self.assertEqual(report["unscorable_samples"],2)
        self.assertEqual(report["scorable_samples"],1)
        self.assertEqual(report["accuracy_including_unknown_as_error"],0)
        extractor=TeacherExtractor.__new__(TeacherExtractor)
        extractor.video_sha="test";extractor.folder=Path("test");extractor.metadata={}
        extractor.tracks={(10,1):dict(x_px=10,y_px=20)};extractor.activity={(10,1):{}}
        record=extractor.extract(dict(frame=10,slot=1,human_label="unknown",prediction="forward"),{})
        self.assertEqual(record["extraction_status"],"label_saved_unknown")
        self.assertIsNone(record["head_target"])
        self.assertIsNone(record["behavior"])


if __name__=="__main__":unittest.main()
