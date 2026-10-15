import csv
from contextlib import closing
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from automatic_head import load_model,sha256
from cumulative_learning import LearningStore,active_model_path,ensure_store,motion_features,behavior_prediction,training_frames
from analyze_flies import DEFAULTS,analyze

HERE=Path(__file__).parent


class CumulativeTests(unittest.TestCase):
    def test_import_other_pc_teachers_preserves_history_and_does_not_overwrite_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);local=root/'local';remote=root/'remote';local.mkdir();remote.mkdir()
            learner,seed=self.make_store(local);other,_=self.make_store(remote)
            learner.update(self.record(seed));first=active_model_path(local);original=first.read_bytes()
            other.update(dict(self.record(seed),key='other-pc'))
            # Another PC may already own the next numeric model filename.
            (local/'models/model_000002.json').write_text('retained remote model')
            self.assertEqual(learner.import_history(other.db),1)
            self.assertEqual(learner.import_history(other.db),0)
            self.assertEqual(first.read_bytes(),original)
            self.assertEqual((local/'models/model_000002.json').read_text(),'retained remote model')
            model=load_model(active_model_path(local));self.assertEqual(model['learning_version'],3)
            self.assertEqual(model['cumulative_labels'],2)
            with closing(sqlite3.connect(other.db)) as db:self.assertEqual(db.execute('SELECT count(*) FROM examples').fetchone()[0],1)

    def make_store(self,path):
        shutil.copyfile(HERE/"learning"/"seed_model.json",path/"seed_model.json")
        return LearningStore(path),load_model(path/"seed_model.json")

    def record(self,seed,label="forward"):
        appearance=np.asarray(seed["training_features"][0])
        feature=motion_features(appearance,[1,0],5,0,3,seed["feature_scale"])
        return dict(key="case-1",human_label=label,video_sha256="new-video",frame=42,slot=1,
                    appearance=appearance.tolist(),behavior=feature.tolist(),head_target=None,
                    axis=[1,0],velocity=[5,0],axis_ratio=3,extraction_status="behavior_learned")

    def test_update_persists_deduplicates_and_revises(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);store,seed=self.make_store(p);seed_hash=sha256(p/"seed_model.json")
            record=self.record(seed);outcome=store.update(record);self.assertEqual(outcome["version"],1)
            first=active_model_path(p);first_bytes=first.read_bytes()
            self.assertEqual(store.update(record)["status"],"already_registered")
            self.assertEqual(store.update(dict(record,human_label="backward"))["version"],2)
            self.assertEqual(first.read_bytes(),first_bytes)
            self.assertEqual(sha256(p/"seed_model.json"),seed_hash)
            model=load_model(active_model_path(p));self.assertEqual(model["cumulative_labels"],1)
            self.assertEqual(model["behavior_memory"][0]["label"],"backward")
            self.assertEqual(training_frames(model,"new-video"),[42])

    def test_motion_descriptor_axis_sign_invariance(self):
        seed=load_model(HERE/"learning"/"seed_model.json");f=np.asarray(seed["training_features"][0])
        a=motion_features(f,[1,0],5,2,3,seed["feature_scale"])
        b=motion_features(-f,[-1,0],5,2,3,seed["feature_scale"])
        np.testing.assert_allclose(a,b)

    def test_restart_from_another_directory_uses_saved_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);store,seed=self.make_store(p)
            store.update(self.record(seed))
            expected=sha256(active_model_path(p))
            code=("from cumulative_learning import LearningStore,active_model_path; "
                  "from automatic_head import sha256; "
                  "LearningStore(); print(sha256(active_model_path()))")
            env=dict(os.environ,FLY_ACTIVITY_LEARNING_DIR=str(p),PYTHONPATH=str(HERE.resolve()))
            restarted=subprocess.run([sys.executable,"-c",code],cwd=p.parent,env=env,text=True,capture_output=True,check=True)
            self.assertEqual(restarted.stdout.strip(),expected)

    def test_migrate_newest_legacy_history_and_retain_original(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);older=p/"old";newer=p/"new";older.mkdir();newer.mkdir()
            old_store,seed=self.make_store(older);new_store,_=self.make_store(newer)
            old_store.update(self.record(seed));new_store.update(self.record(seed))
            new_store.update(dict(self.record(seed),human_label="backward"))
            original=sha256(active_model_path(newer));target=p/"user-data"/"learning"
            ensure_store(target,legacy_dirs=[older,newer])
            self.assertEqual(sha256(active_model_path(target)),original)
            self.assertEqual(sha256(active_model_path(newer)),original)
            with closing(sqlite3.connect(target/"examples.sqlite3")) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM revisions").fetchone()[0],1)
            migrated=LearningStore(target)
            migrated.update(dict(self.record(seed),human_label="stopped"))
            ensure_store(target,legacy_dirs=[newer])
            self.assertEqual(load_model(active_model_path(target))["learning_version"],3)

    def test_recover_missing_corrupt_and_stale_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);store,seed=self.make_store(p)
            store.update(self.record(seed));old_pointer=(p/"current.json").read_text()
            store.update(dict(self.record(seed),human_label="backward"))
            for pointer in (None,"{broken",old_pointer):
                if pointer is None:(p/"current.json").unlink()
                else:(p/"current.json").write_text(pointer)
                self.assertEqual(load_model(active_model_path(p))["learning_version"],2)
                self.assertEqual(json.loads((p/"current.json").read_text())["version"],2)

    def test_interruption_after_commit_recovers_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);store,seed=self.make_store(p)
            store.update(self.record(seed))
            with patch("cumulative_learning.active_model_path",side_effect=OSError("interrupted publication")):
                with self.assertRaises(OSError):store.update(dict(self.record(seed),human_label="backward"))
            restarted=LearningStore(p)
            self.assertEqual(load_model(active_model_path(p))["behavior_memory"][0]["label"],"backward")
            self.assertEqual(restarted.update(dict(self.record(seed),human_label="backward"))["version"],2)

    def test_failed_model_write_keeps_previous_committed_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);store,seed=self.make_store(p)
            store.update(self.record(seed))
            with patch("cumulative_learning.atomic_json",side_effect=OSError("disk full")):
                with self.assertRaises(OSError):store.update(dict(self.record(seed),human_label="backward"))
            self.assertEqual(load_model(active_model_path(p))["learning_version"],1)
            self.assertEqual(store.update(self.record(seed))["status"],"already_registered")

    def test_far_examples_do_not_force_correction(self):
        seed=load_model(HERE/"learning"/"seed_model.json");record=self.record(seed)
        model=dict(seed,behavior_memory=[dict(features=record["behavior"],label="forward")])
        label,confidence=behavior_prediction(np.asarray(record["behavior"])+30,model)
        self.assertIsNone(label);self.assertLess(confidence,.8)

    def test_unscorable_not_teacher_and_frozen_prediction_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);store,seed=self.make_store(p)
            record=dict(self.record(seed),human_label="unscorable",behavior=None,appearance=None,
                        frozen_prediction="backward",extraction_status="label_saved_not_trainable")
            outcome=store.update(record);self.assertFalse(outcome["learned"])
            model=load_model(active_model_path(p));self.assertEqual(len(model["behavior_memory"]),0)
            self.assertEqual(record["frozen_prediction"],"backward")

    def test_learned_behavior_overrides_bad_baseline_without_changing_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);store,seed=self.make_store(p);record=self.record(seed)
            store.update(record);model=load_model(active_model_path(p))
            rows=[dict(frame=k,time_s=k/10,slot=1,segment=1,x_px=k*.5,y_px=0,status="observed",
                       axis_dx=1,axis_dy=0,axis_ratio=3,head_probability_axis_positive=.1) for k in range(11)]
            features=np.tile(record["appearance"],(11,1,1))
            out=p/"result";out.mkdir()
            c=dict(DEFAULTS,expected_flies=1,smooth_window_s=0,direction_move_start_speed_px_s=3,
                   direction_stop_start_speed_px_s=1)
            analyze(rows,c,out,appearance_cache=features,behavior_model=model)
            with (out/"activity.csv").open(encoding="utf-8-sig") as f:activity=list(csv.DictReader(f))
            self.assertTrue(all(r["baseline_locomotion_direction"]=="backward" for r in activity[1:]))
            self.assertTrue(all(r["locomotion_direction"]=="forward" for r in activity[1:]))
            self.assertEqual(float(activity[-1]["x_smooth_px"]),5)
            self.assertEqual(activity[0]["locomotion_direction"],"unknown")


if __name__=="__main__":unittest.main()
