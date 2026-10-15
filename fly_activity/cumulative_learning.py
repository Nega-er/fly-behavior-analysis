"""Persistent supervised updates, immutable model versions and frozen evaluation."""
import argparse
from contextlib import closing
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from automatic_head import appearance_features,load_model,sha256
from video_setup import foreground

HERE=Path(__file__).resolve().parent


def default_store():
    override=os.environ.get("FLY_ACTIVITY_LEARNING_DIR")
    if override:return Path(override).expanduser().resolve()
    return HERE/"learning"


STORE=default_store()
CLASSES=("forward","backward","stopped","lateral")


def atomic_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=".model_",suffix=".json",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(data,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def committed_version(store):
    db=Path(store)/"examples.sqlite3"
    if db.exists():
        with closing(sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True,timeout=30)) as connection:
            row=connection.execute("SELECT version,model_file FROM versions ORDER BY version DESC LIMIT 1").fetchone()
            if row:return row
        return None
    pointer=Path(store)/"current.json"
    if pointer.exists():
        record=json.loads(pointer.read_text(encoding="utf-8"))
        return record["version"],record["model_file"]
    return None


def ensure_store(store=STORE,legacy_dirs=None):
    """Import legacy history once, without moving or deleting the original data."""
    store=Path(store)
    if not (store/"examples.sqlite3").exists() and not (store/"current.json").exists():
        sources=legacy_dirs
        if sources is None:
            sources=(HERE.parent.parent/"fly_activity"/"learning",) if store.resolve()==STORE.resolve() else ()
        candidates=[]
        for source in sources:
            source=Path(source)
            if source.resolve()==store.resolve():continue
            version=committed_version(source)
            if version:
                model=(source/version[1]).resolve()
                if not model.is_relative_to(source.resolve()):raise ValueError("Invalid legacy learning model")
                load_model(model)
                candidates.append((version[0],source))
        if store.exists() and not candidates:
            if not (store/"seed_model.json").exists():
                atomic_json(store/"seed_model.json",load_model(HERE/"learning"/"seed_model.json"))
            return store
        store.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix=".learning_import_",dir=store.parent))
        try:
            if candidates:
                _,source=max(candidates,key=lambda item:item[0])
                for name in ("models","current.json","seed_model.json"):
                    original=source/name
                    if original.is_dir():shutil.copytree(original,staging/name)
                    elif original.is_file():shutil.copy2(original,staging/name)
                db=source/"examples.sqlite3"
                if db.exists():
                    with closing(sqlite3.connect(db.resolve().as_uri()+"?mode=ro",uri=True,timeout=30)) as src, closing(sqlite3.connect(staging/"examples.sqlite3")) as dst:
                        src.backup(dst)
                atomic_json(staging/"migration.json",dict(source=str(source.resolve()),imported_utc=datetime.now(timezone.utc).isoformat()))
            if not (staging/"seed_model.json").exists():
                shutil.copy2(HERE/"learning"/"seed_model.json",staging/"seed_model.json")
            if not store.exists():
                try:staging.rename(store)
                except OSError:
                    if not store.is_dir():raise
            else:
                # A checkout already contains the seed. Publish the database last
                # and retain all original files in the legacy folder.
                for original in sorted(staging.rglob("*")):
                    if not original.is_file() or original.name=="examples.sqlite3":continue
                    target=store/original.relative_to(staging)
                    if target.exists():
                        if target.name!="seed_model.json" and sha256(target)!=sha256(original):
                            raise ValueError(f"Existing learning file differs: {target}")
                        continue
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(original,target)
                if (staging/"examples.sqlite3").exists():
                    os.replace(staging/"examples.sqlite3",store/"examples.sqlite3")
        finally:
            if staging.exists():shutil.rmtree(staging)
    if not (store/"seed_model.json").exists():
        atomic_json(store/"seed_model.json",load_model(HERE/"learning"/"seed_model.json"))
    return store


def active_model_path(store=STORE):
    store=ensure_store(store);pointer=store/"current.json"
    version=committed_version(store)
    if version:
        number,name=version;path=(store/name).resolve()
        if not path.is_relative_to(store.resolve()) or not path.is_file():raise ValueError("Invalid current learning model")
        model=load_model(path)
        record=dict(version=number,model_file=name,updated_utc=model["created_utc"],
                    labels=model["cumulative_labels"],behavior_samples=model["behavior_training_samples"],
                    head_samples=model["cumulative_head_samples"])
        try:current=json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError,ValueError):current=None
        if current!=record:atomic_json(pointer,record)
        return path
    if pointer.exists():
        raise ValueError("Learning pointer has no committed model; retain the learning folder for recovery")
    seed=store/"seed_model.json"
    return seed if seed.exists() else Path(__file__).with_name("head_model.json")


def training_frames(model,video_sha):
    sources=model.get("training_sources",{})
    if video_sha in sources:return sorted(set(sources[video_sha]))
    if model.get("training_video_sha256")==video_sha:return sorted({int(k["frame"]) for k in model.get("training_keys",[])})
    return []


def motion_features(appearance,axis,vx,vy,ratio,scale):
    if appearance is None or not np.all(np.isfinite(appearance)):return None
    axis=np.asarray(axis,float);length=np.linalg.norm(axis)
    if not np.all(np.isfinite(axis)) or length<.5 or not np.isfinite(vx+vy+ratio):return None
    axis/=length;speed=float(np.hypot(vx,vy))
    cosine=float(np.dot(axis,[vx,vy])/speed) if speed>1e-8 else 0.
    normalized=np.clip(np.asarray(appearance)/np.asarray(scale),-8,8)
    return np.r_[normalized*cosine,np.log1p(speed)/2,abs(cosine),np.log1p(ratio)/2]


def behavior_prediction(features,model):
    memory=model.get("behavior_memory",[])
    if features is None or not memory:return None,0.
    x=np.array([r["features"] for r in memory]);delta=x-features
    distance2=.65*np.mean(delta[:,:44]**2,axis=1)+.35*np.mean(delta[:,44:]**2,axis=1)
    nearest=np.argsort(distance2)[:5]
    weights=np.exp(-distance2[nearest]/(2*.3**2))
    scores={label:sum(w for i,w in zip(nearest,weights) if memory[int(i)]["label"]==label) for label in CLASSES}
    label=max(scores,key=scores.get)
    agreement=scores[label]/max(sum(scores.values()),1e-12)
    similarity=float(weights[0]) if len(weights) else 0.
    confidence=similarity*agreement
    return (label if confidence>=.8 else None),confidence


def fit_head(seed,records):
    x=list(seed.get("training_features",[]));y=list(seed.get("training_labels",[]))
    if not x:raise ValueError("Seed model lacks original training data; initialize the learning store first")
    for record in records:
        if record.get("head_target") is not None and record.get("appearance") is not None:
            x.append(record["appearance"]);y.append(record["head_target"])
    x=np.asarray(x,float);y=np.asarray(y,float)
    scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-3);z=x/scale
    def objective(w):
        score=z@w
        return (float(np.mean(np.logaddexp(0,score)-y*score)+.12*np.dot(w,w)/2),
                z.T@(expit(score)-y)/len(y)+.12*w)
    fit=minimize(objective,np.zeros(z.shape[1]),jac=True,method="L-BFGS-B")
    if not fit.success:raise ValueError(f"Cumulative model fit failed: {fit.message}")
    model=dict(seed,feature_scale=scale.tolist(),weights=fit.x.tolist(),training_samples=len(y))
    model["training_agreement"]=float(np.mean((expit(z@fit.x)>=.5)==y))
    model["behavior_memory"]=[]
    sources={seed["training_video_sha256"]:sorted({int(k["frame"]) for k in seed["training_keys"]})}
    for r in records:
        if r.get("behavior") is not None:
            # Retain raw appearance so normalization stays consistent after head updates.
            feature=motion_features(r["appearance"],r["axis"],*r["velocity"],r["axis_ratio"],scale)
            model["behavior_memory"].append(dict(features=feature.tolist(),label=r["human_label"],key=r["key"]))
            sources.setdefault(r["video_sha256"],[]).append(r["frame"])
    model["training_sources"]={k:sorted(set(v)) for k,v in sources.items()}
    return model


class LearningStore:
    def __init__(self,folder=STORE):
        self.folder=ensure_store(folder)
        self.db=self.folder/"examples.sqlite3"
        with closing(self.connect()) as connection, connection:
            connection.execute("CREATE TABLE IF NOT EXISTS examples (key TEXT PRIMARY KEY, payload TEXT NOT NULL, label TEXT NOT NULL, updated TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS revisions (id INTEGER PRIMARY KEY, key TEXT, previous_payload TEXT, created TEXT)")
            connection.execute("CREATE TABLE IF NOT EXISTS versions (version INTEGER PRIMARY KEY, model_file TEXT NOT NULL, created TEXT NOT NULL)")

    def connect(self):return sqlite3.connect(self.db,timeout=30)

    def update(self,record):
        now=datetime.now(timezone.utc).isoformat();seed=load_model(self.folder/"seed_model.json")
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            old=connection.execute("SELECT payload FROM examples WHERE key=?",(record["key"],)).fetchone()
            if old:
                previous=json.loads(old[0])
                if previous["human_label"]==record["human_label"]:
                    current=connection.execute("SELECT MAX(version) FROM versions").fetchone()[0] or 0
                    return dict(status="already_registered",version=current,learned=previous.get("behavior") is not None)
                connection.execute("INSERT INTO revisions(key,previous_payload,created) VALUES (?,?,?)",(record["key"],old[0],now))
            connection.execute("INSERT OR REPLACE INTO examples VALUES (?,?,?,?)",
                               (record["key"],json.dumps(record,ensure_ascii=False),record["human_label"],now))
            records=[json.loads(r[0]) for r in connection.execute("SELECT payload FROM examples ORDER BY key")]
            model=fit_head(seed,records)
            version=max(connection.execute("SELECT MAX(version) FROM versions").fetchone()[0] or 0,
                        max((int(p.stem.split('_')[-1]) for p in (self.folder/'models').glob('model_*.json') if p.stem.split('_')[-1].isdigit()),default=0))+1
            name=f'models/model_{version:06d}.json'
            model.update(learning_version=version,created_utc=now,
                         cumulative_labels=len(records),behavior_training_samples=len(model["behavior_memory"]),
                         cumulative_head_samples=sum(r.get("head_target") is not None for r in records))
            atomic_json(self.folder/name,model)
            connection.execute("INSERT INTO versions VALUES (?,?,?)",(version,name,now))
        # The database commit is authoritative. Recover the pointer on startup if
        # the process exits between committing and publishing this cache.
        active_model_path(self.folder)
        return dict(status=record["extraction_status"],version=version,learned=record.get("behavior") is not None)

    def import_history(self,source):
        """Union teachers from an archived PC history; preserve corrections as revisions."""
        with closing(sqlite3.connect(Path(source).resolve().as_uri()+'?mode=ro',uri=True)) as remote:
            incoming=remote.execute('SELECT key,payload,label,updated FROM examples ORDER BY key').fetchall()
        changed=0
        for key,payload,label,updated in incoming:
            with closing(self.connect()) as local:
                previous=local.execute('SELECT label,updated FROM examples WHERE key=?',(key,)).fetchone()
            if previous and previous[0]==label:continue
            if previous and previous[1]>=updated:
                with closing(self.connect()) as local,local:
                    if not local.execute('SELECT 1 FROM revisions WHERE key=? AND previous_payload=?',(key,payload)).fetchone():
                        local.execute('INSERT INTO revisions(key,previous_payload,created) VALUES (?,?,?)',(key,payload,updated))
                continue
            self.update(json.loads(payload));changed+=1
            with closing(self.connect()) as local,local:
                local.execute('UPDATE examples SET updated=? WHERE key=?',(updated,key))
        return changed


class TeacherExtractor:
    """Match the original evaluated detection, not a guessed target after re-tracking."""
    def __init__(self,result_dir):
        import csv
        self.folder=Path(result_dir)
        self.metadata=json.loads((self.folder/"metadata.json").read_text(encoding="utf-8"))
        self.config=json.loads((self.folder/"config_used.json").read_text(encoding="utf-8"))
        self.config.setdefault("body_detection_mode","background_diff")
        self.video=Path(self.metadata.get("direction_video",self.metadata.get("source","")))
        self.video_sha=self.metadata.get("video_sha256") or sha256(self.video)
        def rows(name):
            with (self.folder/name).open(encoding="utf-8-sig",newline="") as f:
                return {(int(r["frame"]),int(r["slot"])):r for r in csv.DictReader(f)}
        self.tracks=rows("tracks.csv");self.activity=rows("activity.csv")
        self.cache=None
        if (self.folder/"appearance_features.npz").exists():
            with np.load(self.folder/"appearance_features.npz",allow_pickle=False) as data:self.cache=data["features"]
        self.cap=None;self.background=None

    def appearance(self,key,axis):
        if self.cache is not None:
            value=self.cache[key[0],key[1]-1]
            if np.all(np.isfinite(value)):return value
        if self.cap is None:
            self.cap=cv2.VideoCapture(str(self.video))
            bg=cv2.imread(str(self.folder/"background.png"),0)
            if bg is None:raise ValueError("Original background unavailable for teaching")
            self.background=cv2.GaussianBlur(bg,(self.config["blur_size"],)*2,0)
        self.cap.set(1,key[0]);ok,image=self.cap.read()
        if not ok:raise ValueError("Cannot read evaluated frame")
        gray=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(self.config["blur_size"],)*2,0)
        difference,mask=foreground(gray,self.background,self.config)
        n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
        target=self.tracks[key]
        center=np.array([float(target["x_px"]),float(target["y_px"])])
        if n<2:return None
        i=1+int(np.argmin(np.linalg.norm(centers[1:]-center,axis=1)))
        if np.linalg.norm(centers[i]-center)>3:return None
        x,y,w,h=stats[i,:4];yy,xx=np.nonzero(labels[y:y+h,x:x+w]==i)
        return appearance_features(np.column_stack((xx,yy)),difference[yy+y,xx+x],axis)

    def extract(self,sample,seed):
        frame,slot=int(sample["frame"]),int(sample["slot"]);key=frame,slot;track=self.tracks[key];activity=self.activity[key]
        label=sample["human_label"]
        position=[track.get("x_px",""),track.get("y_px","")]
        identity=f'{self.video_sha}:{frame}:{position[0]}:{position[1]}'
        if not position[0]:identity+=f':unknown_slot_{slot}'
        record=dict(key=hashlib.sha256(identity.encode()).hexdigest(),video_sha256=self.video_sha,
                    frame=frame,slot=slot,result_dir=str(self.folder.resolve()),human_label=label,
                    frozen_prediction=sample["prediction"],original_model_sha256=self.metadata.get("head_model_sha256"),
                    appearance=None,behavior=None,head_target=None,extraction_status="label_saved_not_trainable")
        if label=="unknown":
            record["extraction_status"]="label_saved_unknown"
            return record
        if label not in CLASSES:return record
        if track["status"]!="observed" or activity.get("valid_interval")!="1":return record
        axis=np.array([float(track["axis_dx"]),float(track["axis_dy"])])
        ratio=float(track["axis_ratio"])
        if ratio<self.config["direction_min_axis_ratio"]:return record
        current_time=float(activity["time_s"]);window=self.config["speed_window_s"]
        points=[]
        for k in range(frame,-1,-1):
            row=self.activity.get((k,slot))
            if row is None or row.get("segment")!=activity["segment"] or not row.get("x_smooth_px"):break
            if k<frame and self.activity[k+1,slot].get("valid_interval")!="1":break
            points.append(row)
            if current_time-float(row["time_s"])>=window:break
        if len(points)<2:return record
        past=points[-1];duration=current_time-float(past["time_s"])
        if duration<=0:return record
        from analyze_flies import window_speed
        points=points[::-1];times=np.array([float(p["time_s"]) for p in points])
        positions=np.array([[float(p["x_smooth_px"]),float(p["y_smooth_px"])] for p in points])
        delta=np.vstack((np.full((1,2),np.nan),np.diff(positions,axis=0)))
        velocity=np.array([window_speed(delta[:,j],times,window)[-1] for j in range(2)])
        speed=np.linalg.norm(velocity)
        if label in ("forward","backward") and speed<self.config["direction_min_speed_px_s"]:
            record["extraction_status"]="label_saved_motion_too_small";return record
        appearance=self.appearance(key,axis)
        if appearance is None:return record
        feature=motion_features(appearance,axis,*velocity,ratio,seed["feature_scale"])
        record.update(appearance=appearance.tolist(),behavior=feature.tolist(),axis=axis.tolist(),
                      velocity=velocity.tolist(),axis_ratio=ratio,extraction_status="behavior_learned")
        cosine=float(np.dot(axis/np.linalg.norm(axis),velocity)/speed) if speed else 0.
        if label in ("forward","backward") and abs(cosine)>=.6:
            target=int(cosine>0)
            record.update(head_target=target if label=="forward" else 1-target,extraction_status="head_and_behavior_learned")
        return record

    def close(self):
        if self.cap is not None:self.cap.release()


def import_labels(result_dir,store=STORE):
    import csv
    learner=LearningStore(store);seed=load_model(Path(store)/"seed_model.json");extractor=TeacherExtractor(result_dir)
    summary={};last_version=0
    try:
        for name in ("human_evaluation.csv","human_learning.csv"):
            path=Path(result_dir)/name
            if not path.exists():continue
            with path.open(encoding="utf-8-sig",newline="") as f:
                for sample in csv.DictReader(f):
                    if sample.get("human_label"):
                        outcome=learner.update(extractor.extract(sample,seed));last_version=outcome["version"]
                        summary[outcome["status"]]=summary.get(outcome["status"],0)+1
    finally:extractor.close()
    atomic_json(Path(result_dir)/"learning_status.json",dict(last_model_version=last_version,import_counts=summary,
                                                            original_prediction_model=extractor.metadata.get("head_model_sha256"),
                                                            predictions_frozen=True,historical_labels_imported=True))
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-results",type=Path,nargs="+",required=True)
    parser.add_argument("--store",type=Path,default=STORE)
    args=parser.parse_args()
    for folder in args.import_results:print(folder,json.dumps(import_labels(folder,args.store),ensure_ascii=False))
