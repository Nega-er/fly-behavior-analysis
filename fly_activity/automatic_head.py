"""Appearance-only head polarity model; never use movement as a head label."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def appearance_features(points, contrast, axis):
    """Reflection-odd normalized silhouette/contrast profiles (44 features).

    Reversing an unsigned body axis negates every feature. Consequently the
    predicted two-end probabilities swap exactly, independent of eigenvector sign.
    """
    p = np.asarray(points, float)
    axis = np.asarray(axis, float)
    p = p-p.mean(axis=0)
    axis = axis/np.linalg.norm(axis)
    x, y = p@axis, p@np.array([-axis[1],axis[0]])
    sx, sy = max(np.std(x),1e-6), max(np.std(y),1e-6)
    x, y = x/sx, np.abs(y)/sy
    edges_x = np.linspace(-4,4,9)
    edges_y = [0,.5,1,1.5,2,5]
    weights = np.maximum(np.asarray(contrast,float),0)
    if not np.any(weights):
        weights = np.ones(len(p))
    shape = np.histogram2d(x,y,bins=[edges_x,edges_y])[0]/len(p)
    intensity = np.histogram2d(x,y,bins=[edges_x,edges_y],weights=weights)[0]/weights.sum()
    profile = np.r_[(shape[4:]-shape[:4][::-1]).ravel(),
                     (intensity[4:]-intensity[:4][::-1]).ravel()]
    moments = [np.mean(x**3), np.mean(x*y*y), np.average(x,weights=weights),
               np.mean(np.tanh(x)*y)]
    return np.r_[profile,moments]


def load_model(path):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("feature_version") != 1 or len(model.get("weights",[])) != 44:
        raise ValueError("Unsupported head appearance model")
    if len(model.get("feature_scale",[])) != 44 or np.any(np.asarray(model["feature_scale"]) <= 0):
        raise ValueError("Invalid model feature scale")
    return model


def predict_probability(features, model):
    z = np.asarray(features,float)/np.asarray(model["feature_scale"])
    return float(expit(z@np.asarray(model["weights"])))


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def train(video, tracks_path, anchors_path, config_path, out):
    from direction import body_axis, read_anchors
    from analyze_flies import DEFAULTS
    from video_setup import foreground
    config = dict(DEFAULTS, **json.loads(Path(config_path).read_text(encoding="utf-8")))
    bg_path=Path(config["background_path"])
    if not bg_path.is_absolute():
        bg_path=Path(config_path).resolve().parent/bg_path
    bg=cv2.imread(str(bg_path),0)
    if bg is None:
        raise ValueError("Training requires the original background image")
    kernel=(config["blur_size"],)*2
    bg=cv2.GaussianBlur(bg,kernel,0)
    with Path(tracks_path).open(encoding="utf-8-sig",newline="") as f:
        rows={(int(r["frame"]),int(r["slot"])):r for r in csv.DictReader(f)}
    anchors=read_anchors(anchors_path)
    cap=cv2.VideoCapture(str(video))
    features,labels,training_keys=[],[],[]
    try:
        for key,head in anchors.items():
            row=rows[key]
            cap.set(cv2.CAP_PROP_POS_FRAMES,key[0]);ok,image=cap.read()
            if not ok:
                raise ValueError(f"Cannot read training frame {key[0]}")
            gray=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),kernel,0)
            difference,mask=foreground(gray,bg,config)
            _,components,stats,centers=cv2.connectedComponentsWithStats(mask)
            center=np.array([float(row["x_px"]),float(row["y_px"])])
            component=1+int(np.argmin(np.linalg.norm(centers[1:]-center,axis=1)))
            if np.linalg.norm(centers[component]-center)>3:
                continue
            yy,xx=np.nonzero(components==component)
            points=np.column_stack((xx,yy))
            axis,ratio=body_axis(points)
            if ratio<config["direction_min_axis_ratio"]:
                continue
            features.append(appearance_features(points,difference[yy,xx],axis))
            labels.append(int(np.dot(head-center,axis)>0))
            training_keys.append(dict(frame=key[0],slot=key[1]))
    finally:
        cap.release()
    if len(features)<20:
        raise ValueError("At least 20 usable human head labels are required for this model")
    x,y=np.asarray(features),np.asarray(labels)
    feature_scale=np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-3)
    x=x/feature_scale
    # No intercept: reflected features must produce complementary probabilities.
    regularization=.12
    def loss(w):
        scores=x@w
        value=np.mean(np.logaddexp(0,scores)-y*scores)+regularization*np.dot(w,w)/2
        gradient=x.T@(expit(scores)-y)/len(y)+regularization*w
        return value,gradient
    fit=minimize(loss,np.zeros(x.shape[1]),jac=True,method="L-BFGS-B")
    if not fit.success:
        raise ValueError(f"Model fit failed: {fit.message}")
    model=dict(feature_version=1,method="regularized appearance logistic regression; no movement features",
               feature_scale=feature_scale.tolist(),weights=fit.x.tolist(),
               training_video_sha256=sha256(video),training_video_name=Path(video).name,
               training_samples=len(y),training_keys=training_keys,
               training_features=np.asarray(features).tolist(),training_labels=np.asarray(labels).tolist(),
               training_agreement=float(np.mean((expit(x@fit.x)>=.5)==y)),
               note="Training agreement is not an independent accuracy estimate. Evaluate on blinded human labels and preferably different videos.")
    Path(out).write_text(json.dumps(model,indent=2,ensure_ascii=False),encoding="utf-8")
    return model


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("video","tracks","anchors","config","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    a=parser.parse_args()
    if a.out.exists():
        parser.error("Choose a new model path; existing models are not overwritten")
    model=train(a.video,a.tracks,a.anchors,a.config,a.out)
    print(json.dumps({k:model[k] for k in ("training_samples","training_agreement","note")},ensure_ascii=False))
