"""Blinded human assessment of frozen automatic predictions."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from automatic_head import sha256

HUMAN_CLASSES=["forward","backward","stopped","lateral"]
PREDICTED_CLASSES=HUMAN_CLASSES+["unknown"]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def save_csv(path,rows,fields):
    path=Path(path)
    fd,temporary=tempfile.mkstemp(prefix=".evaluation_",suffix=".csv",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8-sig",newline="") as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            f.flush();os.fsync(f.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def metrics(rows):
    completed=[r for r in rows if r.get("human_label")]
    scorable=[r for r in completed if r["human_label"] in HUMAN_CLASSES]
    matrix={h:{p:0 for p in PREDICTED_CLASSES} for h in HUMAN_CLASSES}
    for r in scorable:
        if r["prediction"] not in PREDICTED_CLASSES:
            raise ValueError("Invalid frozen prediction")
        matrix[r["human_label"]][r["prediction"]]+=1
    n=len(scorable);known=sum(r["prediction"]!="unknown" for r in scorable)
    correct=sum(r["prediction"]==r["human_label"] for r in scorable)
    classes={}
    for label in HUMAN_CLASSES:
        tp=matrix[label][label]
        predicted=sum(matrix[h][label] for h in HUMAN_CLASSES)
        reference=sum(matrix[label].values())
        precision=tp/predicted if predicted else None
        recall=tp/reference if reference else None
        f1=(2*tp/(predicted+reference)) if predicted+reference else None
        classes[label]=dict(precision=precision,recall=recall,f1=f1,reference_samples=reference)
    supported=[classes[h]["f1"] for h in HUMAN_CLASSES if classes[h]["reference_samples"]]
    return dict(completed_samples=len(completed),scorable_samples=n,
                unscorable_samples=sum(r.get("human_label") in ("unknown","unscorable") for r in completed),
                human_unknown_samples=sum(r.get("human_label")=="unknown" for r in completed),
                accuracy_including_unknown_as_error=correct/n if n else None,
                prediction_coverage=known/n if n else None,
                accuracy_on_classified_samples=correct/known if known else None,
                macro_f1_present_reference_classes=float(np.mean(supported)) if supported else None,
                classes=classes,confusion_matrix=matrix,
                note="Unknown predictions count as errors for scorable human labels. Unidentifiable target animals are unscorable, so this is not an independent detection/identity accuracy test. Nearby video samples are correlated; sample counts are not independent experimental replicates.")


def make_samples(result_dir,count=80,seed=0,sampling="random"):
    if sampling not in ("random","learning"):raise ValueError("Invalid sampling mode")
    result_dir=Path(result_dir)
    metadata=json.loads((result_dir/"metadata.json").read_text(encoding="utf-8"))
    if metadata.get("head_mode")!="automatic_appearance":
        raise ValueError("Evaluate an automatic-head run, not a manually corrected head run")
    rows=read_csv(result_dir/"activity.csv")
    times=sorted({float(r["time_s"]) for r in rows})
    frame_dt=float(np.median(np.diff(times)))
    # Keep test clips away from human frames used to fit this video's head model.
    exclusions=set()
    radius=max(1,int(np.ceil(.4/frame_dt)))
    for frame in metadata.get("training_frame_indices",[]):
        exclusions.update(range(max(0,frame-radius),frame+radius+1))
    config_path=result_dir/"config_used.json"
    threshold=json.loads(config_path.read_text(encoding="utf-8")).get("automatic_head_min_probability",.6) if config_path.exists() else .6
    candidates=[r for r in rows if int(r["frame"])>0 and int(r["frame"]) not in exclusions]
    if not candidates:
        raise ValueError("No independent test clips remain in this video. Choose another video for evaluation.")
    rng=np.random.default_rng(seed)
    reasons={}
    if sampling=="learning":
        candidates=[r for r in candidates if r.get("x_smooth_px","") and r.get("y_smooth_px","")
                    and r.get("valid_interval")=="1"
                    and np.isfinite(float(r["x_smooth_px"])+float(r["y_smooth_px"]))]
        if not candidates:raise ValueError("学習用に確認できる追跡区間がありません。別の結果を選択してください。")
        def uncertain(row):
            try:value=float(row.get("automatic_head_confidence", ""))
            except (ValueError,TypeError):return False
            return np.isfinite(value) and value<threshold
        pools=[("unknown",[i for i,r in enumerate(candidates) if r["locomotion_direction"]=="unknown"]),
               ("backward",[i for i,r in enumerate(candidates) if r["locomotion_direction"]=="backward"]),
               ("low_head_confidence",[i for i,r in enumerate(candidates) if uncertain(r)]),
               ("random",list(range(len(candidates))))]
        selected=[];selected_times={}
        def add(pool,reason,limit):
            added=0
            for index in rng.permutation(pool):
                index=int(index);row=candidates[index];slot=int(row["slot"]);time=float(row["time_s"])
                if index in reasons or any(abs(time-t)<.6-1e-9 for t in selected_times.get(slot,[])):continue
                selected.append(index);reasons[index]=reason;selected_times.setdefault(slot,[]).append(time)
                added+=1
                if added>=limit:break
        for position,(reason,pool) in enumerate(pools):
            quota=count//4+int(position<count%4)
            if quota:add(pool,reason,quota)
        if len(selected)<count:add(list(range(len(candidates))),"random_fill",count-len(selected))
        rng.shuffle(selected)
    else:
        selected=rng.choice(len(candidates),size=min(count,len(candidates)),replace=False)
    # Keep predictions hidden in both modes; only random mode includes failed tracking.
    samples=[]
    for index in selected:
        r=candidates[int(index)]
        samples.append(dict(sample_id=len(samples)+1,frame=r["frame"],slot=r["slot"],
                            time_s=r["time_s"],prediction=r["locomotion_direction"],human_label="",
                            selection_reason=reasons.get(int(index),"random")))
    return samples,dict(activity_sha256=sha256(result_dir/"activity.csv"),
                         video_sha256=metadata.get("video_sha256"),seed=seed,
                         requested_samples=count,sample_count=len(samples),
                         same_video_as_training=metadata.get("same_video_as_head_training",False),
                         excluded_training_buffer_s=.4,
                         sampling_mode=sampling,
                         selection_counts={reason:sum(r["selection_reason"]==reason for r in samples) for reason in sorted({r["selection_reason"] for r in samples})},
                         low_head_confidence_threshold=threshold,
                         minimum_same_slot_spacing_s=.6 if sampling=="learning" else None,
                         sampling="priority Unknown/backward/low head confidence plus random; not unbiased accuracy" if sampling=="learning" else "uniform over frame-slot pairs, predictions hidden from reviewer",
                         note="Same-video testing, even with unused clips, is not independent-video generalization testing.")


def write_report(result_dir,rows,provenance):
    result_dir=Path(result_dir)
    prefix="learning" if provenance.get("sampling_mode")=="learning" else "evaluation"
    report=metrics(rows);report["provenance"]=provenance
    report["representative_accuracy"]=prefix=="evaluation"
    (result_dir/f"{prefix}_metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    matrix=[dict(human=h,**report["confusion_matrix"][h]) for h in HUMAN_CLASSES]
    save_csv(result_dir/f"{prefix}_confusion.csv",matrix,["human"]+PREDICTED_CLASSES)
    accuracy=report["accuracy_including_unknown_as_error"]
    summary="未評価" if accuracy is None else f'{accuracy:.1%}'
    text=(("# 優先抽出した学習例の入力集計\n\n動画全体の精度を表す値ではありません。\n\n" if prefix=="learning" else "# 自動前進・後退の人手評価\n\n")
          +f"判定可能な人手評価：{report['scorable_samples']}件\n\n"
          f"unknownも誤りに含めた一致率：{summary}\n\n"
          f"人のUnknown・判断不能：{report['unscorable_samples']}件（精度の分母から除外）\n\n"
          "クラス別の適合率・再現率・F1、判定率、混同行列は対応するmetrics.jsonとconfusion.csvに保存。\n\n"
          +("自動予測は評価元モデルのまま固定しています。評価入力は累積学習へ使用し、新しいモデルの精度とは区別します。\n\n"
           if provenance.get("cumulative_learning_enabled") else "自動予測は変更していません。この評価は学習に使用しません。\n\n")
          +("この評価は学習元と同じ動画の未使用区間です。別動画への精度とは区別してください。\n"
            if provenance.get("same_video_as_training") else "学習元とは異なる動画の評価です。\n"))
    (result_dir/f"{prefix}_report.md").write_text(text,encoding="utf-8")
    return report


def review_clip_plan(activity,frame,slot,fps,width,height,last_frame):
    """Crop the uninterrupted target trajectory; never guess across tracking gaps."""
    def point(k):
        row=activity.get((k,slot))
        if row is None:return None
        try:
            x,y=float(row["x_smooth_px"]),float(row["y_smooth_px"])
        except (KeyError,TypeError,ValueError):return None
        if not np.isfinite(x+y) or not (0<=x<width and 0<=y<height):return None
        return x,y
    center=point(frame)
    if center is None:return None
    segment=activity[frame,slot].get("segment")
    points={frame:center};start=end=frame
    radius=max(1,round(.3*fps))
    for step in (-1,1):
        for offset in range(1,radius+1):
            k=frame+step*offset
            if not 0<=k<=last_frame:break
            p=point(k)
            if p is None or activity[k,slot].get("segment")!=segment:break
            points[k]=p
            start=min(start,k);end=max(end,k)
    # A fixed crop preserves visible movement while covering the whole trajectory.
    def axis_bounds(values,size):
        low,high=min(values),max(values)
        span=min(size,max(180,int(np.ceil(high-low))+90))
        origin=max(0,min(int((low+high-span)/2),size-span))
        return origin,origin+span
    x0,x1=axis_bounds([p[0] for p in points.values()],width)
    y0,y1=axis_bounds([p[1] for p in points.values()],height)
    return dict(start=start,end=end,bounds=(x0,y0,x1,y1),points=points)


def review(result_dir,count=80,seed=0,smoke_test=False,learn=True,sampling="random"):
    import tkinter as tk
    from tkinter import ttk,messagebox
    import cv2
    from PIL import Image
    from tk_images import photo_image
    from cumulative_learning import LearningStore,TeacherExtractor,STORE,atomic_json
    from automatic_head import load_model
    result_dir=Path(result_dir)
    samples_file=result_dir/("human_learning.csv" if sampling=="learning" else "human_evaluation.csv")
    provenance_file=result_dir/("learning_sampling.json" if sampling=="learning" else "evaluation_sampling.json")
    if samples_file.exists():
        samples=read_csv(samples_file)
        # A still-open older reviewer may truncate its CSV again on closing.
        recovery=samples_file.with_suffix(".csv.recovered")
        if not samples and recovery.exists():samples=read_csv(recovery)
        provenance=json.loads(provenance_file.read_text(encoding="utf-8"))
        if provenance["activity_sha256"]!=sha256(result_dir/"activity.csv"):
            raise ValueError("Automatic predictions changed since sampling; use a new result directory")
    else:
        samples,provenance=make_samples(result_dir,count,seed,sampling)
        provenance_file.write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding="utf-8")
    metadata=json.loads((result_dir/"metadata.json").read_text(encoding="utf-8"))
    video=Path(metadata.get("direction_video",metadata.get("source","")))
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError("Original video is unavailable for human evaluation")
    fps=cap.get(5);width,height=int(cap.get(3)),int(cap.get(4))
    activity={(int(r["frame"]),int(r["slot"])):r for r in read_csv(result_dir/"activity.csv")}
    last_frame=max(k[0] for k in activity)
    pending=[i for i,r in enumerate(samples) if not r["human_label"]]
    root=tk.Tk();root.title("人手学習（優先抽出・自動判定は非表示）" if sampling=="learning" else "人手による精度評価（自動判定は非表示）")
    if smoke_test:root.withdraw()
    prompt=ttk.Label(root,text="");prompt.pack(padx=15,pady=10)
    target_message=tk.StringVar()
    ttk.Label(root,textvariable=target_message,wraplength=650).pack(padx=15,pady=4)
    image_label=ttk.Label(root);image_label.pack(padx=15,pady=5)
    ttk.Label(root,text="白い円の個体について、中央時刻の行動を判定してください。自動の答えは表示しません。\n"
                        "F 前進 / B 後退 / S 停止 / L 横方向等 / U Unknown（判定不能）。閉じても続きから再開できます。").pack(padx=15,pady=8)
    ttk.Label(root,text="Unknown・後退候補・頭尾低確信度を優先。全体精度の評価にはランダム抽出を使ってください。" if sampling=="learning" else "動画全体からランダム抽出した評価用の場面です。",wraplength=650).pack(pady=4)
    learning_enabled=tk.BooleanVar(value=learn)
    ttk.Checkbutton(root,text="入力ごとに累積学習する（OFFなら評価専用）",variable=learning_enabled).pack(pady=4)
    learning_message=tk.StringVar(value="評価時の自動予測は固定し、更新モデルは次回の解析から使います。")
    ttk.Label(root,textvariable=learning_message,wraplength=650).pack(pady=4)
    learner=None;extractor=None;seed_model=None
    position=[0];clip=[];clip_index=[0];job=[None];target_available=[False]
    scoring_buttons=[]
    fields=["sample_id","frame","slot","time_s","prediction","human_label","learning_status","learned_model_version","selection_reason"]
    def save():
        save_csv(samples_file,samples,fields);write_report(result_dir,samples,provenance)
    def show_frame():
        if not clip:
            return
        image_label.photo=photo_image(Image.fromarray(clip[clip_index[0] % len(clip)]))
        image_label.configure(image=image_label.photo)
        clip_index[0]+=1
        job[0]=root.after(max(20,round(1000/fps)),show_frame)
    def load_sample():
        if job[0]:
            root.after_cancel(job[0]);job[0]=None
        if position[0]>=len(pending):
            save();messagebox.showinfo("評価完了","精度集計を保存しました。",parent=root);root.destroy();return
        sample=samples[pending[position[0]]]
        frame,slot=int(sample["frame"]),int(sample["slot"])
        prompt.configure(text=f"評価 {position[0]+1}/{len(pending)}  個体表示番号 {slot}  中央時刻 {float(sample['time_s']):.2f}秒")
        clip.clear();clip_index[0]=0
        image_label.configure(image="");image_label.photo=None
        target_available[0]=False
        plan=review_clip_plan(activity,frame,slot,fps,width,height,last_frame)
        if plan is not None:
            x0,y0,x1,y1=plan["bounds"]
            cap.set(1,plan["start"])
            for k in range(plan["start"],plan["end"]+1):
                ok,img=cap.read()
                if not ok:break
                x,y=plan["points"][k]
                cv2.circle(img,(round(x),round(y)),17,(255,255,255),1)
                crop=img[y0:y1,x0:x1]
                if crop.size==0:continue
                factor=min(520/crop.shape[1],420/crop.shape[0])
                crop=cv2.resize(crop,None,fx=factor,fy=factor)
                cv2.putText(crop,"CENTER" if k==frame else f'{k/fps:.2f}s',(8,25),0,.6,(255,255,255),1)
                clip.append(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
                if k==frame:target_available[0]=True
        if not target_available[0]:
            clip.clear()
            target_message.set("評価時刻の対象個体を確認できません（座標欠測または動画読込失敗）。「Unknown U」で次へ進んでください。")
        else:
            target_message.set("白い円が対象個体です。追跡が連続している区間だけを拡大表示しています。個体を見分けられない場合は「Unknown U」を選んでください。")
        for button in scoring_buttons:button.configure(state="normal" if target_available[0] else "disabled")
        show_frame()
    def label(value):
        nonlocal learner,extractor,seed_model
        if position[0]>=len(pending):return
        if value not in ("unknown","unscorable") and not target_available[0]:return
        sample=samples[pending[position[0]]];sample["human_label"]=value;save()
        if learning_enabled.get():
            provenance["cumulative_learning_enabled"]=True
            try:
                if learner is None:
                    learner=LearningStore();extractor=TeacherExtractor(result_dir);seed_model=load_model(STORE/"seed_model.json")
                outcome=learner.update(extractor.extract(sample,seed_model))
                sample["learning_status"]=outcome["status"];sample["learned_model_version"]=outcome["version"]
                learning_message.set(f"モデル v{outcome['version']} に保存しました。"+("教師データとして学習しました。" if outcome["learned"] else "この例は学習条件を満たさないため評価のみ保存しました。"))
                atomic_json(result_dir/"learning_status.json",dict(last_model_version=outcome["version"],last_status=outcome["status"],
                                                                  original_prediction_model=metadata.get("head_model_sha256"),
                                                                  predictions_frozen=True))
            except Exception as e:
                sample["learning_status"]="learning_failed_label_saved"
                learning_message.set("評価は保存済みです。学習の更新に失敗しました。")
                messagebox.showerror("学習更新エラー",str(e),parent=root)
        else:sample["learning_status"]="evaluation_only"
        provenance_file.write_text(json.dumps(provenance,indent=2,ensure_ascii=False),encoding="utf-8")
        save();position[0]+=1;load_sample()
    buttons=ttk.Frame(root);buttons.pack(pady=10)
    for text,value,key in [("前進 F","forward","f"),("後退 B","backward","b"),("停止 S","stopped","s"),
                           ("横方向等 L","lateral","l"),("Unknown U","unknown","u")]:
        button=ttk.Button(buttons,text=text,command=lambda v=value:label(v));button.pack(side="left",padx=3)
        if value not in ("unknown","unscorable"):scoring_buttons.append(button)
        root.bind(key,lambda event,v=value:label(v))
        root.bind(key.upper(),lambda event,v=value:label(v))
    root.protocol("WM_DELETE_WINDOW",lambda:(save(),root.destroy()))
    save();load_sample()
    try:
        if smoke_test:
            root.update_idletasks();root.update();root.destroy()
            print("Blinded review GUI constructed and sample clip rendered; no human labels submitted.")
        else:root.mainloop()
    finally:
        cap.release()
        if extractor is not None:extractor.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--samples",type=int,default=80)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--sampling",choices=("random","learning"),default="random",help="Random evaluation or priority learning examples")
    parser.add_argument("--smoke-test",action="store_true")
    parser.add_argument("--no-learn",action="store_true",help="Independent evaluation only")
    a=parser.parse_args()
    if a.samples<1:parser.error("samples must be positive")
    review(a.results,a.samples,a.seed,a.smoke_test,not a.no_learn,a.sampling)
