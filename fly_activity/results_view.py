"""Read-only desktop result dashboard and local preview player."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import tkinter as tk
from tkinter import ttk,messagebox

import cv2
from PIL import Image
from tk_images import photo_image

LABELS={"forward":"前進","backward":"後退","stopped":"停止","lateral":"横方向等","unknown":"不明"}
REASONS={"tracking_or_activity_unknown":"追跡または移動状態が不明", "tracking_missing":"個体の検出・対応付けが欠測",
         "body_axis_unreliable":"体軸の形状が不明瞭", "automatic_head_uncertain":"自動の頭尾推定が不確実",
         "head_unconfirmed":"頭側の確認なし", "heading_discontinuity":"体軸の急な変化",
         "insufficient_heading_supported_motion":"頭向きが有効な移動区間が不足"}


def number(value):
    try:
        result=float(value)
        return result if math.isfinite(result) else None
    except (TypeError,ValueError):return None


def formatted(value,percent=False,digits=2):
    value=number(value)
    if value is None:return "—"
    return f'{value:.1%}' if percent else f'{value:,.{digits}f}'


def json_file(path,required=False):
    if not path.exists():
        if required:raise ValueError(f"結果ファイルがありません: {path.name}")
        return {}
    with path.open(encoding="utf-8") as f:return json.load(f)


def csv_rows(path):
    if not path.exists():return []
    with path.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))


def load_results(folder,progress=None):
    """Load summary files and stream unknown reasons without mutating results."""
    if progress:progress("load",0,4,"集計ファイルを読込")
    folder=Path(folder).resolve()
    group=json_file(folder/"summary_group.json",True)
    slots=csv_rows(folder/"summary_by_slot.csv")
    directions=csv_rows(folder/"direction_summary.csv")
    totals={name:sum(number(r.get(name+"_s")) or 0 for r in directions) for name in LABELS}
    total_time=sum(totals.values())
    reasons=Counter();last_time={}
    activity_path=folder/"activity.csv"
    if progress:progress("load",1,4,"行動の時系列を読込")
    if activity_path.exists():
        with activity_path.open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                slot=row["slot"];time=number(row.get("time_s"))
                dt=max(0,time-last_time.get(slot,time)) if time is not None else 0
                last_time[slot]=time
                if row.get("locomotion_direction",row.get("state"))=="unknown":
                    reason=row.get("direction_reason",row.get("activity_reason","unknown"))
                    reasons[reason]+=dt
    if progress:progress("load",2,4,"表示用の図を準備")
    rate_image=folder/"movement_rate.png" if (folder/"movement_rate.png").exists() else None
    if rate_image is None and (folder/"time_bins.csv").exists():
        from movement_rate import summarize_rates,plot_rates
        bins=csv_rows(folder/"time_bins.csv")
        if bins:rate_image=plot_rates(summarize_rates(bins,int(group["expected_flies"])),bins)
    statistics=json_file(folder/"period_statistics.json")
    if not statistics:
        from period_statistics import compute_period_statistics
        if activity_path.exists():
            with activity_path.open(encoding="utf-8-sig",newline="") as stream:
                statistics=compute_period_statistics(csv.DictReader(stream),group.get("duration_s"),group.get("unit","px"))
        else:
            statistics=compute_period_statistics([],group.get("duration_s"),group.get("unit","px"))
    if progress:progress("load",3,4,"結果画面を準備")
    return dict(folder=folder,group=group,slots=slots,directions=directions,
                metadata=json_file(folder/"metadata.json"),config=json_file(folder/"config_used.json"),
                population=json_file(folder/"population_estimate.json"),period_statistics=statistics,
                identity_events=csv_rows(folder/'identity_events.csv'),
                evaluation=json_file(folder/"evaluation_metrics.json"),
                learning_status=json_file(folder/"learning_status.json"),
                learning_input=json_file(folder/"learning_metrics.json"),
                totals=totals,total_direction_fly_seconds=total_time,
                unknown_fraction=totals["unknown"]/total_time if total_time else None,
                reasons=reasons,rate_image=rate_image,images={name:folder/name for name in ("report.png","direction_report.png") if (folder/name).exists()},
                previews={name:folder/name for name in ("direction_preview.mp4","tracking_preview.mp4") if (folder/name).exists()})


class ImagePanel(ttk.Frame):
    def __init__(self,parent):
        super().__init__(parent);self.original=None;self.photo=None;self.timer=None
        top=ttk.Frame(self);top.pack(fill="x",padx=5,pady=5)
        self.scale=tk.StringVar(value="幅に合わせる")
        selector=ttk.Combobox(top,textvariable=self.scale,values=["幅に合わせる","50%","100%","150%"],state="readonly",width=15)
        selector.pack(side="left");selector.bind("<<ComboboxSelected>>",lambda event:self.render())
        self.caption=ttk.Label(top);self.caption.pack(side="left",padx=10)
        area=ttk.Frame(self);area.pack(fill="both",expand=True)
        self.canvas=tk.Canvas(area,background="white",highlightthickness=0)
        ys=ttk.Scrollbar(area,orient="vertical",command=self.canvas.yview)
        xs=ttk.Scrollbar(area,orient="horizontal",command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=ys.set,xscrollcommand=xs.set)
        self.canvas.grid(row=0,column=0,sticky="nsew");ys.grid(row=0,column=1,sticky="ns");xs.grid(row=1,column=0,sticky="ew")
        area.rowconfigure(0,weight=1);area.columnconfigure(0,weight=1)
        self.canvas.bind("<Configure>",self.resize)

    def load(self,path):
        self.canvas.delete("all");self.photo=None;self.original=None
        if path is None:
            self.caption.configure(text="この結果にはグラフがありません。")
            return
        if isinstance(path,Image.Image):self.original=path.convert("RGB")
        else:
            with Image.open(path) as image:self.original=image.convert("RGB")
        self.caption.configure(text="上下・左右にスクロールできます。");self.render()

    def resize(self,event):
        if self.timer:self.after_cancel(self.timer)
        self.timer=self.after(100,self.render)

    def render(self):
        self.timer=None
        if self.original is None:return
        w,h=self.original.size
        factor=max(120,self.canvas.winfo_width()-10)/w if self.scale.get()=="幅に合わせる" else float(self.scale.get().rstrip("%"))/100
        image=self.original.resize((max(1,round(w*factor)),max(1,round(h*factor))),Image.Resampling.LANCZOS)
        self.photo=photo_image(image);self.canvas.delete("all")
        self.canvas.create_image(5,5,image=self.photo,anchor="nw")
        self.canvas.configure(scrollregion=(0,0,image.width+10,image.height+10))


class VideoPlayer:
    def __init__(self,parent,path):
        self.cap=cv2.VideoCapture(str(path))
        if not self.cap.isOpened():raise ValueError("確認動画を開けません。")
        self.fps=self.cap.get(5);self.frames=int(self.cap.get(7))
        if self.fps<=0 or self.frames<1:self.cap.release();raise ValueError("確認動画の時刻情報が不正です。")
        self.window=tk.Toplevel(parent);self.window.title("結果の確認動画："+Path(path).name);self.window.geometry("820x730")
        self.job=None;self.playing=False;self.frame=0
        self.image=ttk.Label(self.window,anchor="center");self.image.pack(fill="both",expand=True,padx=10,pady=8)
        controls=ttk.Frame(self.window);controls.pack(fill="x",padx=10,pady=8)
        self.button=ttk.Button(controls,text="再生",command=self.toggle);self.button.pack(side="left")
        self.rate=tk.StringVar(value="1")
        ttk.Label(controls,text="速度").pack(side="left",padx=(10,3))
        ttk.Combobox(controls,textvariable=self.rate,values=["0.25","0.5","1","2"],width=5,state="readonly").pack(side="left")
        self.position=tk.DoubleVar(value=0)
        slider=ttk.Scale(controls,from_=0,to=self.frames-1,variable=self.position);slider.pack(side="left",fill="x",expand=True,padx=10)
        slider.bind("<ButtonPress-1>",lambda event:self.pause());slider.bind("<ButtonRelease-1>",lambda event:self.seek())
        self.time=ttk.Label(controls,width=22);self.time.pack(side="left")
        ttk.Label(self.window,text="緑矢印＝頭向き、水色矢印＝変位（方向確認動画）。再生位置を指定して比較できます。").pack(pady=5)
        self.window.protocol("WM_DELETE_WINDOW",self.close);self.window.update_idletasks();self.show(0)

    def show(self,frame,seek=True):
        if seek:self.cap.set(1,frame)
        ok,img=self.cap.read()
        if not ok:self.pause();return
        self.frame=frame;image=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        image.thumbnail((max(200,self.window.winfo_width()-25),max(200,self.window.winfo_height()-110)))
        self.image.photo=photo_image(image);self.image.configure(image=self.image.photo)
        self.position.set(frame);self.time.configure(text=f'{frame/self.fps:.2f} / {(self.frames-1)/self.fps:.2f} 秒')

    def seek(self):self.show(int(round(self.position.get())))

    def pause(self):
        self.playing=False;self.button.configure(text="再生")
        if self.job:self.window.after_cancel(self.job);self.job=None

    def toggle(self):
        if self.playing:self.pause();return
        if self.frame>=self.frames-1:self.show(0)
        self.playing=True;self.button.configure(text="停止");self.tick()

    def tick(self):
        self.job=None
        if not self.playing:return
        if self.frame>=self.frames-1:self.pause();return
        self.show(self.frame+1,seek=False)
        if self.playing:self.job=self.window.after(max(10,round(1000/(self.fps*float(self.rate.get())))),self.tick)

    def close(self):self.pause();self.cap.release();self.window.destroy()


def table(parent,columns):
    frame=ttk.Frame(parent);frame.pack(fill="both",expand=True,padx=8,pady=8)
    tree=ttk.Treeview(frame,columns=[c[0] for c in columns],show="headings")
    for key,label,width in columns:tree.heading(key,text=label);tree.column(key,width=width,minwidth=65,anchor="center",stretch=False)
    sy=ttk.Scrollbar(frame,orient="vertical",command=tree.yview);sx=ttk.Scrollbar(frame,orient="horizontal",command=tree.xview)
    tree.configure(yscrollcommand=sy.set,xscrollcommand=sx.set)
    tree.grid(row=0,column=0,sticky="nsew");sy.grid(row=0,column=1,sticky="ns");sx.grid(row=1,column=0,sticky="ew")
    frame.rowconfigure(0,weight=1);frame.columnconfigure(0,weight=1)
    return tree


def replace_rows(tree,rows):
    tree.delete(*tree.get_children())
    for values in rows:tree.insert("","end",values=values)


class ResultsView(ttk.Frame):
    def __init__(self,parent,on_browse,on_refresh,on_evaluate,on_open,on_rerun=None):
        super().__init__(parent);self.data=None;self.players=[]
        bar=ttk.Frame(self);bar.pack(fill="x",padx=8,pady=8)
        for text,action in [("結果を読込",on_browse),("再読込",on_refresh),("人手学習・評価",on_evaluate),("保存フォルダ",on_open)]:
            ttk.Button(bar,text=text,command=action).pack(side="left",padx=3)
        if on_rerun:ttk.Button(bar,text="最新モデルで再解析",command=on_rerun).pack(side="left",padx=3)
        self.video_choice=tk.StringVar(value="direction_preview.mp4")
        self.video_select=ttk.Combobox(bar,textvariable=self.video_choice,values=[],state="readonly",width=23);self.video_select.pack(side="left",padx=8)
        ttk.Button(bar,text="動画を表示",command=self.play_video).pack(side="left")
        self.path=ttk.Label(self,text="結果未選択：解析を実行するか、結果フォルダを読み込んでください。",wraplength=870)
        self.path.pack(fill="x",padx=12,pady=4)
        tabs=ttk.Notebook(self);tabs.pack(fill="both",expand=True,padx=8,pady=8)
        overview=ttk.Frame(tabs);graphs=ttk.Frame(tabs);individual=ttk.Frame(tabs);quality=ttk.Frame(tabs);evaluation=ttk.Frame(tabs);identity=ttk.Frame(tabs)
        for name,frame in [("概要",overview),("グラフ",graphs),("個体別",individual),("追跡品質",quality),("ID再対応付け",identity),("精度評価",evaluation)]:tabs.add(frame,text=name)
        statistics=ttk.Frame(tabs);tabs.add(statistics,text="3区間の統計検定")
        self.stats_text=ttk.Label(statistics,justify="left",wraplength=850);self.stats_text.pack(anchor="nw",padx=15,pady=10)
        stats_tabs=ttk.Notebook(statistics);stats_tabs.pack(fill="both",expand=True,padx=8,pady=8)
        period_tab=ttk.Frame(stats_tabs);pair_tab=ttk.Frame(stats_tabs);slot_tab=ttk.Frame(stats_tabs)
        for title,frame in [("集団の要約",period_tab),("区間同士の比較",pair_tab),("各個体の平均移動距離",slot_tab)]:stats_tabs.add(frame,text=title)
        self.period_table=table(period_tab,[("period","区間",150),("span","区間長(s)",120),("n","個体数",90),("mean","平均移動距離/秒",170),("sd","標準偏差",130),("coverage","最低有効観測率",150)])
        self.pair_table=table(pair_tab,[("pair","区間の比較",240),("difference","平均差（左−右）",160),("p","p値",120),("holm","Holm補正p値",150),("result","判定（5%）",160)])
        self.period_slot_table=table(slot_tab,[("slot","表示番号",90),("first","0〜30秒",130),("second","30〜60秒",130),("third","60秒以降",130),("coverage","有効観測率（3区間）",240),("included","検定対象",110)])
        ttk.Label(statistics,text="平均移動距離＝測定距離÷有効観測秒（停止を含む）。欠測を距離0として扱いません。\n同じIDを対応させた探索的検定です。ID一致と個体の独立性を仮定します。同一容器の結果だけで母集団全体へ一般化できません。",wraplength=850).pack(anchor="w",padx=12,pady=8)
        self.overview=ttk.Label(overview,justify="left",font=("Helvetica",13),wraplength=850);self.overview.pack(anchor="nw",padx=20,pady=15)
        self.direction_table=table(overview,[("label","行動",130),("time","合計時間（個体秒）",175),("ratio","全個体時間の割合",175),("distance","測定距離",160)])
        ttk.Label(overview,text="距離は有効な区間の合計です。欠測中の経路は含みません。\nIDは位置・外見で保持します。接触後の曖昧な対応は欠測として扱い、動画で確認できます。",wraplength=850).pack(anchor="w",padx=15,pady=10)
        plot_tabs=ttk.Notebook(graphs);plot_tabs.pack(fill="both",expand=True)
        self.distance_image=ImagePanel(plot_tabs);self.direction_image=ImagePanel(plot_tabs)
        self.rate_image=ImagePanel(plot_tabs)
        plot_tabs.add(self.distance_image,text="移動距離・移動停止");plot_tabs.add(self.direction_image,text="前進・後退")
        plot_tabs.add(self.rate_image,text="単位時間あたりの移動量")
        ttk.Label(individual,text="距離・速度の単位は概要に表示。前進後退の時間は方向判定用、移動時間は従来の移動停止判定用です。",wraplength=850).pack(anchor="w",padx=10,pady=8)
        self.slot_table=table(individual,[("slot","表示番号",80),("distance","測定距離",115),("moving","移動時間(s)",115),
                                          ("forward","前進(s)",105),("backward","後退(s)",105),("stopped","停止(s・方向用)",130),
                                          ("unknown","不明(s)",100),("coverage","有効観測率",110),("segments","追跡区間数",110)])
        self.quality_text=ttk.Label(quality,justify="left",wraplength=850);self.quality_text.pack(anchor="nw",padx=15,pady=12)
        self.reason_table=table(quality,[("reason","不明の理由",450),("time","合計時間（個体秒）",220)])
        ttk.Label(identity,text='見失った後に同じIDへ戻した区間です。行を選び、欠測前から動画を確認できます。',wraplength=850).pack(anchor='w',padx=12,pady=8)
        ttk.Button(identity,text='選択した再対応付けを動画で確認',command=self.play_identity_event).pack(anchor='w',padx=12)
        self.identity_table=table(identity,[('time','再対応時刻(s)',140),('id','ID',80),('gap','観測間隔(s)',140),('distance','再検出位置の差(px)',180),('segment','追跡区間',100)])
        self.eval_text=ttk.Label(evaluation,justify="left",wraplength=850);self.eval_text.pack(anchor="nw",padx=15,pady=12)
        self.class_table=table(evaluation,[("label","人手の行動",125),("n","評価数",95),("precision","適合率",125),("recall","再現率",125),("f1","F1",125)])
        self.confusion_table=table(evaluation,[("truth","人手＼自動",130)]+[(k,LABELS[k],105) for k in LABELS])
        ttk.Label(evaluation,text="自動の不明は誤りに含めます。人が判断不能な試料は精度の分母から除外し、件数を併記します。\n人手評価は自動予測を変更しません。時間的に近い評価試料は独立した実験反復ではありません。",wraplength=850).pack(anchor="w",padx=12,pady=8)

    def show(self,data):
        self.data=data;g=data["group"];m=data["metadata"];n=number(g.get("expected_flies"));duration=number(g.get("duration_s"))
        unit=g.get("unit","px");mode="自動（外見モデル）" if m.get("head_mode")=="automatic_appearance" else "頭側入力または旧形式"
        source=Path(m.get("direction_video",m.get("source","動画情報なし"))).name
        self.path.configure(text=str(data["folder"]))
        self.overview.configure(text=f"動画：{source}\n"
                                     f"個体数：{formatted(n,digits=0)}　観測時間：{formatted(duration)} 秒　頭尾判定：{mode}\n"
                                     f"総測定距離：{formatted(g.get('measured_distance'))} {unit}　有効観測率：{formatted(g.get('interval_coverage'),True)}\n"
                                     f"距離 / 有効個体秒：{formatted(g.get('distance_per_valid_fly_second'))} {unit}/個体秒\n"
                                     f"方向が不明な割合：{formatted(data['unknown_fraction'],True) if data['unknown_fraction'] is not None else '集計なし'}")
        if m.get("learning_version") is not None:
            self.overview.configure(text=self.overview.cget("text")+f"\n解析に使用した学習モデル：v{m['learning_version']}（行動教師 {m.get('behavior_training_samples',0)}件）")
        statistics=data.get("period_statistics",{})
        if statistics.get("status")=="ok":
            verdict="有意差あり" if statistics["omnibus_significant"] else "有意差を検出せず"
            self.overview.configure(text=self.overview.cget("text")+f"\n3区間の比較：{verdict}（p={statistics['omnibus_p']:.6g}、n={statistics['n']}）。詳細は「3区間の統計検定」。")
        elif statistics:
            self.overview.configure(text=self.overview.cget("text")+"\n3区間の比較：検定不可。理由は「3区間の統計検定」に表示。")
        total=data["total_direction_fly_seconds"]
        replace_rows(self.direction_table,[(LABELS[k],formatted(data["totals"][k]),formatted(data["totals"][k]/total,True) if total else "—",
                                           formatted(sum(number(r.get(k+"_measured_distance")) or 0 for r in data["directions"]))+" "+unit)
                                          for k in LABELS] if data["directions"] else [])
        lookup={str(r["slot"]):r for r in data["directions"]}
        rows=[]
        for s in data["slots"]:
            d=lookup.get(str(s["slot"]),{})
            coverage=(number(s.get("valid_interval_s"))/duration) if duration and number(s.get("valid_interval_s")) is not None else None
            rows.append((s["slot"],formatted(s.get("smoothed_distance")),formatted(s.get("moving_s")),
                         formatted(d.get("forward_s")),formatted(d.get("backward_s")),formatted(d.get("stopped_s")),
                         formatted(d.get("unknown_s")),formatted(coverage,True) if coverage is not None else "—",s.get("segment_count","—")))
        replace_rows(self.slot_table,rows)
        self.distance_image.load(data["images"].get("report.png"));self.direction_image.load(data["images"].get("direction_report.png"))
        self.rate_image.load(data.get("rate_image"))
        self.show_statistics(data.get("period_statistics",{}))
        pop=data["population"]
        population_text=(f"自動個体数推定：{pop.get('estimated_flies','—')}（最頻値 {pop.get('modal_count',pop.get('estimated_flies','—'))}、検出範囲 {pop.get('count_min','—')}〜{pop.get('count_max','—')}）\n" if pop else "")
        identity_text=(f"IDを保持した再対応付け：{m.get('identity_reacquisitions',0)} 回　静止本体の補助検出：{m.get('stationary_recovery_frames',0)} フレーム\n"
                       "これらは処理件数です。同一個体との一致率は、人手確認前には未評価です。\n" if m.get('identity_method') else "")
        if (number(m.get('above_requested_population_fraction')) or 0)>=.2:
            identity_text+='指定数を超える検出が継続しています。個体数と容器の反射・検出条件を確認してください。\n'
        self.quality_text.configure(text=f"有効観測：{formatted(g.get('valid_fly_seconds'))} / {formatted(g.get('possible_fly_seconds'))} 個体秒\n"
                                         f"全個体を同時に追跡できた時間の割合：{formatted(g.get('all_seven_interval_fraction'),True)}\n"
                                         f"動画確認対象：{m.get('review_frames','—')} フレーム　繰り返し画像：{m.get('repeated_encoded_frames','—')} フレーム\n"
                                         +population_text+identity_text+"下表は方向不明の理由別時間です。同じ時間の複数個体はそれぞれ加算しています。")
        replace_rows(self.reason_table,[(REASONS.get(reason,reason),formatted(seconds)) for reason,seconds in data["reasons"].most_common()])
        replace_rows(self.identity_table,[(formatted(r['time_s']),r['slot'],formatted(r['identity_gap_s']),formatted(r['identity_match_distance_px']),r['segment']) for r in data.get('identity_events',[])])
        self.show_evaluation(data["evaluation"],data.get("learning_status",{}),data.get("learning_input",{}))
        names=list(data["previews"]);self.video_select.configure(values=names);self.video_choice.set(names[0] if names else "")

    def show_statistics(self,s):
        text=f"0〜30秒・30〜60秒・60秒以降の比較　単位：{s.get('rate_unit','—')}\n3区間すべてに有効観測がある個体：{s.get('n',0)} / {s.get('total_slots',0)}（欠測区間のある個体は検定から除外）\n"
        if s.get("status")=="ok":
            verdict="有意差あり" if s["omnibus_significant"] else "有意差を検出せず"
            text+=f"全体：{verdict}（対応のある平均差の置換検定、p={s['omnibus_p']:.6g}、有意水準5%）\n区間同士：対応のある両側置換検定、3比較をHolm補正。"
            if not s.get("omnibus_exact"):text+=f" 全体p値は乱数固定・{s['resamples']:,}回の近似です。"
        else:text+=s.get("reason","検定結果はありません。")
        self.stats_text.configure(text=text)
        replace_rows(self.period_table,[(p["label"],formatted(p["duration_s"]),p["n"],formatted(p["mean"]),formatted(p["sd"]),formatted(p["min_coverage"],True)) for p in s.get("periods",[])])
        replace_rows(self.pair_table,[(p["a"]+" vs "+p["b"],formatted(p["mean_difference"]),f'{p["p"]:.6g}',f'{p["p_holm"]:.6g}',"有意差あり" if p["significant"] else "有意差を検出せず") for p in s.get("pairs",[])])
        replace_rows(self.period_slot_table,[(r["slot"],*[formatted(p["mean_distance_per_s"]) for p in r["periods"]]," / ".join(formatted(p["coverage"],True) for p in r["periods"]),"対象" if r["included"] else "除外") for r in s.get("individuals",[])])

    def play_identity_event(self):
        chosen=self.identity_table.selection()
        if not chosen or not self.data:return
        row=self.data['identity_events'][self.identity_table.index(chosen[0])]
        path=self.data['previews'].get('tracking_preview.mp4')
        if not path:return
        player=VideoPlayer(self,path);self.players.append(player)
        start=float(row['time_s'])-float(row['identity_gap_s'])-1
        player.show(max(0,int(start*player.fps)))

    def show_evaluation(self,e,learning_status=None,learning_input=None):
        n=e.get("scorable_samples",0)
        if not n:
            text=f"精度：未評価（判定可能な人手評価：{n}件、判断不能：{e.get('unscorable_samples',0)}件）。\n入力設定で「評価用（ランダム）」を選び、上の「人手学習・評価」から評価すると集計が反映されます。"
        else:
            text=(f"完了：{e.get('completed_samples',0)}件　判定可能：{n}件　判断不能：{e.get('unscorable_samples',0)}件\n"
                  f"全体一致率（自動不明も誤り）：{formatted(e.get('accuracy_including_unknown_as_error'),True)}\n"
                  f"自動の判定率：{formatted(e.get('prediction_coverage'),True)}　判定できた試料の一致率：{formatted(e.get('accuracy_on_classified_samples'),True)}\n"
                  f"Macro F1：{formatted(e.get('macro_f1_present_reference_classes'))}")
        if learning_input:
            text+=f"\n学習用の優先抽出：入力 {learning_input.get('completed_samples',0)}件 / 人のUnknown {learning_input.get('human_unknown_samples',0)}件（全体精度の評価とは別）。"
        if e.get("provenance",{}).get("same_video_as_training"):
            text+="\n学習元と同じ動画の未使用区間による評価です。別動画への精度とは区別してください。"
        if e.get("provenance",{}).get("cumulative_learning_enabled"):
            text+="\nこの表は評価元モデルの固定予測との比較です。評価入力は次のモデルの学習に使用されます。"
        if learning_status:
            text+=f"\n評価を保存した累積モデル：v{learning_status.get('last_model_version','—')}。この表は更新後モデルの精度ではありません。"
        self.eval_text.configure(text=text)
        replace_rows(self.class_table,[(LABELS[k],v.get("reference_samples",0),formatted(v.get("precision"),True),
                                       formatted(v.get("recall"),True),formatted(v.get("f1"))) for k,v in e.get("classes",{}).items()])
        matrix=e.get("confusion_matrix",{})
        replace_rows(self.confusion_table,[(LABELS[k],*[row.get(c,0) for c in LABELS]) for k,row in matrix.items()])

    def play_video(self):
        if not self.data:return
        path=self.data["previews"].get(self.video_choice.get())
        if not path:messagebox.showinfo("動画なし","この結果には確認動画がありません。",parent=self.winfo_toplevel());return
        try:self.players.append(VideoPlayer(self.winfo_toplevel(),path))
        except Exception as e:messagebox.showerror("動画エラー",str(e),parent=self.winfo_toplevel())
