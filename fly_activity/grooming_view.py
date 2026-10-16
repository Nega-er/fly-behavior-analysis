"""Human-readable inspection of experimental image motion and manual labels."""
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from grooming import read_rows
from results_view import ImagePanel,VideoPlayer,table,replace_rows

LABELS={'grooming':'グルーミング候補','non_grooming':'非グルーミング','unknown':'判定不能',
        'groom_head':'グルーミング頭','groom_body':'グルーミング体','stopped':'停止',
        'forward':'前進','backward':'後退','flying':'飛んでいる','':'未入力'}
REASONS={'body_axis_unreliable':'体軸が不確か','body_not_resolved':'胴体を分離できない','neighbor_overlap':'他個体が近い',
         'tracking_missing':'追跡欠測','crop_clipped':'画像端で切出し不能','background_motion':'背景・反射の動き',
         'illumination_jump':'照明の大きな変化','turning':'旋回中','locomotion_overlap':'移動との区別を保留',
         'nonspecific_motion':'動きの種類を識別できない','localized_appendage_motion_candidate':'局所的な動きが継続',
         'one_end_appendage_motion_candidate':'体軸の片側に局所的な動き',
         'quiet_body_and_appendages':'胴体・周囲とも変化が小さい','new_tracking_segment':'追跡開始直後',
         'repeated_encoded_frame':'同じ画像の繰返し','observation_gap':'観測間隔が長い','no_observation':'観測なし',
         'insufficient_coverage':'有効な観測時間が不足','mixed_or_nonspecific_motion':'継続時間・動きの種類が未確定',
         'sustained_local_motion_candidate':'局所的な動きが継続','sustained_quiet':'変化が小さい状態が継続'}


class GroomingWindow:
    def __init__(self,parent,folder):
        self.folder=Path(folder);self.players=[]
        self.metadata=json.loads((self.folder/'grooming_metadata.json').read_text(encoding='utf-8'))
        report=json.loads((self.folder/'grooming_validation.json').read_text(encoding='utf-8'))
        self.rows=read_rows(self.folder/'grooming_intervals.csv')
        self.window=tk.Toplevel(parent);self.window.title('グルーミング試作：画像差分と人の判定');self.window.geometry('1100x820')
        ttk.Label(self.window,text='試作の固定ルールです。局所的な動きがグルーミングかは人の確認が必要です。頭／体の自動分類は行いません。',wraplength=1040).pack(anchor='w',padx=12,pady=6)
        def metric(name):
            value=report.get(name)
            return f'{value:.1%}' if value is not None else '未評価'
        text=f"正解付き {report['labeled_intervals']}区間（グルーミング {report['positive_intervals']}／非グルーミング {report['negative_intervals']}） / 判定不能 {report['unknown_predictions']}区間\n"
        text+=f"候補の適合率 {metric('precision')} / 再現率 {metric('recall')} / 判定できた割合 {metric('prediction_coverage')}。判定不能も見逃しに含みます。"
        ttk.Label(self.window,text=text,wraplength=1040).pack(anchor='w',padx=12,pady=4)
        if report.get('evaluation_independent') is False:
            ttk.Label(self.window,text='判定条件の改善に使った開発データとの比較です。独立した動画での精度検証は未実施です。',wraplength=1040).pack(anchor='w',padx=12)
        controls=ttk.Frame(self.window);controls.pack(fill='x',padx=12,pady=5)
        ttk.Label(controls,text='個体ID').pack(side='left')
        self.slot=tk.StringVar(value=str(self.metadata['included_slots'][0]))
        selector=ttk.Combobox(controls,textvariable=self.slot,values=list(map(str,self.metadata['included_slots'])),state='readonly',width=7)
        selector.pack(side='left',padx=5);selector.bind('<<ComboboxSelected>>',lambda e:self.refresh())
        path=self.folder/'grooming_preview.mp4'
        if path.exists():ttk.Button(controls,text=f"補正画像・差分・人の判定を動画で確認（ID {self.metadata['preview_slot']}）",command=self.play).pack(side='left',padx=6)
        tabs=ttk.Notebook(self.window);tabs.pack(fill='both',expand=True,padx=10,pady=5)
        self.graph=ImagePanel(tabs);tabs.add(self.graph,text='時系列と部位別の変化量');self.graph.load(self.folder/'grooming_report.png')
        intervals=ttk.Frame(tabs);tabs.add(intervals,text='1秒ごとの人手比較・特徴量')
        self.table=table(intervals,[('time','開始秒',70),('rule','試作判定',150),('human','人の判定',150),
                                  ('coverage','有効観測率',100),('periphery','体の周囲の変化率',140),('core','胴体の変化率',130),('reason','判定理由',300)])
        ttk.Label(self.window,text='変化率は判定不能の区間でも測定できた値を表示します。有効観測率が低い区間の数値は参考値です。\nEnd A/Bは体軸の両端で、頭／体を意味しません。手入力との対応は動画・追跡CSVのハッシュで確認します。\n'+str(self.folder),wraplength=1040).pack(anchor='w',padx=12,pady=5)
        self.refresh()
        self.window.protocol('WM_DELETE_WINDOW',self.close)

    def refresh(self):
        def percent(value):return f'{float(value):.1%}' if value!='' else '—'
        def reason(row):
            main=REASONS.get(row['reason'],row['reason'])
            quality=REASONS.get(row.get('dominant_frame_reason'),row.get('dominant_frame_reason',''))
            return main+(f'（{quality}）' if quality and row['grooming_state']=='unknown' else '')
        replace_rows(self.table,[(r['start_s'],LABELS[r['grooming_state']],LABELS.get(r['human_label'],r['human_label']),
                                 percent(r['valid_fraction']),percent(r['periphery_change']),percent(r['core_change']),
                                 reason(r))
                                for r in self.rows if r['slot']==self.slot.get()])

    def play(self):self.players.append(VideoPlayer(self.window,self.folder/'grooming_preview.mp4'))

    def close(self):
        for player in self.players:
            if player.window.winfo_exists():player.close()
        if self.graph.timer:self.graph.after_cancel(self.graph.timer);self.graph.timer=None
        self.window.destroy()
