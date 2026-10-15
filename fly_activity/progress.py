"""Stage progress shared by the analysis worker and the desktop display."""
import tkinter as tk
from tkinter import ttk

from progress_protocol import PREFIX
STAGES = (
    ("prepare", "動画準備", 8, "#457b9d"),
    ("track", "個体追跡", 52, "#238b8e"),
    ("analyze", "行動解析", 15, "#647ac2"),
    ("save", "集計・図の保存", 8, "#b07832"),
    ("preview", "確認動画", 12, "#9b609c"),
    ("load", "結果表示", 5, "#538544"),
)



class StackedProgress(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.caption = tk.StringVar(value="待機中")
        ttk.Label(self, textvariable=self.caption).pack(anchor="w")
        self.canvas = tk.Canvas(self, height=24, highlightthickness=0, background="#f4f5f7")
        self.canvas.pack(fill="x", pady=(4, 4))
        self.legend = ttk.Frame(self)
        self.legend.pack(fill="x")
        self.stages = STAGES
        self.values = {s[0]: 0 for s in self.stages}
        self.failed = False
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.begin()

    def begin(self, stages=STAGES):
        self.stages = stages
        self.values = {s[0]: 0 for s in stages}
        self.failed = False
        for child in self.legend.winfo_children():
            child.destroy()
        for key, label, weight, color in stages:
            tk.Label(self.legend, text="●", fg=color).pack(side="left")
            ttk.Label(self.legend, text=label).pack(side="left", padx=(0, 12))
        self.caption.set("待機中")
        self.draw()

    def update_progress(self, stage, current=0, total=1, detail=""):
        keys = [s[0] for s in self.stages]
        if stage not in keys:
            return
        index = keys.index(stage)
        for key in keys[:index]:
            self.values[key] = 1
        fraction = min(1, max(0, current / total)) if total > 0 else 0
        self.values[stage] = max(self.values[stage], fraction)
        label = self.stages[index][1]
        self.caption.set(f"全体 約{self.percent():.0f}% ｜ {label} {self.values[stage]:.0%}"
                         + (f" ｜ {detail}" if detail else ""))
        self.draw()

    def percent(self):
        return 100 * sum(weight * self.values[key] for key, _, weight, _ in self.stages) / sum(s[2] for s in self.stages)

    def finish(self):
        self.values = {s[0]: 1 for s in self.stages}
        self.caption.set("完了 100%")
        self.draw()

    def fail(self):
        self.failed = True
        self.caption.set(f"処理中断 ｜ 全体 約{self.percent():.0f}%（完了した処理を表示）")
        self.draw()

    def draw(self):
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        total = sum(s[2] for s in self.stages)
        left = 0
        for key, label, weight, color in self.stages:
            right = left + width * weight / total
            self.canvas.create_rectangle(left, 1, right, 23, fill="#e1e5eb", outline="")
            filled = left + (right-left) * self.values[key]
            if filled > left:
                self.canvas.create_rectangle(left, 1, filled, 23, fill=color, outline="")
            self.canvas.create_line(right, 1, right, 23, fill="white", width=2)
            left = right
