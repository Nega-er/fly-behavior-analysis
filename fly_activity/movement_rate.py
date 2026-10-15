"""Measured locomotion per time bin, with missing-data coverage."""
from collections import defaultdict
from io import BytesIO
from pathlib import Path
import numpy as np


def summarize_rates(bins,expected_flies):
    grouped=defaultdict(list)
    for row in bins:grouped[float(row["start_s"]),float(row["end_s"])].append(row)
    result=[]
    for (start,end),rows in sorted(grouped.items()):
        span=end-start
        if span<=0:raise ValueError("Time bins must have positive duration")
        distance=sum(float(r["distance"]) for r in rows)
        valid=sum(float(r["valid_s"]) for r in rows)
        result.append(dict(start_s=start,end_s=end,window_s=span,
                           measured_distance=distance,valid_fly_seconds=valid,
                           distance_per_wall_s=distance/span if valid>0 else "",
                           mean_distance_per_valid_fly_second=distance/valid if valid>0 else "",
                           interval_coverage=valid/(expected_flies*span),unit=rows[0]["unit"]))
    return result


def plot_rates(rates,bins,destination=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    if not rates:raise ValueError("No time bins available")
    centers=np.array([(r["start_s"]+r["end_s"])/2 for r in rates])
    def values(name):return np.array([float(r[name]) if r[name]!="" else np.nan for r in rates])
    unit=rates[0]["unit"]
    fig,axes=plt.subplots(3,1,figsize=(13,9),sharex=True,constrained_layout=True)
    axes[0].plot(centers,values("distance_per_wall_s"),color="#3074af",marker=".")
    axes[0].set(title="Measured group distance per elapsed second",ylabel=f'{unit}/s (group)')
    by_slot=defaultdict(list)
    for row in bins:by_slot[int(row["slot"])].append(row)
    for slot,rows in sorted(by_slot.items()):
        rows=sorted(rows,key=lambda r:float(r["start_s"]))
        t=[(float(r["start_s"])+float(r["end_s"]))/2 for r in rows]
        speed=[float(r["distance"])/float(r["valid_s"]) if float(r["valid_s"])>0 else np.nan for r in rows]
        axes[1].plot(t,speed,alpha=.4,linewidth=1,label=f'Slot {slot}')
    axes[1].plot(centers,values("mean_distance_per_valid_fly_second"),color="black",linewidth=2.3,label="Valid-time weighted mean")
    axes[1].set(title="Distance rate normalized by valid observation time",ylabel=f'{unit}/s (per fly)')
    if len(by_slot)<=15:axes[1].legend(ncol=min(5,len(by_slot)+1),fontsize=8)
    axes[2].plot(centers,values("interval_coverage"),color="#329866",marker=".")
    axes[2].set(title="Valid observation coverage",ylabel="Fraction",xlabel="Time (s)",ylim=(0,1.05))
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("Locomotion rate over time | gaps are not zero movement")
    if destination is not None:
        fig.savefig(Path(destination),dpi=150);plt.close(fig);return None
    stream=BytesIO();fig.savefig(stream,format="png",dpi=150);plt.close(fig);stream.seek(0)
    with Image.open(stream) as image:result=image.convert("RGB")
    return result
