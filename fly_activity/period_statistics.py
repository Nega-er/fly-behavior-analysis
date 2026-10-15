"""Paired comparisons of per-fly measured distance rates in three time periods."""
import csv
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.stats import permutation_test

PERIODS=("0〜30秒","30〜60秒","60秒以降")
RESAMPLES=19999
SEED=20261014


def permutation_randomness(seed):
    """Use the supported SciPy keyword without changing the seeded generator."""
    keyword="rng" if "rng" in inspect.signature(permutation_test).parameters else "random_state"
    return {keyword:np.random.default_rng(seed)}


def holm_adjust(pvalues):
    values=np.asarray(pvalues,float);order=np.argsort(values)
    adjusted=np.empty(len(values));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(values)-rank)*values[index]))
        adjusted[index]=previous
    return adjusted.tolist()


def mean_difference_statistic(*samples,axis=0):
    means=np.stack([np.mean(sample,axis=axis) for sample in samples],axis=-1)
    return np.sum((means-np.mean(means,axis=-1,keepdims=True))**2,axis=-1)


def paired_mean_statistic(a,b,axis=0):
    return np.mean(a-b,axis=axis)


def compute_period_statistics(rows,duration,unit="px",resamples=RESAMPLES):
    """Split valid frame intervals at exact boundaries; never count gaps as zero."""
    duration=float(duration or 0)
    edges=(0.,30.,60.,duration)
    spans=[max(0.,min(duration,edges[i+1])-edges[i]) for i in range(3)]
    data={};last={};has_fields=False
    for row in rows:
        if not {"slot","time_s","valid_interval","distance"}.issubset(row):continue
        has_fields=True;slot=int(row["slot"]);time=float(row["time_s"])
        values=data.setdefault(slot,np.zeros((3,2)))
        previous=last.get(slot,time);last[slot]=time
        if str(row["valid_interval"])!="1" or time<=previous:continue
        distance=float(row["distance"])
        if not np.isfinite(distance) or distance<0:raise ValueError("Invalid movement distance")
        for period in range(3):
            overlap=max(0.,min(time,edges[period+1])-max(previous,edges[period]))
            if overlap:
                values[period,0]+=distance*overlap/(time-previous)
                values[period,1]+=overlap
    individuals=[];complete=[];included=[]
    for slot,values in sorted(data.items()):
        rates=[float(d/v) if v>0 else None for d,v in values]
        usable=all(rate is not None for rate in rates)
        if usable:complete.append(rates);included.append(slot)
        individuals.append(dict(slot=slot,included=usable,periods=[
            dict(label=PERIODS[i],distance=float(d),valid_s=float(v),
                 coverage=float(v/spans[i]) if spans[i]>0 else None,mean_distance_per_s=rates[i])
            for i,(d,v) in enumerate(values)]))
    result=dict(schema_version=1,unit=unit,rate_unit=unit+"/s",alpha=.05,
                metric="measured_distance / valid_observation_seconds (including stopped intervals)",
                boundaries_s=[0,30,60,duration],individuals=individuals,
                total_slots=len(data),n=len(complete),included_slots=included,
                excluded_slots=[r["slot"] for r in individuals if not r["included"]],
                seed=SEED,resamples=resamples,periods=[],pairs=[],
                method="paired within-individual permutation test of mean differences; Holm correction for 3 exact paired comparisons",
                assumptions="Period labels must be exchangeable under the null; display slots must retain identity. Animals in one arena may not be independent. Exploratory within-video inference, not replicated-arena population inference.")
    for i,label in enumerate(PERIODS):
        values=[row[i] for row in complete]
        result["periods"].append(dict(label=label,duration_s=spans[i],n=len(values),
                                      mean=float(np.mean(values)) if values else None,
                                      sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
                                      min_coverage=min((r["periods"][i]["coverage"] for r in individuals if r["included"]),default=None)))
    if not has_fields:
        result.update(status="unavailable",reason="移動距離と有効区間のデータがありません。")
        return result
    if duration<=60:
        result.update(status="insufficient",reason="60秒以降の観測がないため、3区間の検定はできません。")
        return result
    if len(complete)<2:
        result.update(status="insufficient",reason="3区間すべてで観測できた個体が2個体未満のため、検定できません。")
        return result
    values=np.asarray(complete);samples=tuple(values[:,i] for i in range(3))
    omnibus=permutation_test(samples,mean_difference_statistic,permutation_type="samples",
                             vectorized=True,alternative="greater",n_resamples=resamples,
                             batch=256,**permutation_randomness(SEED))
    result.update(status="ok",omnibus_statistic=float(omnibus.statistic),
                  omnibus_p=float(omnibus.pvalue),omnibus_significant=bool(omnibus.pvalue<.05),
                  omnibus_exact=bool(6**len(complete)<=resamples))
    for a,b in ((0,1),(0,2),(1,2)):
        # Exhaustive swaps for small populations; bounded resampling for larger videos.
        pair=permutation_test((samples[a],samples[b]),paired_mean_statistic,
                              permutation_type="samples",vectorized=True,alternative="two-sided",
                              n_resamples=max(resamples,65536),batch=256,**permutation_randomness(SEED+a*3+b))
        result["pairs"].append(dict(a=PERIODS[a],b=PERIODS[b],mean_difference=float(pair.statistic),
                                    p=float(pair.pvalue),exact=bool(2**len(complete)<=max(resamples,65536))))
    for pair,adjusted in zip(result["pairs"],holm_adjust([p["p"] for p in result["pairs"]])):
        pair.update(p_holm=adjusted,significant=bool(adjusted<.05))
    return result


def save_period_statistics(folder,result):
    folder=Path(folder)
    (folder/"period_statistics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    fields=["slot","included","period","duration_s","distance","valid_s","coverage","mean_distance_per_s","unit"]
    with (folder/"period_individuals.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for individual in result["individuals"]:
            for i,period in enumerate(individual["periods"]):
                writer.writerow(dict(slot=individual["slot"],included=individual["included"],period=period["label"],
                                     duration_s=result["periods"][i]["duration_s"],distance=period["distance"],
                                     valid_s=period["valid_s"],coverage=period["coverage"],
                                     mean_distance_per_s=period["mean_distance_per_s"],unit=result["rate_unit"]))
