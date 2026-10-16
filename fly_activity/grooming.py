"""Experimental, body-aligned image motion; independent of locomotion labels.

These rules are candidates for human validation, not a trained grooming detector.
End A/B have no anatomical meaning: unsigned axes are made temporally continuous.
"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

PARAMETERS=dict(crop_size=72,pixel_difference=12,body_contrast=25,body_core_fraction=.7,min_axis_ratio=1.5,
                max_turn_deg=18,max_speed_body_lengths_s=.5,max_frame_gap_s=.25,
                max_illumination_step=25,max_background_change=.15,
                quiet_fraction=.025,peripheral_fraction=.06,max_core_fraction=.10,
                min_coverage=.6,min_grooming_fraction=.35,min_bout_s=.25,max_bout_gap_s=.12,
                localized_fraction=.09,speed_window_s=.2)
STATES=('non_grooming','grooming','unknown')
REGIONS=('core','periphery','end_a','end_b')
VERSION='aligned_difference_v2_unvalidated'


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def save_rows(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def aligned_patch(gray,center,axis,size):
    """Sample directly from the source; avoid rotating an already clipped square."""
    axis=np.asarray(axis,float);axis/=np.linalg.norm(axis)
    side=np.array([-axis[1],axis[0]])
    origin=np.asarray(center)-((size-1)/2)*(axis+side)
    transform=np.column_stack((axis,side,origin)).astype(np.float32)
    return cv2.warpAffine(gray,transform,(size,size),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT)


def body_geometry(patch,contrast,core_fraction=.7):
    border=np.r_[patch[0],patch[-1],patch[:,0],patch[:,-1]]
    background=float(np.median(border));midpoint=(patch.shape[0]-1)/2
    middle=patch[int(midpoint)-12:int(midpoint)+13,int(midpoint)-12:int(midpoint)+13]
    bright=float(np.percentile(middle,99));dark=float(np.percentile(middle,1))
    # Match the tracker's body-core principle: avoid attaching dim wings or a
    # container reflection to the bright body; support dark flies symmetrically.
    if bright-background>=background-dark:
        mask=((patch>=background+max(contrast,core_fraction*(bright-background)))).astype(np.uint8)
    else:mask=((patch<=background-max(contrast,core_fraction*(background-dark)))).astype(np.uint8)
    n,labels,stats,centers=cv2.connectedComponentsWithStats(mask,8)
    choices=[k for k in range(1,n) if stats[k,cv2.CC_STAT_AREA]>=12 and np.linalg.norm(centers[k]-midpoint)<10]
    if not choices:return None
    target=min(choices,key=lambda k:np.linalg.norm(centers[k]-midpoint))
    y,x=np.nonzero(labels==target)
    major=max(4.,float(np.quantile(np.abs(x-midpoint),.9)))
    minor=max(2.,float(np.quantile(np.abs(y-midpoint),.9)))
    if major>patch.shape[0]/3 or minor>patch.shape[0]/4:return None
    return major,minor


def region_masks(size,geometry):
    major,minor=geometry
    y,x=np.mgrid[:size,:size].astype(float);x-=(size-1)/2;y-=(size-1)/2
    core=(x/(major*.8))**2+(y/(minor*.8))**2<=1
    outer=(x/(major*1.8))**2+(y/(minor*3))**2<=1
    periphery=outer&~core
    return dict(core=core,periphery=periphery,end_a=periphery&(x>=0),end_b=periphery&(x<0),background=~outer)


def difference_features(previous,current,masks,p):
    raw=current.astype(float)-previous.astype(float)
    illumination=float(np.median(raw[masks['background']]))
    difference=np.abs(raw-illumination)
    features=dict(illumination_step=illumination)
    for name,mask in masks.items():
        features[name+'_change']=float(np.mean(difference[mask]>=p['pixel_difference']))
        if name!='background':features[name+'_energy']=float(np.mean(difference[mask])/255)
    return features,difference


def classify(features,speed,turn,p):
    if abs(features['illumination_step'])>p['max_illumination_step']:return 'unknown','illumination_jump'
    if features['background_change']>p['max_background_change']:return 'unknown','background_motion'
    if turn>p['max_turn_deg']:return 'unknown','turning'
    if speed>p['max_speed_body_lengths_s']:return 'unknown','locomotion_overlap'
    core=features['core_change'];peripheral=features['periphery_change']
    if max(core,peripheral)<p['quiet_fraction']:return 'non_grooming','quiet_body_and_appendages'
    if peripheral>=p['peripheral_fraction'] and core<p['max_core_fraction'] and peripheral>1.4*core:
        return 'grooming','localized_appendage_motion_candidate'
    local=max(features.get('end_a_change',0),features.get('end_b_change',0))
    if (local>=p['localized_fraction'] and peripheral>=p['peripheral_fraction']/2
            and core<p['max_core_fraction'] and local>1.4*core):
        return 'grooming','one_end_appendage_motion_candidate'
    return 'unknown','nonspecific_motion'


def movement_speed(positions,body_length):
    """Net movement over a short window suppresses centroid jitter from strokes."""
    if len(positions)<2:return 0.
    start,first=positions[0];end,last=positions[-1]
    return float(np.linalg.norm(last-first)/(body_length*(end-start)))


def aggregate(frames,slots,duration,p,labels=None):
    labels=labels or {};grouped=defaultdict(list)
    # A measurement represents (previous decoded observation, current observation].
    for row in frames:
        start=row['time_s']-row['dt_s'];end=row['time_s']
        if row['dt_s']<=0:continue
        for second in range(max(0,math.floor(start+1e-9)),min(math.ceil(duration),math.ceil(end-1e-9))):
            overlap=max(0,min(end,second+1)-max(start,second))
            if overlap:grouped[row['slot'],second].append((row,overlap))
    intervals=[]
    for slot in slots:
        for second in range(math.ceil(duration-1e-9)):
            span=min(duration,second+1)-second;items=grouped[slot,second]
            valid=sum(dt for row,dt in items if row['valid'])
            positive=sum(dt for row,dt in items if row['valid'] and row['candidate']=='grooming')
            quiet=sum(dt for row,dt in items if row['valid'] and row['candidate']=='non_grooming')
            # Leg strokes include short pauses. Bridge only measured, valid
            # pauses; neither pauses nor rejected observations count as evidence.
            run=longest=gap=0.;last_end=None
            for row,dt in items:
                start=max(second,row['time_s']-row['dt_s'])
                if last_end is not None and abs(start-last_end)>1e-6:run=gap=0.
                last_end=min(second+1,row['time_s'])
                if not row['valid']:run=gap=0.
                elif row['candidate']=='grooming':
                    run+=dt;gap=0.;longest=max(longest,run)
                else:
                    gap+=dt
                    if gap>p['max_bout_gap_s']:run=0.
            state='unknown';reason='insufficient_coverage'
            if valid/span>=p['min_coverage']:
                if positive/valid>=p['min_grooming_fraction'] and longest>=p['min_bout_s']:
                    state='grooming';reason='sustained_local_motion_candidate'
                elif quiet/valid>=.75:state='non_grooming';reason='sustained_quiet'
                else:reason='mixed_or_nonspecific_motion'
            human=labels.get(f'{slot}:{second}','')
            truth='grooming' if human in ('groom_head','groom_body') else ('non_grooming' if human in ('stopped','forward','backward','flying') else '')
            record=dict(slot=slot,start_s=second,end_s=min(duration,second+1),duration_s=span,
                        grooming_state=state,reason=reason,valid_fraction=valid/span,
                        measurement_fraction=sum(dt for row,dt in items if row['periphery_change']!='')/span,
                        dominant_frame_reason=Counter(row['reason'] for row,dt in items).most_common(1)[0][0] if items else 'no_observation',
                        grooming_candidate_fraction=positive/valid if valid else 0.,longest_candidate_s=longest,
                        human_label=human,human_grooming=truth,agreement=int(state==truth) if truth else '')
            for name in REGIONS:
                for metric in ('change','energy'):
                    key=name+'_'+metric
                    measured=[(row,dt) for row,dt in items if row[key]!='']
                    total=sum(dt for row,dt in measured)
                    record[key]=sum(float(row[key])*dt for row,dt in measured)/total if total else ''
            intervals.append(record)
    return intervals


def validation_report(intervals):
    labeled=[r for r in intervals if r['human_grooming']]
    count=Counter((r['human_grooming'],r['grooming_state']) for r in labeled)
    tp=count['grooming','grooming'];fp=count['non_grooming','grooming']
    positives=sum(r['human_grooming']=='grooming' for r in labeled)
    return dict(method=VERSION,trained=False,thresholds_fitted_to_labels=True,evaluation_independent=False,labeled_intervals=len(labeled),
                positive_intervals=positives,negative_intervals=len(labeled)-positives,
                unknown_predictions=sum(r['grooming_state']=='unknown' for r in labeled),
                precision=tp/(tp+fp) if tp+fp else None,recall=tp/positives if positives else None,
                accuracy_including_unknown_as_error=sum(r['grooming_state']==r['human_grooming'] for r in labeled)/len(labeled) if labeled else None,
                prediction_coverage=sum(r['grooming_state']!='unknown' for r in labeled)/len(labeled) if labeled else None,
                confusion=[dict(human=a,predicted=b,count=count[a,b]) for a in STATES[:2] for b in STATES],
                note='Unvalidated rule prototype revised using development labels; not an independent accuracy estimate. Unknown predictions count as misses; unlabeled/human Unknown are excluded. Head/body are not predicted.')


def save_comparison_examples(intervals,provenance,learning):
    """Append immutable labeled feature examples, separately from the active classifier."""
    directory=Path(learning)/'grooming/examples';count=0
    for row in intervals:
        if not row['human_grooming']:continue
        record=dict(schema_version=1,feature_version=VERSION,purpose='comparison_not_model_training',
                    video_sha256=provenance['video_sha256'],tracks_sha256=provenance['tracks_sha256'],
                    parameters=provenance['parameters'],interval=row)
        encoded=json.dumps(record,ensure_ascii=False,sort_keys=True,allow_nan=False)
        key=hashlib.sha256(encoded.encode()).hexdigest();directory.mkdir(parents=True,exist_ok=True)
        target=directory/(key+'.json')
        try:
            with target.open('x',encoding='utf-8') as stream:stream.write(encoded)
            count+=1
        except FileExistsError:
            if target.read_text(encoding='utf-8')!=encoded:raise ValueError('比較教師ファイルの内容が一致しません。')
    return count


def plot_comparison(intervals,slots,out):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    fig=Figure(figsize=(13,max(6,len(slots)*.6+3)),layout='constrained');FigureCanvasAgg(fig)
    timeline,features=fig.subplots(2,1,gridspec_kw={'height_ratios':[max(1,len(slots)*.5),1]})
    codes={'non_grooming':0,'grooming':1,'unknown':2,'':3};seconds=max(int(r['start_s']) for r in intervals)+1
    matrix=np.full((len(slots)*2,seconds),3)
    for i,slot in enumerate(slots):
        for row in intervals:
            if row['slot']!=slot:continue
            matrix[2*i,int(row['start_s'])]=codes[row['grooming_state']]
            matrix[2*i+1,int(row['start_s'])]=codes[row['human_grooming']]
    timeline.imshow(matrix,aspect='auto',interpolation='nearest',extent=(0,seconds,len(matrix),0),vmin=0,vmax=3,
                    cmap=ListedColormap(['#5489a7','#da9543','#999999','#f0f0f0']))
    timeline.set(yticks=np.arange(len(matrix))+.5,yticklabels=[f'ID {s} {kind}' for s in slots for kind in ('rule','human')],
                 xlabel='Time (s)',title='Prototype comparison: blue non-grooming / orange grooming / gray unknown / white unlabeled')
    target=next((r['slot'] for r in intervals if r['human_grooming']),slots[0])
    rows=[r for r in intervals if r['slot']==target]
    for name in ('periphery','core','end_a','end_b'):
        features.plot([r['start_s']+.5 for r in rows],[float(r[name+'_change']) if r[name+'_change']!='' else np.nan for r in rows],label=name)
    labeled=[r for r in rows if r['human_grooming']]
    bounds=(max(0,min(r['start_s'] for r in labeled)-1),max(r['end_s'] for r in labeled)+1) if labeled else (0,max(r['end_s'] for r in rows))
    for row in labeled:
        features.axvspan(row['start_s'],row['end_s'],color='#da9543' if row['human_grooming']=='grooming' else '#5489a7',alpha=.12)
    features.set(xlabel='Time (s)',ylabel='Changed-pixel fraction',ylim=(0,1),xlim=bounds,
                 title=f'ID {target}: regional differences (shading = human labels; low-quality measurements are diagnostic only)')
    features.legend();fig.savefig(Path(out)/'grooming_report.png',dpi=140)


def run_prototype(video,tracks,out,manual_session=None,excluded_slots=(),parameters=None,progress=None,learning=None,preview=True):
    video=Path(video).resolve();tracks=Path(tracks).resolve();out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('既存の出力は上書きしません。新しいフォルダを指定してください。')
    p=dict(PARAMETERS,**(parameters or {}))
    if set(p)!=set(PARAMETERS) or any(not math.isfinite(float(v)) or v<=0 for v in p.values()):raise ValueError('画像差分パラメータが不正です。')
    if int(p['crop_size'])!=p['crop_size'] or p['crop_size']<24 or p['crop_size']>256:raise ValueError('切出しサイズは24〜256の整数です。')
    if any(p[k]>1 for k in ('quiet_fraction','peripheral_fraction','localized_fraction','max_core_fraction','min_coverage','min_grooming_fraction','max_background_change','body_core_fraction')):
        raise ValueError('割合のパラメータは0より大きく1以下です。')
    if p['quiet_fraction']>=p['peripheral_fraction']:raise ValueError('静止判定の閾値はグルーミング候補より小さくしてください。')
    video_hash=sha256(video);tracks_hash=sha256(tracks);labels={};manual_hash=None
    if manual_session:
        manual_bytes=Path(manual_session).read_bytes();manual_hash=hashlib.sha256(manual_bytes).hexdigest()
        data=json.loads(manual_bytes)
        if data.get('schema_version')!=1 or data.get('video_sha256')!=video_hash or data.get('tracks_sha256')!=tracks_hash:
            raise ValueError('手入力と動画・追跡データが一致しません。異なる追跡の同じIDは対応付けません。手入力アプリの比較ボタンを使ってください。')
        labels=data.get('labels',{})
    source=read_rows(tracks)
    lookup={}
    for row in source:
        key=int(row['frame']),int(row['slot'])
        if key in lookup or key[0]<0:raise ValueError('追跡データのフレーム・IDが不正または重複しています。')
        lookup[key]=row
    all_slots=sorted({s for _,s in lookup});slots=[s for s in all_slots if s not in excluded_slots]
    if not slots:raise ValueError('解析対象の個体がありません。')
    frames=max(k for k,_ in lookup)+1
    if len(lookup)!=frames*len(all_slots):raise ValueError('欠測を含む全フレーム・全IDの追跡座標が必要です。')
    cap=cv2.VideoCapture(str(video));writer=None
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(3));height=int(cap.get(4))
        if not cap.isOpened() or not math.isfinite(fps) or fps<=0:raise ValueError('動画のFPSを確認できません。')
        for (k,_),row in lookup.items():
            if abs(float(row['time_s'])-k/fps)>1e-4:raise ValueError('追跡座標と動画のFPS・時刻が一致しません。')
        duration=frames/fps;out.mkdir(parents=True,exist_ok=True)
        preview_slot=next((int(key.split(':')[0]) for key in labels if int(key.split(':')[0]) in slots),slots[0])
        if preview:
            writer=cv2.VideoWriter(str(out/'grooming_preview.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(768,320))
            if not writer.isOpened():raise ValueError('グルーミング確認動画を作成できません。')
        records=[];previous={};geometries={};last_gray=None;size=int(p['crop_size']);half=math.ceil(size/math.sqrt(2))
        for frame in range(frames):
            ok,image=cap.read()
            if not ok:raise ValueError(f'動画が追跡データより短くなっています：frame {frame}')
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);time=frame/fps
            decoded_repeat=last_gray is not None and np.array_equal(gray,last_gray)
            last_gray=gray
            for slot in slots:
                row=lookup[frame,slot];record=dict(frame=frame,time_s=time,slot=slot,dt_s=0.,valid=0,candidate='unknown',reason='tracking_missing',
                                                  speed_body_lengths_s='',instant_speed_body_lengths_s='',turn_deg='',illumination_step='',background_change='',body_length_px='',body_width_px='',alignment_reliable=0,
                                                  tracking_image_updated=str(row.get('image_updated','1')),decoded_repeat=int(decoded_repeat))
                for name in REGIONS:record.update({name+'_change':'',name+'_energy':''})
                patch=raw_patch=diff=None
                try:
                    if row['status']!='observed':raise ValueError('tracking_missing')
                    center=np.array([float(row['x_px']),float(row['y_px'])]);axis=np.array([float(row['axis_dx']),float(row['axis_dy'])])
                    ratio=float(row['axis_ratio'])
                    if not np.all(np.isfinite(np.r_[center,axis,ratio])) or np.linalg.norm(axis)<.5:raise ValueError('body_axis_unreliable')
                    if min(center[0],center[1],width-1-center[0],height-1-center[1])<half:raise ValueError('crop_clipped')
                    axis/=np.linalg.norm(axis);old=previous.get(slot)
                    reliable=ratio>=p['min_axis_ratio'];record['alignment_reliable']=int(reliable)
                    if not reliable and old is not None:axis=old['axis']
                    if old is not None and np.dot(axis,old['axis'])<0:axis=-axis
                    patch=aligned_patch(gray,center,axis,size)
                    raw_patch=cv2.getRectSubPix(gray,(size,size),tuple(center.astype(float)))
                    geometry=geometries.get(slot) or body_geometry(patch,p['body_contrast'],p['body_core_fraction'])
                    if geometry is None:raise ValueError('body_not_resolved')
                    record.update(body_length_px=geometry[0]*2,body_width_px=geometry[1]*2)
                    for other in all_slots:
                        other_row=lookup[frame,other]
                        if other==slot or other_row['status']!='observed':continue
                        location=np.array([float(other_row['x_px']),float(other_row['y_px'])])
                        if np.linalg.norm(location-center)<max(geometry)*4:raise ValueError('neighbor_overlap')
                    geometries[slot]=geometry
                    # Tracking may suppress tiny local leg motion after blurring.
                    # Only discard its repeat flag when decoded images agree too.
                    if str(row.get('image_updated','1')).lower() in ('0','false') and decoded_repeat:
                        record['reason']='repeated_encoded_frame'
                    else:
                        continuous=old is not None and old['segment']==row['segment'] and time-old['time']<=p['max_frame_gap_s']
                        positions=[(t,c) for t,c in old['positions'] if t>=time-p['speed_window_s']] if continuous else []
                        if continuous and not positions:positions=[old['positions'][-1]]
                        positions.append((time,center.copy()))
                        previous[slot]=dict(patch=patch,axis=axis,center=center,time=time,segment=row['segment'],reliable=reliable,positions=positions)
                        if old is None or old['segment']!=row['segment']:record['reason']='new_tracking_segment'
                        elif time-old['time']>p['max_frame_gap_s']:record['reason']='observation_gap'
                        else:
                            dt=time-old['time'];record['dt_s']=dt
                            features,diff=difference_features(old['patch'],patch,region_masks(size,geometry),p)
                            instant_speed=float(np.linalg.norm(center-old['center'])/(2*geometry[0]*dt))
                            speed=movement_speed(positions,2*geometry[0])
                            turn=float(np.degrees(np.arccos(np.clip(np.dot(axis,old['axis']),-1,1))))
                            state,reason=classify(features,speed,turn,p)
                            if not reliable or not old['reliable']:state,reason='unknown','body_axis_unreliable'
                            record.update(features,speed_body_lengths_s=speed,instant_speed_body_lengths_s=instant_speed,turn_deg=turn,candidate=state,reason=reason,
                                          valid=int(reason not in ('illumination_jump','background_motion','turning','locomotion_overlap','body_axis_unreliable')))
                except (KeyError,TypeError,ValueError) as error:
                    previous.pop(slot,None);record['reason']=str(error)
                records.append(record)
                if writer is not None and slot==preview_slot:
                    panels=[]
                    for picture in (raw_patch,patch,diff):
                        if picture is None:panel=np.zeros((240,256,3),np.uint8)
                        elif picture is diff:
                            panel=cv2.applyColorMap(np.clip(picture*4,0,255).astype(np.uint8),cv2.COLORMAP_TURBO);panel=cv2.resize(panel,(256,240))
                        else:panel=cv2.cvtColor(cv2.resize(picture,(256,240)),cv2.COLOR_GRAY2BGR)
                        panels.append(panel)
                    display=np.zeros((320,768,3),np.uint8);display[:240]=np.hstack(panels)
                    for x,title in ((8,'Centered raw'),(264,'Body-aligned'),(520,'Compensated difference')):
                        cv2.putText(display,title,(x,260),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1)
                    human=labels.get(f'{slot}:{int(time)}','unlabeled') or 'unlabeled'
                    cv2.putText(display,f'ID {slot} {time:.2f}s human={human} candidate={record["candidate"]}',(8,284),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
                    cv2.putText(display,record['reason'][:80],(8,307),cv2.FONT_HERSHEY_SIMPLEX,.48,(180,200,255),1);writer.write(display)
            if progress and (frame%20==0 or frame==frames-1):progress('load',frame+1,frames,'グルーミングの画像差分を計算')
        intervals=aggregate(records,slots,duration,p,labels);validation=validation_report(intervals)
        provenance=dict(method=VERSION,video=str(video),tracks=str(tracks),video_sha256=video_hash,tracks_sha256=tracks_hash,
                        manual_session=str(Path(manual_session).resolve()) if manual_session else None,
                        manual_session_sha256=manual_hash,
                        parameters=p,included_slots=slots,excluded_slots=list(excluded_slots),fps=fps,frames=frames,preview_slot=preview_slot,
                        bout_note='longest_candidate_s counts positive evidence only in a bout; short valid pauses may bridge strokes but add no evidence.',
                        region_note='End A/B are unsigned body-axis halves, not head/body labels. Locomotion results are unchanged.')
        save_rows(out/'grooming_frames.csv',records);save_rows(out/'grooming_intervals.csv',intervals)
        summary=[]
        for slot in slots:
            for state in STATES:
                summary.append(dict(slot=slot,grooming_state=state,duration_s=sum(r['duration_s'] for r in intervals if r['slot']==slot and r['grooming_state']==state)))
        save_rows(out/'grooming_summary.csv',summary);plot_comparison(intervals,slots,out)
        provenance['comparison_examples_added']=save_comparison_examples(intervals,provenance,learning) if learning and labels else 0
        dump(out/'grooming_validation.json',validation);dump(out/'grooming_metadata.json',provenance)
        return out
    finally:
        cap.release()
        if writer is not None:writer.release()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True);parser.add_argument('--tracks',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--manual-session',type=Path)
    parser.add_argument('--learning',type=Path);parser.add_argument('--no-preview',action='store_true')
    parser.add_argument('--parameters',type=Path,help='JSON overrides; thresholds remain experimental')
    args=parser.parse_args()
    run_prototype(args.video,args.tracks,args.out,args.manual_session,learning=args.learning,preview=not args.no_preview,
                  parameters=json.loads(args.parameters.read_text(encoding='utf-8')) if args.parameters else None)


if __name__=='__main__':main()
