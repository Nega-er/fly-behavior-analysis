"""Video-specific background, ROI and provisional population estimate."""
from collections import Counter
import cv2
import numpy as np


def temporal_background(samples, config):
    """A low/high temporal percentile preserves animals that stop for most frames."""
    q = config.get('background_percentile', 10)
    polarity = config['foreground_polarity']
    return np.percentile(samples, q if polarity == 'bright' else 100-q if polarity == 'dark' else 50,
                         axis=0).astype(np.uint8)


def recover_stationary(gray, mask, tracker, frame, fps, info):
    """Spatial body cores only around previously observed stationary identities.

    Searching the whole image here would count static rim highlights as animals.
    Never synthesize a centroid when there is no body-shaped image evidence.
    """
    if not info['body_core_applied'] or tracker.c['foreground_polarity'] == 'both':
        return mask, 0
    result = mask.copy()
    brightness = 255-gray if tracker.c['foreground_polarity'] == 'dark' else gray
    for i in range(tracker.n):
        if tracker.last_frame[i] < 0 or not tracker.stationary(i):
            continue
        if (frame-tracker.last_frame[i])/fps > tracker.c.get('stationary_identity_hold_s',120):
            continue
        x,y = np.round(tracker.pos[i]).astype(int)
        # Existing foreground at the anchor needs no rescue.
        if mask[max(0,y-5):y+6,max(0,x-5):x+6].any():
            continue
        size = max(31, int(np.sqrt(tracker.c['max_area_px'])*2)+1)|1
        radius = size
        x0,y0 = max(0,x-radius),max(0,y-radius)
        crop = brightness[y0:min(gray.shape[0],y+radius+1),x0:min(gray.shape[1],x+radius+1)]
        contrast = cv2.morphologyEx(crop,cv2.MORPH_TOPHAT,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
        candidates = ((contrast>=tracker.c['threshold']) & (crop>=info['body_core_threshold'])).astype(np.uint8)
        n,labels,stats,centers = cv2.connectedComponentsWithStats(candidates)
        eligible = [j for j in range(1,n) if tracker.c['min_area_px']<=stats[j,4]<=tracker.c['max_area_px']
                    and np.linalg.norm(centers[j]+[x0,y0]-tracker.pos[i])<=8]
        if len(eligible)==1:
            target=result[y0:y0+crop.shape[0],x0:x0+crop.shape[1]]
            target[labels==eligible[0]]=1
    result &= (roi_mask(gray.shape,tracker.c)>0).astype(np.uint8)
    return result, int(np.count_nonzero(result & ~mask))


def roi_mask(shape, config):
    roi=np.ones(shape,np.uint8)*255
    if config.get("roi_rect_px") is not None:
        x,y,w,h=map(int,config["roi_rect_px"])
        if w<=0 or h<=0 or x<0 or y<0 or x+w>shape[1] or y+h>shape[0]:
            raise ValueError("ROI must be inside the video")
        roi[:]=0;roi[y:y+h,x:x+w]=255
    if config["arena_center_px"] is not None:
        circle=np.zeros(shape,np.uint8)
        cv2.circle(circle,tuple(map(int,config["arena_center_px"])),int(config["arena_radius_px"]),255,-1)
        roi=cv2.bitwise_and(roi,circle)
    return roi


def foreground(gray, background, config, diagnostics=False):
    difference=(cv2.subtract(gray,background) if config["foreground_polarity"]=="bright" else
                cv2.subtract(background,gray) if config["foreground_polarity"]=="dark" else
                cv2.absdiff(gray,background))
    mask=((difference>=config["threshold"]) & (roi_mask(gray.shape,config)>0)).astype(np.uint8)
    mode=config.get("body_detection_mode","auto")
    core_threshold=config.get("body_min_brightness",0)
    applied=False
    if mode!="background_diff":
        # Work on body candidates before the area upper limit is applied:
        # bright body+wing masks may be larger than the final body core.
        brightness=255-gray if config["foreground_polarity"]=="dark" else gray
        if not core_threshold:
            count,labels,stats,_=cv2.connectedComponentsWithStats(mask)
            peaks=[]
            for i in range(1,count):
                if config["min_area_px"]<=stats[i,4]<=config["max_area_px"]*4:
                    x,y,w,h=stats[i,:4]
                    component=labels[y:y+h,x:x+w]==i
                    peaks.append(int(brightness[y:y+h,x:x+w][component].max()))
            reference=float(np.percentile(peaks,75,method="higher")) if peaks else 0.
            if mode=="body_core" or reference>=config.get("body_core_reference_min",200):
                core_threshold=max(config["threshold"],reference*config.get("body_core_fraction",.7))
        if core_threshold:
            mask&=(brightness>=core_threshold).astype(np.uint8)
            applied=True
    info=dict(body_core_applied=int(applied),body_core_threshold=float(core_threshold or 0))
    return (difference,mask,info) if diagnostics else (difference,mask)


def estimate_population(counts):
    counts=[int(x) for x in counts]
    nonzero=[x for x in counts if x>0]
    if not nonzero:
        raise ValueError("No animals detected. Adjust ROI, contrast threshold or area limits.")
    histogram=Counter(counts)
    positive=Counter(nonzero)
    modal=max(positive,key=lambda n:(positive[n],n))
    # Use a repeatedly supported upper count, rather than the most common count
    # during contacts. Isolated split/noise peaks do not raise the population.
    number=max([modal]+[n for n,frequency in positive.items()
                        if frequency>=max(3,int(np.ceil(len(counts)*.3)))])
    if number>500:
        raise ValueError("Too many detected components. Adjust segmentation before tracking.")
    return dict(estimated_flies=number,modal_count=modal,modal_fraction=histogram[modal]/len(counts),
                selected_count_fraction=histogram[number]/len(counts),method='repeated_supported_upper_count',
                count_min=min(counts),count_max=max(counts),sample_frames=len(counts),
                histogram=dict(sorted(histogram.items())),
                warning="Provisional component count, not verified animal count. Contacts, stationary animals and reflections can bias it.")


def prepare_video(video,config,progress=None):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    if count<2 or not np.isfinite(fps) or fps<=0:
        raise ValueError("Video needs at least two frames and a valid FPS")
    samples=[];indices=[]
    try:
        sample_indices=np.unique(np.linspace(0,count-1,config["background_samples"]).astype(int))
        for sample_number,k in enumerate(sample_indices):
            cap.set(1,int(k));ok,image=cap.read()
            if ok:
                samples.append(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY));indices.append(int(k))
            if progress:progress("prepare",sample_number+1,2*len(sample_indices)+1,"背景用画像を読込")
    finally:
        cap.release()
    if not samples:
        raise ValueError("No sampled frames decoded")
    if config["background_path"]:
        background=cv2.imread(str(config["background_path"]),0)
        if background is None or background.shape!=samples[0].shape:
            raise ValueError("Background does not match this video's dimensions")
    else:
        background=temporal_background(samples,config)
    kernel=(config["blur_size"],)*2;blurred_bg=cv2.GaussianBlur(background,kernel,0)
    counts=[];sample_rows=[]
    for sample_number,(index,gray) in enumerate(zip(indices,samples)):
        _,mask=foreground(cv2.GaussianBlur(gray,kernel,0),blurred_bg,config)
        n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
        valid=[i for i in range(1,n) if config["min_area_px"]<=stats[i,4]<=config["max_area_px"]]
        number=len(valid);counts.append(number)
        sample_rows.append(dict(frame=index,time_s=index/fps,detected_components=number))
        if progress:progress("prepare",len(sample_indices)+sample_number+1,2*len(sample_indices)+1,"個体候補を検出")
    population=estimate_population(counts)
    if progress:progress("prepare",1,1,f"個体候補数 {population['estimated_flies']} を確認")
    return background,population,sample_rows
