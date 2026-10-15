#!/usr/bin/env python3
"""Conservative shared-arena tracking and activity analysis; see README.md."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fly_activity_mpl"))
import cv2
import numpy as np
from scipy.ndimage import median_filter
from identity_tracker import Tracker
from direction import body_axis, read_anchors, analyze_direction, plot_direction, export_direction_preview, remap_anchors
from video_setup import prepare_video, roi_mask, foreground, temporal_background, recover_stationary
from movement_rate import summarize_rates,plot_rates
from automatic_head import load_model, appearance_features, predict_probability, sha256
from cumulative_learning import active_model_path,training_frames,motion_features,behavior_prediction

DEFAULTS = dict(expected_flies=7, background_path=None, background_samples=31,
                threshold=30, blur_size=5, min_area_px=40, max_area_px=500,
                foreground_polarity="bright",
                arena_center_px=None, arena_radius_px=None, arena_diameter_mm=None,
                max_speed_px_s=600, collision_distance_px=28, assignment_margin_px=4,
                smooth_window_s=.1, speed_window_s=.2, move_start_speed=12, stop_start_speed=6,
                confirmation_s=.15, bin_s=1, repeat_mean_difference=0.05,
                repeat_max_difference=8, max_measurement_dt_s=.2, direction_min_axis_ratio=1.5,
                direction_max_axis_step_deg=40, direction_min_cosine=.5,
                direction_min_speed_px_s=3, direction_move_start_speed_px_s=6,
                direction_stop_start_speed_px_s=3, direction_confirmation_s=.1,
                roi_rect_px=None, automatic_head_min_probability=.6,
                body_detection_mode="auto",body_min_brightness=0,
                body_core_reference_min=200,body_core_fraction=.7,
                background_percentile=10,identity_hold_s=15,stationary_identity_hold_s=120,
                identity_recovery_radius_px=240,stationary_recovery_radius_px=24,
                identity_appearance_weight=20)


def write_csv(path, rows, fields=None):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_config(c):
    if not isinstance(c["expected_flies"], int) or c["expected_flies"] < 1:
        raise ValueError("expected_flies must be a positive integer")
    if c["blur_size"] < 1 or c["blur_size"] % 2 != 1:
        raise ValueError("blur_size must be positive and odd")
    if not 0 < c["min_area_px"] < c["max_area_px"]:
        raise ValueError("Invalid area limits")
    if not 0 <= c["stop_start_speed"] < c["move_start_speed"]:
        raise ValueError("Require 0 <= stop_start_speed < move_start_speed")
    for key in ("max_speed_px_s", "confirmation_s", "bin_s", "collision_distance_px",
                "identity_hold_s", "stationary_identity_hold_s", "identity_recovery_radius_px",
                "stationary_recovery_radius_px"):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not 0 <= c["background_percentile"] < 50 or c["identity_appearance_weight"] < 0:
        raise ValueError("Invalid background percentile or appearance weight")
    if c["smooth_window_s"] < 0 or c["assignment_margin_px"] < 0:
        raise ValueError("Window and margin must be nonnegative")
    if c["foreground_polarity"] not in ("bright", "dark", "both"):
        raise ValueError("foreground_polarity must be bright, dark or both")
    if c["speed_window_s"] <= 0:
        raise ValueError("speed_window_s must be positive")
    if c["max_measurement_dt_s"] <= 0 or c["repeat_mean_difference"] < 0 or c["repeat_max_difference"] < 0:
        raise ValueError("Invalid repeated-frame parameters")
    if not (1 < c["direction_min_axis_ratio"] and 0 < c["direction_max_axis_step_deg"] < 90
            and 0 < c["direction_min_cosine"] <= 1 and c["direction_min_speed_px_s"] >= 0):
        raise ValueError("Invalid direction thresholds")
    if not (0 <= c["direction_stop_start_speed_px_s"] < c["direction_move_start_speed_px_s"]
            and c["direction_confirmation_s"] > 0):
        raise ValueError("Invalid direction activity thresholds")
    if not .5 < c["automatic_head_min_probability"] < 1:
        raise ValueError("automatic_head_min_probability must be between .5 and 1")
    if c["body_detection_mode"] not in ("auto","background_diff","body_core"):
        raise ValueError("Invalid body_detection_mode")
    if not (0<=c["body_min_brightness"]<=255 and 0<c["body_core_fraction"]<1
            and 0<c["body_core_reference_min"]<=255):
        raise ValueError("Invalid body core brightness parameters")
    if c["arena_diameter_mm"] is not None:
        if c["arena_diameter_mm"] <= 0 or not c["arena_radius_px"] or c["arena_radius_px"] <= 0:
            raise ValueError("mm conversion requires positive diameter and arena_radius_px")
    if (c["arena_center_px"] is None) != (c["arena_radius_px"] is None):
        raise ValueError("Set both arena_center_px and arena_radius_px")


def track_video(video, config, out, prepared_background=None, head_model=None, progress=None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(3)), int(cap.get(4))
    if not math.isfinite(fps) or fps <= 0 or count < 2:
        raise ValueError("Video requires valid FPS and at least two frames")
    bg_path = config["background_path"]
    if prepared_background is not None:
        background=prepared_background
    elif bg_path:
        background = cv2.imread(str(bg_path), cv2.IMREAD_GRAYSCALE)
        if background is None or background.shape != (height, width):
            raise ValueError("Background image missing or dimensions do not match video")
    else:
        samples = []
        for index in np.unique(np.linspace(0, count - 1, config["background_samples"]).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, image = cap.read()
            if ok:
                samples.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        if not samples:
            raise ValueError("No background samples could be read")
        background = temporal_background(samples, config)
    cv2.imwrite(str(out / "background.png"), background)
    kernel = (config["blur_size"],) * 2
    background = cv2.GaussianBlur(background, kernel, 0)
    roi = roi_mask((height,width),config)
    writer = cv2.VideoWriter(str(out / "tracking_preview.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise ValueError("Cannot create preview video")
    tracker = Tracker(config)
    appearance_cache=np.full((count,tracker.n,44),np.nan,dtype=np.float32) if head_model is not None else None
    tracks, quality = [], []
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame = 0
    previous_gray, last_update_frame = None, 0
    repeated_frames = 0
    last_body_info = None
    detection_rows = []
    colors = [(40, 220, 255), (255, 160, 50), (100, 255, 100), (255, 80, 200),
              (60, 100, 255), (255, 255, 80), (180, 120, 255)]
    while True:
        ok, image = cap.read()
        if not ok:
            break
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), kernel, 0)
        image_updated = True
        if previous_gray is not None:
            change = cv2.absdiff(gray, previous_gray)
            image_updated = not (float(change.mean()) <= config["repeat_mean_difference"]
                                 and int(change.max()) <= config["repeat_max_difference"])
        measurement_dt = min(config["max_measurement_dt_s"],
                             max(1/fps, (frame-last_update_frame)/fps)) if image_updated else 1/fps
        if image_updated:
            last_update_frame = frame
        else:
            repeated_frames += 1
        previous_gray = gray
        difference,mask,body_info=foreground(gray,background,config,diagnostics=True)
        if body_info['body_core_applied']:
            last_body_info = body_info
        mask, recovered_pixels = recover_stationary(gray, mask, tracker, frame, fps, last_body_info or body_info)
        n, labels, stats, centers = cv2.connectedComponentsWithStats(mask, 8)
        valid = [i for i in range(1, n) if config["min_area_px"] <= stats[i, 4] <= config["max_area_px"]]
        oversized = sum(stats[i, 4] > config["max_area_px"] for i in range(1, n))
        points = centers[valid]
        descriptors=[]
        for component in valid:
            sx,sy,sw,sh=stats[component,:4]
            yy,xx=np.nonzero(labels[sy:sy+sh,sx:sx+sw]==component)
            _,ratio=body_axis(np.column_stack((xx,yy)))
            values=gray[yy+sy,xx+sx].astype(float)
            descriptors.append([np.log(len(xx)),np.log(max(1,ratio)),values.mean()/255,values.std()/255])
        rows, close, usable = tracker.update(points, frame, fps, measurement_dt, image_updated, descriptors)
        for j,(point,descriptor) in enumerate(zip(points,descriptors)):
            detection_rows.append(dict(frame=frame,time_s=frame/fps,detection=j+1,x_px=float(point[0]),y_px=float(point[1]),
                                       area_px=float(np.exp(descriptor[0])),axis_ratio=float(np.exp(descriptor[1])),
                                       mean_brightness=descriptor[2]*255,std_brightness=descriptor[3]*255))
        for row in rows:
            row.update(axis_dx="", axis_dy="", axis_ratio="", measurement_dt_s=measurement_dt,
                       image_updated=int(image_updated), head_probability_axis_positive="",
                       automatic_head_confidence="")
            if row["status"] != "observed":
                continue
            nearest = int(np.argmin(np.linalg.norm(points - [row["x_px"], row["y_px"]], axis=1)))
            component = valid[nearest]
            sx, sy, sw, sh = stats[component, :4]
            yy, xx = np.nonzero(labels[sy:sy+sh, sx:sx+sw] == component)
            axis, ratio = body_axis(np.column_stack((xx, yy)))
            row.update(axis_dx=float(axis[0]), axis_dy=float(axis[1]), axis_ratio=ratio)
            if head_model is not None and np.all(np.isfinite(axis)):
                features=appearance_features(np.column_stack((xx,yy)),difference[yy+sy,xx+sx],axis)
                if frame>=len(appearance_cache):
                    appearance_cache=np.concatenate((appearance_cache,np.full((512,tracker.n,44),np.nan,np.float32)))
                appearance_cache[frame,row["slot"]-1]=features
                probability=predict_probability(features,head_model)
                row.update(head_probability_axis_positive=probability,
                           automatic_head_confidence=max(probability,1-probability))
        tracks.extend(rows)
        observed = sum(r["status"] == "observed" for r in rows)
        quality.append(dict(frame=frame, time_s=frame / fps, detections=len(points),
                            usable_detections=usable, observed_slots=observed,
                            close_detections=close, oversized_components=oversized,
                            extra_detections=max(0, usable - observed),
                            identity_reacquisitions=sum(r["identity_state"]=="reacquired" for r in rows),
                            reserved_missing_ids=sum(r["identity_state"]=="missing_reserved" for r in rows),
                            stationary_recovery_pixels=recovered_pixels,
                            body_core_applied=body_info["body_core_applied"],
                            body_core_threshold=body_info["body_core_threshold"],
                            review_required=int(observed != tracker.n or close > 0 or oversized > 0 or usable > tracker.n)))
        for point in points:
            cv2.circle(image, tuple(point.astype(int)), 15, (128, 128, 128), 1)
        for row in rows:
            if row["status"] != "observed":
                i=row['slot']-1
                if tracker.last_frame[i]>=0:
                    x,y=tracker.pos[i].astype(int)
                    cv2.drawMarker(image,(x,y),(140,140,140),cv2.MARKER_TILTED_CROSS,14,1)
                    cv2.putText(image,f'{i+1}:missing',(x+12,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(140,140,140),1)
                continue
            x, y = int(row["x_px"]), int(row["y_px"])
            color = colors[(row["slot"] - 1) % len(colors)]
            cv2.circle(image, (x, y), 9, color, 2)
            axis = np.array([row["axis_dx"], row["axis_dy"]])
            if np.all(np.isfinite(axis)):
                cv2.line(image, tuple((np.array([x,y])-18*axis).astype(int)),
                         tuple((np.array([x,y])+18*axis).astype(int)), (255,255,0), 1)
            cv2.putText(image, f'{row["slot"]}:{row["segment"]}', (x + 12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2)
        cv2.putText(image, f'{frame/fps:.2f}s  observed {observed}/{tracker.n}  slot:segment',
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .65, (100, 255, 255), 2)
        writer.write(image)
        if frame == 0:
            cv2.imwrite(str(out / "first_frame_overlay.jpg"), image)
        frame += 1
        if progress and (frame == 1 or frame % max(1, count//200) == 0):
            progress("track",frame,count,f"{frame:,} / {count:,} フレーム")
    cap.release()
    writer.release()
    if frame < 2:
        raise ValueError("Fewer than two frames decoded")
    write_csv(out / "tracks.csv", tracks)
    event_fields=['frame','time_s','slot','segment','identity_gap_s','identity_match_distance_px',
                  'identity_previous_x_px','identity_previous_y_px','x_px','y_px']
    write_csv(out/'identity_events.csv',[{k:r[k] for k in event_fields} for r in tracks if r['identity_state']=='reacquired'],event_fields)
    write_csv(out / "detections.csv",detection_rows,
              ['frame','time_s','detection','x_px','y_px','area_px','axis_ratio','mean_brightness','std_brightness'])
    if appearance_cache is not None:np.savez_compressed(out/"appearance_features.npz",features=appearance_cache[:frame],feature_version=1)
    write_csv(out / "quality.csv", quality)
    if progress:progress("track",1,1,"座標の保存完了")
    return tracks, dict(fps=fps, frames_decoded=frame, frames_reported=count,
                        timing_source="frame/FPS (constant-frame-rate assumption)",
                        width=width, height=height,
                        review_frames=sum(q["review_required"] for q in quality),
                        repeated_encoded_frames=repeated_frames,
                        identity_method="reserved_position_appearance",
                        identity_reacquisitions=sum(q["identity_reacquisitions"] for q in quality),
                        reserved_missing_id_frames=sum(q["reserved_missing_ids"] for q in quality),
                        above_requested_population_fraction=sum(q['usable_detections']>tracker.n for q in quality)/frame,
                        stationary_recovery_frames=sum(q["stationary_recovery_pixels"]>0 for q in quality))


def contiguous_runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def classify_intervals(speed, dt, config):
    """Index k represents the interval ending at frame k; no missing bridges."""
    states = np.full(len(speed), "unknown", dtype=object)
    state, pending, start, elapsed = "unknown", None, 0, 0.
    for k, value in enumerate(speed):
        if not np.isfinite(value):
            state, pending, elapsed = "unknown", None, 0.
            continue
        candidate = ("moving" if value >= config["move_start_speed"] else
                     "stopped" if value <= config["stop_start_speed"] else None)
        if candidate is None or candidate == state:
            pending, elapsed = None, 0.
        else:
            if pending != candidate:
                pending, start, elapsed = candidate, k, 0.
            elapsed += dt[k]
            if elapsed + 1e-10 >= config["confirmation_s"]:
                state = candidate
                states[start:k + 1] = state
                pending, elapsed = None, 0.
        states[k] = state
    return states


def window_speed(distance, times, window_s):
    """Trailing time-weighted speed within each contiguous measured run.

    Repeated encoded frames must not turn one movement into isolated bursts.
    Partial intervals at the left window edge are weighted by their duration.
    """
    result = np.full(len(times), np.nan)
    for a, b in contiguous_runs(np.isfinite(distance)):
        for k in range(a, b):
            left = max(times[a-1], times[k]-window_s)
            j = max(a, int(np.searchsorted(times, left, side="right")))
            lengths = np.maximum(0., times[j:k+1]-np.maximum(times[j-1:k], left))
            interval_dt = times[j:k+1]-times[j-1:k]
            result[k] = float(np.sum(distance[j:k+1]*lengths/interval_dt)/(times[k]-left))
    return result


def analyze(tracks, config, out, head_anchors=None,appearance_cache=None,behavior_model=None,progress=None):
    head_anchors = head_anchors or {}
    n = config["expected_flies"]
    by_frame = {}
    for row in tracks:
        row = dict(row)
        frame, slot = int(row["frame"]), int(row["slot"])
        if slot not in range(1, n + 1) or frame < 0 or (frame, slot) in by_frame:
            raise ValueError("Invalid or duplicate frame/slot")
        by_frame[frame, slot] = row
    frames = max(k[0] for k in by_frame) + 1
    if frames < 2 or len(by_frame) != frames * n:
        raise ValueError("CSV requires one row per frame per slot, including unknown rows")
    for key in head_anchors:
        if key not in by_frame or by_frame[key]["status"] != "observed":
            raise ValueError(f"Head anchor must refer to an observed frame/slot: {key}")
    times = np.array([float(by_frame[k, 1]["time_s"]) for k in range(frames)])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Timestamps must be finite and strictly increasing")
    dt = np.r_[0., np.diff(times)]
    duration = times[-1] - times[0]
    scale = config["arena_diameter_mm"] / (2 * config["arena_radius_px"]) if config["arena_diameter_mm"] else 1.
    unit = "mm" if config["arena_diameter_mm"] else "px"
    window = max(1, int(round(config["smooth_window_s"] / np.median(dt[1:]))))
    if window % 2 == 0:
        window += 1
    series, summaries, bouts, segments = [], [], [], []
    all_distances = np.zeros((n, frames))
    all_states = np.empty((n, frames), object)
    all_valid = np.zeros((n, frames), bool)
    all_directions = np.empty((n, frames), object)
    direction_summary = []
    for slot in range(1, n + 1):
        if progress:progress("analyze",slot-1,n,f"個体 {slot} / {n} を解析")
        xy = np.full((frames, 2), np.nan)
        seg = np.zeros(frames, int)
        for k in range(frames):
            r = by_frame[k, slot]
            if abs(float(r["time_s"]) - times[k]) > 1e-8:
                raise ValueError("All slots must share frame timestamps")
            if r["status"] == "observed":
                xy[k] = float(r["x_px"]), float(r["y_px"])
                seg[k] = int(r["segment"])
                if seg[k] <= 0 or not np.all(np.isfinite(xy[k])):
                    raise ValueError("Observed rows require finite coordinates and positive segment")
            elif r["status"] != "unknown":
                raise ValueError("status must be observed or unknown")
        smooth = xy.copy()
        measurement_dt = np.array([float(by_frame[k,slot].get("measurement_dt_s", dt[k]) or dt[k])
                                   for k in range(frames)])
        if not np.all(np.isfinite(measurement_dt)) or np.any(measurement_dt < 0):
            raise ValueError("Invalid measurement_dt_s")
        gate_dt = np.maximum(dt, np.minimum(measurement_dt, config["max_measurement_dt_s"]))
        # Split at both missing coordinates and segment changes.
        good = np.isfinite(xy[:, 0])
        boundaries = [0] + [k for k in range(1, frames)
                            if not (good[k] and good[k-1] and seg[k] == seg[k-1])
                            or np.linalg.norm(xy[k]-xy[k-1]) > config["max_speed_px_s"] * gate_dt[k]] + [frames]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            if good[a]:
                smooth[a:b] = median_filter(xy[a:b], size=(min(window, 2*((b-a)//2)+1), 1), mode="nearest")
                segments.append(dict(slot=slot, segment=int(seg[a]), start_s=times[a],
                                     end_s=times[b-1], observed_frames=b-a))
        valid = np.r_[False, good[1:] & good[:-1] & (seg[1:] == seg[:-1])]
        raw = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1)] * scale
        distance = np.r_[np.nan, np.linalg.norm(np.diff(smooth, axis=0), axis=1)] * scale
        # External CSVs also undergo physical jump rejection; split downstream state.
        valid &= np.nan_to_num(raw) <= config["max_speed_px_s"] * scale * gate_dt + 1e-10
        distance[~valid] = np.nan
        raw[~valid] = np.nan
        instant_speed = np.divide(distance, dt, out=np.full(frames, np.nan), where=dt > 0)
        speed = window_speed(distance, times, config["speed_window_s"])
        state_config = dict(config, move_start_speed=config["move_start_speed"] * scale,
                            stop_start_speed=config["stop_start_speed"] * scale)
        state = classify_intervals(speed, dt, state_config)
        slot_rows = [by_frame[k, slot] for k in range(frames)]
        headings, head_sources, longitudinal, cosine, directions, reasons, direction_activity = analyze_direction(
            slot_rows, smooth, times, valid, state, config, scale, head_anchors, window_speed, classify_intervals)
        baseline_directions=directions.copy();learning_confidence=np.zeros(frames)
        if behavior_model is not None and behavior_model.get("behavior_memory") and appearance_cache is not None and not head_anchors:
            delta=np.vstack((np.full((1,2),np.nan),np.diff(smooth,axis=0)));delta[~valid]=np.nan
            vx=window_speed(delta[:,0],times,config["speed_window_s"]);vy=window_speed(delta[:,1],times,config["speed_window_s"])
            for k in range(frames):
                if not valid[k] or not np.isfinite(vx[k]+vy[k]):continue
                row=slot_rows[k]
                try:axis=[float(row["axis_dx"]),float(row["axis_dy"])];ratio=float(row["axis_ratio"])
                except (KeyError,ValueError):continue
                if ratio<config["direction_min_axis_ratio"]:continue
                feature=motion_features(appearance_cache[k,slot-1],axis,vx[k],vy[k],ratio,behavior_model["feature_scale"])
                prediction,confidence=behavior_prediction(feature,behavior_model);learning_confidence[k]=confidence
                if prediction is not None:
                    directions[k]=prediction;reasons[k]="cumulative_human_correction"
                    direction_activity[k]="stopped" if prediction=="stopped" else "moving"
        all_directions[slot-1] = directions
        ds = dict(slot=slot, unit=unit, heading_observed_frames=int(np.isfinite(headings[:, 0]).sum()))
        for name in ("forward", "backward", "lateral", "stopped", "unknown"):
            selected = directions == name
            ds[name+"_s"] = float(dt[selected].sum())
            ds[name+"_measured_distance"] = float(np.nansum(distance[selected]))
        direction_summary.append(ds)
        measured = float(dt[valid].sum())
        moving, stopped = state == "moving", state == "stopped"
        move_t, stop_t = float(dt[moving].sum()), float(dt[stopped].sum())
        total = float(np.nansum(distance))
        moving_distance = float(np.nansum(distance[moving]))
        summaries.append(dict(slot=slot, segment_count=sum(s["slot"] == slot for s in segments),
                              observed_frame_fraction=float(good.mean()),
                              valid_interval_s=measured, unmeasured_interval_s=duration-measured,
                              classified_s=move_t+stop_t, unclassified_valid_s=measured-move_t-stop_t,
                              raw_distance=float(np.nansum(raw)), smoothed_distance=total,
                              moving_distance=moving_distance, moving_s=move_t, stopped_s=stop_t,
                              moving_fraction_classified=move_t/(move_t+stop_t) if move_t+stop_t else "",
                              distance_per_valid_s=total/measured if measured else "",
                              mean_moving_speed=moving_distance/move_t if move_t else "", unit=unit,
                              identity_note="reserved IDs matched by position/appearance; gaps excluded; identity after contacts requires human verification"))
        cumulative = np.cumsum(np.nan_to_num(distance))
        all_distances[slot-1] = np.nan_to_num(distance)
        all_states[slot-1] = state
        all_valid[slot-1] = valid
        for k in range(frames):
            series.append(dict(frame=k, time_s=times[k], slot=slot, segment=int(seg[k]),
                               x_smooth_px=float(smooth[k, 0]) if good[k] else "",
                               y_smooth_px=float(smooth[k, 1]) if good[k] else "",
                               valid_interval=int(valid[k]),
                               distance=float(distance[k]) if valid[k] else "",
                               speed=float(speed[k]) if valid[k] else "",
                               instantaneous_speed=float(instant_speed[k]) if valid[k] else "", state=state[k],
                               activity_reason=("speed_hysteresis" if state[k] != "unknown" else
                                                "tracking_missing" if not good[k] else
                                                "segment_start_or_invalid_interval" if not valid[k] else
                                                "speed_not_confirmed"),
                               head_dx=float(headings[k,0]) if np.isfinite(headings[k,0]) else "",
                               head_dy=float(headings[k,1]) if np.isfinite(headings[k,1]) else "",
                               head_source=head_sources[k],
                               automatic_head_confidence=slot_rows[k].get("automatic_head_confidence",""),
                               longitudinal_speed=float(longitudinal[k]) if np.isfinite(longitudinal[k]) else "",
                               direction_cosine=float(cosine[k]) if np.isfinite(cosine[k]) else "",
                               locomotion_direction=directions[k], direction_reason=reasons[k],
                               baseline_locomotion_direction=baseline_directions[k],learning_confidence=learning_confidence[k],
                               direction_activity_state=direction_activity[k],
                               cumulative_measured_distance=float(cumulative[k]), unit=unit))
        for behavior in ("moving", "stopped"):
            for a, b in contiguous_runs(state == behavior):
                bouts.append(dict(slot=slot, segment=int(seg[a]), state=behavior,
                                  start_s=times[a-1], end_s=times[b-1], duration_s=float(dt[a:b].sum()),
                                  distance=float(np.nansum(distance[a:b])), unit=unit,
                                  left_censored=int(a == 1 or state[a-1] == "unknown"),
                                  right_censored=int(b == frames or state[b] == "unknown")))
    if progress:
        progress("analyze",1,1,"行動解析完了")
        progress("save",0,1,"時間集計を作成")
    bins = []
    for start in np.arange(times[0], times[-1], config["bin_s"]):
        end = min(start + config["bin_s"], times[-1])
        # Proportionally allocate intervals crossing a bin boundary.
        overlap = np.maximum(0., np.minimum(times[1:], end) - np.maximum(times[:-1], start))
        weights = np.r_[0., overlap / dt[1:]]
        interval_time = np.r_[0., overlap]
        for slot in range(n):
            bins.append(dict(start_s=float(start), end_s=float(end), slot=slot+1,
                             distance=float(np.sum(all_distances[slot]*weights)),
                             valid_s=float(np.sum(all_valid[slot]*interval_time)),
                             moving_s=float(np.sum((all_states[slot]=="moving")*interval_time)),
                             stopped_s=float(np.sum((all_states[slot]=="stopped")*interval_time)), unit=unit))
    write_csv(out / "activity.csv", series)
    write_csv(out / "summary_by_slot.csv", summaries)
    write_csv(out / "segments.csv", segments, ["slot", "segment", "start_s", "end_s", "observed_frames"])
    write_csv(out / "bouts.csv", bouts, ["slot", "segment", "state", "start_s", "end_s", "duration_s", "distance", "unit", "left_censored", "right_censored"])
    write_csv(out / "time_bins.csv", bins)
    rate_rows=summarize_rates(bins,n)
    write_csv(out/"movement_rate.csv",rate_rows)
    if progress:progress("save",1,4,"移動量の図を作成")
    plot_rates(rate_rows,bins,out/"movement_rate.png")
    write_csv(out / "direction_summary.csv", direction_summary)
    measured_time = sum(s["valid_interval_s"] for s in summaries)
    group = dict(expected_flies=n, duration_s=duration, unit=unit,
                 measured_distance=float(all_distances.sum()),
                 valid_fly_seconds=measured_time,
                 possible_fly_seconds=n*duration,
                 interval_coverage=measured_time/(n*duration),
                 distance_per_valid_fly_second=float(all_distances.sum()/measured_time) if measured_time else None,
                 all_seven_interval_fraction=float(dt[np.all(all_valid, axis=0)].sum()/duration),
                 smoothing_window_frames=window,
                 warning="Measured distance excludes missing/ambiguous intervals; not a complete-path estimate. Slots are not verified identities across segments. Movement thresholds are provisional.")
    (out / "summary_group.json").write_text(json.dumps(group, indent=2, ensure_ascii=False), encoding="utf-8")
    from period_statistics import compute_period_statistics,save_period_statistics
    save_period_statistics(out,compute_period_statistics(series,duration,unit))
    if progress:progress("save",2,4,"距離と移動停止の図を作成")
    plot_report(times, all_distances, all_states, summaries, group, out)
    if progress:progress("save",3,4,"前進後退の図を作成")
    plot_direction(times, all_directions, out,
                   automatic=behavior_model is not None or any(str(r["head_source"]).startswith("auto") for r in series))
    if progress:progress("save",1,1,"集計と図の保存完了")
    return group


def plot_report(times, distances, states, summaries, group, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for i, row in enumerate(distances):
        axes[0, 0].plot(times, np.cumsum(row), label=f'Slot {i+1}')
    axes[0, 0].set(title="Cumulative measured distance (gaps excluded)", xlabel="Time (s)", ylabel=group["unit"])
    axes[0, 0].legend(ncol=2)
    code = np.array([[{"unknown": 0, "stopped": 1, "moving": 2}[v] for v in row] for row in states])
    from matplotlib.colors import ListedColormap
    mesh = axes[0, 1].pcolormesh(times, np.arange(len(states)+1)+.5, code[:, 1:],
                                  cmap=ListedColormap(["#aaaaaa", "#3878b6", "#e8952f"]), vmin=0, vmax=2, shading="flat")
    axes[0, 1].set(title="Activity: gray unknown / blue stopped / orange moving", xlabel="Time (s)", ylabel="Slot", yticks=range(1, len(states)+1))
    axes[1, 0].bar(range(1, len(states)+1), [s["smoothed_distance"] for s in summaries])
    axes[1, 0].set(title="Distance by display slot (identity may change)", xlabel="Slot", ylabel=group["unit"])
    axes[1, 1].bar(range(1, len(states)+1), [s["valid_interval_s"]/group["duration_s"] for s in summaries])
    axes[1, 1].set(title="Valid interval coverage", xlabel="Slot", ylabel="Fraction", ylim=(0, 1.05))
    fig.suptitle(f'7-fly activity | measured {group["measured_distance"]:.1f} {group["unit"]} | coverage {group["interval_coverage"]:.1%}'.replace("7-fly", f'{len(states)}-fly'))
    fig.savefig(out / "report.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--tracks", type=Path, help="Re-analyze this program's tracks.csv")
    parser.add_argument("--progress", action="store_true", help="Emit stage progress for the desktop app")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--flies", help="auto or positive integer; overrides config")
    parser.add_argument("--head-model", type=Path, help="Use a fixed model instead of the latest cumulative model")
    parser.add_argument("--out", type=Path, required=True, help="New/empty output directory")
    parser.add_argument("--arena-diameter-mm", type=float)
    parser.add_argument("--head-anchors", type=Path, help="CSV: frame,slot,head_x_px,head_y_px")
    parser.add_argument("--head-anchor-tracks", type=Path, help="Original tracks for spatial transfer of existing head annotations")
    parser.add_argument("--direction-video", type=Path, help="Original video for head/motion overlay")
    args = parser.parse_args()
    from progress_protocol import emit_progress
    progress=emit_progress if args.progress else None
    if progress:progress("prepare",0,1,"動画とモデルを準備")
    args.head_model=args.head_model or active_model_path()
    config = DEFAULTS.copy()
    if args.config:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        unknown = set(loaded) - set(config)
        if unknown:
            parser.error(f"Unknown config keys: {sorted(unknown)}")
        config.update(loaded)
        if config["background_path"]:
            config["background_path"] = str((args.config.resolve().parent / config["background_path"]).resolve())
    if args.arena_diameter_mm is not None:
        config["arena_diameter_mm"] = args.arena_diameter_mm
    requested_flies=args.flies or (str(config["expected_flies"]) if args.config else "auto")
    prepared_background=None
    population=None;population_samples=None
    model=None;appearance_cache=None
    if args.video:
        prepared_background,population,population_samples=prepare_video(args.video,config,progress)
        config["expected_flies"]=population["estimated_flies"] if requested_flies=="auto" else int(requested_flies)
    elif requested_flies=="auto":
        with args.tracks.open(encoding="utf-8-sig",newline="") as f:
            config["expected_flies"]=max(int(r["slot"]) for r in csv.DictReader(f))
    else:
        config["expected_flies"]=int(requested_flies)
    validate_config(config)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be new or empty; existing results are never overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config_used.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.video:
        if progress:progress("track",0,1,"個体の追跡を開始")
        model=load_model(args.head_model) if not args.head_anchors else None
        tracks, metadata = track_video(args.video, config, args.out, prepared_background, model,progress)
        metadata["source"] = str(args.video.resolve())
        metadata["video_sha256"]=sha256(args.video)
        metadata["population_mode"]=requested_flies
        (args.out/"population_estimate.json").write_text(json.dumps(population,indent=2,ensure_ascii=False),encoding="utf-8")
        write_csv(args.out/"population_samples.csv",population_samples)
        if model is not None:
            metadata["head_mode"]="automatic_appearance"
            metadata["head_model"]=str(args.head_model.resolve())
            metadata["head_model_sha256"]=sha256(args.head_model)
            metadata["training_video_sha256"]=model["training_video_sha256"]
            metadata["training_frame_indices"]=training_frames(model,metadata["video_sha256"])
            metadata["same_video_as_head_training"]=bool(metadata["training_frame_indices"])
            metadata["learning_version"]=model.get("learning_version",0)
            metadata["behavior_training_samples"]=len(model.get("behavior_memory",[]))
        if (args.out/"appearance_features.npz").exists():
            with np.load(args.out/"appearance_features.npz",allow_pickle=False) as data:appearance_cache=data["features"]
    else:
        with args.tracks.open(encoding="utf-8-sig", newline="") as f:
            tracks = list(csv.DictReader(f))
        metadata = dict(source=str(args.tracks.resolve()), timing_source="CSV time_s")
        prior_path=args.tracks.parent/"metadata.json"
        if prior_path.exists() and not args.head_anchors:
            prior=json.loads(prior_path.read_text(encoding="utf-8"))
            if prior.get("head_mode")=="automatic_appearance":
                for key in ("head_mode","head_model","head_model_sha256","video_sha256",
                            "training_video_sha256","same_video_as_head_training","training_frame_indices"):
                    if key in prior:metadata[key]=prior[key]
        if not tracks:
            parser.error("Empty tracks CSV")
    anchors = read_anchors(args.head_anchors)
    if args.head_anchor_tracks:
        if not args.head_anchors:
            parser.error("--head-anchor-tracks requires --head-anchors")
        with args.head_anchor_tracks.open(encoding="utf-8-sig", newline="") as f:
            original_tracks = list(csv.DictReader(f))
        anchors, audit = remap_anchors(anchors, original_tracks, tracks)
        write_csv(args.out/"head_anchor_transfer.csv", audit,
                  ["frame","old_slot","new_slot","center_error_px","status"])
        write_csv(args.out/"head_anchors_remapped.csv",
                  [dict(frame=k[0],slot=k[1],head_x_px=float(h[0]),head_y_px=float(h[1])) for k,h in anchors.items()],
                  ["frame","slot","head_x_px","head_y_px"])
    if args.head_anchors:
        metadata["head_anchors_source"] = str(args.head_anchors.resolve())
        import shutil
        shutil.copyfile(args.head_anchors, args.out / "head_anchors_used.csv")
    result = analyze(tracks, config, args.out, anchors,appearance_cache,model,progress)
    preview_source = args.direction_video or args.video
    if preview_source is None and args.tracks:
        previous_metadata = args.tracks.parent / "metadata.json"
        if previous_metadata.exists():
            previous = json.loads(previous_metadata.read_text(encoding="utf-8"))
            candidate = Path(previous.get("direction_video", previous.get("source", "")))
            if candidate.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv") and candidate.is_file():
                preview_source = candidate
    if progress:progress("preview",0,1,"確認動画を作成")
    if preview_source is not None:
        export_direction_preview(preview_source, args.out, config,progress)
        metadata["direction_video"] = str(preview_source.resolve())
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if progress:progress("preview",1,1,"解析結果の保存完了")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
