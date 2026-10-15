"""Body-relative locomotion. Head polarity never comes from displacement."""
import csv
import math
from pathlib import Path

import numpy as np


def body_axis(points):
    """Unsigned major axis of foreground pixels: eigenvector sign is arbitrary."""
    points = np.asarray(points, float)
    if len(points) < 5:
        return np.array([np.nan, np.nan]), 0.
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
    ratio = math.sqrt(max(eigenvalues[1], 0.) / max(eigenvalues[0], 1e-9))
    return eigenvectors[:, 1], ratio


def read_anchors(path):
    anchors = {}
    if path is None:
        return anchors
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"frame", "slot", "head_x_px", "head_y_px"} <= set(reader.fieldnames or []):
            raise ValueError("Head CSV needs frame,slot,head_x_px,head_y_px")
        for row in reader:
            key = int(row["frame"]), int(row["slot"])
            head = np.array([float(row["head_x_px"]), float(row["head_y_px"])])
            if key in anchors or key[0] < 0 or key[1] < 1 or not np.all(np.isfinite(head)):
                raise ValueError(f"Invalid/duplicate head anchor: {key}")
            anchors[key] = head
    return anchors


def remap_anchors(anchors, original_tracks, new_tracks, tolerance_px=2):
    """Transfer only unique same-frame, same-position detections; no ID guesses."""
    old = {(int(r["frame"]),int(r["slot"])):r for r in original_tracks}
    new_by_frame = {}
    for row in new_tracks:
        if row["status"] == "observed":
            new_by_frame.setdefault(int(row["frame"]), []).append(row)
    mapped, audit = {}, []
    for key, head in anchors.items():
        if key not in old or old[key]["status"] != "observed":
            raise ValueError(f"Head anchor does not match original tracks: {key}")
        center = np.array([float(old[key]["x_px"]),float(old[key]["y_px"])])
        candidates = []
        for row in new_by_frame.get(key[0],[]):
            error = float(np.linalg.norm(center-[float(row["x_px"]),float(row["y_px"])]))
            if error <= tolerance_px:
                candidates.append((row,error))
        record = dict(frame=key[0],old_slot=key[1],new_slot="",center_error_px="",status="not_transferred")
        if len(candidates) == 1:
            row,error = candidates[0]
            new_key = key[0],int(row["slot"])
            if new_key in mapped:
                raise ValueError(f"Multiple head anchors map to same detection: {new_key}")
            mapped[new_key] = head
            record.update(new_slot=new_key[1],center_error_px=error,status="transferred")
        audit.append(record)
    return mapped,audit


def resolve_headings(rows, anchors, config):
    """Resolve 180-degree polarity using a clicked head then axis continuity.

    Reset at gaps, segment changes, unreliable axes and abrupt rotations.
    No forward-walking assumption or motion-based sign selection is allowed.
    """
    headings = np.full((len(rows), 2), np.nan)
    sources = np.full(len(rows), "head_unconfirmed", dtype=object)
    previous, previous_segment = None, None
    automatic = not bool(anchors) and any(r.get("head_probability_axis_positive","") != "" for r in rows)
    for k, row in enumerate(rows):
        key = int(row["frame"]), int(row["slot"])
        segment = int(row["segment"])
        if row["status"] != "observed":
            sources[k] = "tracking_missing"
            previous, previous_segment = None, None
            continue
        if segment != previous_segment:
            previous = None
        try:
            axis = np.array([float(row.get("axis_dx", "")), float(row.get("axis_dy", ""))])
            ratio = float(row.get("axis_ratio", ""))
        except (ValueError, TypeError):
            axis, ratio = np.full(2, np.nan), 0.
        if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < .5 or ratio < config["direction_min_axis_ratio"]:
            sources[k] = "body_axis_unreliable"
            previous, previous_segment = None, segment
            continue
        axis /= np.linalg.norm(axis)
        if key in anchors:
            center = np.array([float(row["x_px"]), float(row["y_px"])])
            delta = anchors[key] - center
            length = np.linalg.norm(delta)
            if length < 2 or abs(float(np.dot(delta/length, axis))) < math.cos(math.radians(45)):
                raise ValueError(f"Head anchor not near a body-axis endpoint: {key}")
            axis *= 1 if np.dot(delta, axis) > 0 else -1
            sources[k] = "manual_anchor"
        elif previous is not None:
            alignment = float(np.dot(axis, previous))
            if abs(alignment) < math.cos(math.radians(config["direction_max_axis_step_deg"])):
                previous = None
                if not automatic:
                    sources[k] = "heading_discontinuity"
                    previous_segment = segment
                    continue
            if previous is not None:
                axis *= 1 if alignment >= 0 else -1
                sources[k] = "auto_axis_propagated" if automatic else "axis_propagated"
        else:
            if not automatic:
                previous_segment = segment
                continue
        if automatic:
            try:
                probability=float(row.get("head_probability_axis_positive",""))
            except (ValueError,TypeError):
                probability=np.nan
            confidence=max(probability,1-probability) if np.isfinite(probability) else 0.
            if confidence>=config["automatic_head_min_probability"]:
                # Probability refers to the original unsigned axis, not the
                # continuity-flipped copy. This prevents sign convention errors.
                raw_axis=np.array([float(row["axis_dx"]),float(row["axis_dy"])])
                axis=raw_axis/np.linalg.norm(raw_axis)*(1 if probability>=.5 else -1)
                sources[k]="automatic_appearance"
            elif previous is None:
                sources[k]="automatic_head_uncertain"
                previous_segment=segment
                continue
        headings[k] = axis
        previous, previous_segment = axis, segment
    return headings, sources


def classify_direction(cosine, state, minimum_cosine):
    if state == "stopped":
        return "stopped"
    if state != "moving" or not np.isfinite(cosine):
        return "unknown"
    if cosine >= minimum_cosine:
        return "forward"
    if cosine <= -minimum_cosine:
        return "backward"
    return "lateral"


def analyze_direction(rows, xy, times, valid, state, config, scale, anchors, window_speed, classify_motion=None):
    headings, sources = resolve_headings(rows, anchors, config)
    signed_distance = np.full(len(rows), np.nan)
    supported_distance = np.full(len(rows), np.nan)
    for k in range(1, len(rows)):
        if not valid[k] or not np.all(np.isfinite(headings[k-1:k+1])):
            continue
        h = headings[k-1] + headings[k]
        if np.linalg.norm(h) < 1e-8:
            continue
        h /= np.linalg.norm(h)
        delta = (xy[k]-xy[k-1]) * scale
        signed_distance[k] = np.dot(delta, h)
        supported_distance[k] = np.linalg.norm(delta)
    longitudinal_speed = window_speed(signed_distance, times, config["speed_window_s"])
    path_speed = window_speed(supported_distance, times, config["speed_window_s"])
    if classify_motion is not None:
        center_distance = np.r_[np.nan, np.linalg.norm(np.diff(xy, axis=0), axis=1)] * scale
        center_distance[~valid] = np.nan
        center_speed = window_speed(center_distance, times, config["speed_window_s"])
        motion_config = dict(config,
                             move_start_speed=config["direction_move_start_speed_px_s"] * scale,
                             stop_start_speed=config["direction_stop_start_speed_px_s"] * scale,
                             confirmation_s=config["direction_confirmation_s"])
        state = classify_motion(center_speed, np.r_[0., np.diff(times)], motion_config)
    cosine = np.divide(longitudinal_speed, path_speed, out=np.full(len(rows), np.nan),
                       where=path_speed > config["direction_min_speed_px_s"] * scale)
    cosine = np.clip(cosine, -1, 1)
    labels = np.array([classify_direction(c, s, config["direction_min_cosine"])
                       for c, s in zip(cosine, state)], dtype=object)
    reasons = sources.copy()
    for k, label in enumerate(labels):
        if label == "stopped":
            reasons[k] = "center_stopped"
        elif state[k] == "unknown":
            reasons[k] = "tracking_or_activity_unknown"
        elif label != "unknown":
            reasons[k] = "body_relative_projection"
        elif np.all(np.isfinite(headings[k])):
            reasons[k] = "insufficient_heading_supported_motion"
    return headings, sources, longitudinal_speed, cosine, labels, reasons, state


def export_direction_preview(video, out, config, progress=None):
    """Visual diagnosis without changing head labels or motion classification."""
    import cv2
    out = Path(out)
    with (out/"activity.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    lookup = {(int(r["frame"]), int(r["slot"])): r for r in rows}
    slots=sorted({int(r['slot']) for r in rows})
    frames = max(int(r["frame"]) for r in rows)+1
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise ValueError(f"Cannot open direction preview source: {video}")
    width, height = int(cap.get(3)), int(cap.get(4))
    writer = cv2.VideoWriter(str(out/"direction_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width,height))
    if not writer.isOpened():
        raise ValueError("Cannot create direction preview")
    colors = {"unknown":(160,160,160), "stopped":(190,110,50), "forward":(80,210,60),
              "backward":(60,60,240), "lateral":(200,100,170)}
    lag = max(1, round(config["speed_window_s"]*fps))
    try:
        for frame in range(frames):
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Direction video ends before CSV frame {frame}")
            for slot in slots:
                row = lookup[frame,slot]
                if not row["x_smooth_px"]:
                    continue
                center = np.array([float(row["x_smooth_px"]),float(row["y_smooth_px"])])
                start = tuple(center.astype(int))
                if row["head_dx"]:
                    h = np.array([float(row["head_dx"]),float(row["head_dy"])])
                    cv2.arrowedLine(image, start, tuple((center+25*h).astype(int)), (0,255,0), 2)
                past_frame = max(0,frame-lag)
                past = lookup[past_frame,slot]
                if past["x_smooth_px"] and all(lookup[k,slot]["valid_interval"] == "1"
                                               for k in range(past_frame+1,frame+1)):
                    motion = center-np.array([float(past["x_smooth_px"]),float(past["y_smooth_px"])])
                    length = np.linalg.norm(motion)
                    if length > .5:
                        cv2.arrowedLine(image,start,tuple((center+25*motion/length).astype(int)),(255,255,0),2)
                label = row["locomotion_direction"]
                cv2.putText(image,f'{slot}:{row["segment"]} {label}', (start[0]+12,start[1]+18),
                            cv2.FONT_HERSHEY_SIMPLEX,.45,colors[label],1)
            cv2.putText(image, f'{frame/fps:.2f}s  GREEN: head  CYAN: displacement',
                        (20,35),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2)
            writer.write(image)
            if progress and (frame == 0 or (frame+1) % max(1,frames//200) == 0):
                progress("preview",frame+1,frames,f"{frame+1:,} / {frames:,} フレーム")
    finally:
        cap.release()
        writer.release()


def plot_direction(times, labels, out, automatic=False,slot_ids=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    names = ["unknown", "stopped", "forward", "backward", "lateral"]
    colors = ["#aaaaaa", "#3878b6", "#2ca25f", "#d73027", "#9566b3"]
    values = np.array([[names.index(x) for x in row] for row in labels])
    fig, ax = plt.subplots(figsize=(13, 4), constrained_layout=True)
    mesh = ax.pcolormesh(times, np.arange(len(labels)+1)+.5, values[:, 1:],
                         cmap=ListedColormap(colors), vmin=-.5, vmax=4.5, shading="flat")
    if slot_ids is not None:ax.set_yticks(range(1,len(labels)+1),labels=slot_ids)
    bar = fig.colorbar(mesh, ax=ax, ticks=range(5))
    bar.ax.set_yticklabels(names)
    ax.set(title="Automatic body-relative direction (appearance model)" if automatic else "Body-relative direction (manual head anchors)",
           xlabel="Time (s)", ylabel="Slot", yticks=range(1, len(labels)+1))
    fig.savefig(Path(out)/"direction_report.png", dpi=150)
    plt.close(fig)
