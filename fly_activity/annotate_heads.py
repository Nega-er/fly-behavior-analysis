#!/usr/bin/env python3
"""Click the anatomical head once per reliable tracking segment."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from analyze_flies import write_csv


def requests(rows, min_ratio=1.5, max_step_deg=40):
    selected, previous = [], {}
    for row in sorted(rows, key=lambda r: (int(r["frame"]), int(r["slot"]))):
        slot, segment = int(row["slot"]), int(row["segment"])
        if row["status"] != "observed":
            previous.pop(slot, None)
            continue
        try:
            if float(row.get("axis_ratio", "")) < min_ratio:
                previous.pop(slot, None)
                continue
            axis = np.array([float(row["axis_dx"]), float(row["axis_dy"])])
            if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) < .5:
                previous.pop(slot, None)
                continue
        except (ValueError, KeyError):
            previous.pop(slot, None)
            continue
        axis /= np.linalg.norm(axis)
        old = previous.get(slot)
        if old is None or old[0] != segment or abs(np.dot(old[1], axis)) < np.cos(np.radians(max_step_deg)):
            selected.append(row)
        previous[slot] = segment, axis
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Head CSV already exists; choose a new output path")
    with args.tracks.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    todo = requests(rows)
    if not todo:
        parser.error("No reliable body axes. Re-run video tracking or adjust axis threshold.")
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        parser.error("Cannot open video")
    anchors, stop = [], False
    window = "Head confirmation: click head | N skip | Q save and quit"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    try:
        for index, row in enumerate(todo):
            frame, slot = int(row["frame"]), int(row["slot"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode annotation frame {frame}")
            x, y = float(row["x_px"]), float(row["y_px"])
            x0, y0 = max(0, int(x)-45), max(0, int(y)-45)
            x1, y1 = min(image.shape[1], int(x)+46), min(image.shape[0], int(y)+46)
            factor = 6
            crop = cv2.resize(image[y0:y1, x0:x1], None, fx=factor, fy=factor)
            canvas = cv2.copyMakeBorder(crop, 70, 0, 0, 0, cv2.BORDER_CONSTANT)
            center = np.array([(x-x0)*factor, (y-y0)*factor+70])
            axis = np.array([float(row["axis_dx"]), float(row["axis_dy"])])
            cv2.line(canvas, tuple((center-18*factor*axis).astype(int)),
                     tuple((center+18*factor*axis).astype(int)), (255,255,0), 1)
            cv2.circle(canvas, tuple(center.astype(int)), 3, (0,255,255), 1)
            cv2.putText(canvas, f'{index+1}/{len(todo)} slot {slot} segment {row["segment"]} frame {frame}',
                        (8,25), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            cv2.putText(canvas, 'Click HEAD. If unclear: N. Save/quit: Q.',
                        (8,52), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            choice = []
            def click(event, mx, my, flags, userdata):
                if event == cv2.EVENT_LBUTTONDOWN and my >= 70:
                    head = np.array([x0+mx/factor, y0+(my-70)/factor])
                    delta = head-np.array([x,y])
                    length = np.linalg.norm(delta)
                    if length >= 2 and abs(np.dot(delta/length, axis)) >= np.cos(np.pi/4):
                        choice[:] = [head]
            cv2.setMouseCallback(window, click)
            cv2.imshow(window, canvas)
            while not choice:
                key = cv2.waitKey(30) & 255
                if key in (ord('n'), ord('N')):
                    break
                if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    stop = True
                    break
            if choice:
                anchors.append(dict(frame=frame, slot=slot, head_x_px=float(choice[0][0]), head_y_px=float(choice[0][1])))
            if stop:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, anchors, ["frame", "slot", "head_x_px", "head_y_px"])
    Path(str(args.out)+".json").write_text(json.dumps(dict(tracks_source=str(args.tracks.resolve()),
                                                          video_source=str(args.video.resolve())), indent=2), encoding="utf-8")
    print(f'Saved {len(anchors)} confirmed head anchors to {args.out}')


if __name__ == "__main__":
    main()
