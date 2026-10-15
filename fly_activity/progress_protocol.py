"""Line-based progress protocol; analysis does not require a GUI runtime."""
import json

PREFIX = "FLY_PROGRESS "


def emit_progress(stage, current=0, total=1, detail=""):
    print(PREFIX + json.dumps(dict(stage=stage, current=current, total=total,
                                  detail=detail), ensure_ascii=False), flush=True)
