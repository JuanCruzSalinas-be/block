import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

CONFIG = {
    # Frames of history kept per track
    "history_len": 30,
    # Moving-average window used to smooth box centers (removes YOLO jitter)
    "smooth_window": 5,
    # Frames between the two points used to measure velocity
    "velocity_step": 5,
    # Below this speed (box heights / second) heading and turn rate are ignored
    "min_moving_speed": 0.3,
    # Drop a track after this many frames without seeing it
    "max_missing_frames": 30,
}


@dataclass
class Metrics:
    speed: float            # box heights per second (perspective-corrected)
    speed_ratio: float      # speed / median speed of moving cars this frame
    heading: Optional[float] # degrees, 0 = right, 90 = down (image coords)
    turn_rate: Optional[float] # degrees per second, signed
    age: int                # frames this track has been seen
    height: float           # smoothed box height in pixels
    accel: Optional[float]  # box heights per second^2, negative = slowing down
    cx: float               # smoothed box center
    cy: float
    box_change: float       # relative raw box height change over one velocity step


def _wrap_angle(degrees):
    return (degrees + 180) % 360 - 180


class MotionTracker:
    """Keeps a short position history per track ID and derives motion metrics."""

    def __init__(self, fps, config=None):
        self.fps = fps
        self.cfg = {**CONFIG, **(config or {})}
        self.history = {}    # track_id -> deque of (frame_idx, cx, cy, height)
        self.last_seen = {}  # track_id -> frame_idx
        self.age = {}        # track_id -> frames seen

    @property
    def min_frames(self):
        # Need two velocity segments of smoothed points to measure turn rate
        return self.cfg["smooth_window"] + 2 * self.cfg["velocity_step"]

    def update(self, frame_idx, track_ids, boxes):
        """Add this frame's boxes and return {track_id: Metrics} for ready tracks."""
        for track_id, (x1, y1, x2, y2) in zip(track_ids, boxes):
            if track_id not in self.history:
                self.history[track_id] = deque(maxlen=self.cfg["history_len"])
                self.age[track_id] = 0
            elif frame_idx - self.last_seen[track_id] > 1:
                # Smoothing assumes consecutive frames; averaging across a detection
                # gap fakes a speed spike, so start the history over
                self.history[track_id].clear()
            self.history[track_id].append(
                (frame_idx, (x1 + x2) / 2, (y1 + y2) / 2, y2 - y1)
            )
            self.last_seen[track_id] = frame_idx
            self.age[track_id] += 1

        self._drop_stale(frame_idx)

        raw = {}
        for track_id in track_ids:
            metrics = self._compute(track_id)
            if metrics is not None:
                raw[track_id] = metrics

        # Compare each car to the median of the cars that are actually moving
        moving = [m[0] for m in raw.values() if m[0] >= self.cfg["min_moving_speed"]]
        median = float(np.median(moving)) if moving else 0.0

        return {
            track_id: Metrics(
                speed=speed,
                speed_ratio=speed / median if median > 0 else 0.0,
                heading=heading,
                turn_rate=turn_rate,
                age=self.age[track_id],
                height=height,
                accel=accel,
                cx=cx,
                cy=cy,
                box_change=box_change,
            )
            for track_id, (speed, heading, turn_rate, height, accel, cx, cy, box_change) in raw.items()
        }

    def _drop_stale(self, frame_idx):
        stale = [
            track_id for track_id, seen in self.last_seen.items()
            if frame_idx - seen > self.cfg["max_missing_frames"]
        ]
        for track_id in stale:
            del self.history[track_id]
            del self.last_seen[track_id]
            del self.age[track_id]

    def _smoothed(self, points):
        window = self.cfg["smooth_window"]
        arr = np.array(points, dtype=float)
        kernel = np.ones(window) / window
        # Columns: frame_idx, cx, cy, height -> smooth everything except frame_idx
        smoothed = [arr[window - 1:, 0]]
        for col in range(1, 4):
            smoothed.append(np.convolve(arr[:, col], kernel, mode="valid"))
        return np.stack(smoothed, axis=1)

    def _velocity(self, a, b):
        """Speed (box heights/s) and heading (deg) between two smoothed points."""
        dt = (b[0] - a[0]) / self.fps
        if dt <= 0:
            return 0.0, None
        dx, dy = b[1] - a[1], b[2] - a[2]
        height = max((a[3] + b[3]) / 2, 1.0)
        speed = math.hypot(dx, dy) / height / dt
        return speed, math.degrees(math.atan2(dy, dx))

    def _compute(self, track_id):
        points = self.history[track_id]
        if len(points) < self.min_frames:
            return None

        s = self._smoothed(points)
        step = self.cfg["velocity_step"]
        now, mid, old = s[-1], s[-1 - step], s[-1 - 2 * step]

        speed, heading = self._velocity(mid, now)
        prev_speed, prev_heading = self._velocity(old, mid)

        min_speed = self.cfg["min_moving_speed"]
        if speed < min_speed:
            heading = None

        dt = (now[0] - mid[0]) / self.fps
        accel = (speed - prev_speed) / dt if dt > 0 else None

        turn_rate = None
        if heading is not None and prev_speed >= min_speed and prev_heading is not None:
            turn_rate = _wrap_angle(heading - prev_heading) / dt

        # Sudden box resizing (occlusion, trailer, merged detections) fakes motion
        box_change = abs(points[-1][3] / max(points[-1 - step][3], 1.0) - 1)

        return speed, heading, turn_rate, now[3], accel, now[1], now[2], box_change
