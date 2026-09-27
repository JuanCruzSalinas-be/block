import csv
import hashlib
import shutil
import string
from datetime import datetime
from pathlib import Path

import cv2

FIELDS = [
    "plate_id",
    "track_id",
    "reason",
    "camera_id",
    "location",
    "video",
    "video_date",
    "frame",
    "time_s",
    "value",
    "speed",
    "speed_ratio",
    "turn_rate",
    "box",
    "snapshot",
    "recorded_at",
]


def plate_id(camera_id, video, track_id):
    """Simulated license plate, e.g. "KRT4821".

    Track IDs restart in every video, so the plate is derived from the camera,
    video and track ID together. The same car in the same video always gets the
    same plate, which keeps re-runs and database imports consistent.
    """
    digest = hashlib.sha1(f"{camera_id}|{video}|{track_id}".encode()).digest()
    letters = "".join(string.ascii_uppercase[b % 26] for b in digest[:3])
    digits = int.from_bytes(digest[3:7], "big") % 10_000
    return f"{letters}{digits:04d}"


class FlagRegistry:
    """Dataset of flagged vehicles, one row per video + plate + reason.

    Re-running a video replaces that video's rows, so the CSV always reflects
    the latest thresholds instead of piling up duplicates.
    """

    def __init__(self, csv_path="results/flagged_vehicles.csv", snapshot_dir="results/snapshots"):
        self.csv_path = Path(csv_path)
        self.snapshot_dir = Path(snapshot_dir)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.rows = []
        if self.csv_path.exists():
            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                # Rows from before the camera/date columns existed can't be
                # matched to a plate anymore, so start the file fresh
                if set(FIELDS) <= set(reader.fieldnames or []):
                    self.rows = list(reader)

        self.camera_id = self.location = self.video = self.video_date = None

    def begin_video(self, video, camera_id, location, video_date):
        """Clear earlier records and snapshots for this video."""
        self.video, self.camera_id = video, camera_id
        self.location, self.video_date = location, video_date

        self.rows = [r for r in self.rows if r["video"] != video]
        shutil.rmtree(self._video_dir(), ignore_errors=True)
        self._video_dir().mkdir(parents=True)
        self._write()

    def plate(self, track_id):
        return plate_id(self.camera_id, self.video, track_id)

    def record(self, frame, frame_idx, fps, track_id, flag, box, metrics):
        plate = self.plate(track_id)
        reason_slug = flag.reason.lower().replace(" ", "_")
        snapshot = self._video_dir() / f"{plate}_{reason_slug}.jpg"
        cv2.imwrite(str(snapshot), _crop(frame, box))

        self.rows.append({
            "plate_id": plate,
            "track_id": track_id,
            "reason": flag.reason,
            "camera_id": self.camera_id,
            "location": self.location,
            "video": self.video,
            "video_date": self.video_date,
            "frame": frame_idx,
            "time_s": round(frame_idx / fps, 2),
            "value": round(flag.value, 3),
            "speed": round(metrics.speed, 3),
            "speed_ratio": round(metrics.speed_ratio, 3),
            "turn_rate": "" if metrics.turn_rate is None else round(metrics.turn_rate, 2),
            "box": " ".join(str(int(v)) for v in box),
            "snapshot": str(snapshot),
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        })
        self._write()
        return plate

    def _video_dir(self):
        return self.snapshot_dir / self.video

    def _write(self):
        # Rewrite the whole file so the CSV is complete even if the run is stopped
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)


def _crop(frame, box, pad=0.15):
    x1, y1, x2, y2 = box
    pad_x, pad_y = (x2 - x1) * pad, (y2 - y1) * pad
    h, w = frame.shape[:2]
    x1, y1 = max(int(x1 - pad_x), 0), max(int(y1 - pad_y), 0)
    x2, y2 = min(int(x2 + pad_x), w), min(int(y2 + pad_y), h)
    return frame[y1:y2, x1:x2]
