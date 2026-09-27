import argparse
import shutil
import time
from datetime import date, datetime
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

import ingest
from hazards import HazardDetector
from motion import MotionTracker
from registry import FlagRegistry

VEHICLE_CLASSES = [2, 3, 5, 7]
VIDEO_EXTENSIONS = {".ts", ".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test the vehicle tracking model on video files or a webcam."
    )
    parser.add_argument(
        "sources",
        nargs="*",
        default=["videos"],
        help="Video files, folders of videos, or a webcam index (e.g. 0).",
    )
    parser.add_argument("--model", default="yolo26s.pt")
    # ByteTrack ignores detections below its track_low_thresh (0.1) anyway
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--camera", default="CAM01", help="Camera ID stored with each flag.")
    parser.add_argument("--location", default="unknown", help="Where the camera is, e.g. 'Main St & 5th'.")
    parser.add_argument("--db", default="data/incidents.db", help="Incident database to update.")
    parser.add_argument("--snapshots", default="data/snapshots", help="Where the database keeps snapshots.")
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Delete the database and its snapshots before running, so it only holds this run.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Only write results/flagged_vehicles.csv; don't update the database.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write annotated videos to the results/ folder.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't open a preview window.",
    )
    return parser.parse_args()


def collect_sources(sources):
    collected = []

    for source in sources:
        # Webcam index
        if source.isdigit():
            collected.append(int(source))
            continue

        path = Path(source)

        if path.is_dir():
            collected.extend(
                str(p) for p in sorted(path.iterdir())
                if p.suffix.lower() in VIDEO_EXTENSIONS
            )
        elif path.is_file():
            collected.append(str(path))
        else:
            print(f"Skipping missing source: {source}")

    return collected


def extract_tracks(result):
    if result.boxes is None or result.boxes.id is None:
        return [], [], []
    boxes = result.boxes.xyxy.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist()
    confidences = result.boxes.conf.cpu().numpy()
    return boxes, track_ids, confidences


def format_metrics(m):
    if m is None:
        return "warming up"
    turn = f"{m.turn_rate:+.0f}d/s" if m.turn_rate is not None else "--"
    return f"spd {m.speed:.2f} x{m.speed_ratio:.1f} turn {turn}"


def draw_detections(frame, boxes, track_ids, confidences, metrics, hazards, registry):
    for box, track_id, confidence in zip(boxes, track_ids, confidences):
        x1, y1, x2, y2 = map(int, box)

        reasons = hazards.reasons(track_id)
        color = (0, 0, 255) if reasons else (0, 255, 0)
        label = f"Vehicle {track_id} {confidence:.2f}"
        if reasons:
            label = f"{registry.plate(track_id)} {' + '.join(reasons)}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )
        cv2.putText(
            frame,
            format_metrics(metrics.get(track_id)),
            (x1, y2 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        frame,
        f"Tracked Vehicles: {len(track_ids)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )


def print_flag_summary(hazards, registry):
    by_reason = {}
    for track_id, flags in hazards.flags.items():
        for reason in flags:
            by_reason.setdefault(reason, []).append(track_id)

    print(f"\nFlagged vehicles:      {len(hazards.flags)}")
    for reason, ids in sorted(by_reason.items()):
        plates = ", ".join(registry.plate(i) for i in sorted(ids))
        print(f"  {reason:<12} {len(ids):3d}  {plates}")


def save_to_db(conn, registry, args):
    """Replace this video's incidents in the database with the latest run."""
    source = {
        "camera_id": registry.camera_id,
        "location": registry.location,
        "video": registry.video,
        "video_date": registry.video_date,
    }
    rows = [r for r in registry.rows if r["video"] == registry.video]
    with conn:
        ingest.ingest_video(conn, source, rows, Path(args.snapshots))
        ingest.refresh_vehicles(conn)
    videos, vehicles, incidents = ingest.summary(conn)
    print(f"Database {args.db}: {videos} videos, {vehicles} vehicles, {incidents} incidents")


def run_on_source(model, source, device, args, registry, conn):
    """Run tracking on one video. Returns False if the user pressed q."""
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Could not open video: {source}")
        return True

    print(f"\n=== Testing on: {source} ===")

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    name = f"webcam_{source}" if isinstance(source, int) else Path(source).stem
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    motion = MotionTracker(video_fps)
    hazards = HazardDetector(video_fps)
    if isinstance(source, int):
        video_date = date.today().isoformat()
    else:
        # The file's modification time is the best available guess at when it was recorded
        video_date = datetime.fromtimestamp(Path(source).stat().st_mtime).date().isoformat()
    registry.begin_video(name, args.camera, args.location, video_date)

    writer = None
    if args.save:
        out_path = out_dir / f"{name}_tracked.mp4"
        fps = video_fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        print(f"Saving to: {out_path}")

    # Reset tracker state so IDs don't carry over between videos
    if model.predictor is not None:
        model.predictor.trackers = None

    frames = 0
    total_detections = 0
    max_in_frame = 0
    unique_ids = set()
    start = time.time()
    keep_going = True

    while True:
        success, frame = cap.read()

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="vehicle_tracker.yaml",
            conf=args.conf,
            imgsz=args.imgsz,
            max_det=500,
            classes=VEHICLE_CLASSES,
            device=device,
            verbose=False,
        )

        boxes, track_ids, confidences = extract_tracks(results[0])
        metrics = motion.update(frames, track_ids, boxes)
        boxes_by_id = dict(zip(track_ids, boxes))
        # Runs before drawing, so snapshots are taken from the clean frame
        for track_id, flag in hazards.update(frames, metrics):
            plate = registry.record(
                frame, frames, video_fps, track_id, flag,
                boxes_by_id[track_id], metrics[track_id],
            )
            print(
                f"[{frames / video_fps:7.2f}s] {plate} flagged: "
                f"{flag.reason} ({flag.value:+.2f})"
            )
        draw_detections(frame, boxes, track_ids, confidences, metrics, hazards, registry)
        vehicle_count = len(track_ids)

        frames += 1
        total_detections += vehicle_count
        max_in_frame = max(max_in_frame, vehicle_count)
        unique_ids.update(track_ids)

        elapsed = time.time() - start
        fps = frames / elapsed if elapsed > 0 else 0
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if writer is not None:
            writer.write(frame)

        if not args.no_show:
            cv2.imshow("Vehicle Tracking Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                keep_going = False
                break

    elapsed = time.time() - start
    cap.release()
    if writer is not None:
        writer.release()

    print(f"Frames processed:      {frames}")
    print(f"Average FPS:           {frames / elapsed if elapsed > 0 else 0:.1f}")
    print(f"Unique vehicles seen:  {len(unique_ids)}")
    print(f"Max vehicles in frame: {max_in_frame}")
    print(f"Avg vehicles / frame:  {total_detections / frames if frames else 0:.2f}")

    print_flag_summary(hazards, registry)
    print(f"Flag dataset:          {registry.csv_path}")
    if conn is not None:
        save_to_db(conn, registry, args)

    return keep_going


def main():
    args = parse_args()

    sources = collect_sources(args.sources)
    if not sources:
        print("No valid video sources found.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)

    model = YOLO(args.model)
    registry = FlagRegistry()

    conn = None
    if not args.no_db:
        if args.reset_db:
            Path(args.db).unlink(missing_ok=True)
            shutil.rmtree(args.snapshots, ignore_errors=True)
            print(f"Cleared {args.db} and {args.snapshots}")
        conn = ingest.connect(args.db)

    try:
        for source in sources:
            if not run_on_source(model, source, device, args, registry, conn):
                break
    finally:
        if conn is not None:
            conn.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
