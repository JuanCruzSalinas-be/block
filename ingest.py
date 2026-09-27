"""Load flagged_vehicles.csv into the incident database.

Usage:
    python3 ingest.py                      # loads results/flagged_vehicles.csv
    python3 ingest.py path/to/flags.csv --db data/incidents.db

Each video in the CSV replaces that video's earlier incidents, so re-running a
video with new thresholds and ingesting again never creates duplicates.
Snapshots are copied into data/snapshots/<camera>/<video>/ so the database doesn't depend on
results/, which test.py overwrites.
"""
import argparse
import csv
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id   TEXT PRIMARY KEY,   -- camera_id/video
    camera_id   TEXT NOT NULL,
    location    TEXT,
    video       TEXT NOT NULL,
    video_date  TEXT,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vehicles (
    plate_id    TEXT PRIMARY KEY,
    first_seen  TEXT,
    last_seen   TEXT
);

CREATE TABLE IF NOT EXISTS incidents (
    incident_id INTEGER PRIMARY KEY,
    plate_id    TEXT NOT NULL REFERENCES vehicles(plate_id),
    source_id   TEXT NOT NULL REFERENCES sources(source_id),
    reason      TEXT NOT NULL,
    value       REAL,               -- the measurement that triggered the rule
    speed       REAL,
    speed_ratio REAL,
    turn_rate   REAL,
    frame       INTEGER,
    time_s      REAL,               -- seconds into the video
    box         TEXT,
    snapshot    TEXT,
    UNIQUE (source_id, plate_id, reason)
);

CREATE INDEX IF NOT EXISTS incidents_plate ON incidents(plate_id);
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Load flagged vehicles into the incident database.")
    parser.add_argument("csv", nargs="?", default="results/flagged_vehicles.csv")
    parser.add_argument("--db", default="data/incidents.db")
    parser.add_argument("--snapshots", default="data/snapshots")
    return parser.parse_args()


def number(value, cast=float):
    return None if value in ("", None) else cast(value)


def connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def copy_snapshot(row, source_id, snapshot_dir):
    src = Path(row["snapshot"])
    if not src.exists():
        print(f"  Missing snapshot for {row['plate_id']}: {src}")
        return None
    reason_slug = row["reason"].lower().replace(" ", "_")
    dest = snapshot_dir / source_id / f"{row['plate_id']}_{reason_slug}.jpg"
    shutil.copy2(src, dest)
    return str(dest)


def ingest(conn, rows, snapshot_dir):
    now = datetime.now().isoformat(timespec="seconds")

    by_source = {}
    for row in rows:
        by_source.setdefault(f"{row['camera_id']}/{row['video']}", []).append(row)

    with conn:  # one transaction: all or nothing
        for source_id, source_rows in by_source.items():
            first = source_rows[0]
            conn.execute("DELETE FROM incidents WHERE source_id = ?", (source_id,))
            shutil.rmtree(snapshot_dir / source_id, ignore_errors=True)
            (snapshot_dir / source_id).mkdir(parents=True)

            conn.execute(
                """INSERT INTO sources (source_id, camera_id, location, video, video_date, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(source_id) DO UPDATE SET
                       location = excluded.location,
                       video_date = excluded.video_date,
                       ingested_at = excluded.ingested_at""",
                (source_id, first["camera_id"], first["location"], first["video"],
                 first["video_date"], now),
            )

            for row in source_rows:
                conn.execute(
                    "INSERT OR IGNORE INTO vehicles (plate_id) VALUES (?)", (row["plate_id"],)
                )
                conn.execute(
                    """INSERT INTO incidents (plate_id, source_id, reason, value, speed,
                           speed_ratio, turn_rate, frame, time_s, box, snapshot)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (row["plate_id"], source_id, row["reason"], number(row["value"]),
                     number(row["speed"]), number(row["speed_ratio"]), number(row["turn_rate"]),
                     number(row["frame"], int), number(row["time_s"]), row["box"],
                     copy_snapshot(row, source_id, snapshot_dir)),
                )
            print(f"  {source_id}: {len(source_rows)} incidents")

        # Keep the vehicles table in step with the incidents that remain
        conn.execute("DELETE FROM vehicles WHERE plate_id NOT IN (SELECT plate_id FROM incidents)")
        conn.execute(
            """UPDATE vehicles SET
                   first_seen = (SELECT MIN(s.video_date) FROM incidents i
                                 JOIN sources s USING (source_id) WHERE i.plate_id = vehicles.plate_id),
                   last_seen  = (SELECT MAX(s.video_date) FROM incidents i
                                 JOIN sources s USING (source_id) WHERE i.plate_id = vehicles.plate_id)"""
        )

    return len(by_source)


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"No CSV found at {csv_path}. Run test.py first.")
        return

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{csv_path} has no flagged vehicles to load.")
        return
    if "camera_id" not in rows[0]:
        print(f"{csv_path} was made by an older version of test.py. Re-run test.py first.")
        return

    conn = connect(args.db)
    print(f"Loading {len(rows)} rows from {csv_path}")
    sources = ingest(conn, rows, Path(args.snapshots))

    totals = conn.execute(
        "SELECT (SELECT COUNT(*) FROM sources), (SELECT COUNT(*) FROM vehicles), "
        "(SELECT COUNT(*) FROM incidents)"
    ).fetchone()
    conn.close()

    print(f"Updated {sources} video(s). Database {args.db} now holds "
          f"{totals[0]} videos, {totals[1]} vehicles, {totals[2]} incidents.")


if __name__ == "__main__":
    main()
