"""Build a per-vehicle driver risk report from the incident database.

Usage:
    python3 report.py                  # writes data/reports/driver_risk_report_<date>.csv
    python3 report.py --top 20         # also changes how many rows are printed

The risk score is a weighted count of incidents. Weights and tier cut-offs are
at the top of this file.
"""
import argparse
import csv
import sqlite3
from datetime import date
from pathlib import Path

from hazards import (
    HARD_BRAKING, SHARP_TURN, SPEEDING, STOPPED, TAILGATING, WEAVING, WRONG_WAY,
)

# How much each hazard adds to a vehicle's risk score
WEIGHTS = {
    WRONG_WAY: 10,
    TAILGATING: 3,
    WEAVING: 3,
    SPEEDING: 3,
    SHARP_TURN: 2,
    HARD_BRAKING: 2,   # can be defensive driving, so weighted lower
    STOPPED: 1,        # often a breakdown rather than bad driving
}

# Minimum score for each tier, highest first
TIERS = [(10, "HIGH"), (5, "MEDIUM"), (0, "LOW")]


def parse_args():
    parser = argparse.ArgumentParser(description="Per-vehicle driver risk report.")
    parser.add_argument("--db", default="data/incidents.db")
    parser.add_argument("--out", default=None, help="Output CSV path.")
    parser.add_argument("--top", type=int, default=10, help="Rows to print.")
    return parser.parse_args()


def tier(score):
    return next(name for minimum, name in TIERS if score >= minimum)


def build_report(conn):
    rows = conn.execute(
        """SELECT i.plate_id, i.reason, i.snapshot, s.camera_id, v.first_seen, v.last_seen
           FROM incidents i
           JOIN sources s USING (source_id)
           JOIN vehicles v USING (plate_id)"""
    ).fetchall()

    vehicles = {}
    for plate, reason, snapshot, camera, first_seen, last_seen in rows:
        v = vehicles.setdefault(plate, {
            "plate_id": plate,
            "counts": {reason: 0 for reason in WEIGHTS},
            "cameras": set(),
            "first_seen": first_seen,
            "last_seen": last_seen,
            "worst": (-1, None, None),  # (weight, reason, snapshot)
        })
        v["counts"][reason] = v["counts"].get(reason, 0) + 1
        v["cameras"].add(camera)
        weight = WEIGHTS.get(reason, 1)
        if weight > v["worst"][0]:
            v["worst"] = (weight, reason, snapshot)

    report = []
    for v in vehicles.values():
        score = sum(WEIGHTS.get(r, 1) * n for r, n in v["counts"].items())
        row = {
            "plate_id": v["plate_id"],
            "risk_score": score,
            "risk_tier": tier(score),
            "incidents": sum(v["counts"].values()),
        }
        row.update({r.lower().replace(" ", "_"): n for r, n in v["counts"].items()})
        row.update({
            "cameras": " ".join(sorted(v["cameras"])),
            "first_seen": v["first_seen"],
            "last_seen": v["last_seen"],
            "worst_incident": v["worst"][1],
            "evidence_snapshot": v["worst"][2] or "",
        })
        report.append(row)

    report.sort(key=lambda r: (-r["risk_score"], r["plate_id"]))
    return report


def main():
    args = parse_args()
    if not Path(args.db).exists():
        print(f"No database at {args.db}. Run ingest.py first.")
        return

    conn = sqlite3.connect(args.db)
    report = build_report(conn)
    conn.close()

    if not report:
        print("The database has no incidents yet.")
        return

    out = Path(args.out or f"data/reports/driver_risk_report_{date.today().isoformat()}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)

    tiers = {name: sum(1 for r in report if r["risk_tier"] == name) for _, name in TIERS}
    print(f"{len(report)} vehicles: " + ", ".join(f"{n} {name}" for name, n in tiers.items()))
    print(f"\n{'plate':<9}{'score':>6}  {'tier':<7}{'incidents':>9}  worst incident")
    for r in report[:args.top]:
        print(f"{r['plate_id']:<9}{r['risk_score']:>6}  {r['risk_tier']:<7}"
              f"{r['incidents']:>9}  {r['worst_incident']}")
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
