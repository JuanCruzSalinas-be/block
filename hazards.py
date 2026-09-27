import math
from collections import defaultdict, deque
from dataclasses import dataclass

from motion import CONFIG as MOTION_CONFIG

CONFIG = {
    # Speed must be this many times the median speed of moving cars
    "speed_ratio_threshold": 2.5,
    # Absolute turn rate in degrees per second
    "turn_rate_threshold": 45.0,
    # A rule must hold for this many consecutive frames before flagging
    "confirm_frames": 5,
    # The median speed isn't meaningful with fewer moving cars than this
    "min_moving_cars": 3,
    # Ignore cars smaller than this (px): far-away boxes jitter too much to judge
    "min_box_height": 30,
    # Skip frames where the box height changed more than this fraction in one
    # velocity step; the apparent motion comes from the detection, not the car
    "max_box_change": 0.2,

    # Hard braking: deceleration in box heights / s^2, from at least this speed
    "brake_decel_threshold": 4.0,
    "brake_confirm_frames": 3,    # braking is brief, so it needs fewer frames
    "brake_min_speed": 1.5,

    # Wrong way: heading differs from the usual traffic direction in that area
    "flow_cell_px": 40,           # grid cell size; small enough that a cell covers one lane
    "flow_decay": 0.995,          # how fast old traffic directions are forgotten
    "flow_min_samples": 30,       # cell needs this much traffic before it's trusted
    "flow_min_consistency": 0.7,  # 0..1, low = mixed directions (two-way), skip
    "wrong_way_angle": 135.0,     # degrees away from the usual direction
    "wrong_way_seconds": 1.5,     # must drive against traffic this long

    # Weaving: repeated left/right swings within a time window
    "weave_window_s": 3.0,
    "weave_min_turn": 15.0,       # deg/s; smaller turns count as driving straight
    "weave_min_swings": 4,        # direction changes needed within the window

    # Tailgating: time gap to the car ahead, in seconds
    "tailgate_headway_s": 0.3,
    "tailgate_min_gap": 0.25,     # car lengths; closer than this is a towed trailer
    "tailgate_min_speed": 1.0,    # both cars must be moving at least this fast
    "tailgate_max_heading_diff": 20.0,
    "tailgate_max_lateral": 0.6,  # max side offset, in box heights (same lane)

    # Stopped in lane: was moving, then stopped while traffic keeps flowing
    "stopped_speed": 0.15,
    "stopped_seconds": 3.0,
}

SPEEDING = "SPEEDING"
SHARP_TURN = "SHARP TURN"
HARD_BRAKING = "HARD BRAKING"
WRONG_WAY = "WRONG WAY"
WEAVING = "WEAVING"
TAILGATING = "TAILGATING"
STOPPED = "STOPPED IN LANE"


@dataclass
class Flag:
    reason: str
    frame: int   # frame the flag was confirmed on
    value: float # the measurement that triggered the rule


def _angle_diff(a, b):
    return abs((a - b + 180) % 360 - 180)


class FlowMap:
    """Learns the usual direction of traffic in each grid cell of the frame."""

    def __init__(self, cell_px, decay):
        self.cell_px = cell_px
        self.decay = decay
        self.cells = {}  # (col, row) -> [sum_x, sum_y, weight, samples]

    def _cell(self, cx, cy):
        return int(cx // self.cell_px), int(cy // self.cell_px)

    def lookup(self, cx, cy):
        """Returns (direction in degrees, consistency 0..1, samples) or None."""
        cell = self.cells.get(self._cell(cx, cy))
        if cell is None or cell[2] == 0:
            return None
        sx, sy, weight, samples = cell
        consistency = math.hypot(sx, sy) / weight
        return math.degrees(math.atan2(sy, sx)), consistency, samples

    def add(self, cx, cy, heading):
        cell = self.cells.setdefault(self._cell(cx, cy), [0.0, 0.0, 0.0, 0])
        rad = math.radians(heading)
        cell[0] = cell[0] * self.decay + math.cos(rad)
        cell[1] = cell[1] * self.decay + math.sin(rad)
        cell[2] = cell[2] * self.decay + 1
        cell[3] += 1


class HazardDetector:
    """Applies hazard rules to motion metrics. Flags stick for the whole video."""

    def __init__(self, fps, config=None):
        self.fps = fps
        self.cfg = {**CONFIG, **(config or {})}
        self.streaks = defaultdict(int)  # (track_id, reason) -> consecutive frames
        self.flags = {}                  # track_id -> {reason: Flag}
        self.signals = {}                # track_id -> latest derived values, for logging

        self.flow = FlowMap(self.cfg["flow_cell_px"], self.cfg["flow_decay"])
        self.turn_history = {}           # track_id -> deque of turn rates
        self.was_moving = set()          # tracks that have driven normally at some point
        self.confirm = {
            STOPPED: int(self.cfg["stopped_seconds"] * fps),
            HARD_BRAKING: self.cfg["brake_confirm_frames"],
            WRONG_WAY: int(self.cfg["wrong_way_seconds"] * fps),
        }

    def update(self, frame_idx, metrics):
        """Evaluate this frame's {track_id: Metrics}. Returns the newly confirmed Flags."""
        cfg = self.cfg
        moving = sum(
            1 for m in metrics.values() if m.speed >= MOTION_CONFIG["min_moving_speed"]
        )
        judged = {
            track_id: m for track_id, m in metrics.items()
            if m.height >= cfg["min_box_height"] and m.box_change <= cfg["max_box_change"]
        }
        headways = self._headways(judged)
        new_flags = []
        self.signals = {}  # only cars judged this frame

        for track_id, m in judged.items():
            flow_dev = self._flow_deviation(m)
            swings = self._swings(track_id, m)
            headway = headways.get(track_id)

            if m.speed >= cfg["brake_min_speed"]:
                self.was_moving.add(track_id)

            self.signals[track_id] = {
                "flow_dev": flow_dev,
                "swings": swings,
                "headway": headway,
            }

            checks = {
                SPEEDING: (
                    moving >= cfg["min_moving_cars"]
                    and m.speed_ratio >= cfg["speed_ratio_threshold"],
                    m.speed_ratio,
                ),
                SHARP_TURN: (
                    m.turn_rate is not None
                    and abs(m.turn_rate) >= cfg["turn_rate_threshold"],
                    m.turn_rate,
                ),
                HARD_BRAKING: (
                    m.accel is not None
                    and m.accel <= -cfg["brake_decel_threshold"]
                    # Speed one velocity step ago must have been real driving speed
                    and m.speed - m.accel * MOTION_CONFIG["velocity_step"] / self.fps
                    >= cfg["brake_min_speed"],
                    m.accel,
                ),
                WRONG_WAY: (
                    flow_dev is not None and flow_dev >= cfg["wrong_way_angle"],
                    flow_dev,
                ),
                WEAVING: (swings >= cfg["weave_min_swings"], swings),
                TAILGATING: (
                    headway is not None and headway <= cfg["tailgate_headway_s"],
                    headway,
                ),
                STOPPED: (
                    track_id in self.was_moving
                    and m.speed <= cfg["stopped_speed"]
                    and moving >= cfg["min_moving_cars"],
                    self.streaks[(track_id, STOPPED)] / self.fps,
                ),
            }

            for reason, (triggered, value) in checks.items():
                key = (track_id, reason)
                self.streaks[key] = self.streaks[key] + 1 if triggered else 0

                needed = self.confirm.get(reason, cfg["confirm_frames"])
                already = reason in self.flags.get(track_id, {})
                if self.streaks[key] >= needed and not already:
                    flag = Flag(reason, frame_idx, value)
                    self.flags.setdefault(track_id, {})[reason] = flag
                    new_flags.append((track_id, flag))

            # Learn traffic direction after judging, so a car isn't compared to itself
            if m.heading is not None:
                self.flow.add(m.cx, m.cy, m.heading)

        return new_flags

    def reasons(self, track_id):
        return list(self.flags.get(track_id, {}))

    def _flow_deviation(self, m):
        """Degrees between this car's heading and the usual direction where it is."""
        if m.heading is None:
            return None
        found = self.flow.lookup(m.cx, m.cy)
        if found is None:
            return None
        direction, consistency, samples = found
        if samples < self.cfg["flow_min_samples"] or consistency < self.cfg["flow_min_consistency"]:
            return None
        return _angle_diff(m.heading, direction)

    def _swings(self, track_id, m):
        """How many times the car switched between turning left and right recently."""
        window = max(int(self.cfg["weave_window_s"] * self.fps), 2)
        history = self.turn_history.setdefault(track_id, deque(maxlen=window))
        history.append(m.turn_rate)

        signs = [
            1 if t > 0 else -1 for t in history
            if t is not None and abs(t) >= self.cfg["weave_min_turn"]
        ]
        return sum(1 for a, b in zip(signs, signs[1:]) if a != b)

    def _headways(self, judged):
        """Seconds to the car directly ahead in the same lane, per following car."""
        cfg = self.cfg
        movers = [
            (track_id, m) for track_id, m in judged.items()
            if m.heading is not None and m.speed >= cfg["tailgate_min_speed"]
        ]
        headways = {}

        for f_id, f in movers:
            ux, uy = math.cos(math.radians(f.heading)), math.sin(math.radians(f.heading))
            for l_id, l in movers:
                if l_id == f_id:
                    continue
                if _angle_diff(f.heading, l.heading) > cfg["tailgate_max_heading_diff"]:
                    continue

                dx, dy = l.cx - f.cx, l.cy - f.cy
                along = dx * ux + dy * uy
                lateral = abs(dx * uy - dy * ux)
                size = (f.height + l.height) / 2

                # Leader must be ahead in the same lane; boxes that overlap
                # heavily are more likely a double detection than a real car
                if along <= 0.8 * size or lateral > cfg["tailgate_max_lateral"] * size:
                    continue

                gap = (along - size) / size  # bumper-to-bumper, in car lengths
                if gap < cfg["tailgate_min_gap"]:
                    continue
                headway = gap / f.speed
                if f_id not in headways or headway < headways[f_id]:
                    headways[f_id] = headway

        return headways
