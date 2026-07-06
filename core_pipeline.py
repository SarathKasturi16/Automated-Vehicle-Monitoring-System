import cv2
import numpy as np
import time
import subprocess
import sqlite3
import datetime
import uuid
import os
import tempfile  # To store files temporarily
from collections import deque

from ultralytics import YOLO
import supervision as sv  # It converts YOLO's output into a format that ByteTrack understands.
from supervision.geometry.core import Point


MODEL_PATH = "yolov8x.pt"

VEHICLE_CLASS_NAMES = {"car", "motorcycle", "bus", "truck"}

PIXEL_TO_METER = 0.05
SPEED_LIMIT_KMPH = 40.0
MAX_REALISTIC_SPEED = 200.0

DB_NAME = "traffic_vehicles.db"

MIN_FRAMES_TO_LOG = 10  # Very short detections are often false positives
LOST_GRACE_FRAMES = 12  # If ByteTrack loses a vehicle wait 12 Frames
REID_DISTANCE_THRESH = 90

DEFAULT_FPS_FALLBACK = 30.0

# EMA = Exponential Moving Average
SPEED_EMA_ALPHA = 0.3  # This controls how smooth the speed number is for each vehicle


PROGRESS_UPDATE_EVERY_N_FRAMES = 5


# DATABASE
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute(
        "PRAGMA journal_mode=WAL;"
    )  # WAL = Write Ahead Logging -> for temporary log
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicles (
            run_id TEXT, 
            vehicle_id INTEGER,
            class TEXT,
            entry_time TEXT,
            exit_time TEXT,
            avg_speed REAL,
            max_speed REAL,
            direction TEXT,
            frames_seen INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def insert_vehicle(row):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO vehicles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row)
    conn.commit()
    conn.close()


# OpenCV saves video as mp4v, which browsers can't play well.
# This converts it to browser-friendly H.264 using ffmpeg.


def fix_video_for_streamlit(path):
    out = path.replace(".mp4", "_streamlit.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                path,
                "-vcodec",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                out,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        if os.path.exists(out) and os.path.getsize(out) > 0:
            # Transcode succeeded the raw mp4v intermediate is no longer
            # needed and would otherwise sit on disk forever.
            try:
                os.remove(path)
            except OSError:
                pass
            return out
    except Exception:
        pass
    return path


# Main Pipeline
def run_traffic_analytics(
    video_path,
    pixel_to_meter=PIXEL_TO_METER,
    speed_limit_kmph=SPEED_LIMIT_KMPH,
    progress_callback=None,
):
    # Part 1:Setup
    init_db()
    run_id = uuid.uuid4().hex[
        :12
    ]  # unique ID for this run, so multiple runs don't mix up in DB

    start_time_real = datetime.datetime.now()

    model = YOLO(MODEL_PATH)
    vehicle_class_ids = [
        idx for idx, name in model.names.items() if name in VEHICLE_CLASS_NAMES
    ]

    # Part 2: Opening the video + preparing output writer

    cap = cv2.VideoCapture(video_path)  # Open the video
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get the dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0 or np.isnan(fps):
        fps = DEFAULT_FPS_FALLBACK

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0  # Used for Progress Bar

    raw_output_path = os.path.join(  # Create a temporary output file.
        tempfile.gettempdir(), f"traffic_output_{run_id}.mp4"
    )

    # object that writes processed frames to output video
    writer = cv2.VideoWriter(
        raw_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    # Part 3: Set up line, tracker, and tracking variables

    lane_y = height // 2
    # Used to determine IN/OUT
    line_zone = sv.LineZone(Point(0, lane_y), Point(width, lane_y))
    line_annotator = sv.LineZoneAnnotator()  # Draw the line on the output video.
    tracker = sv.ByteTrack(frame_rate=int(round(fps)))  # frame-to-frame object tracker

    next_vehicle_id = 1  # initialize the vehicle id
    vehicles = {}
    track_to_vehicle = {}  # maps ByteTracks track_id -> stable vehicle_id
    last_positions = {}
    last_seen_frame = {}
    frame_idx = 0
    fps_window = deque(maxlen=30)  # Stores last 30 processing FPS values

    # Part 4: loop — read frame, detect, track
    """
                Workflow
                
                Read Frame
                    ↓
            YOLO Detection
                    ↓
            Supervision Conversion
                    ↓
            ByteTrack Tracking
                    ↓
            Vehicle ID Assignment
                    ↓
            Speed Calculation
                    ↓
            Draw Bounding Boxes
                    ↓
                Save Frame
    """

    while True:
        ret, frame = cap.read()  # ret is boolean
        if not ret:
            break

        frame_idx += 1
        start_t = time.time()

        # run YOLO detection
        results = model(frame, conf=0.25, iou=0.5, classes=vehicle_class_ids)[0]
        detections = sv.Detections.from_ultralytics(
            results
        )  # convert to supervision format
        detections = tracker.update_with_detections(detections)  # assign ByteTrack IDs

        # Part 5: Loop through each detected vehicle in this frame

        used_vehicle_ids = set()

        if detections.tracker_id is not None:
            for i, track_id in enumerate(detections.tracker_id):
                if track_id is None:
                    continue

                track_id = int(track_id)
                x1, y1, x2, y2 = detections.xyxy[i]  # bounding box coords
                cx, cy = (
                    int((x1 + x2) / 2),
                    int((y1 + y2) / 2),
                )  # center point of the box
                cls = model.names[int(detections.class_id[i])]

                # Part 6: Stable ID matching (nearest-match re-identification)

                """ByteTrack gives Track ID
                            │
                            ▼
                    Already mapped?
                         │
                    ┌────┴────┐
                    │         │
                    Yes        No
                    │         │
                    ▼         ▼
                 Use ID   Search nearby
                                │
                                ▼
                        Found Match?
                            │      │
                           Yes     No
                            │      │
                            ▼      ▼
                     Reuse ID  Create New ID
            """

                if track_id not in track_to_vehicle:  # If this ByteTrack ID is new
                    matched_vid = None
                    best_dist = REID_DISTANCE_THRESH

                    # try to match it to a recently lost vehicle nearby
                    for vid, (px, py) in last_positions.items():
                        # Vehicle has already been matched in this frame So Don't match it again
                        if vid in used_vehicle_ids:
                            continue
                        if vehicles[vid]["class"] != cls:
                            continue
                        # too long back (may be its new vehicle)
                        if frame_idx - last_seen_frame.get(vid, 0) > LOST_GRACE_FRAMES:
                            continue
                        dist = np.hypot(cx - px, cy - py)  # distance to candidate
                        if dist < best_dist:
                            best_dist = dist
                            matched_vid = vid  # closest match so far

                    if matched_vid is not None:
                        vehicle_id = matched_vid  # reuse existing vehicle ID
                    else:  # no match found this is a genuinely new vehicle
                        vehicle_id = next_vehicle_id
                        next_vehicle_id += 1
                        # give time after the video started
                        relative_time = frame_idx / fps
                        vehicle_entry = start_time_real + datetime.timedelta(
                            seconds=relative_time
                        )
                        vehicles[vehicle_id] = {
                            "class": cls,
                            "entry_time": vehicle_entry.isoformat(),
                            "speeds": [],
                            "max_speed": 0.0,
                            "smoothed_speed": None,  # EMA state
                            "direction": None,
                            "frames": 0,
                            "start_y": cy,
                            "end_y": cy,
                            "last_seen_frame": frame_idx,
                        }
                    # map bytetrack id with stable id
                    track_to_vehicle[track_id] = vehicle_id

                else:
                    vehicle_id = track_to_vehicle[track_id]  # already known reuse

                used_vehicle_ids.add(
                    vehicle_id
                )  # Marks this vehicle as already processed in the current frame.

                # Part 7: Update this vehicle's stats
                v = vehicles[vehicle_id]
                v["frames"] += 1
                v["end_y"] = cy
                v["last_seen_frame"] = frame_idx

                # Part 8: Speed calculation
                speed = 0.0
                if vehicle_id in last_positions:
                    px, py = last_positions[vehicle_id]

                    delta_frames = frame_idx - last_seen_frame[vehicle_id]
                    if delta_frames < 1:
                        delta_frames = 1

                    pixel_distance = np.hypot(
                        cx - px, cy - py
                    )  # how far it moved (in pixels)
                    raw_speed = (
                        pixel_distance * pixel_to_meter * fps * 3.6
                    ) / delta_frames  # convert: pixels -> meters -> per-second -> km/h

                    if 0 < raw_speed < MAX_REALISTIC_SPEED:
                        # good reading -> apply EMA smoothing
                        if v["smoothed_speed"] is None:
                            v["smoothed_speed"] = raw_speed
                        else:
                            v["smoothed_speed"] = (
                                SPEED_EMA_ALPHA * raw_speed
                                + (1 - SPEED_EMA_ALPHA) * v["smoothed_speed"]
                            )
                        speed = v["smoothed_speed"]
                        v["speeds"].append(speed)
                        v["max_speed"] = max(v["max_speed"], speed)
                    else:
                        # bad/outlier reading -> just reuse last good smoothed value
                        speed = v["smoothed_speed"] or 0.0

                    # Part 9: Direction (IN/OUT) decision
                    if v["direction"] is None:
                        if py < lane_y and cy >= lane_y:
                            v["direction"] = "OUT"
                        elif py > lane_y and cy <= lane_y:
                            v["direction"] = "IN"

                # Part 10: Save position, draw the box
                last_positions[vehicle_id] = (cx, cy)  # update last known position
                last_seen_frame[vehicle_id] = frame_idx  # draw bounding box

                color = (0, 0, 255) if speed > speed_limit_kmph else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(
                    frame,
                    f"ID:{vehicle_id} {cls} {speed:.1f}km/h",
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        # Part 11: Clean up lost vehicles

        stale_vids = [
            vid
            for vid, lf in last_seen_frame.items()
            if frame_idx - lf > LOST_GRACE_FRAMES
        ]
        for vid in stale_vids:
            last_positions.pop(vid, None)
            last_seen_frame.pop(vid, None)

        # Part 12: Draw line, write frame, report progress
        line_zone.trigger(detections)
        frame = line_annotator.annotate(frame, line_zone)
        writer.write(frame)
        fps_window.append(1 / (time.time() - start_t))

        if progress_callback and frame_idx % PROGRESS_UPDATE_EVERY_N_FRAMES == 0:
            avg_fps = sum(fps_window) / len(fps_window) if fps_window else 0
            progress_callback(frame_idx, total_frames, avg_fps)

    cap.release()
    writer.release()
    # ---------------loop ends-------------

    if progress_callback:
        avg_fps = sum(fps_window) / len(fps_window) if fps_window else 0
        progress_callback(frame_idx, total_frames, avg_fps)

    # Part 13: Save each vehicle's summary to database
    for vid, v in vehicles.items():
        if v["frames"] < MIN_FRAMES_TO_LOG:
            continue  # ignore vehicles seen too briefly (likely false detections)

        # fallback if it never crossed the line: guess direction from overall movement
        if v["direction"] is None:
            v["direction"] = "OUT" if v["end_y"] > v["start_y"] else "IN"

        # use vehicle's own last frame, not the shared dict
        last_frame = v.get("last_seen_frame", frame_idx)
        relative_exit_time = last_frame / fps
        exit_time = start_time_real + datetime.timedelta(seconds=relative_exit_time)

        avg_speed = None
        max_speed = None
        if v["speeds"]:
            valid_speeds = [s for s in v["speeds"] if 0 < s < MAX_REALISTIC_SPEED]
            if valid_speeds:
                avg_speed = round(float(np.mean(valid_speeds)), 2)
                max_speed = round(float(max(valid_speeds)), 2)

        # save one row per vehicle to the database
        insert_vehicle(
            (
                run_id,
                vid,
                v["class"],
                v["entry_time"],
                exit_time.isoformat(),
                avg_speed,
                max_speed,
                v["direction"],
                v["frames"],
            )
        )

    # Part 14: Return summary
    return {
        "run_id": run_id,
        "output_video": fix_video_for_streamlit(raw_output_path),
        "fps": sum(fps_window) / len(fps_window) if fps_window else 0,
        "db": DB_NAME,
        "pixel_to_meter": pixel_to_meter,
        "speed_limit_kmph": speed_limit_kmph,
    }
