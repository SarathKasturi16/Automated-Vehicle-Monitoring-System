import cv2
import numpy as np
import time
import subprocess
import sqlite3
import datetime
from collections import deque

from ultralytics import YOLO
import supervision as sv
from supervision.geometry.core import Point


# CONFIG
PIXEL_TO_METER = 0.05
SPEED_LIMIT_KMPH = 40.0
MAX_REALISTIC_SPEED = 200.0  # Safety filter
DB_NAME = "traffic_vehicles.db"

MIN_FRAMES_TO_LOG = 10  # Eliminates: False positives and Partial detections
LOST_GRACE_FRAMES = 12  # How long a vehicle can disappear
REID_DISTANCE_THRESH = 90  # To match lost vehicle to reappearing detection


# DATABASE
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS vehicles")
    cur.execute("""
        CREATE TABLE vehicles (
            vehicle_id INTEGER,
            class TEXT,
            entry_time TEXT,  
            exit_time TEXT,
            avg_speed REAL,
            max_speed REAL,
            direction TEXT,
            frames_seen INTEGER
        )
    """)
    conn.commit()
    conn.close()


# Vehicle details insertion
def insert_vehicle(row):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("INSERT INTO vehicles VALUES (?, ?, ?, ?, ?, ?, ?, ?)", row)
    conn.commit()
    conn.close()


# Video fix
def fix_video_for_streamlit(path):
    out = path.replace(".mp4", "_streamlit.mp4")
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
    )
    return out


# MAIN PIPELINE
def run_traffic_analytics(video_path):
    init_db()

    # Get real start time for accurate timestamps
    start_time_real = datetime.datetime.now()
    start_time_video = time.time()

    # SEGMENTATION MODEL
    model = YOLO("yolov8x-seg.pt")

    # Get video attributes
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = cv2.VideoWriter(  # Writes annotated output video
        "traffic_output_raw.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    lane_y = height // 2
    line_zone = sv.LineZone(
        Point(0, lane_y), Point(width, lane_y)
    )  # Virtual line for IN / OUT detection
    line_annotator = sv.LineZoneAnnotator()

    tracker = sv.ByteTrack()

    # Key terms
    next_vehicle_id = (
        1  # Every time a new physical vehicle is detected,vehicle_id = next_vehicle_id
    )
    vehicles = {}  # A dictionary keyed by vehicle_id
    track_to_vehicle = {}  # ByteTrack track_id → persistent vehicle_id
    last_positions = {}  # vehicle_id → (x, y)
    last_seen_frame = {}  # vehicle_id → frame_index
    frame_idx = 0  # Current frame number (global counter)
    fps_window = deque(maxlen=30)  # Last 30 instantaneous FPS measurements

    while True:
        ret, frame = (
            cap.read()
        )  # ret → a boolean (success flag), frame → the actual video frame (image)
        if not ret:  # Stops at video end
            break

        frame_idx += 1
        start_t = time.time()

        results = model(frame, conf=0.25, iou=0.5)[
            0
        ]  # Runs deep learning inference to YOLO on
        detections = sv.Detections.from_ultralytics(
            results
        )  # Converts YOLO output to Supervision format
        detections = tracker.update_with_detections(detections)  # Assigns tracker IDs

        used_vehicle_ids = set()

        # Per-Detection Processing
        for i, track_id in enumerate(detections.tracker_id):
            if track_id is None:  # Skip untracked detections
                continue

            track_id = int(track_id)
            x1, y1, x2, y2 = detections.xyxy[i]
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cls = model.names[int(detections.class_id[i])]

            # Stable ID Assignment
            if track_id not in track_to_vehicle:
                matched_vid = None
                
                '''When a new tracker ID appears, try to see if it is actually a previously seen vehicle that temporarily disappeared.
                Match only if:
                It hasn’t already been used this frame
                It’s the same vehicle class
                It disappeared only briefly
                It reappeared close to its last position'''
                for vid, (px, py) in last_positions.items(): #Checking every vehicle thats is seen before
                    if vid in used_vehicle_ids:  #If this vehicle ID is already used in this frame then skip it
                        continue
                    if vehicles[vid]["class"] != cls:
                        continue
                    if frame_idx - last_seen_frame.get(vid, 0) > LOST_GRACE_FRAMES:#If this vehicle disappeared too long ago, don’t try to resurrect it.
                        continue
                    if np.hypot(cx - px, cy - py) < REID_DISTANCE_THRESH:
                        matched_vid = vid #Only match if the vehicle reappeared close to where it was last seen
                        break

                if matched_vid is not None:
                    vehicle_id = matched_vid
                else:
                    vehicle_id = next_vehicle_id
                    next_vehicle_id += 1
                    # Calculate relative time for accurate timestamp
                    relative_time = time.time() - start_time_video
                    vehicle_entry = start_time_real + datetime.timedelta(
                        seconds=relative_time
                    )

                    vehicles[vehicle_id] = {
                        "class": cls,
                        "entry_time": vehicle_entry.isoformat(),  # Store as ISO string
                        "speeds": [],
                        "max_speed": 0.0,
                        "direction": None,
                        "frames": 0,
                        "start_y": cy,
                        "end_y": cy,
                    }

                track_to_vehicle[track_id] = vehicle_id
            else:
                vehicle_id = track_to_vehicle[track_id]

            used_vehicle_ids.add(vehicle_id)

            v = vehicles[vehicle_id]
            v["frames"] += 1
            v["end_y"] = cy

            speed = 0.0
            if vehicle_id in last_positions:
                px, py = last_positions[vehicle_id]
                
                # Calculate speed in km/h
                pixel_distance = np.hypot(cx - px, cy - py) #Euclidean displacement
                speed = pixel_distance * PIXEL_TO_METER * fps * 3.6 #Converts pixels/frame → km/h
                # Validate speed is realistic
                if 0 < speed < MAX_REALISTIC_SPEED:
                    v["speeds"].append(speed)
                    v["max_speed"] = max(v["max_speed"], speed)
                elif speed >= MAX_REALISTIC_SPEED:
                    # Skip unrealistic speed spike
                    pass

                #Direction Detection
                if v["direction"] is None: #Has the direction of this vehicle already been determined or not?
                    if py < lane_y and cy >= lane_y:
                        v["direction"] = "OUT" #The vehicle moved from top to bottom across the line.
                    elif py > lane_y and cy <= lane_y:
                        v["direction"] = "IN" #The vehicle moved from bottom to top across the line

            last_positions[vehicle_id] = (cx, cy)
            last_seen_frame[vehicle_id] = frame_idx

            color = (0, 0, 255) if speed > SPEED_LIMIT_KMPH else (0, 255, 0) #color to red if the vehicle is speeding and green if it is within the speed limit
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

        line_zone.trigger(detections)
        frame = line_annotator.annotate(frame, line_zone)
        writer.write(frame)
        fps_window.append(1 / (time.time() - start_t))

    cap.release()
    writer.release()

    # ---------------- FINAL DB WRITE ----------------
    for vid, v in vehicles.items():
        if v["frames"] < MIN_FRAMES_TO_LOG: #Rejects weak detections
            continue

        if v["direction"] is None:
            v["direction"] = (
                "OUT" if v["end_y"] > v["start_y"] else "IN"
            )

        # Calculate exit time
        relative_exit_time = time.time() - start_time_video
        exit_time = start_time_real + datetime.timedelta(seconds=relative_exit_time)

        # Calculate average speed if we have valid speed data
        avg_speed = None
        max_speed = None
        if v["speeds"]:
            # Filter out any remaining unrealistic speeds
            valid_speeds = [s for s in v["speeds"] if 0 < s < MAX_REALISTIC_SPEED]
            if valid_speeds:
                avg_speed = round(float(np.mean(valid_speeds)), 2)
                max_speed = round(float(max(valid_speeds)), 2)

        insert_vehicle(
            (
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

    return {
        "output_video": fix_video_for_streamlit("traffic_output_raw.mp4"),
        "fps": sum(fps_window) / len(fps_window) if fps_window else 0,
        "db": DB_NAME,
    }
