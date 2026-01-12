import streamlit as st
import tempfile  # Used to create temporary files on disk
import sqlite3
import pandas as pd

from core_pipeline import run_traffic_analytics


# FRONT END PAGE
st.set_page_config(
    page_title="Traffic Analytics System",
    layout="wide",
)

st.title("🚦 Traffic Analytics System")
st.caption(
    "YOLO-based Vehicle Detection • Tracking • Speed Estimation • "
    "Vehicle-Centric Traffic Analytics"
)


# VIDEO UPLOAD
uploaded_video = st.file_uploader("Upload a traffic video", type=["mp4", "avi", "mov"])

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    st.subheader("📥 Input Video")
    st.video(video_path)

    # RUN PIPELINE
    if st.button("▶ Run Traffic Analytics"):
        with st.spinner("Processing video... Please wait"):
            result = run_traffic_analytics(video_path)

        st.success("Analysis completed successfully")

        # OUTPUT VIDEO
        st.subheader("📽️ Annotated Output Video")
        st.video(result["output_video"])

        # SUMMARY METRICS
        col1 = st.columns(1)[0]

        with col1:
            st.metric("Processing FPS", f"{result['fps']:.2f}")

        # DATABASE ANALYTICS (VEHICLE-CENTRIC)
        st.subheader("🚘 Vehicle Summary (Database)")

        conn = sqlite3.connect(result["db"])
        df = pd.read_sql("SELECT * FROM vehicles", conn)
        conn.close()

        if df.empty:
            st.warning("No vehicle records found.")
        else:
            try:
                df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
                df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
            except (ValueError, TypeError, pd.errors.ParserError) as e:
                st.warning(f"Using fallback timestamp conversion. Error: {str(e)}")
                try:
                    df["entry_time"] = pd.to_datetime(
                        df["entry_time"], unit="s", errors="coerce"
                    )
                    df["exit_time"] = pd.to_datetime(
                        df["exit_time"], unit="s", errors="coerce"
                    )
                except (ValueError, TypeError) as e2:
                    st.error(f"Could not convert timestamps: {str(e2)}")
                    pass

            # Filter out unrealistic speeds
            df = df[(df["max_speed"].isna()) | (df["max_speed"] <= 200.0)]

            # Fill NaN values for display
            df["avg_speed"] = df["avg_speed"].fillna(0)
            df["max_speed"] = df["max_speed"].fillna(0)

            # Order columns
            df = df[
                [
                    "vehicle_id",
                    "class",
                    "avg_speed",
                    "max_speed",
                    "direction",
                    "frames_seen",
                    "entry_time",
                    "exit_time",
                ]
            ]

            st.dataframe(
                df.sort_values("vehicle_id"),
                use_container_width=True,
            )

            # STATISTICS
            st.subheader("📊 Statistics")

            if len(df) > 0:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    valid_speeds = df[df["avg_speed"] > 0]["avg_speed"]
                    if len(valid_speeds) > 0:
                        avg_speed_all = round(valid_speeds.mean(), 2)
                    else:
                        avg_speed_all = 0
                    st.metric("Average Speed (km/h)", avg_speed_all)

                with col2:
                    max_speed = round(df["max_speed"].max(), 2)
                    st.metric("Maximum Speed (km/h)", max_speed)

                with col3:
                    speeding_count = len(df[df["max_speed"] > 40.0])
                    st.metric("Speeding Vehicles", speeding_count)

                with col4:
                    avg_frames = round(df["frames_seen"].mean(), 2)
                    st.metric("Avg Frames Seen", avg_frames)

        # SIDEBAR SUMMARY
        with st.sidebar:
            st.subheader("⚙️ Run Summary")

            if not df.empty:
                st.metric("Total Vehicles", len(df))
                st.metric(
                    "IN Vehicles",
                    int((df["direction"] == "IN").sum()),
                )
                st.metric(
                    "OUT Vehicles",
                    int((df["direction"] == "OUT").sum()),
                )
            else:
                st.metric("Total Vehicles", 0)
                st.info("No vehicles detected in the video")
