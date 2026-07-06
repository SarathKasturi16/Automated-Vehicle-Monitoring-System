import streamlit as st
import tempfile
import sqlite3
import pandas as pd
import os
from core_pipeline import run_traffic_analytics

st.set_page_config(
    page_title="Vehicle Vision",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@600;800&display=swap');
    h1, h2, h3, h4 { font-family: 'Outfit', sans-serif !important; }
    .header-title {
        background: linear-gradient(135deg, #60a5fa, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .header-subtitle { color: #94a3b8; margin-top: 4px; }
    div.stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# SESSION STATE
if "analysis_results" not in st.session_state:
    st.session_state["analysis_results"] = None
if "temp_video_path" not in st.session_state:
    st.session_state["temp_video_path"] = None
if "uploaded_file_name" not in st.session_state:
    st.session_state["uploaded_file_name"] = None


def cleanup_temp_video():
    path = st.session_state["temp_video_path"]
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except Exception:
            pass
    st.session_state["temp_video_path"] = None


# HEADER
st.markdown('<p class="header-title"> Vehicle Vision </p>', unsafe_allow_html=True)
st.markdown(
    '<p class="header-subtitle">YOLO-based vehicle detection, tracking, direction sorting & speed estimation</p>',
    unsafe_allow_html=True,
)
st.divider()


# SIDEBAR
with st.sidebar:
    st.markdown("### Platform Control")
    uploaded_video = st.file_uploader(
        "Upload raw traffic footage", type=["mp4", "avi", "mov"]
    )

    if uploaded_video:
        if st.session_state["uploaded_file_name"] != uploaded_video.name:
            cleanup_temp_video()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                st.session_state["temp_video_path"] = tmp.name
            st.session_state["uploaded_file_name"] = uploaded_video.name
            st.session_state["analysis_results"] = None

        st.info(f" Active File: {st.session_state['uploaded_file_name']}")

        st.markdown("###  Calibration")
        pixel_to_meter = st.slider(
            "Pixel-to-meter ratio",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            help="Real-world meters represented by one pixel of motion. "
            "Only accurate for a roughly top-down camera with no perspective "
            "distortion -- this is a manual estimate, not a calibrated measurement.",
        )
        speed_limit = st.slider(
            "Speed limit for violation flagging (km/h)",
            min_value=10,
            max_value=120,
            value=40,
            step=5,
        )

        run_clicked = st.button("▶ Run Traffic Analytics")

        if run_clicked:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(frame_idx, total_frames, current_fps):
                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    status_text.caption(
                        f"Frame {frame_idx}/{total_frames} · {current_fps:.1f} FPS"
                    )
                else:
                    status_text.caption(
                        f"Frame {frame_idx} · {current_fps:.1f} FPS (total length unknown)"
                    )

            try:
                result = run_traffic_analytics(
                    st.session_state["temp_video_path"],
                    pixel_to_meter=pixel_to_meter,
                    speed_limit_kmph=speed_limit,
                    progress_callback=update_progress,
                )
                st.session_state["analysis_results"] = result
                progress_bar.progress(1.0)
                status_text.empty()
                st.success("Analysis completed successfully!")
            except Exception as e:
                st.error(f"Pipeline error: {e}")

# MAIN DISPLAY
if not uploaded_video:
    st.info("Upload a traffic monitoring video from the sidebar to begin processing.")
else:
    results = st.session_state["analysis_results"]

    if results is None:
        st.subheader(" Video Source Preview")
        st.video(st.session_state["temp_video_path"])
        st.caption(
            "Set calibration values and click 'Run Traffic Analytics' in the sidebar."
        )
    else:
        conn = sqlite3.connect(results["db"])
        df = pd.read_sql(
            "SELECT * FROM vehicles WHERE run_id = ?", conn, params=(results["run_id"],)
        )
        conn.close()

        df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
        df["avg_speed"] = df["avg_speed"].fillna(0)
        df["max_speed"] = df["max_speed"].fillna(0)

        df_display = df[
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
        ].copy()

        tab_feed, tab_analytics, tab_logs = st.tabs(
            [" Video Feeds", " Analytics Dashboard", " Database Logs"]
        )

        # ---------------- TAB 1: VIDEO FEEDS ----------------
        with tab_feed:
            col_in, col_out = st.columns(2)
            with col_in:
                st.markdown("** Input Source**")
                st.video(st.session_state["temp_video_path"])
            with col_out:
                st.markdown("** Annotated Tracking Output**")
                st.video(results["output_video"])
                st.caption(
                    "Bounding boxes, direction line, and per-vehicle speed overlay."
                )

        # ---------------- TAB 2: ANALYTICS DASHBOARD ----------------
        with tab_analytics:
            if df.empty:
                st.warning("No vehicles were detected to generate metrics.")
            else:
                speed_limit_used = results["speed_limit_kmph"]

                total_vehicles = len(df)
                in_count = int((df["direction"] == "IN").sum())
                out_count = int((df["direction"] == "OUT").sum())

                valid_speeds = df[df["avg_speed"] > 0]["avg_speed"]
                avg_speed = (
                    round(valid_speeds.mean(), 1) if not valid_speeds.empty else 0.0
                )
                max_speed = round(df["max_speed"].max(), 1)

                # Uses the SAME limit that was applied during this run, instead
                # of a separately hardcoded number that could drift out of sync.
                speeding_count = len(df[df["max_speed"] > speed_limit_used])
                speeding_ratio = round((speeding_count / total_vehicles) * 100, 1)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric(
                    "Total Vehicles", total_vehicles, f"IN {in_count} · OUT {out_count}"
                )
                c2.metric("Average Speed", f"{avg_speed} km/h")
                c3.metric("Max Speed Recorded", f"{max_speed} km/h")
                c4.metric(
                    "Speeding Violators",
                    speeding_count,
                    f"{speeding_ratio}% of traffic",
                )
                st.caption(
                    f"Calibration used for this run: {results['pixel_to_meter']} m/px · "
                    f"speed limit {speed_limit_used} km/h."
                )

                st.divider()
                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown("**Vehicle Class Distribution**")
                    st.bar_chart(df["class"].value_counts())
                with col_chart2:
                    st.markdown("**Directional Flow (IN / OUT)**")
                    st.bar_chart(df["direction"].value_counts())

        # ---------------- TAB 3: DATABASE LOGS ----------------
        with tab_logs:
            if df_display.empty:
                st.warning("No vehicle entries for this run.")
            else:
                csv_data = df_display.to_csv(index=False).encode("utf-8")
                st.download_button(
                    " Download Logs (CSV)",
                    data=csv_data,
                    file_name=f"traffic_analytics_{results['run_id']}.csv",
                    mime="text/csv",
                )
                st.dataframe(
                    df_display.sort_values("vehicle_id"), use_container_width=True
                )
                st.caption(
                    f"Showing {len(df_display)} records for this run "
                    f"(run_id: {results['run_id']}) · Pipeline speed: {results['fps']:.2f} FPS."
                )
