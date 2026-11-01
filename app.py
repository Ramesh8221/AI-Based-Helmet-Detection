"""AI-Based Helmet Detection for Riders using Computer Vision.

Streamlit Web Dashboard supporting Image, Video, Live Webcam detection,
and persistent SQLite historical audit logging.
"""

from pathlib import Path
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st

import config
from detector import HelmetDetector, HelmetDetectorError
from database import db
from utils import video_utils

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern design aesthetics
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 14px 0 rgba(0, 0, 0, 0.25);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 4px;
    }
    
    /* Status Badges */
    .status-badge-safe {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22C55E;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 10px 0;
    }
    .status-badge-violation {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 10px 0;
    }
    .status-badge-info {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3B82F6;
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 10px 0;
    }
    .status-badge-warning {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 10px 16px;
        border-radius: 8px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Detector Singleton / Cache
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_detector(model_file_str: str) -> HelmetDetector:
    """Caches the HelmetDetector model in memory for fast inference."""
    path_obj = Path(model_file_str) if model_file_str else None
    return HelmetDetector(path_obj)


def get_detector() -> HelmetDetector:
    """Retrieves current detector based on selected session state model."""
    selected_model = st.session_state.get("selected_model_path", "")
    return load_cached_detector(selected_model)


# Initialize Database
db.init_db()


# ---------------------------------------------------------
# Sidebar Navigation & Settings
# ---------------------------------------------------------
st.sidebar.title("🪖 Helmet AI Sentinel")
st.sidebar.caption("Computer Vision & YOLO Deep Learning")

# Discover available models in models directory
available_models = list(config.MODELS_DIR.glob("*.pt"))
fallback_model = config.FALLBACK_MODEL_PATH if config.FALLBACK_MODEL_PATH.exists() else None

model_options = {}
for m in available_models:
    model_options[f"models/{m.name}"] = str(m)

if fallback_model and str(fallback_model) not in model_options.values():
    model_options[fallback_model.name] = str(fallback_model)

if not model_options:
    model_options["yolov8n.pt (Auto-download)"] = "yolov8n.pt"

default_selection = list(model_options.keys())[0]
if "selected_model_key" not in st.session_state:
    st.session_state["selected_model_key"] = default_selection
    st.session_state["selected_model_path"] = model_options[default_selection]

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🖼️ Image Detection",
        "🎥 Video Detection",
        "📹 Live Webcam",
        "📊 Detection History",
        "⚙️ Model & Settings",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Controls")

conf_slider = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.95,
    value=config.DEFAULT_CONF_THRESHOLD,
    step=0.05,
    help="Minimum score for recognizing riders and helmets.",
)

iou_slider = st.sidebar.slider(
    "IoU NMS Threshold",
    min_value=0.10,
    max_value=0.90,
    value=config.DEFAULT_IOU_THRESHOLD,
    step=0.05,
    help="Intersection-over-Union threshold for bounding box non-max suppression.",
)

# Fetch current detector and inspect
try:
    current_detector = get_detector()
    model_info = current_detector.get_model_info()
    model_is_helmet = model_info["is_helmet_specific"]
except Exception as e:
    current_detector = None
    model_info = {}
    model_is_helmet = False

st.sidebar.markdown("---")
st.sidebar.subheader("Active Model Status")
if model_is_helmet:
    st.sidebar.success(f"🪖 Dedicated Helmet Model\n({model_info.get('model_name')})")
else:
    st.sidebar.info(
        f"ℹ️ Base YOLO Model\n({model_info.get('model_name', 'yolov8n.pt')})\n"
        "Detects riders & traffic. Add custom helmet weights in models/ for dedicated helmet classification."
    )


# ---------------------------------------------------------
# Helper Rendering Functions
# ---------------------------------------------------------
def render_metrics_cards(counts: dict, has_violations: bool):
    """Renders 4 responsive metric summary cards."""
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Detections</div>
                <div class="metric-value">{counts.get('total', 0)}</div>
                <div class="metric-sub">Objects localized</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Helmets Detected</div>
                <div class="metric-value" style="color: #22C55E;">{counts.get('helmet', 0)}</div>
                <div class="metric-sub">Compliant riders</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Violations (No Helmet)</div>
                <div class="metric-value" style="color: #EF4444;">{counts.get('violations', 0)}</div>
                <div class="metric-sub">Suspected non-compliance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        riders = counts.get("riders", 0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Riders / Vehicles</div>
                <div class="metric-value" style="color: #38BDF8;">{riders + counts.get('motorcycles', 0)}</div>
                <div class="metric-sub">Traffic elements</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_status_banner(result: dict):
    """Renders informative status banner based on compliance and model mode."""
    counts = result["counts"]
    violations = counts.get("violations", 0)
    helmets = counts.get("helmet", 0)
    is_helmet_spec = result.get("is_helmet_specific", False)

    if not is_helmet_spec:
        st.markdown(
            f"""
            <div class="status-badge-warning">
                ⚠️ <b>Base YOLO Detection Active:</b> {result.get('status_message', '')}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif violations > 0:
        st.markdown(
            f"""
            <div class="status-badge-violation">
                🚨 <b>HELMET VIOLATION DETECTED:</b> {violations} rider(s) without helmet detected!
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif helmets > 0:
        st.markdown(
            f"""
            <div class="status-badge-safe">
                ✅ <b>COMPLIANT:</b> All {helmets} detected rider(s) are wearing protective helmets.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="status-badge-info">
                ℹ️ <b>No Riders Detected:</b> No riders or helmet objects found in the current frame.
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# PAGE 1: OVERVIEW & SYSTEM DASHBOARD
# =========================================================
if page == "🏠 Overview":
    st.title("🪖 AI-Based Helmet Detection for Riders")
    st.markdown(
        "An intelligent Computer Vision application powered by **YOLO (Ultralytics)** and **OpenCV** "
        "to monitor motorcycle rider helmet compliance, flag safety violations, and maintain an audit history."
    )

    stats = db.fetch_summary_stats()
    render_metrics_cards(
        {
            "total": stats.get("total_objects", 0),
            "helmet": stats.get("total_helmets", 0),
            "violations": stats.get("total_violations", 0),
            "riders": stats.get("total_scans", 0),
        },
        has_violations=stats.get("total_violations", 0) > 0,
    )

    st.markdown("### 🚀 Key Features")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.info(
            "**🖼️ Image Processing**\n\n"
            "Upload any road or traffic photograph to instantly detect riders, helmets, and violations. "
            "Download annotated images with one click."
        )
    with col_b:
        st.success(
            "**🎥 Video Processing**\n\n"
            "Analyze recorded CCTV traffic footage frame-by-frame. "
            "Real-time progress bars, preview streaming, and MP4 video generation."
        )
    with col_c:
        st.warning(
            "**📹 Live Webcam & Audit**\n\n"
            "Direct camera feed monitoring with real-time HUD overlays and an integrated SQLite detection history database."
        )

    st.markdown("### 📋 Quick-Start Guide for Testing")
    st.markdown(
        """
        1. Navigate to **🖼️ Image Detection** in the left sidebar.
        2. You can use the built-in **Sample Test Images** (`rider_with_helmet.jpg` or `rider_no_helmet.jpg`) or upload your own photo.
        3. Navigate to **🎥 Video Detection** to test the sample traffic video (`sample_traffic.mp4`).
        4. Inspect historical records under **📊 Detection History**.
        """
    )


# =========================================================
# PAGE 2: IMAGE DETECTION
# =========================================================
elif page == "🖼️ Image Detection":
    st.title("🖼️ Image Helmet & Rider Detection")
    st.markdown("Upload a traffic photo or select a built-in sample image to run object detection.")

    # Input Mode Selection
    input_choice = st.radio(
        "Image Source:",
        ["📁 Upload from Computer", "🧪 Use Built-in Sample Image"],
        horizontal=True,
    )

    image_to_process = None
    image_name = "input_image.jpg"

    if input_choice == "📁 Upload from Computer":
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=config.ALLOWED_IMAGE_TYPES,
            help="Supported formats: JPG, JPEG, PNG, WEBP, BMP",
        )
        if uploaded_file is not None:
            try:
                pil_img = Image.open(uploaded_file)
                image_to_process = np.array(pil_img.convert("RGB"))
                image_name = uploaded_file.name
            except Exception as e:
                st.error(f"Error loading uploaded image: {e}")
    else:
        # Sample images
        sample_files = list(config.SAMPLES_DIR.glob("*.jpg")) + list(config.SAMPLES_DIR.glob("*.png"))
        if sample_files:
            sample_names = [f.name for f in sample_files]
            selected_sample_name = st.selectbox("Select a sample image:", sample_names)
            chosen_sample_path = config.SAMPLES_DIR / selected_sample_name
            if chosen_sample_path.exists():
                pil_img = Image.open(chosen_sample_path)
                image_to_process = np.array(pil_img.convert("RGB"))
                image_name = selected_sample_name
        else:
            st.warning("No sample images found in data/samples. Please upload an image.")

    if image_to_process is not None:
        st.markdown("---")
        # Run detection
        if current_detector is None:
            st.error("Detector is not loaded. Please verify model configuration in Settings.")
        else:
            with st.spinner("Analyzing image with YOLO..."):
                t_start = time.time()
                # Detector expects BGR numpy array
                bgr_input = cv2.cvtColor(image_to_process, cv2.COLOR_RGB2BGR)
                result = current_detector.detect(
                    bgr_input,
                    conf_threshold=conf_slider,
                    iou_threshold=iou_slider,
                )
                inference_time_ms = round((time.time() - t_start) * 1000, 1)

            # Display Status Banner & Metrics
            render_status_banner(result)
            render_metrics_cards(result["counts"], result["has_violations"])

            # Side-by-side Image View
            col_orig, col_annotated = st.columns(2)
            with col_orig:
                st.subheader("📷 Original Image")
                st.image(image_to_process, use_container_width=True)

            with col_annotated:
                st.subheader("🎯 Detection & Annotation HUD")
                annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_container_width=True)

            # Save annotated image for download
            output_filename = f"annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_name}"
            output_filepath = config.OUTPUTS_DIR / output_filename
            cv2.imwrite(str(output_filepath), result["annotated_image"])

            # Encode for Streamlit download button
            success, encoded_img = cv2.imencode(".jpg", result["annotated_image"])
            if success:
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=encoded_img.tobytes(),
                    file_name=output_filename,
                    mime="image/jpeg",
                )

            # Auto-save record to SQLite database
            avg_conf = (
                np.mean([d["confidence"] for d in result["detections"]])
                if result["detections"]
                else 0.0
            )
            db.save_detection_record(
                source_type="Image",
                filename=image_name,
                total_detections=result["counts"]["total"],
                helmet_count=result["counts"]["helmet"],
                no_helmet_count=result["counts"]["no_helmet"],
                violation_count=result["counts"]["violations"],
                avg_confidence=float(avg_conf),
                details=result["detections"],
                output_path=str(output_filepath),
            )

            # Detection Table Breakdown
            st.markdown("### 🔍 Detection Breakdown")
            if result["detections"]:
                det_records = []
                for idx, det in enumerate(result["detections"], 1):
                    det_records.append(
                        {
                            "#": idx,
                            "Category": det["category"].capitalize(),
                            "Raw Class": det["raw_label"],
                            "Confidence": f"{det['confidence'] * 100:.1f}%",
                            "Bounding Box (x1, y1, x2, y2)": str(det["box"]),
                            "Violation": "🚨 YES" if det["is_violation"] else "✅ Compliant / N/A",
                        }
                    )
                df_det = pd.DataFrame(det_records)
                st.dataframe(df_det, use_container_width=True)
                st.caption(f"Inference latency: {inference_time_ms} ms")
            else:
                st.info("No targets detected above the selected confidence threshold.")


# =========================================================
# PAGE 3: VIDEO DETECTION
# =========================================================
elif page == "🎥 Video Detection":
    st.title("🎥 Video Helmet & Violation Detection")
    st.markdown("Upload a recorded traffic video or test with the sample video.")

    video_choice = st.radio(
        "Video Source:",
        ["📁 Upload Video File", "🧪 Use Built-in Sample Video"],
        horizontal=True,
    )

    video_input_path = None
    input_video_name = "test_video.mp4"

    if video_choice == "📁 Upload Video File":
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=config.ALLOWED_VIDEO_TYPES,
            help="Supported formats: MP4, AVI, MOV, MKV",
        )
        if uploaded_video is not None:
            temp_input_path = config.OUTPUTS_DIR / f"temp_upload_{uploaded_video.name}"
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_video.read())
            video_input_path = temp_input_path
            input_video_name = uploaded_video.name
    else:
        sample_video_path = config.SAMPLES_DIR / "sample_traffic.mp4"
        if sample_video_path.exists():
            video_input_path = sample_video_path
            input_video_name = "sample_traffic.mp4"
            st.info(f"Using built-in sample video: `{sample_video_path.name}`")
        else:
            st.warning("No sample video found in data/samples. Please upload a video.")

    if video_input_path and Path(video_input_path).exists():
        v_info = video_utils.get_video_info(video_input_path)
        if not v_info["is_valid"]:
            st.error(f"Cannot read video: {v_info.get('error')}")
        else:
            col_v1, col_v2, col_v3 = st.columns(3)
            col_v1.metric("Resolution", f"{v_info['width']} x {v_info['height']}")
            col_v2.metric("FPS", f"{v_info['fps']}")
            col_v3.metric("Duration / Frames", f"{v_info['duration_sec']}s ({v_info['frame_count']} f)")

            frame_skip = st.slider(
                "Processing Speed (Frame Step)",
                min_value=1,
                max_value=5,
                value=1,
                help="1 = process every single frame (highest accuracy). 2 or 3 = faster processing.",
            )

            if st.button("🚀 Start Video Analysis", type="primary"):
                if current_detector is None:
                    st.error("Detector is not loaded.")
                else:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    preview_col1, preview_col2 = st.columns(2)
                    preview_frame_holder = preview_col1.empty()
                    live_stats_holder = preview_col2.empty()

                    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_video_path = config.OUTPUTS_DIR / f"processed_{timestamp_suffix}_{input_video_name}"

                    def on_frame_progress(current_f, total_f, annotated_f, stats):
                        pct = min(1.0, current_f / max(1, total_f))
                        progress_bar.progress(pct)
                        status_text.text(f"Processing frame {current_f} / {total_f} ({int(pct * 100)}%)...")
                        # Show occasional preview update
                        if current_f % (frame_skip * 3) == 0:
                            rgb_prev = cv2.cvtColor(annotated_f, cv2.COLOR_BGR2RGB)
                            preview_frame_holder.image(rgb_prev, caption=f"Live HUD - Frame {current_f}", use_container_width=True)
                            live_stats_holder.markdown(
                                f"""
                                <div class="metric-card">
                                    <div class="metric-title">Live Video Stream</div>
                                    <p><b>Frame Helmets:</b> {stats.get('helmets_current', 0)}</p>
                                    <p><b>Frame Violations:</b> {stats.get('violations_current', 0)}</p>
                                    <p><b>Cumulative Violations:</b> <span style="color:#EF4444;font-weight:bold;">{stats.get('total_violations_seen', 0)}</span></p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with st.spinner("Processing video frames..."):
                        t0 = time.time()
                        v_summary = video_utils.process_video_file(
                            input_path=video_input_path,
                            output_path=output_video_path,
                            detector=current_detector,
                            conf_threshold=conf_slider,
                            iou_threshold=iou_slider,
                            progress_callback=on_frame_progress,
                            frame_skip=frame_skip,
                        )
                        proc_duration = round(time.time() - t0, 1)

                    progress_bar.progress(1.0)
                    status_text.success(f"Video processing finished in {proc_duration}s!")

                    # Render summary cards
                    render_metrics_cards(
                        {
                            "total": v_summary["total_helmets_detected"] + v_summary["total_no_helmets_detected"],
                            "helmet": v_summary["total_helmets_detected"],
                            "violations": v_summary["total_violations_detected"],
                            "riders": v_summary["violation_frames_count"],
                        },
                        has_violations=v_summary["has_violations"],
                    )

                    # Save record to database
                    db.save_detection_record(
                        source_type="Video",
                        filename=input_video_name,
                        total_detections=v_summary["processed_frames"],
                        helmet_count=v_summary["total_helmets_detected"],
                        no_helmet_count=v_summary["total_no_helmets_detected"],
                        violation_count=v_summary["total_violations_detected"],
                        avg_confidence=0.0,
                        details=[v_summary],
                        output_path=str(output_video_path),
                    )

                    st.markdown("### 📥 Download Processed Video")
                    if output_video_path.exists():
                        with open(output_video_path, "rb") as vf:
                            video_bytes = vf.read()
                        st.download_button(
                            label="⬇️ Download Processed Video (MP4)",
                            data=video_bytes,
                            file_name=f"helmet_detected_{input_video_name}",
                            mime="video/mp4",
                        )


# =========================================================
# PAGE 4: LIVE WEBCAM
# =========================================================
elif page == "📹 Live Webcam":
    st.title("📹 Live Webcam Helmet Monitoring")
    st.markdown(
        "Monitor helmet compliance in real-time through your computer's webcam or connected USB camera."
    )

    cam_mode = st.radio(
        "Camera Mode:",
        ["📸 Instant Camera Snapshot (Browser/Built-in)", "🔴 Continuous Live Stream (Local OpenCV)"],
        horizontal=True,
    )

    if cam_mode == "📸 Instant Camera Snapshot (Browser/Built-in)":
        st.caption("Capture a quick snapshot directly using your browser camera.")
        cam_photo = st.camera_input("Take a snapshot for helmet inspection")
        if cam_photo is not None:
            pil_img = Image.open(cam_photo)
            rgb_arr = np.array(pil_img.convert("RGB"))
            bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

            if current_detector:
                result = current_detector.detect(
                    bgr_arr,
                    conf_threshold=conf_slider,
                    iou_threshold=iou_slider,
                )
                render_status_banner(result)
                render_metrics_cards(result["counts"], result["has_violations"])

                annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, caption="Annotated Webcam Capture", use_container_width=True)

                # Save record to DB
                db.save_detection_record(
                    source_type="Webcam Snapshot",
                    filename="webcam_capture.jpg",
                    total_detections=result["counts"]["total"],
                    helmet_count=result["counts"]["helmet"],
                    no_helmet_count=result["counts"]["no_helmet"],
                    violation_count=result["counts"]["violations"],
                    details=result["detections"],
                )
    else:
        st.caption("Continuous local video capture. Press 'Start Webcam' to begin.")
        cam_index = st.number_input("Camera Index (Default: 0)", min_value=0, max_value=5, value=0)
        start_button = st.button("▶️ Start Webcam Stream", type="primary")
        stop_button = st.button("⏹️ Stop Stream")

        frame_window = st.empty()
        status_bar = st.empty()

        if start_button:
            cap = cv2.VideoCapture(int(cam_index))
            if not cap.isOpened():
                st.error(
                    f"Unable to access camera index {cam_index}. "
                    "Please check if another application is using the camera or ensure a webcam is connected."
                )
            else:
                st.success("Webcam stream connected! Press 'Stop Stream' or reload to terminate.")
                fps_counter = 0
                prev_time = time.time()

                try:
                    while cap.isOpened() and not stop_button:
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("Failed to receive video frame from webcam.")
                            break

                        fps_counter += 1
                        curr_time = time.time()
                        fps_display = round(1.0 / (curr_time - prev_time + 1e-6), 1)
                        prev_time = curr_time

                        if current_detector:
                            res = current_detector.detect(
                                frame,
                                conf_threshold=conf_slider,
                                iou_threshold=iou_slider,
                            )
                            ann_frame = res["annotated_image"]

                            # Draw HUD FPS banner
                            cv2.putText(
                                ann_frame,
                                f"FPS: {fps_display}",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 255, 0),
                                2,
                            )

                            rgb_frame = cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB)
                            frame_window.image(rgb_frame, use_container_width=True)

                            # Live status
                            counts = res["counts"]
                            if counts.get("violations", 0) > 0:
                                status_bar.error(f"🚨 VIOLATION ALERT: {counts['violations']} Rider without Helmet!")
                            elif counts.get("helmet", 0) > 0:
                                status_bar.success(f"✅ Helmet Compliant ({counts['helmet']} detected)")
                            else:
                                status_bar.info("Scanning for riders...")
                finally:
                    cap.release()


# =========================================================
# PAGE 5: DETECTION HISTORY
# =========================================================
elif page == "📊 Detection History":
    st.title("📊 Helmet Compliance History & Audit Logs")
    st.markdown("Persistent detection logs stored in SQLite database (`detection_history.db`).")

    stats = db.fetch_summary_stats()
    render_metrics_cards(
        {
            "total": stats.get("total_objects", 0),
            "helmet": stats.get("total_helmets", 0),
            "violations": stats.get("total_violations", 0),
            "riders": stats.get("total_scans", 0),
        },
        has_violations=stats.get("total_violations", 0) > 0,
    )

    records = db.fetch_all_records(limit=200)

    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.subheader(f"Recent Detections ({len(records)} entries)")
    with col_h2:
        if st.button("🗑️ Clear All History", type="secondary"):
            db.clear_all_records()
            st.rerun()

    if records:
        df = pd.DataFrame(records)
        # Rename columns for presentation
        display_df = df.copy()
        if "details_json" in display_df.columns:
            display_df.drop(columns=["details_json"], inplace=True)

        st.dataframe(display_df, use_container_width=True)

        # CSV Export
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export Audit Log as CSV",
            data=csv_data,
            file_name=f"helmet_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No detection history recorded yet. Run Image, Video, or Webcam detections to populate logs.")


# =========================================================
# PAGE 6: MODEL & SYSTEM SETTINGS
# =========================================================
elif page == "⚙️ Model & Settings":
    st.title("⚙️ Model Taxonomy & System Configuration")
    st.markdown("Inspect deep learning model weights, class names, and configure paths.")

    if current_detector:
        m_info = current_detector.get_model_info()

        st.markdown("### 🧠 Active Model Details")
        st.write(f"**Weights File:** `{m_info.get('model_path')}`")
        st.write(f"**Total Classes:** `{m_info.get('classes_count')}`")
        st.write(f"**Dedicated Helmet Model:** `{'Yes' if m_info.get('is_helmet_specific') else 'No'}`")
        st.info(m_info.get("mode_description", ""))

        st.markdown("### 🏷️ Model Class Names (Taxonomy)")
        classes_dict = m_info.get("class_names", {})
        if classes_dict:
            items = list(classes_dict.items())
            df_classes = pd.DataFrame(
                [{"Class ID": cid, "Class Name": name} for cid, name in items]
            )
            st.dataframe(df_classes, use_container_width=True, height=250)

    st.markdown("---")
    st.markdown("### 🔄 Switch or Upload Custom Model Weights")

    # Select from existing
    selected_key = st.selectbox(
        "Choose an existing model:",
        list(model_options.keys()),
        index=list(model_options.keys()).index(st.session_state.get("selected_model_key", list(model_options.keys())[0])),
    )

    if st.button("Apply Selected Model"):
        st.session_state["selected_model_key"] = selected_key
        st.session_state["selected_model_path"] = model_options[selected_key]
        st.success(f"Model updated to `{selected_key}`! Reloading detector...")
        st.cache_resource.clear()
        st.rerun()

    # Upload new .pt file
    uploaded_model = st.file_uploader("Upload custom PyTorch (.pt) weights file", type=["pt"])
    if uploaded_model is not None:
        save_target = config.MODELS_DIR / uploaded_model.name
        with open(save_target, "wb") as f:
            f.write(uploaded_model.read())
        st.success(f"Custom weights saved to `{save_target}`! You can now select it above.")
        st.rerun()
