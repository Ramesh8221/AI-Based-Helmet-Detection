# 🪖 AI-Based Helmet Detection for Riders using Computer Vision

An end-to-end Computer Vision and Deep Learning application built with **Python**, **YOLO (Ultralytics)**, **OpenCV**, and **Streamlit** to detect motorcycle riders, check for helmet compliance, flag traffic safety violations, and maintain an audit history in **SQLite**.

---

## 📌 Project Overview
Wearing safety helmets is vital for motorcycle riders. This project provides an automated visual inspection sentinel capable of:
1. **Detecting Riders & Helmets** in static traffic images.
2. **Processing Video Footage** frame-by-frame from street CCTV and dashcams.
3. **Live Camera Monitoring** via webcam snapshot or real-time streaming HUD.
4. **Transparent Model Inspection**: Inspects the deep learning model's class taxonomy before inference to ensure scientific honesty and explainability.
5. **Historical Audit Logging**: Automatically records every detection session, timestamp, violation count, and confidence scores into a persistent SQLite database.

---

## 🏗️ Folder Structure

```
AI-Based-Helmet-Detection/
│
├── app.py                  # Main Streamlit Web Application Dashboard
├── detector.py             # YOLO Inference Engine, Taxonomy Inspector & Annotations
├── config.py               # Central System Configuration, Paths & Color Themes
├── requirements.txt        # Python package dependencies
├── README.md               # Detailed beginner-friendly user manual
├── .gitignore              # Ignored files (virtual environment, cache, etc.)
│
├── models/                 # Model weights directory
│   ├── README.md           # Model architecture, class taxonomy & setup guide
│   └── yolov8n.pt          # Baseline YOLOv8 nano model (auto-configured)
│
├── data/
│   ├── create_samples.py   # Script to generate sample test assets
│   └── samples/            # Built-in sample test images and videos
│       ├── rider_with_helmet.jpg
│       ├── rider_no_helmet.jpg
│       └── sample_traffic.mp4
│
├── outputs/                # Processed images and exported annotated MP4 videos
├── database/
│   └── db.py               # SQLite database operations (audit logs & statistics)
│
└── utils/
    ├── __init__.py
    └── video_utils.py      # Video decoding, progress tracking & MP4 encoding
```

---

## ⚙️ Quick Installation Guide (Windows, VS Code & PowerShell)

Follow these exact steps in your **VS Code PowerShell Terminal**:

### Step 1: Open the Project in VS Code
1. Launch **Visual Studio Code**.
2. Go to **File** > **Open Folder...** and select the `AI-Based-Helmet-Detection` folder.
3. Open a new terminal by pressing `Ctrl + ~` (or **Terminal** > **New Terminal**).

### Step 2: Create a Virtual Environment
```powershell
python -m venv .venv
```
*(If `python` is not recognized, use `py -3.11 -m venv .venv` or specify your installed Python path).*

### Step 3: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```

> [!TIP]
> **PowerShell Execution Policy Error Fix:**  
> If you see an error saying *"cannot be loaded because running scripts is disabled on this system"*, run this one-time command in PowerShell to bypass the restriction for your session:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

### Step 4: Upgrade Pip & Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Launch the Application
```powershell
streamlit run app.py
```

The Streamlit web dashboard will automatically open in your default browser at:  
👉 **`http://localhost:8501`**

---

## 🧠 AI Model Setup & Taxonomy

### Automatic Model Inspection
The system is built with scientific integrity in mind:
* When initialized, `detector.py` checks the loaded model's classes (`model.names`).
* **Dedicated Helmet Model** (`helmet_yolov8.pt`): Directly classifies objects into `Helmet` (Green) and `No Helmet` (Red violation).
* **Base YOLO Model** (`yolov8n.pt`): Localizes `person` (rider) and `motorcycle` objects, informing the user that dedicated helmet weights can be placed in `models/` for automated helmet classification.

### Adding a Dedicated Helmet-Trained Model
1. Obtain YOLOv8 helmet weights from [Roboflow Universe](https://universe.roboflow.com/search?q=helmet+detection) or [Kaggle](https://www.kaggle.com/).
2. Place the weights file inside the `models/` folder as `models/helmet_yolov8.pt`.
3. Open the **Model & Settings** tab in the web dashboard and select your new model, or upload it directly through the UI.

---

## 🧪 Testing Image, Video, and Webcam Detection

### 1. Test Image Detection
1. Click **🖼️ Image Detection** in the sidebar.
2. Select **🧪 Use Built-in Sample Image** and choose:
   * `rider_with_helmet.jpg` to test helmet compliance.
   * `rider_no_helmet.jpg` to test violation detection.
3. View the side-by-side detection HUD and click **⬇️ Download Annotated Image**.

### 2. Test Video Detection
1. Click **🎥 Video Detection** in the sidebar.
2. Select **🧪 Use Built-in Sample Video** (`sample_traffic.mp4`) or upload an MP4/AVI file.
3. Click **🚀 Start Video Analysis**. Watch the live frame preview and progress bar.
4. Download the generated annotated video when complete.

### 3. Test Live Webcam
1. Click **📹 Live Webcam** in the sidebar.
2. Choose **📸 Instant Camera Snapshot** to take an instant photo with your browser camera, or select **🔴 Continuous Live Stream** for real-time video stream detection.

### 4. Review Audit Logs
1. Click **📊 Detection History** to view all recorded sessions, violation counts, and compliance rates.
2. Click **⬇️ Export Audit Log as CSV** to download a spreadsheet report.

---

## 🛠️ Troubleshooting Common Issues

* **Camera Access Denied:** Ensure Windows Settings > Privacy & Security > Camera allows desktop apps to access the camera.
* **Video Codec Error:** If an uploaded video fails to decode, convert it to standard `.mp4` format (H.264 / AAC).
* **Port Conflict:** If port 8501 is busy, run:
  ```powershell
  streamlit run app.py --server.port 8502
  ```

---

## 🎓 Academic / Project Viva Questions & Answers
* **Q1: Why use YOLOv8 over earlier object detectors (e.g., R-CNN, SSD)?**  
  *Answer:* YOLOv8 is a single-stage detector that predicts bounding boxes and class probabilities in a single forward pass, providing real-time FPS suitable for video streams and edge deployment.
* **Q2: What is Non-Maximum Suppression (IoU)?**  
  *Answer:* NMS eliminates duplicate overlapping bounding boxes for the same rider by calculating the Intersection-over-Union ratio and retaining only the highest-confidence prediction.
* **Q3: How does the system handle model class discrepancy?**  
  *Answer:* The engine programmatically inspects the class dictionary (`model.names`) upon loading. If helmet classes are absent, it alerts the operator instead of hallucinating results.

---

## 📄 License
This project is open-source and built for educational and research purposes.
