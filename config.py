"""Configuration module for AI-Based Helmet Detection System.

Centralizes paths, thresholds, class labels, color schemes, and system constants.
"""

from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
SAMPLES_DIR = DATA_DIR / "samples"
OUTPUTS_DIR = BASE_DIR / "outputs"
DATABASE_DIR = BASE_DIR / "database"

# Ensure essential directories exist
for folder in [MODELS_DIR, DATA_DIR, SAMPLES_DIR, OUTPUTS_DIR, DATABASE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = DATABASE_DIR / "detection_history.db"

# Default Model Paths
# Primary: Custom helmet detection model (if user provides best.pt / helmet_yolov8.pt)
# Fallback: Official YOLOv8 nano model for general object detection & rider localization
HELMET_MODEL_FILENAME = "helmet_yolov8.pt"
FALLBACK_MODEL_FILENAME = "yolov8n.pt"

CUSTOM_MODEL_PATH = MODELS_DIR / HELMET_MODEL_FILENAME
FALLBACK_MODEL_PATH = BASE_DIR / FALLBACK_MODEL_FILENAME
FALLBACK_IN_MODELS = MODELS_DIR / FALLBACK_MODEL_FILENAME

# Detection Hyperparameters
DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
MIN_CONFIDENCE = 0.10
MAX_CONFIDENCE = 1.00

# Class name definitions for flexible helmet model mapping
HELMET_LABELS = {
    "helmet",
    "with_helmet",
    "with helmet",
    "with-helmet",
    "wearing_helmet",
    "hard hat",
    "hardhat",
    "with_hardhat",
}

NO_HELMET_LABELS = {
    "no_helmet",
    "no helmet",
    "no-helmet",
    "without_helmet",
    "without helmet",
    "without-helmet",
    "head",
    "bare_head",
    "no-hardhat",
    "without_hardhat",
}

RIDER_LABELS = {
    "person",
    "rider",
    "motorcyclist",
    "driver",
    "biker",
}

VEHICLE_LABELS = {
    "motorcycle",
    "motorbike",
    "bike",
    "bicycle",
    "scooter",
}

# Color palette in BGR (for OpenCV drawing) and HEX (for UI rendering)
COLORS_BGR = {
    "helmet": (70, 210, 70),       # Vibrant Green
    "no_helmet": (40, 40, 235),     # Alert Red
    "rider": (235, 170, 30),        # Cyan / Light Blue
    "motorcycle": (20, 140, 240),   # Orange
    "neutral": (180, 180, 180),     # Gray
}

COLORS_HEX = {
    "helmet": "#46D246",
    "no_helmet": "#EB2828",
    "rider": "#1EAEEB",
    "motorcycle": "#F08C14",
    "neutral": "#B4B4B4",
}

# Supported file formats
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
ALLOWED_VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]

# Application metadata
APP_TITLE = "AI Helmet Detection System"
APP_SUBTITLE = "Real-Time Motorcycle Rider Helmet Compliance & Violation Monitoring"
VERSION = "1.0.0"
