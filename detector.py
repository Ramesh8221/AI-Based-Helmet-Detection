"""YOLO-based Helmet Detection Engine.

Handles model loading, architecture inspection, class mapping, inference,
and visual bounding box annotations.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

import config


class HelmetDetectorError(Exception):
    """Custom exception raised for detector-specific errors."""
    pass


class HelmetDetector:
    """Encapsulates YOLO inference, class taxonomy inspection, and bounding box rendering."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initializes the detector with a specified model path or auto-discovers one."""
        self.model: Optional[YOLO] = None
        self.model_path: Optional[Path] = None
        self.class_names: Dict[int, str] = {}
        self.is_helmet_specific: bool = False
        self.helmet_class_ids: List[int] = []
        self.no_helmet_class_ids: List[int] = []
        self.rider_class_ids: List[int] = []
        self.vehicle_class_ids: List[int] = []

        self.load_model(model_path)

    def find_available_model_path(self, requested_path: Optional[Union[str, Path]] = None) -> Path:
        """Finds a valid model weights file from requested path, models dir, or workspace."""
        if requested_path:
            p = Path(requested_path)
            if p.exists() and p.is_file():
                return p

        # Check for custom helmet model in models/
        if config.CUSTOM_MODEL_PATH.exists():
            return config.CUSTOM_MODEL_PATH

        # Check for any .pt file inside models/ directory
        pt_files_in_models = list(config.MODELS_DIR.glob("*.pt"))
        if pt_files_in_models:
            # Prefer files with 'helmet' or 'best' in name
            for candidate in pt_files_in_models:
                if any(k in candidate.name.lower() for k in ["helmet", "best", "safety"]):
                    return candidate
            return pt_files_in_models[0]

        # Check for fallback model in workspace root or models/
        if config.FALLBACK_MODEL_PATH.exists():
            return config.FALLBACK_MODEL_PATH

        if config.FALLBACK_IN_MODELS.exists():
            return config.FALLBACK_IN_MODELS

        # If yolov8n.pt can be auto-downloaded by ultralytics
        return Path("yolov8n.pt")

    def load_model(self, model_path: Optional[Union[str, Path]] = None) -> None:
        """Loads the YOLO model and inspects its classes."""
        resolved_path = self.find_available_model_path(model_path)
        self.model_path = resolved_path

        try:
            self.model = YOLO(str(resolved_path))
        except Exception as e:
            raise HelmetDetectorError(
                f"Failed to load YOLO model from '{resolved_path}'. "
                f"Please ensure valid .pt weights exist. Details: {e}"
            )

        # Inspect class names
        raw_names = self.model.names if hasattr(self.model, "names") else {}
        if isinstance(raw_names, list):
            self.class_names = {i: name for i, name in enumerate(raw_names)}
        elif isinstance(raw_names, dict):
            self.class_names = raw_names
        else:
            self.class_names = {}

        self._inspect_and_categorize_classes()

    def _inspect_and_categorize_classes(self) -> None:
        """Analyzes class labels in the model to determine if it is helmet-specific

        or standard COCO.
        """
        self.helmet_class_ids = []
        self.no_helmet_class_ids = []
        self.rider_class_ids = []
        self.vehicle_class_ids = []

        for class_id, raw_name in self.class_names.items():
            name_clean = str(raw_name).strip().lower().replace("_", " ").replace("-", " ")

            # Check helmet labels
            if any(hl.replace("_", " ").replace("-", " ") in name_clean for hl in config.HELMET_LABELS):
                # Ensure it's not a negative phrase like 'no helmet'
                if not any(nhl.replace("_", " ").replace("-", " ") in name_clean for nhl in config.NO_HELMET_LABELS):
                    self.helmet_class_ids.append(class_id)
                    continue

            # Check no-helmet / head labels
            if any(nhl.replace("_", " ").replace("-", " ") in name_clean for nhl in config.NO_HELMET_LABELS):
                self.no_helmet_class_ids.append(class_id)
                continue

            # Check rider labels
            if any(rl.replace("_", " ").replace("-", " ") in name_clean for rl in config.RIDER_LABELS):
                self.rider_class_ids.append(class_id)
                continue

            # Check vehicle labels
            if any(vl.replace("_", " ").replace("-", " ") in name_clean for vl in config.VEHICLE_LABELS):
                self.vehicle_class_ids.append(class_id)
                continue

        # A model is dedicated helmet model if it specifically defines helmet or no-helmet classes
        self.is_helmet_specific = bool(self.helmet_class_ids or self.no_helmet_class_ids)

    def get_model_info(self) -> Dict[str, Any]:
        """Returns structured metadata about the currently loaded model."""
        model_name = self.model_path.name if self.model_path else "Unknown"
        return {
            "model_path": str(self.model_path) if self.model_path else "None",
            "model_name": model_name,
            "classes_count": len(self.class_names),
            "class_names": self.class_names,
            "is_helmet_specific": self.is_helmet_specific,
            "helmet_classes": [self.class_names.get(i) for i in self.helmet_class_ids],
            "no_helmet_classes": [self.class_names.get(i) for i in self.no_helmet_class_ids],
            "mode_description": (
                "Dedicated Helmet Detection Model (Classes directly verify Helmet / No Helmet)"
                if self.is_helmet_specific
                else "Standard YOLO COCO Model (Localizes riders/motorcycles; dedicated helmet-trained model recommended for direct helmet classification)"
            ),
        }

    def detect(
        self,
        image_input: Union[np.ndarray, Image.Image],
        conf_threshold: float = config.DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = config.DEFAULT_IOU_THRESHOLD,
    ) -> Dict[str, Any]:
        """Runs inference on the provided image and draws annotations.

        Parameters:
            image_input: NumPy BGR/RGB array or PIL Image.
            conf_threshold: Minimum detection confidence score (0.0 to 1.0).
            iou_threshold: IoU threshold for Non-Max Suppression.

        Returns:
            Dictionary with annotated image, detections list, summary counts,
            and status message.
        """
        if self.model is None:
            raise HelmetDetectorError("Model is not initialized.")

        # Convert input to BGR numpy array
        if isinstance(image_input, Image.Image):
            rgb_arr = np.array(image_input.convert("RGB"))
            bgr_image = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            bgr_image = image_input.copy()
            # If grayscale, convert to BGR
            if len(bgr_image.shape) == 2:
                bgr_image = cv2.cvtColor(bgr_image, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError("Unsupported image type. Provide a NumPy array or PIL Image.")

        h, w = bgr_image.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Empty image provided for detection.")

        # Run inference
        results = self.model.predict(
            source=bgr_image,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        helmet_count = 0
        no_helmet_count = 0
        rider_count = 0
        motorcycle_count = 0

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy_list = boxes.xyxy.cpu().numpy()
            conf_list = boxes.conf.cpu().numpy()
            cls_list = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy_list)):
                x1, y1, x2, y2 = map(int, xyxy_list[i])
                # Clip to image boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                confidence = float(conf_list[i])
                class_id = int(cls_list[i])
                raw_label = self.class_names.get(class_id, f"Class {class_id}")

                # Determine object semantic category
                if self.is_helmet_specific:
                    if class_id in self.helmet_class_ids:
                        category = "helmet"
                        display_label = f"Helmet {confidence:.2f}"
                        is_violation = False
                        helmet_count += 1
                    elif class_id in self.no_helmet_class_ids:
                        category = "no_helmet"
                        display_label = f"NO HELMET {confidence:.2f}"
                        is_violation = True
                        no_helmet_count += 1
                    elif class_id in self.rider_class_ids:
                        category = "rider"
                        display_label = f"Rider {confidence:.2f}"
                        is_violation = False
                        rider_count += 1
                    elif class_id in self.vehicle_class_ids:
                        category = "motorcycle"
                        display_label = f"Motorcycle {confidence:.2f}"
                        is_violation = False
                        motorcycle_count += 1
                    else:
                        category = "neutral"
                        display_label = f"{raw_label} {confidence:.2f}"
                        is_violation = False
                else:
                    # Standard COCO Model Behavior
                    if class_id in self.rider_class_ids:  # 'person'
                        category = "rider"
                        display_label = f"Rider/Person {confidence:.2f}"
                        is_violation = False
                        rider_count += 1
                    elif class_id in self.vehicle_class_ids:  # 'motorcycle'
                        category = "motorcycle"
                        display_label = f"Motorcycle {confidence:.2f}"
                        is_violation = False
                        motorcycle_count += 1
                    else:
                        # Skip irrelevant non-traffic COCO classes if desired, or keep as neutral
                        category = "neutral"
                        display_label = f"{raw_label} {confidence:.2f}"
                        is_violation = False

                detections.append(
                    {
                        "box": [x1, y1, x2, y2],
                        "confidence": round(confidence, 4),
                        "class_id": class_id,
                        "raw_label": raw_label,
                        "category": category,
                        "display_label": display_label,
                        "is_violation": is_violation,
                    }
                )

        # Draw annotations on image
        annotated_image = self._render_annotations(bgr_image, detections)

        total_detections = len(detections)
        violations_count = no_helmet_count

        # Formulate status message
        if not self.is_helmet_specific:
            if rider_count == 0 and motorcycle_count == 0:
                status_message = "No riders or motorcycles detected in the frame."
            else:
                status_message = (
                    f"Detected {rider_count} rider(s) and {motorcycle_count} motorcycle(s). "
                    "Note: Loaded model is base YOLO COCO. For automated Helmet vs No-Helmet "
                    "classification, place a helmet-trained weights file in 'models/'."
                )
        else:
            if total_detections == 0:
                status_message = "No objects detected."
            elif violations_count > 0:
                status_message = f"ALERT: {violations_count} helmet violation(s) detected!"
            elif helmet_count > 0:
                status_message = f"COMPLIANT: All {helmet_count} detected rider(s) are wearing helmets."
            else:
                status_message = "No riders or helmets detected in this frame."

        return {
            "annotated_image": annotated_image,
            "detections": detections,
            "counts": {
                "total": total_detections,
                "helmet": helmet_count,
                "no_helmet": no_helmet_count,
                "riders": rider_count,
                "motorcycles": motorcycle_count,
                "violations": violations_count,
            },
            "has_violations": violations_count > 0,
            "is_helmet_specific": self.is_helmet_specific,
            "status_message": status_message,
        }

    def _render_annotations(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draws bounding boxes and labels onto the image."""
        canvas = image.copy()
        h, w = canvas.shape[:2]

        # Adaptive thickness based on image resolution
        box_thickness = max(2, int(min(h, w) / 300))
        font_scale = max(0.45, min(h, w) / 1000.0)
        font_thickness = max(1, int(box_thickness / 2))

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            category = det["category"]
            label = det["display_label"]
            is_violation = det["is_violation"]

            # Choose color
            color = config.COLORS_BGR.get(category, config.COLORS_BGR["neutral"])

            # Draw outer rectangle
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, box_thickness)

            # Draw corner accents for professional HUD feel
            corner_len = min(20, int((x2 - x1) / 4), int((y2 - y1) / 4))
            accent_thickness = box_thickness + 1
            if corner_len > 4:
                # Top-Left
                cv2.line(canvas, (x1, y1), (x1 + corner_len, y1), color, accent_thickness)
                cv2.line(canvas, (x1, y1), (x1, y1 + corner_len), color, accent_thickness)
                # Top-Right
                cv2.line(canvas, (x2, y1), (x2 - corner_len, y1), color, accent_thickness)
                cv2.line(canvas, (x2, y1), (x2, y1 + corner_len), color, accent_thickness)
                # Bottom-Left
                cv2.line(canvas, (x1, y2), (x1 + corner_len, y2), color, accent_thickness)
                cv2.line(canvas, (x1, y2), (x1, y2 - corner_len), color, accent_thickness)
                # Bottom-Right
                cv2.line(canvas, (x2, y2), (x2 - corner_len, y2), color, accent_thickness)
                cv2.line(canvas, (x2, y2), (x2, y2 - corner_len), color, accent_thickness)

            # Measure text size
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Position badge above bounding box if space allows, otherwise inside
            badge_y1 = max(0, y1 - text_h - 10)
            badge_y2 = y1 if y1 - text_h - 10 >= 0 else y1 + text_h + 10
            badge_x2 = min(w - 1, x1 + text_w + 12)

            # Background pill for label
            cv2.rectangle(canvas, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

            # Text color (White text for high contrast)
            text_color = (255, 255, 255)
            text_baseline_y = badge_y2 - 5 if y1 - text_h - 10 >= 0 else badge_y2 - 5

            cv2.putText(
                canvas,
                label,
                (x1 + 6, text_baseline_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                font_thickness,
                cv2.LINE_AA,
            )

        return canvas
