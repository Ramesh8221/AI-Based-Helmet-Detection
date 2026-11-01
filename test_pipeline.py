"""Automated test suite verifying the complete helmet detection pipeline,

database operations, and input validation.
"""

from pathlib import Path
import cv2
import numpy as np

import config
from detector import HelmetDetector, HelmetDetectorError
from database import db
from utils import video_utils


def test_database_operations():
    """Tests SQLite database logging and analytics retrieval."""
    print("Testing SQLite Database operations...")
    db.init_db()
    rec_id = db.save_detection_record(
        source_type="Test_Unit",
        filename="test_mock.jpg",
        total_detections=2,
        helmet_count=1,
        no_helmet_count=1,
        violation_count=1,
        avg_confidence=0.88,
        details=[{"mock": True}],
    )
    assert rec_id > 0, "Failed to insert database record"

    records = db.fetch_all_records(limit=10)
    assert len(records) > 0, "Database records list is empty"
    assert records[0]["filename"] == "test_mock.jpg", "Database record filename mismatch"

    stats = db.fetch_summary_stats()
    assert stats["total_scans"] > 0, "Summary stats total_scans should be > 0"
    print("  ✓ Database operations passed.")


def test_detector_initialization():
    """Tests model loading, class taxonomy inspection, and fallback behavior."""
    print("Testing HelmetDetector initialization...")
    detector = HelmetDetector()
    model_info = detector.get_model_info()
    assert model_info["classes_count"] > 0, "Model has 0 classes loaded"
    print(f"  ✓ Model loaded: {model_info['model_name']} ({model_info['classes_count']} classes)")
    print(f"  ✓ Mode: {model_info['mode_description']}")
    return detector


def test_image_detection(detector: HelmetDetector):
    """Tests inference and bounding box annotations on sample images."""
    print("Testing Image Detection pipeline...")
    sample_helmet_path = config.SAMPLES_DIR / "rider_with_helmet.jpg"
    assert sample_helmet_path.exists(), f"Sample image missing: {sample_helmet_path}"

    img = cv2.imread(str(sample_helmet_path))
    assert img is not None, "Failed to load sample image with cv2"

    result = detector.detect(img, conf_threshold=0.25)
    assert "annotated_image" in result, "annotated_image missing from result"
    assert "counts" in result, "counts missing from result"
    assert result["annotated_image"].shape == img.shape, "Annotated image shape differs from input"
    print(f"  ✓ Image detection successful. Detections count: {result['counts']['total']}")


def test_video_processing(detector: HelmetDetector):
    """Tests frame-by-frame video inference pipeline and MP4 output generation."""
    print("Testing Video Processing pipeline...")
    sample_video_path = config.SAMPLES_DIR / "sample_traffic.mp4"
    assert sample_video_path.exists(), f"Sample video missing: {sample_video_path}"

    test_out_video = config.OUTPUTS_DIR / "test_out_video.mp4"
    summary = video_utils.process_video_file(
        input_path=sample_video_path,
        output_path=test_out_video,
        detector=detector,
        conf_threshold=0.25,
        frame_skip=2,
    )
    assert summary["processed_frames"] > 0, "No frames processed in video"
    assert test_out_video.exists(), "Output video file was not generated"
    assert test_out_video.stat().st_size > 0, "Output video file is empty (0 bytes)"
    print(f"  ✓ Video pipeline passed. Processed {summary['processed_frames']} frames.")


def test_error_handling(detector: HelmetDetector):
    """Tests graceful handling of invalid images and empty inputs."""
    print("Testing Error Handling & Input Validation...")
    # 1. Empty image input
    empty_canvas = np.zeros((0, 0, 3), dtype=np.uint8)
    try:
        detector.detect(empty_canvas)
        assert False, "Should have raised ValueError on empty image"
    except ValueError:
        print("  ✓ Gracefully caught empty image error.")

    # 2. Invalid video path
    v_info = video_utils.get_video_info("non_existent_video_path.mp4")
    assert not v_info["is_valid"], "Non-existent video should be flagged as invalid"
    print("  ✓ Gracefully caught invalid video path.")


if __name__ == "__main__":
    print("=" * 60)
    print("STARTING COMPLETE HELMET DETECTION PIPELINE TEST SUITE")
    print("=" * 60)
    test_database_operations()
    det = test_detector_initialization()
    test_image_detection(det)
    test_video_processing(det)
    test_error_handling(det)
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY! (100% PASS)")
    print("=" * 60)
