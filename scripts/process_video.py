from pathlib import Path

from advanced_lane_finding.calibration import calibrate_camera
from advanced_lane_finding.config import load_config
from advanced_lane_finding.video import process_video

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.toml"
CALIBRATION_DIRECTORY = PROJECT_ROOT / "data" / "camera_calibration"
INPUT_VIDEO_PATH = PROJECT_ROOT / "data" / "test_videos" / "project_video.mp4"
OUTPUT_VIDEO_PATH = PROJECT_ROOT / "outputs" / "videos" / "project_video_processed.mp4"


def main() -> None:
    config = load_config(CONFIG_PATH)
    calibration_paths = sorted(CALIBRATION_DIRECTORY.glob("calibration*.jpg"))

    if not calibration_paths:
        raise RuntimeError(f"No calibration images found in: {CALIBRATION_DIRECTORY}")

    calibration = calibrate_camera(calibration_paths, config.calibration)
    result = process_video(
        INPUT_VIDEO_PATH,
        OUTPUT_VIDEO_PATH,
        calibration,
        config,
    )

    print(f"Calibration RMS error: {calibration.rms_error:.4f}")
    print(f"Frames processed: {result.frame_count}")
    print(f"Validated detections: {result.detected_frames}")
    print(f"Missed detections: {result.missed_frames}")
    print(f"Output video: {result.output_path}")


if __name__ == "__main__":
    main()
