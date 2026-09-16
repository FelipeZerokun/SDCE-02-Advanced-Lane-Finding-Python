from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from advanced_lane_finding.calibration import (
    calibrate_camera,
)
from advanced_lane_finding.config import load_config
from advanced_lane_finding.tracking import LaneTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs/default.toml"
CALIBRATION_DIRECTORY = PROJECT_ROOT / "data" / "camera_calibration"
INPUT_VIDEO_PATH = PROJECT_ROOT / "data" / "test_videos" / "project_video.mp4"
MAX_FRAMES = 300
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "images"

Uint8Image = NDArray[np.uint8]


def _write_image(path: Path, image: Uint8Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write diagnostic image: {path}")


def main() -> None:
    config = load_config(CONFIG_PATH)

    calibration_paths = sorted(
        CALIBRATION_DIRECTORY.glob("calibration*.jpg"),
    )

    if not calibration_paths:
        raise RuntimeError(f"No calibration images found in: {CALIBRATION_DIRECTORY}")

    calibration = calibrate_camera(
        calibration_paths,
        config.calibration,
    )

    capture = cv2.VideoCapture(str(INPUT_VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {INPUT_VIDEO_PATH}")

    tracker = LaneTracker(config.tracking)
    processed = 0
    successful = 0
    missed = 0
    first_miss_frame: int | None = None
    failure_reasons: Counter[str] = Counter()

    try:
        while processed < MAX_FRAMES:
            readable, frame = capture.read()

            if not readable or frame is None:
                break

            image = np.asarray(frame, dtype=np.uint8)
            result = tracker.process_frame(
                image,
                calibration,
                config,
            )

            if result is None:
                missed += 1
                reason = tracker.last_failure_reason or "Unknown failure"
                failure_reasons[reason] += 1

                if first_miss_frame is None:
                    first_miss_frame = processed
                    _write_image(
                        OUTPUT_DIRECTORY / "tracking_first_miss.jpg",
                        image,
                    )
            else:
                successful += 1

                if successful == 1:
                    _write_image(
                        OUTPUT_DIRECTORY / "tracking_first_success.jpg",
                        result.rendered_image,
                    )

                if processed in (149, 185, 299):
                    _write_image(
                        OUTPUT_DIRECTORY / f"tracking_frame_{processed:03d}.jpg",
                        result.rendered_image,
                    )

            processed += 1
    finally:
        capture.release()

    if processed == 0:
        raise RuntimeError("No frames could be read from the video")

    print(f"Calibration RMS error: {calibration.rms_error:.4f}")
    print(f"Frames processed: {processed}")
    print(f"Validated detections: {successful}")
    print(f"Missed detections: {missed}")
    print(f"First missed frame: {first_miss_frame}")
    print(f"Consecutive failures at end: {tracker.consecutive_failures}")
    print("Failure reasons:")

    for reason, count in failure_reasons.most_common():
        print(f"  {count}: {reason}")

    print(f"Saved tracking diagnostics to: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
