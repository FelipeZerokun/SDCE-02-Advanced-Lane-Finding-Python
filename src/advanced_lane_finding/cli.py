from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from advanced_lane_finding.calibration import (
    CalibrationError,
    calibrate_camera,
    load_calibration,
    save_calibration,
)
from advanced_lane_finding.config import (
    AdvancedLaneFindingConfig,
    ConfigError,
    load_config,
)
from advanced_lane_finding.detector import DetectionError
from advanced_lane_finding.pipeline import process_image
from advanced_lane_finding.validation import LaneValidationError
from advanced_lane_finding.video import VideoProcessingError, process_video


class CLIError(RuntimeError):
    """Raised when command-line input or output cannot be handled."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="advanced-lane-finding",
        description="Detect and render road lanes in images and videos.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a TOML configuration file (default: packaged config).",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser("calibrate", help="Calibrate a camera.")
    calibrate.add_argument("calibration_dir", type=Path)
    calibrate.add_argument("output", type=Path)

    image = commands.add_parser("image", help="Process one image.")
    image.add_argument("input", type=Path)
    image.add_argument("output", type=Path)
    image.add_argument("--calibration", type=Path, required=True)

    video = commands.add_parser("video", help="Process a video.")
    video.add_argument("input", type=Path)
    video.add_argument("output", type=Path)
    video.add_argument("--calibration", type=Path, required=True)

    return parser


def _run_calibration(
    calibration_dir: Path,
    output: Path,
    config: AdvancedLaneFindingConfig,
) -> None:
    if not calibration_dir.is_dir():
        raise CalibrationError(
            f"Calibration directory does not exist: {calibration_dir}"
        )

    image_paths = sorted(
        path
        for path in calibration_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    calibration = calibrate_camera(image_paths, config.calibration)
    saved_path = save_calibration(calibration, output)

    print(f"Calibration RMS error: {calibration.rms_error:.4f}")
    print(f"Calibration saved to: {saved_path}")


def _run_image(
    input_path: Path,
    output: Path,
    calibration_path: Path,
    config: AdvancedLaneFindingConfig,
) -> None:
    if output.exists():
        raise CLIError(f"Output path already exists: {output}")

    loaded = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if loaded is None:
        raise CLIError(f"Could not read input image: {input_path}")

    calibration = load_calibration(calibration_path)
    image = np.asarray(loaded, dtype=np.uint8)
    result = process_image(image, calibration, config)

    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), result.rendered_image):
        raise CLIError(f"Could not write output image: {output}")

    print(f"Output image: {output}")


def _run_video(
    input_path: Path,
    output: Path,
    calibration_path: Path,
    config: AdvancedLaneFindingConfig,
) -> None:
    calibration = load_calibration(calibration_path)
    result = process_video(input_path, output, calibration, config)

    print(f"Frames processed: {result.frame_count}")
    print(f"Validated detections: {result.detected_frames}")
    print(f"Missed detections: {result.missed_frames}")
    print(f"Output video: {result.output_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)

        if args.command == "calibrate":
            _run_calibration(args.calibration_dir, args.output, config)
        elif args.command == "image":
            _run_image(args.input, args.output, args.calibration, config)
        elif args.command == "video":
            _run_video(args.input, args.output, args.calibration, config)
        else:
            parser.error(f"Unknown command: {args.command}")

    except (
        CalibrationError,
        CLIError,
        ConfigError,
        DetectionError,
        LaneValidationError,
        VideoProcessingError,
        OSError,
    ) as error:
        parser.exit(1, f"error: {error}\n")

    return 0
