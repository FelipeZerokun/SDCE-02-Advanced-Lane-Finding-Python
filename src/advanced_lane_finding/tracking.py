from __future__ import annotations

from collections import deque
from dataclasses import replace

import numpy as np

from advanced_lane_finding.calibration import CameraCalibration
from advanced_lane_finding.config import AdvancedLaneFindingConfig, TrackingConfig
from advanced_lane_finding.detector import DetectionError, LaneFit
from advanced_lane_finding.perspective import create_perspective_transform
from advanced_lane_finding.pipeline import (
    ColorImage,
    ImagePipelineResult,
    process_image,
    render_validated_fit,
)
from advanced_lane_finding.validation import LaneValidationError, validate_lane_fit


class LaneTracker:
    """Keep valid lane fits and failure state across video frames."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self._fit_history: deque[LaneFit] = deque(
            maxlen=config.history_size,
        )
        self._consecutive_failures = 0
        self._last_failure_reason: str | None = None

    @property
    def has_fit(self) -> bool:
        return bool(self._fit_history)

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    @property
    def last_failure_reason(self) -> str | None:
        return self._last_failure_reason

    def reset(self) -> None:
        """Discard tracking history and return to first-frame behavior."""
        self._fit_history.clear()
        self._consecutive_failures = 0
        self._last_failure_reason = None

    @property
    def current_fit(self) -> LaneFit | None:
        """Return the average of recent valid fits."""
        if not self._fit_history:
            return None

        left_coefficients = np.mean(
            [fit.left_coefficients for fit in self._fit_history],
            axis=0,
        )
        right_coefficients = np.mean(
            [fit.right_coefficients for fit in self._fit_history],
            axis=0,
        )

        return LaneFit(
            left_coefficients=np.asarray(
                left_coefficients,
                dtype=np.float64,
            ),
            right_coefficients=np.asarray(
                right_coefficients,
                dtype=np.float64,
            ),
        )

    def record_success(self, fit: LaneFit) -> None:
        """Store a valid fit and clear the failure count."""
        self._fit_history.append(fit)
        self._consecutive_failures = 0
        self._last_failure_reason = None

    def record_failure(self, reason: str | None = None) -> None:
        """Count a failed frame and eventually discard stale fits."""
        self._last_failure_reason = reason
        self._consecutive_failures += 1

        if self._consecutive_failures >= self.config.maximum_failures:
            self._fit_history.clear()

    def process_frame(
        self,
        image: ColorImage,
        calibration: CameraCalibration,
        pipeline_config: AdvancedLaneFindingConfig,
    ) -> ImagePipelineResult | None:
        """Process one frame and update the tracker after a valid detection."""
        try:
            result = process_image(
                image,
                calibration,
                pipeline_config,
                previous_fit=self.current_fit,
            )
        except (DetectionError, LaneValidationError) as error:
            self.record_failure(f"{type(error).__name__}: {error}")
            return None

        self.record_success(result.lane_fit)

        if len(self._fit_history) == 1:
            return result

        smoothed_fit = self.current_fit
        assert smoothed_fit is not None

        height, width = result.undistorted_image.shape[:2]
        image_size = (width, height)

        validation = validate_lane_fit(
            fit=smoothed_fit,
            image_size=image_size,
            perspective=pipeline_config.perspective,
            measurement=pipeline_config.measurement,
            validation=pipeline_config.validation,
        )

        transform = create_perspective_transform(
            image_size,
            pipeline_config.perspective,
        )
        rendered, measurements = render_validated_fit(
            result.undistorted_image,
            smoothed_fit,
            transform,
            pipeline_config,
        )

        return replace(
            result,
            rendered_image=rendered,
            lane_fit=smoothed_fit,
            validation=validation,
            measurements=measurements,
        )
