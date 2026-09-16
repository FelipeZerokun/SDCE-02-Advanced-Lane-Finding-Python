from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib.resources import files
from math import isfinite, pi
from pathlib import Path
from typing import Any, cast

IntegerRange = tuple[int, int]
FloatRange = tuple[float, float]
Point = tuple[float, float]
Quadrilateral = tuple[Point, Point, Point, Point]
Color = tuple[int, int, int]


class ConfigError(ValueError):
    """Raised when advanced lane-finding configuration is invalid."""


def _require_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ConfigError(f"{name} must be positive")


def _require_range(
    name: str,
    value: tuple[float, float] | tuple[int, int],
    minimum: float,
    maximum: float,
) -> None:
    low, high = value

    if not minimum <= low < high <= maximum:
        raise ConfigError(f"{name} must satisfy {minimum} <= low < high <= {maximum}")


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)

    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid [{name}] section")

    return cast(dict[str, Any], value)


def _require_exact_keys(
    data: dict[str, Any],
    section_name: str,
    expected: set[str],
) -> None:
    actual = set(data)
    missing = expected - actual
    unexpected = actual - expected

    if missing:
        names = ", ".join(sorted(missing))
        raise ConfigError(f"[{section_name}] is missing: {names}")

    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ConfigError(f"[{section_name}] contains unknown settings: {names}")


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")

    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number")

    result = float(value)

    if not isfinite(result):
        raise ConfigError(f"{name} must be finite")

    return result


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{name} must be a string")

    return value


def _integer_range(value: Any, name: str) -> IntegerRange:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{name} must contain exactly two integers")

    return (
        _integer(value[0], f"{name}[0]"),
        _integer(value[1], f"{name}[1]"),
    )


def _float_range(value: Any, name: str) -> FloatRange:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{name} must contain exactly two numbers")

    return (
        _number(value[0], f"{name}[0]"),
        _number(value[1], f"{name}[1]"),
    )


def _point(value: Any, name: str) -> Point:
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"{name} must contain x and y coordinates")

    return (
        _number(value[0], f"{name}[0]"),
        _number(value[1], f"{name}[1]"),
    )


def _quadrilateral(value: Any, name: str) -> Quadrilateral:
    if not isinstance(value, list) or len(value) != 4:
        raise ConfigError(f"{name} must contain exactly four points")

    return (
        _point(value[0], f"{name}[0]"),
        _point(value[1], f"{name}[1]"),
        _point(value[2], f"{name}[2]"),
        _point(value[3], f"{name}[3]"),
    )


def _color(value: Any, name: str) -> Color:
    if not isinstance(value, list) or len(value) != 3:
        raise ConfigError(f"{name} must contain exactly three integers")

    return (
        _integer(value[0], f"{name}[0]"),
        _integer(value[1], f"{name}[1]"),
        _integer(value[2], f"{name}[2]"),
    )


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    chessboard_columns: int
    chessboard_rows: int
    square_size: float

    def __post_init__(self) -> None:
        if self.chessboard_columns < 2:
            raise ConfigError("calibration.chessboard_columns must be at least 2")
        if self.chessboard_rows < 2:
            raise ConfigError("calibration.chessboard_rows must be at least 2")
        _require_positive("calibration.square_size", self.square_size)


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    sobel_kernel: int
    gradient_x: IntegerRange
    gradient_magnitude: IntegerRange
    gradient_direction: FloatRange
    hls_saturation: IntegerRange
    hls_lightness: IntegerRange

    def __post_init__(self) -> None:
        if self.sobel_kernel <= 0 or self.sobel_kernel % 2 == 0:
            raise ConfigError("threshold.sobel_kernel must be a positive odd integer")

        _require_range("threshold.gradient_x", self.gradient_x, 0, 255)
        _require_range(
            "threshold.gradient_magnitude",
            self.gradient_magnitude,
            0,
            255,
        )
        _require_range(
            "threshold.gradient_direction",
            self.gradient_direction,
            0.0,
            pi / 2,
        )
        _require_range("threshold.hls_saturation", self.hls_saturation, 0, 255)
        _require_range("threshold.hls_lightness", self.hls_lightness, 0, 255)


@dataclass(frozen=True, slots=True)
class PerspectiveConfig:
    source: Quadrilateral
    destination: Quadrilateral

    def __post_init__(self) -> None:
        for group_name, points in (
            ("source", self.source),
            ("destination", self.destination),
        ):
            if len(points) != 4:
                raise ConfigError(f"perspective.{group_name} must contain four points")

            for point in points:
                if len(point) != 2:
                    raise ConfigError(
                        f"Each perspective.{group_name} point must contain x and y"
                    )

                x, y = point
                if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
                    raise ConfigError(
                        f"perspective.{group_name} coordinates must be between 0 and 1"
                    )


@dataclass(frozen=True, slots=True)
class SearchConfig:
    windows: int
    margin: int
    minimum_pixels: int
    previous_fit_margin: int

    def __post_init__(self) -> None:
        if self.windows <= 0:
            raise ConfigError("search.windows must be positive")
        if self.margin <= 0:
            raise ConfigError("search.margin must be positive")
        if self.minimum_pixels <= 0:
            raise ConfigError("search.minimum_pixels must be positive")
        if self.previous_fit_margin <= 0:
            raise ConfigError("search.previous_fit_margin must be positive")


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    minimum_lane_pixels: int
    minimum_lane_width_m: float
    maximum_lane_width_m: float
    maximum_width_variation_m: float

    def __post_init__(self) -> None:
        if self.minimum_lane_pixels <= 0:
            raise ConfigError("validation.minimum_lane_pixels must be positive")

        if not 0 < self.minimum_lane_width_m < self.maximum_lane_width_m:
            raise ConfigError(
                "Lane widths must satisfy 0 < minimum_lane_width_m "
                "< maximum_lane_width_m"
            )

        _require_positive(
            "validation.maximum_width_variation_m",
            self.maximum_width_variation_m,
        )


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    history_size: int
    maximum_failures: int

    def __post_init__(self) -> None:
        if self.history_size <= 0:
            raise ConfigError("tracking.history_size must be positive")
        if self.maximum_failures <= 0:
            raise ConfigError("tracking.maximum_failures must be positive")


@dataclass(frozen=True, slots=True)
class MeasurementConfig:
    lane_width_m: float
    visible_road_length_m: float

    def __post_init__(self) -> None:
        _require_positive("measurement.lane_width_m", self.lane_width_m)
        _require_positive(
            "measurement.visible_road_length_m",
            self.visible_road_length_m,
        )


@dataclass(frozen=True, slots=True)
class RenderConfig:
    lane_color_bgr: Color
    text_color_bgr: Color
    overlay_alpha: float
    font_scale: float
    line_thickness: int

    def __post_init__(self) -> None:
        for name, color in (
            ("lane_color_bgr", self.lane_color_bgr),
            ("text_color_bgr", self.text_color_bgr),
        ):
            if len(color) != 3 or any(
                channel < 0 or channel > 255 for channel in color
            ):
                raise ConfigError(
                    f"render.{name} must contain three values between 0 and 255"
                )

        if not 0.0 <= self.overlay_alpha <= 1.0:
            raise ConfigError("render.overlay_alpha must be between 0 and 1")

        _require_positive("render.font_scale", self.font_scale)

        if self.line_thickness <= 0:
            raise ConfigError("render.line_thickness must be positive")


@dataclass(frozen=True, slots=True)
class VideoConfig:
    codec: str

    def __post_init__(self) -> None:
        if len(self.codec) != 4:
            raise ConfigError("video.codec must contain exactly four characters")


@dataclass(frozen=True, slots=True)
class AdvancedLaneFindingConfig:
    calibration: CalibrationConfig
    threshold: ThresholdConfig
    perspective: PerspectiveConfig
    search: SearchConfig
    validation: ValidationConfig
    tracking: TrackingConfig
    measurement: MeasurementConfig
    render: RenderConfig
    video: VideoConfig


def _create_config(
    data: dict[str, Any],
    source: str,
) -> AdvancedLaneFindingConfig:
    section_names = {
        "calibration",
        "threshold",
        "perspective",
        "search",
        "validation",
        "tracking",
        "measurement",
        "render",
        "video",
    }
    _require_exact_keys(data, "configuration", section_names)

    calibration = _section(data, "calibration")
    threshold = _section(data, "threshold")
    perspective = _section(data, "perspective")
    search = _section(data, "search")
    validation = _section(data, "validation")
    tracking = _section(data, "tracking")
    measurement = _section(data, "measurement")
    render = _section(data, "render")
    video = _section(data, "video")

    _require_exact_keys(
        calibration,
        "calibration",
        {"chessboard_columns", "chessboard_rows", "square_size"},
    )
    _require_exact_keys(
        threshold,
        "threshold",
        {
            "sobel_kernel",
            "gradient_x",
            "gradient_magnitude",
            "gradient_direction",
            "hls_saturation",
            "hls_lightness",
        },
    )
    _require_exact_keys(
        perspective,
        "perspective",
        {"source", "destination"},
    )
    _require_exact_keys(
        search,
        "search",
        {"windows", "margin", "minimum_pixels", "previous_fit_margin"},
    )
    _require_exact_keys(
        validation,
        "validation",
        {
            "minimum_lane_pixels",
            "minimum_lane_width_m",
            "maximum_lane_width_m",
            "maximum_width_variation_m",
        },
    )
    _require_exact_keys(
        tracking,
        "tracking",
        {"history_size", "maximum_failures"},
    )
    _require_exact_keys(
        measurement,
        "measurement",
        {"lane_width_m", "visible_road_length_m"},
    )
    _require_exact_keys(
        render,
        "render",
        {
            "lane_color_bgr",
            "text_color_bgr",
            "overlay_alpha",
            "font_scale",
            "line_thickness",
        },
    )
    _require_exact_keys(video, "video", {"codec"})

    try:
        return AdvancedLaneFindingConfig(
            calibration=CalibrationConfig(
                chessboard_columns=_integer(
                    calibration["chessboard_columns"],
                    "calibration.chessboard_columns",
                ),
                chessboard_rows=_integer(
                    calibration["chessboard_rows"],
                    "calibration.chessboard_rows",
                ),
                square_size=_number(
                    calibration["square_size"],
                    "calibration.square_size",
                ),
            ),
            threshold=ThresholdConfig(
                sobel_kernel=_integer(
                    threshold["sobel_kernel"],
                    "threshold.sobel_kernel",
                ),
                gradient_x=_integer_range(
                    threshold["gradient_x"],
                    "threshold.gradient_x",
                ),
                gradient_magnitude=_integer_range(
                    threshold["gradient_magnitude"],
                    "threshold.gradient_magnitude",
                ),
                gradient_direction=_float_range(
                    threshold["gradient_direction"],
                    "threshold.gradient_direction",
                ),
                hls_saturation=_integer_range(
                    threshold["hls_saturation"],
                    "threshold.hls_saturation",
                ),
                hls_lightness=_integer_range(
                    threshold["hls_lightness"],
                    "threshold.hls_lightness",
                ),
            ),
            perspective=PerspectiveConfig(
                source=_quadrilateral(
                    perspective["source"],
                    "perspective.source",
                ),
                destination=_quadrilateral(
                    perspective["destination"],
                    "perspective.destination",
                ),
            ),
            search=SearchConfig(
                windows=_integer(search["windows"], "search.windows"),
                margin=_integer(search["margin"], "search.margin"),
                minimum_pixels=_integer(
                    search["minimum_pixels"],
                    "search.minimum_pixels",
                ),
                previous_fit_margin=_integer(
                    search["previous_fit_margin"],
                    "search.previous_fit_margin",
                ),
            ),
            validation=ValidationConfig(
                minimum_lane_pixels=_integer(
                    validation["minimum_lane_pixels"],
                    "validation.minimum_lane_pixels",
                ),
                minimum_lane_width_m=_number(
                    validation["minimum_lane_width_m"],
                    "validation.minimum_lane_width_m",
                ),
                maximum_lane_width_m=_number(
                    validation["maximum_lane_width_m"],
                    "validation.maximum_lane_width_m",
                ),
                maximum_width_variation_m=_number(
                    validation["maximum_width_variation_m"],
                    "validation.maximum_width_variation_m",
                ),
            ),
            tracking=TrackingConfig(
                history_size=_integer(
                    tracking["history_size"],
                    "tracking.history_size",
                ),
                maximum_failures=_integer(
                    tracking["maximum_failures"],
                    "tracking.maximum_failures",
                ),
            ),
            measurement=MeasurementConfig(
                lane_width_m=_number(
                    measurement["lane_width_m"],
                    "measurement.lane_width_m",
                ),
                visible_road_length_m=_number(
                    measurement["visible_road_length_m"],
                    "measurement.visible_road_length_m",
                ),
            ),
            render=RenderConfig(
                lane_color_bgr=_color(
                    render["lane_color_bgr"],
                    "render.lane_color_bgr",
                ),
                text_color_bgr=_color(
                    render["text_color_bgr"],
                    "render.text_color_bgr",
                ),
                overlay_alpha=_number(
                    render["overlay_alpha"],
                    "render.overlay_alpha",
                ),
                font_scale=_number(
                    render["font_scale"],
                    "render.font_scale",
                ),
                line_thickness=_integer(
                    render["line_thickness"],
                    "render.line_thickness",
                ),
            ),
            video=VideoConfig(
                codec=_string(video["codec"], "video.codec"),
            ),
        )
    except KeyError as error:
        raise ConfigError(
            f"Invalid configuration in {source}: missing {error.args[0]}"
        ) from error


def load_config(
    path: str | Path | None = None,
) -> AdvancedLaneFindingConfig:
    """Load an external configuration or the packaged default."""
    if path is None:
        resource = files("advanced_lane_finding").joinpath("default.toml")
        source = "packaged default configuration"

        try:
            with resource.open("rb") as file:
                data = tomllib.load(file)
        except FileNotFoundError:
            raise ConfigError("Packaged default configuration is missing") from None
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(
                f"Invalid packaged default configuration: {error}"
            ) from error
        except OSError as error:
            raise ConfigError(
                f"Could not read packaged default configuration: {error}"
            ) from error
    else:
        config_path = Path(path)
        source = str(config_path)

        try:
            with config_path.open("rb") as file:
                data = tomllib.load(file)
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {config_path}") from None
        except tomllib.TOMLDecodeError as error:
            raise ConfigError(
                f"Invalid configuration in {config_path}: {error}"
            ) from error
        except OSError as error:
            raise ConfigError(
                f"Could not read configuration {config_path}: {error}"
            ) from error

    return _create_config(data, source)
