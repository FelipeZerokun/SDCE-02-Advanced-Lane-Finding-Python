from pathlib import Path

import pytest

from advanced_lane_finding.config import ConfigError, load_config

PROJECT_ROOT = Path(__file__).parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.toml"


def _modified_config(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> Path:
    contents = DEFAULT_CONFIG.read_text(encoding="utf-8")

    assert original in contents

    path = tmp_path / "modified.toml"
    path.write_text(
        contents.replace(original, replacement),
        encoding="utf-8",
    )
    return path


def test_loads_editable_default_configuration() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config.calibration.chessboard_columns == 9
    assert config.calibration.chessboard_rows == 6
    assert config.threshold.sobel_kernel == 3
    assert config.threshold.gradient_x == (30, 200)
    assert config.perspective.source[0] == (0.195, 0.94)
    assert config.search.windows == 9
    assert config.measurement.lane_width_m == 3.7
    assert config.render.lane_color_bgr == (0, 255, 0)
    assert config.video.codec == "mp4v"


def test_loads_packaged_default_configuration() -> None:
    config = load_config()

    assert config.search.margin == 100
    assert config.tracking.history_size == 5


def test_packaged_default_matches_editable_default() -> None:
    assert load_config() == load_config(DEFAULT_CONFIG)


def test_reports_missing_configuration(tmp_path: Path) -> None:
    missing = tmp_path / "missing.toml"

    with pytest.raises(ConfigError, match="Configuration file not found"):
        load_config(missing)


def test_rejects_malformed_toml(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        "[search\nwindows = 9",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Invalid configuration"):
        load_config(invalid)


def test_rejects_even_sobel_kernel(tmp_path: Path) -> None:
    invalid = _modified_config(
        tmp_path,
        "sobel_kernel = 3",
        "sobel_kernel = 4",
    )

    with pytest.raises(ConfigError, match="positive odd integer"):
        load_config(invalid)


def test_rejects_reversed_threshold(tmp_path: Path) -> None:
    invalid = _modified_config(
        tmp_path,
        "gradient_x = [30, 200]",
        "gradient_x = [200, 30]",
    )

    with pytest.raises(ConfigError, match="threshold.gradient_x"):
        load_config(invalid)


def test_rejects_boolean_integer(tmp_path: Path) -> None:
    invalid = _modified_config(
        tmp_path,
        "windows = 9",
        "windows = true",
    )

    with pytest.raises(ConfigError, match="search.windows must be an integer"):
        load_config(invalid)


def test_rejects_out_of_bounds_coordinate(tmp_path: Path) -> None:
    invalid = _modified_config(
        tmp_path,
        "[0.195, 0.94]",
        "[1.195, 0.94]",
    )

    with pytest.raises(ConfigError, match="coordinates must be between 0 and 1"):
        load_config(invalid)


def test_rejects_unknown_setting(tmp_path: Path) -> None:
    invalid = _modified_config(
        tmp_path,
        'codec = "mp4v"',
        'codec = "mp4v"\nunsupported_setting = true',
    )

    with pytest.raises(ConfigError, match="unknown settings"):
        load_config(invalid)


def test_rejects_invalid_codec(tmp_path: Path) -> None:
    invalid = _modified_config(
        tmp_path,
        'codec = "mp4v"',
        'codec = "invalid"',
    )

    with pytest.raises(ConfigError, match="exactly four characters"):
        load_config(invalid)
