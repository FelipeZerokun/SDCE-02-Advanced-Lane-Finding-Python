from pathlib import Path
from unittest.mock import Mock

import pytest

from advanced_lane_finding import cli
from advanced_lane_finding.config import AdvancedLaneFindingConfig


@pytest.mark.parametrize(
    ("arguments", "helper_name", "expected_paths"),
    [
        (
            ["calibrate", "images", "camera.yml"],
            "_run_calibration",
            (Path("images"), Path("camera.yml")),
        ),
        (
            ["image", "input.jpg", "output.jpg", "--calibration", "camera.yml"],
            "_run_image",
            (Path("input.jpg"), Path("output.jpg"), Path("camera.yml")),
        ),
        (
            ["video", "input.mp4", "output.mp4", "--calibration", "camera.yml"],
            "_run_video",
            (Path("input.mp4"), Path("output.mp4"), Path("camera.yml")),
        ),
    ],
)
def test_dispatches_command(
    arguments: list[str],
    helper_name: str,
    expected_paths: tuple[Path, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    helper = Mock()
    monkeypatch.setattr(cli, helper_name, helper)

    assert cli.main(arguments) == 0

    helper.assert_called_once()
    assert helper.call_args.args[:-1] == expected_paths
    assert isinstance(helper.call_args.args[-1], AdvancedLaneFindingConfig)


def test_image_requires_calibration_argument(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["image", "input.jpg", "output.jpg"])

    assert error.value.code == 2
    assert "--calibration" in capsys.readouterr().err


def test_reports_missing_config_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_config = tmp_path / "missing.toml"

    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "--config",
                str(missing_config),
                "calibrate",
                "images",
                "camera.yml",
            ]
        )

    assert error.value.code == 1
    assert "Configuration file not found" in capsys.readouterr().err
