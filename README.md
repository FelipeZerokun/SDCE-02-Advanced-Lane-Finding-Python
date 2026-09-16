# Advanced Lane Finding — Python

![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A configurable computer-vision pipeline for detecting curved road-lane boundaries in images and MP4 videos. It calibrates the camera, corrects distortion, isolates lane pixels, fits and validates lane boundaries, estimates curvature and vehicle offset, and draws the result onto the original view. The implementation uses OpenCV and NumPy, with `uv` for its Python environment.

This repository modernizes the Advanced Lane Finding project from Udacity's Self-Driving Car Engineer Nanodegree as a typed, tested Python package with a command-line interface. It follows the simpler [Finding Lane Lines — Python project](https://github.com/FelipeZerokun/SDCE-01-Finding-Lane-Lines-Python), but handles curved roads and camera distortion rather than relying on straight-line detection.

## Features

- Calibrates a camera from chessboard photographs and saves reusable calibration data.
- Processes individual road images and complete videos using OpenCV.
- Combines gradient and HLS color thresholds to identify candidate lane pixels.
- Uses a bird's-eye perspective transform and polynomial fits for curved lanes.
- Validates lane geometry and smooths successful fits across video frames.
- Estimates lane curvature and vehicle position relative to the lane center.
- Uses a validated TOML configuration, a reproducible lockfile, tests, linting, and type checking.
- Refuses to overwrite existing calibration files, images, or videos.

## Detection pipeline

For each image or video frame, the pipeline:

1. Corrects camera distortion using a saved chessboard calibration.
2. Combines gradient and color thresholds into a binary lane-pixel mask.
3. Warps the mask into a bird's-eye view.
4. Finds lane pixels with sliding windows, or searches near a previous valid fit for video frames.
5. Fits polynomial boundaries and rejects implausible lane geometry.
6. Calculates lane curvature and the vehicle's offset from lane center.
7. Projects the lane area back onto the camera view and draws the measurements.

Video processing tracks recent valid fits to reduce jitter. If detection fails for one frame, it writes the undistorted frame without a lane overlay rather than dropping the frame.

## Requirements

- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- Git, if cloning the repository

The requested Python version, dependencies, and resolved versions are recorded in `.python-version`, `pyproject.toml`, and `uv.lock`.

## Installation

```console
git clone https://github.com/FelipeZerokun/SDCE-02-Advanced-Lane-Finding-Python.git
cd SDCE-02-Advanced-Lane-Finding-Python
uv python install
uv sync --dev
```

`uv` manages a local `.venv`; activating it is optional when using `uv run`.

## Usage

The repository includes calibration photographs, one sample image, and one sample video. Run these commands from the repository root. They are single-line commands and work in PowerShell.

### Calibrate the camera

```console
uv run advanced-lane-finding calibrate data/camera_calibration outputs/calibration.yml
```

This saves camera parameters that the image and video commands can reuse. The supplied calibration and sample media have matching dimensions; a different camera or image size requires an appropriate calibration.

### Process an image

```console
uv run advanced-lane-finding image data/test_images/test1.jpg outputs/test1_processed.jpg --calibration outputs/calibration.yml
```

### Process a video

```console
uv run advanced-lane-finding video data/test_videos/project_video.mp4 outputs/project_video_processed.mp4 --calibration outputs/calibration.yml
```

The video command reports processed, validated, and missed-frame counts. The output directory is created when needed. Existing output files are not replaced; choose a new output path if you run a command again.

Use `--help` for command options:

```console
uv run advanced-lane-finding --help
uv run advanced-lane-finding calibrate --help
uv run advanced-lane-finding image --help
uv run advanced-lane-finding video --help
```

## Configuration

The CLI uses the packaged default configuration unless you supply a TOML file. The editable [`configs/default.toml`](configs/default.toml) mirrors that default. Place `--config` **before** the subcommand:

```console
uv run advanced-lane-finding --config configs/default.toml image data/test_images/test1.jpg outputs/test1_custom.jpg --calibration outputs/calibration.yml
```

| Section | Purpose |
| --- | --- |
| `calibration` | Chessboard dimensions and square size |
| `threshold` | Gradient and HLS color thresholds |
| `perspective` | Source and destination points for the bird's-eye view |
| `search` | Sliding-window and previous-fit search parameters |
| `validation` | Minimum lane pixels and plausible lane-width limits |
| `tracking` | Fit-history length and failure tolerance |
| `measurement` | Lane width and visible-road scale in meters |
| `render` | Overlay and text appearance |
| `video` | Four-character output codec |

Invalid configuration values and malformed TOML are reported as command-line errors.

## Development

Install the development dependencies and run the quality checks:

```console
uv sync --dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src scripts tests
uv run pytest
uv build
```

To apply formatting, run `uv run ruff format src scripts tests`. The archived implementation under `legacy/` is excluded from Ruff checks.

## Project structure

```text
configs/default.toml           Editable pipeline configuration
data/camera_calibration/       Chessboard calibration photographs
data/test_images/             Tracked sample image
data/test_videos/             Tracked sample video
legacy/                       Original Udacity-era notebook and implementation
outputs/                      Generated calibration, images, and videos (Git-ignored)
scripts/                      Diagnostic and video-processing scripts
src/advanced_lane_finding/    Typed Python package and CLI
tests/                        Automated tests
pyproject.toml                Package metadata and dependencies
uv.lock                       Resolved dependency versions
```

Only the calibration photographs and one sample image and video are tracked as modern test data. Other local test media and generated outputs are ignored by Git.

## Limitations

This is a classical computer-vision pipeline tuned around the supplied camera view and project video. Detection may degrade with heavy shadows or glare, faded or occluded markings, unusual road geometry, different camera placement, or weather and lighting unlike the supplied examples. The perspective, threshold, validation, and measurement settings may need tuning for other cameras or roads.

The curvature and vehicle-offset values are estimates based on configured pixel-to-meter scales. This project is educational and is not suitable for vehicle control.

## Legacy project and attribution

The historical submission is preserved in [`legacy/P2.ipynb`](legacy/P2.ipynb) and [`legacy/Project02_RojasFelipe.py`](legacy/Project02_RojasFelipe.py). They are not used by the modern CLI.

This project is based on Udacity's [Advanced Lane Finding assignment](https://github.com/udacity/CarND-Advanced-Lane-Lines) from the Self-Driving Car Engineer Nanodegree. The original [course lesson](https://learn.udacity.com/nd013?version=3.0.1&partKey=a258e378-e335-4edb-a369-0cd30bbb8409&lessonKey=3a264f11-30a9-4b31-b2e4-8e8f9af52146&conceptKey=7dc9c51e-f6a7-4556-a42f-18d70e716d4f) may require Udacity access.

## License

This project is distributed under the terms of the [MIT License](LICENSE).
