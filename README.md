# Offset-tracking user manual

## Introduction

This manual provides instructions on how to use the various modules and functions available in the `paracelso_wp32` package. The package includes utilities for image normalization, optical flow algorithms, and image coregistration.

## Table of Contents

- [Offset-tracking user manual](#offset-tracking-user-manual)
  - [Introduction](#introduction)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Modules Overview](#modules-overview)
    - [normalize.py](#normalizepy)
    - [metodi.py](#metodipy)
    - [main.py](#mainpy)
    - [interfaces.py](#interfacespy)
    - [coreg.py](#coregpy)
    - [algoritmi.py](#algoritmipy)
  - [Usage](#usage)
    - [Image processing](#image-processing)
    - [Optical Flow](#optical-flow)
    - [Coregistration](#coregistration)
  - [Command Line Interface](#command-line-interface)
    - [Mandatory arguments](#mandatory-arguments)
    - [Common arguments (avaiable to any algorithm)](#common-arguments-avaiable-to-any-algorithm)
    - [Arguments for `OpenCVOpticalFlow`](#arguments-for-opencvopticalflow)
    - [Arguments for `SkiOpticalFlowILK`](#arguments-for-skiopticalflowilk)
    - [Arguments for `SkiOpticalFlowTVL1`](#arguments-for-skiopticalflowtvl1)
    - [Arguments for `SkiPCC_Vector`](#arguments-for-skipcc_vector)
  - [Examples](#examples)
    - [Example 1: Normalizing an Image](#example-1-normalizing-an-image)
    - [Example 2: Performing Optical Flow](#example-2-performing-optical-flow)
    - [Example 3: Coregistering Images](#example-3-coregistering-images)

## Installation

To install the `paracelso_wp32` package, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/paracelso_wp32.git
cd paracelso_wp32
pip install -r requirements.txt
```

## Modules Overview

### normalize.py

This module provides various functions for normalizing and transforming 2D arrays and image data. It includes utilities for applying rolling windows, converting arrays to 8-bit unsigned integers, equalizing histograms of image channels, and performing different types of normalization such as standard normalization, power transformation, and z-score normalization.

### metodi.py

This module provides a factory for creating instances of various optical flow algorithms. It defines an enumeration `AlgFactory` that maps algorithm names to their corresponding classes, and a function `get_algorithm` to retrieve an algorithm instance by its name.

### main.py

This script is the main entry point for executing optical flow algorithms. It parses command-line arguments, loads images, applies normalization if specified, and performs optical flow calculations.

### interfaces.py

This module provides classes and functions for handling and processing images with multiple bands. It includes an enumeration for image bands, a class for representing images, and an abstract base class for optical tracking algorithms.

### coreg.py

This module contains a function for basic pixel coregistration between a target image and a reference image. It aligns the pixels and optionally reprojects the target image to the same CRS as the reference image.

### algoritmi.py

This module contains classes and functions for calculating optical flow between images and performing phase cross-correlation. It provides wrappers for OpenCV and scikit-image optical flow functions and utilities for converting results to GeoDataFrame or DataFrame formats.

## Usage

### Image processing

The `image_processing.py` module provides several processing function for handling images:

- `stepped_rolling_window(array_2d: np.ndarray, win_size: tuple, step_size: tuple = (1, 1)) -> tuple[np.ndarray]`
- `_to_CV8U(a: np.ndarray, cv2_norm_type: int = cv2.NORM_MINMAX) -> np.ndarray`
- `cv2_equalize_channels(array: np.ndarray) -> np.ndarray`
- `rasterio_to_CV2_8U(source: str)`
- `powernorm(band: np.ndarray, gamma: float = 1.0, mask=None, nodata: int | float | None = np.nan) -> np.ndarray`
- `_normalize_band(band, mask=None, nodata: int | float | None = np.nan)`
- `_zscore_band(band, mask=None, nodata: int | float | None = np.nan)`
- `_log_band(band, mask=None, epsilon=1e-5, nodata: int | float | None = np.nan)`
- `_clahe(band, clip_limit: float = 2., kernel_size: int | tuple[int] = 3, mask=None, nodata: int | float | None = np.nan)`

### Optical Flow

The `algoritmi.py` module provides classes for different optical flow algorithms:

- `OpenCVOpticalFlow`
- `SkiOpticalFlowILK`
- `SkiOpticalFlowTVL1`
- `SkiPCC_Vector`

To create an instance of an optical flow algorithm, use the `get_algorithm` function from the `metodi.py` module:

```python
from ot.metodi import get_algorithm

algorithm = get_algorithm("OPENCVOF")
```

### Coregistration

The `coreg.py` module provides a function for basic pixel coregistration:

- `basic_pixel_coregistration(infile: str, match: str, outfile: str)`

## Command Line Interface

The `main.py` script provides a command-line interface for executing optical flow algorithms. The following arguments are available:

- `-ot`, `--algname`: Algorithm to use.
- `-r`, `--reference`: Reference image.
- `-t`, `--target`: Target image.
- `-o`, `--output`: Output file.
- `-b`, `--band`: Band to use, can be either a band name (`str`) or a zero/based index. If not specified, all available bands will (should) be used.
- `--nodata`: NoData value.
- `--lognorm`: Normalize the logarithms of the images.
- `--normalize`: Normalize images based on minimum and maximum values.
- `--zscore`: Normalize images using z-score.
- `--minmax`: Normalize individual bands in the range 0 - 1 based on minimum and maximum values.
- `--clahe`: Normalize individual bands using an adaptive histogram equalization strategy (CLAHE)
- `--flow`: Initial flow guess.
- `--levels`: Number of pyramids.
- `--pyr_scale`: Scale ratio between pyramid levels.
- `--winsize`: Moving window size.
- `--step_size`: Sampling interval.
- `--iterations`: Number of iterations.
- `--poly_n`: Number of pixels for polynomial expansion.
- `--poly_sigma`: Standard deviation of the Gaussian for polynomial expansion.
- `--flags`: Optional operations.
- `--radius`: Moving window size.
- `--num_warp`: Number of iterations for the warp.
- `--gaussian`: Apply a Gaussian filter to the output image.
- `--prefilter`: Apply a pre-filter to the output image.
- `--attachment`: Smooth the final result.
- `--tightness`: Determine the tightness value.
- `--num_iter`: Fixed number of iterations.
- `--tol`: Tolerance for convergence.
- `--phase_norm`: Type of normalization in cross-correlation.
- `--upsmp_fac`: Upsample factor for sub-pixel scale shifts.

### Mandatory arguments
- `-ot`, `--algname`: Algorithm to use.
- `-r`, `--reference`: Reference image.
- `-t`, `--target`: Target image.

### Common arguments (avaiable to any algorithm)
- `-o`, `--output`: Output file.
- `-b`, `--band`: Band to use, can be either a band name (`str`) or a zero/based index. If not specified, all available bands will be used.
- `--nodata`: NoData value.
- `--lognorm`: Normalize the logarithms of the images.
- `--normalize`: Normalize images based on minimum and maximum values.
- `--zscore`: Normalize images using z-score.
- `--minmax`: Normalize individual bands in the range 0 - 1 based on minimum and maximum values.

### Arguments for `OpenCVOpticalFlow`
- [Mandatory arguments](#mandatory-arguments)
- `--flow`: Initial flow guess.
- `--pyr_scale`: Scale ratio between pyramid levels.
- `--levels`: Number of pyramids.
- `--winsize`: Moving window size.
- `--iterations`: Number of iterations.
- `--poly_n`: Number of pixels for polynomial expansion.
- `--poly_sigma`: Standard deviation of the Gaussian for polynomial expansion.
- `--flags`: Optional operations.

### Arguments for `SkiOpticalFlowILK`
- [Mandatory arguments](#mandatory-arguments)
- `--radius`: Moving window size.
- `--num_warp`: Number of iterations for the warp.
- `--gaussian`: Apply a Gaussian filter to the output image.
- `--prefilter`: Apply a pre-filter to the output image.

### Arguments for `SkiOpticalFlowTVL1`
- [Mandatory arguments](#mandatory-arguments)
- `--attachment`: Smooth the final result.
- `--tightness`: Determine the tightness value.
- `--num_warp`: Number of iterations for the warp.
- `--num_iter`: Fixed number of iterations.
- `--tol`: Tolerance for convergence.
- `--prefilter`: Apply a pre-filter to the output image.

### Arguments for `SkiPCC_Vector`
- [Mandatory arguments](#mandatory-arguments)
- `--winsize`: Moving window size.
- `--step_size`: Sampling interval.
- `--phase_norm`: Type of normalization in cross-correlation.
- `--upsmp_fac`: Upsample factor for sub-pixel scale shifts.


## Examples

### Example 1: Normalizing an Image

```python
import numpy as np
from ot.normalize import std_norm

array = np.random.rand(100, 100)
normalized_array = std_norm(array)
print(normalized_array)
```

### Example 2: Performing Optical Flow

```python
from ot.interfaces import Image
from ot.metodi import get_algorithm

reference = Image.from_file("reference.tif")
target = Image.from_file("target.tif")

algorithm = get_algorithm("OPENCVOF")
OTMethod = algorithm.from_dict({
    "pyr_scale": 0.5,
    "levels": 4,
    "winsize": 16,
    "iterations": 5,
    "poly_n": 5,
    "poly_sigma": 1.1,
    "flags": None
})

result = OTMethod(reference=reference.get_band("B1"), target=target.get_band("B1"))
result.to_file("output.tif")
```

### Example 3: Coregistering Images

```python
from ot.coreg import basic_pixel_coregistration

basic_pixel_coregistration("target.tif", "reference.tif", "coregistered_output.tif")
```