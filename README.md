# **Paracelso WP3.2**

# Table of contents

- [Overview](#overview)
  - [Project structure](#project-structure)
  - [Available algorithms](#available-algorithms)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Logging](#logging)
- [Offset-tracking module (`ot`)](#ot-sub-module)
  - [Features](#features)
  - [Modules](#modules)
  - [Usage](#usage-1)
- [Sentinel image processing module (`s1`)](#s1-sub-module)
  - [Features](#features-1)
  - [Graphs](#graphs)
  - [Usage](#usage-2)
- [COSMO-SkyMed image processing module (`cosmo`)](#cosmo-skymed-image-processing)
  - [Modules](#modules-1)
  - [Usage](#usage-3)
- [SNAP GPT module (`snap_gpt`)](#snap-gpt-module)
  - [Graphs](#graphs-1)
  - [Usage](#usage-4)
- [Image Processing Modules (`ot.image_processing`)](#image-processing-modules)
- [Logging module (`log`)](#log-sub-module)
- [Main script (`main.py`)](#ot-main-script)
  - [Command-Line Interface (CLI)](#command-line-interface-cli)

# Overview

This project implements various optical flow algorithms and image processing techniques for offset tracking. It includes modules for handling images, applying preprocessing steps, and executing different optical flow methods.

## Available Algorithms

The following algorithms are available:
- `OPENCVOF`: OpenCV Optical Flow
- `SKIOFILK`: Scikit-Image Optical Flow ILK
- `SKIOFTVL1`: Scikit-Image Optical Flow TVL1
- `SKIPCCV`: Scikit-Image Phase Cross-Correlation Vector

## Project Structure

```
paracelso_wp32/
├── ot/
│   ├── __init__.py
│   ├── algoritmi.py
│   ├── helpmsg.py
│   ├── image_processing/
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── opencv.py
│   │   ├── ski.py
│   ├── interfaces.py
│   ├── main.py
│   ├── metodi.py
│   ├── utils.py
├── README.md
```

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/vincecrit/paracelso_wp32.git
    cd paracelso_wp32
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running the Main Script

To run the main script, use the following command:
```sh
python -m ot.main --algname <algorithm_name> --reference <reference_image> --target <target_image> --output <output_file>
```

### Example

```sh
python -m ot.main --algname OPENCVOF --reference ref.tif --target tar.tif --output output.tif
```

## Logging

The project uses a custom logging setup to log messages to both the console and a file. The log configuration is defined in sub-module `log`.

# OT Sub-Module

The `ot` sub-module provides various functionalities for optical tracking and image processing. It includes algorithms for optical flow, image normalization, and utility functions for handling geospatial data.

## Features

- Optical Flow Algorithms (OpenCV and scikit-image)
- Image Normalization (Min-Max, Z-Score, Logarithmic)
- Geospatial Data Handling (GeoTIFF, GeoPackage)
- Preprocessing Dispatcher for Image Processing

## Modules

### `ot/algoritmi.py`

This module contains classes and functions for optical flow and phase cross-correlation.

- `OpenCVOpticalFlow`: Wraps the `calcOpticalFlowFarneback` function from OpenCV.
- `SkiOpticalFlowILK`: Wraps the `optical_flow_ilk` function from scikit-image.
- `SkiOpticalFlowTVL1`: Wraps the `optical_flow_tvl1` function from scikit-image.
- `SkiPCC_Vector`: Calculates phase cross-correlation vectors.

### `ot/interfaces.py`

This module provides classes and functions for handling and processing images with multiple bands.

- `Image`: Represents an image with multiple bands.
- `OTAlgorithm`: Abstract base class for optical tracking algorithms.

### `ot/utils.py`

This module provides utility functions for handling geospatial data and image processing.

- `basic_pixel_coregistration`: Aligns pixels between a target image and a reference image.
- `rasterio_read`: Reads an image file using rasterio.
- `image_to_raster`: Saves an image as a raster file.
- `geopandas_to_gpkg`: Saves a GeoDataFrame as a GeoPackage file.

### `ot/metodi.py`

This module provides a factory for creating instances of various optical flow algorithms.

- `AlgFactory`: Enumeration for creating algorithm instances.
- `get_method`: Retrieves an algorithm instance by name.

### `ot/helpmsg.py`

This module contains help strings for the argument parser.

### `ot/main.py`

This module is the main entry point for running the optical tracking analysis.

- `get_parser`: Returns the argument parser for the script.
- `load_images`: Loads a pair of images and returns `Image` objects.
- `main`: Main function to run the optical tracking analysis.

### `ot/image_processing`

This sub-module provides image processing functions using both OpenCV and skimage libraries.

- `equalize`: Applies histogram equalization to an image.
- `minmax`: Normalizes an image by its minimum and maximum values.
- `zscore`: Applies z-score normalization to an image.
- `lognorm`: Applies logarithmic normalization to an image.
- `clahe`: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

### `ot/image_processing/common.py`

This module provides common utility functions used by both `opencv.py` and `ski.py` modules.

- `_np_to_cv2_dtype`: Converts numpy data types to OpenCV data types.
- `_get_cv2_norm_name`: Gets the name of an OpenCV normalization type.
- `_cv2_to_np_dtype`: Converts OpenCV data types to numpy data types.
- `_array_verbose`: Logs detailed information about an array.
- `_tofromimage`: Decorator to handle image conversion.
- `_is_cv8u`: Checks if an array is of type `CV_8U`.
- `_is_multiband`: Checks if an array is multiband.
- `_normalize`: Normalizes an array to a specified range.
- `_overwrite_nodata`: Overwrites NoData values in an array.
- `to_single_band_uint8`: Converts an array to a single-band 8-bit unsigned integer format.

### `ot/image_processing/opencv.py`

This module provides image processing functions using the `OpenCV` library.

- `equalize`: Applies basic histogram equalization to an image.
- `minmax`: Normalizes an image by its minimum and maximum values.
- `zscore`: Applies z-score normalization to an image.
- `lognorm`: Applies logarithmic normalization to an image.
- `clahe`: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

### `ot/image_processing/ski.py`

This module provides image processing functions using the `skimage` library.

- `equalize`: Applies histogram equalization to an image.
- `minmax`: Normalizes an image by its minimum and maximum values.
- `zscore`: Applies z-score normalization to an image.
- `lognorm`: Applies logarithmic normalization to an image.
- `clahe`: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

## Usage

To use the optical tracking and image processing functions, import the desired module and call the functions as needed. For example:

```python
from ot.image_processing import dispatcher

# Load an image as a numpy array
image = ...

# Apply histogram equalization
equalized_image = dispatcher.dispatch_process("skimage_equalize", array=image)
```

# S1 Sub-Module

The `s1` sub-module provides functionalities for processing Sentinel-1 satellite images. It includes various processing steps such as orbit file application, thermal noise removal, calibration, debursting, subsetting, multilooking, speckle filtering, terrain correction, and band calculations.

## Features

- Apply Orbit File
- Thermal Noise Removal
- Calibration
- TOPSAR Deburst
- Subsetting
- Multilooking
- Speckle Filtering
- Terrain Correction
- Band Calculations

## Graphs

The sub-module includes several XML graph files that define the processing workflows for Sentinel-1 images. These graphs can be used with the SNAP software to automate the processing steps.

### `s1_slc_noSpeckleFilter.xml`

This graph processes Sentinel-1 SLC images without applying speckle filtering. The steps include:

- Reading the input file
- Applying the orbit file
- Removing thermal noise
- Calibrating the image
- Debursting the TOPSAR image
- Subsetting the image
- Applying multilooking
- Performing terrain correction
- Writing the output to a GeoTIFF file

### `s1_slc_noSpeckleFilter+band3.xml`

This graph processes Sentinel-1 SLC images without applying speckle filtering and includes band calculations. The steps include:

- Reading the input file
- Applying the orbit file
- Removing thermal noise
- Calibrating the image
- Debursting the TOPSAR image
- Subsetting the image
- Applying multilooking
- Performing terrain correction
- Calculating bands (Sigma0_VH, Sigma0_VV, Sigma0_VV/Sigma0_VH)
- Writing the output to a GeoTIFF file

### `s1_slc_default.xml`

This graph processes Sentinel-1 SLC images with default settings. The steps include:

- Reading the input file
- Applying the orbit file
- Removing thermal noise
- Calibrating the image
- Debursting the TOPSAR image
- Subsetting the image
- Applying multilooking
- Applying speckle filtering
- Performing terrain correction
- Writing the output to a GeoTIFF file

### `s1_slc_default+band3.xml`

This graph processes Sentinel-1 SLC images with default settings and includes band calculations. The steps include:

- Reading the input file
- Applying the orbit file
- Removing thermal noise
- Calibrating the image
- Debursting the TOPSAR image
- Subsetting the image
- Applying multilooking
- Applying speckle filtering
- Performing terrain correction
- Calculating bands (Sigma0_VH, Sigma0_VV, Sigma0_VV/Sigma0_VH)
- Writing the output to a GeoTIFF file

### `s1_grd_default.xml`

This graph processes Sentinel-1 GRD images with default settings. The steps include:

- Reading the input file
- Applying the orbit file
- Removing thermal noise
- Performing terrain correction
- Writing the output to a GeoTIFF file

## Usage

To use the processing graphs, open them in the SNAP software and run the graph with the desired input file. The output will be saved in the specified format.

## Dependencies

- SNAP software

Make sure to install the SNAP software before using the processing graphs.


# COSMO-SkyMed Image Processing

This repository contains various modules for processing COSMO-SkyMed satellite images. The modules are still under development and not yet complete.

## Modules

### `cosmo/utils.py`

This module provides utility functions for handling COSMO-SkyMed data. The available functions include:

- `StrChopper`: A class for chopping strings into smaller chunks.
- `str2dt`: Converts a string to a `datetime` object.
- `_get_attributes`: Retrieves attributes from an HDF5 file.
- `_get_group_names`: Retrieves group names from an HDF5 file.
- `_get_attrs_names`: Retrieves attribute names from an HDF5 file.
- `_exploreh5`: Explores the structure of an HDF5 file.
- `batch_to_image`: Exports a list of HDF5 files to image format.
- `footprints_to_geopandas`: Converts footprints of COSMO-SkyMed files to a GeoDataFrame.

### `cosmo/naming.py`

This module provides classes for parsing COSMO-SkyMed filenames. The available classes include:

- `CSKFilename`: Parses filenames for COSMO-SkyMed I generation.
- `CSGFilename`: Parses filenames for COSMO-SkyMed II generation.

### `cosmo/lib.py`

This module provides classes and enumerations for handling COSMO-SkyMed products. The available classes and enumerations include:

- `Polarization`: Enumeration for polarization types.
- `Orbit`: Enumeration for orbit directions.
- `CosmoProduct`: Enumeration for COSMO-SkyMed product generations.
- `Squint`: Enumeration for squint angles.
- `Product`: Base class for COSMO-SkyMed products.
- `CSKProduct`: Class for handling COSMO-SkyMed I generation products.
- `CSGProduct`: Class for handling COSMO-SkyMed II generation products.
- `CSKFile`: Class for handling COSMO-SkyMed HDF5 files.
- `Pols`: Class for handling polarization data.

### `ot/image_processing`

This sub-module provides image processing functions using both OpenCV and skimage libraries. The available functions include:

- `equalize`: Applies histogram equalization to an image.
- `minmax`: Normalizes an image by its minimum and maximum values.
- `zscore`: Applies z-score normalization to an image.
- `lognorm`: Applies logarithmic normalization to an image.
- `clahe`: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

## Usage

To use the image processing functions, import the desired module and call the functions as needed. For example:

```python
from cosmo.utils import batch_to_image

# List of HDF5 files
h5_files = [...]

# Export to JPEG format
batch_to_image(h5_files, wd='output_directory', format='jpeg')
```

# SNAP GPT module

The `snap_gpt` sub-module provides functionalities for processing Sentinel-1 and COSMO-SkyMed satellite images using the SNAP Graph Processing Tool (GPT). It includes various processing steps such as orbit file application, thermal noise removal, calibration, debursting, subsetting, multilooking, speckle filtering, terrain correction, and band calculations.

## Features

- Apply Orbit File
- Thermal Noise Removal
- Calibration
- TOPSAR Deburst
- Subsetting
- Multilooking
- Speckle Filtering
- Terrain Correction
- Band Calculations

## Graphs

The sub-module includes several XML graph files that define the processing workflows for Sentinel-1 and COSMO-SkyMed images. These graphs can be used with the SNAP software to automate the processing steps.

### `s1_slc_noSpeckleFilter.xml`

This graph processes Sentinel-1 SLC images without applying speckle filtering. The steps include:

- Reading the input file
- Applying the orbit file
- Removing thermal noise
- Calibrating the image
- Debursting the TOPSAR image
- Subsetting the image
- Applying multilooking
- Performing terrain correction
- Writing the output to a GeoTIFF file

### `s1_slc_noSpeckleFilter+band3.xml`

This graph processes Sentinel-1 SLC images without applying speckle filtering and includes band calculations. The steps include:

- Reading the input file
- Applying the orbit file
- Removing thermal noise
- Calibrating the image
- Debursting the TOPSAR image
- Subsetting the image
- Applying multilooking
- Performing terrain correction
- Calculating bands (Sigma0_VH, Sigma0_VV, Sigma0_VV/Sigma0_VH)
- Writing the output to a GeoTIFF file

### `s1_slc_default.xml`

This graph processes Sentinel-1 SLC images with default settings. The steps include:

- Reading the input file
- Applying the orbit file
- Removing thermal noise
- Calibrating the image
- Debursting the TOPSAR image
- Subsetting the image
- Applying multilooking
- Applying speckle filtering
- Performing terrain correction
- Writing the output to a GeoTIFF file

### `s1_slc_default+band3.xml`

This graph processes Sentinel-1 SLC images with default settings and includes band calculations. The steps include:

- Reading the input file
- Applying the orbit file
- Removing thermal noise
- Calibrating the image
- Debursting the TOPSAR image
- Subsetting the image
- Applying multilooking
- Applying speckle filtering
- Performing terrain correction
- Calculating bands (Sigma0_VH, Sigma0_VV, Sigma0_VV/Sigma0_VH)
- Writing the output to a GeoTIFF file

### `s1_grd_default.xml`

This graph processes Sentinel-1 GRD images with default settings. The steps include:

- Reading the input file
- Applying the orbit file
- Removing thermal noise
- Performing terrain correction
- Writing the output to a GeoTIFF file

### `csg_scs-b_default.xml`

This graph processes COSMO-SkyMed SCS-B images with default settings. The steps include:

- Reading the input file
- Calibrating the image
- Subsetting the image
- Applying multilooking
- Converting linear to dB
- Performing terrain correction
- Writing the output to a specified format

## Usage

To use the processing graphs, open them in the SNAP software and run the graph with the desired input file. The output will be saved in the specified format.

# Image Processing Modules

This repository contains various modules for image processing using both OpenCV and skimage libraries. Below is a brief description of each module and its functionality.

## Modules

### `ot/image_processing/ski.py`

This module provides image processing functions using the `skimage` library. The available functions include:

- `equalize`: Applies histogram equalization to an image.
- `minmax`: Normalizes an image by its minimum and maximum values.
- `zscore`: Applies z-score normalization to an image.
- `lognorm`: Applies logarithmic normalization to an image.
- `clahe`: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

### `ot/image_processing/opencv.py`

This module provides image processing functions using the `OpenCV` library. The available functions include:

- `equalize`: Applies basic histogram equalization to an image.
- `minmax`: Normalizes an image by its minimum and maximum values.
- `zscore`: Applies z-score normalization to an image.
- `lognorm`: Applies logarithmic normalization to an image.
- `clahe`: Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

### `ot/image_processing/common.py`

This module provides common utility functions used by both `ski.py` and `opencv.py` modules. The functions include:

- `_np_to_cv2_dtype`: Converts numpy data types to OpenCV data types.
- `_get_cv2_norm_name`: Gets the name of an OpenCV normalization type.
- `_cv2_to_np_dtype`: Converts OpenCV data types to numpy data types.
- `_array_verbose`: Logs detailed information about an array.
- `_tofromimage`: Decorator to handle image conversion.
- `_is_cv8u`: Checks if an array is of type `CV_8U`.
- `_is_multiband`: Checks if an array is multiband.
- `_normalize`: Normalizes an array to a specified range.
- `_overwrite_nodata`: Overwrites NoData values in an array.
- `to_single_band_uint8`: Converts an array to a single-band 8-bit unsigned integer format.

### `ot/image_processing/__init__.py`

This module initializes the image processing dispatcher and registers the functions from `ski.py` and `opencv.py` modules. The dispatcher allows for easy access to the image processing functions by their names.

## Usage

To use the image processing functions, import the desired module and call the functions as needed. For example:

```python
from ot.image_processing import ski

# Load an image as a numpy array
image = ...

# Apply histogram equalization
equalized_image = ski.equalize(array=image)
```

# Log module

This sub-module provides logging functionality for the project. It is designed to create log files and console outputs for debugging and monitoring purposes.

## Features

- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console logging
- Timed rotating file logging (daily rotation)
- Customizable log format

## Usage

To use the logging functionality, import the `setup_logger` function and create a logger instance:

```python
from log import setup_logger

logger = setup_logger(__name__)
```

## Configuration

The logging configuration is defined in the `log/__init__.py` file. The main configurations include:

- `LOG_FORMAT`: The format of the log messages.
- `LOG_LEVEL`: The logging level (default is `DEBUG`).
- `LOG_FILE`: The name of the log file.
- `LOG_DIR`: The directory where log files are stored.


# OT Main Script

The `main.py` file is the main entry point for running the optical tracking analysis in the `ot` sub-module. It provides a command-line interface (CLI) for configuring and executing various optical tracking algorithms on input images.

## Features

- Command-line interface for configuring optical tracking analysis
- Support for multiple optical tracking algorithms (OpenCV, scikit-image)
- Image preprocessing options
- Coregistration of images
- Output in various formats (GeoTIFF, GeoPackage, Shapefile)

## Command-Line Interface (CLI)

The CLI allows users to specify various parameters for the optical tracking analysis. Below is a description of the available options:

### Common Parameters

- `-ot`, `--algname`: Name of the algorithm to use (e.g., `OPENCVOF`, `SKIOFILK`, `SKIOFTVL1`, `SKIPCCV`).
- `-r`, `--reference`: Path to the reference image.
- `-t`, `--target`: Path to the target image.
- `-o`, `--output`: Path to the output file (default: `output.tif`).
- `-b`, `--band`: Band to use (if not specified, all available bands will be used).
- `--nodata`: NoData value (default: None).
- `-prep`, `--preprocessing`: Preprocessing method to apply (default: `equalize`).
- `--out_format`: Output format (default: None).

### OpenCV Parameters

- `--flow`: Initial flow guess (requires `--flags` set to 4).
- `--levels`: Number of pyramid layers (default: 4).
- `--pyr_scale`: Image scale to build pyramids (default: 0.5).
- `--winsize`: Averaging window size (default: 4).
- `--step_size`: Sampling interval (default: 1).
- `--iterations`: Number of iterations at each pyramid level (default: 10).
- `--poly_n`: Size of the pixel neighborhood for polynomial expansion (default: 5).
- `--poly_sigma`: Standard deviation of the Gaussian for smoothing derivatives (default: 1.1).
- `--flags`: Optional flags (e.g., 4 for initial flow, 256 for Gaussian filter).

### Scikit-Image Parameters

- `--radius`: Radius of the window (default: 4).
- `--num_warp`: Number of warping iterations (default: 3).
- `--gaussian`: Apply Gaussian filter (default: False).
- `--prefilter`: Apply pre-filtering (default: False).
- `--attachment`: Attachment parameter (default: 10).
- `--tightness`: Tightness parameter (default: 0.3).
- `--num_iter`: Number of iterations (default: 10).
- `--tol`: Tolerance for convergence (default: 1e-4).
- `--phase_norm`: Apply phase normalization in cross-correlation (default: True).
- `--upsmp_fac`: Upsampling factor for sub-pixel accuracy (default: 1.0).

## Usage

To run the optical tracking analysis, use the following command:

```sh
python main.py -ot <algorithm_name> -r <reference_image> -t <target_image> -o <output_file> [options]
```

### Example

```sh
python main.py -ot OPENCVOF -r reference.tif -t target.tif -o output.tif --levels 3 --winsize 5
```

This command runs the optical tracking analysis using the OpenCV optical flow algorithm with the specified parameters.
