# SenseTrack

## Index
- [1. Offset-Tracking Submodule (ot)](#offset-tracking-submodule-ot)
- [2. SNAP-GPT Submodule (snap_gpt)](#snap-gpt-submodule-snap_gpt)
- [3. Sentinel-1 SAR Preprocessing Submodule (sentinel)](#sentinel-1-sar-preprocessing-submodule-sentinel)
- [4. COSMO-SkyMed Submodule (cosmo)](#cosmo-skymed-submodule-cosmo)
- [5. PRISMA Post-Processing and Conversion Submodule (prisma)](#prisma-post-processing-and-conversion-submodule-prisma)

---

## Offset-Tracking Submodule (ot)

### Overview
The `sensetrack.ot` subpackage provides core functionalities for optical flow analysis, image normalization, interface management, and CLI for offset tracking. It is designed to work with satellite images and raster data, offering advanced algorithms and support tools for research and operational applications.

### Module Structure
- `algorithms.py`  
  Implements classes and functions for calculating optical flow and phase cross-correlation between images. Provides wrappers for OpenCV and scikit-image algorithms.
- `cli.py`  
  Implements the Command Line Interface for launching optical flow processes, normalization, and other operations directly from the terminal.
- `helpmsg.py`  
  Contains help messages and text documentation for CLI and main module functions.
- `interfaces.py`  
  Defines classes for image representation, band management, and abstract interfaces for tracking algorithms.
- `lib.py`  
  Support functions and common utilities for image manipulation, format conversions, and recurring mathematical operations.
- `opencvof.py`  
  The `algorithms.OpenCVOpticalFlow` algorithm provides a Python interface to the Farneback dense optical flow method, as implemented in OpenCV’s `calcOpticalFlowFarneback` function. This approach estimates the motion field between two images by analyzing the apparent movement of pixel intensities, producing a dense displacement vector for every pixel. The core of the algorithm relies on constructing image pyramids, which allow it to capture both large and small displacements by progressively analyzing the images at multiple scales. At each level, the algorithm models local neighborhoods with polynomial expansions, enabling it to robustly estimate motion even in the presence of noise or textureless regions. The flexibility of the implementation allows users to fine-tune parameters such as the pyramid scale, window size, number of iterations, and the degree of smoothing, thus balancing accuracy and computational efficiency. After computing the flow, the results are transformed into images representing the horizontal and vertical components of the displacement, as well as the overall magnitude
- `skiofilk.py`
  The `algorithms.SkiOpticalFlowILK` algorithm offers a Python interface to the Inverse Lucas-Kanade (ILK) method for dense optical flow estimation, as implemented in scikit-image’s `optical_flow_ilk` function. This approach is designed to estimate the pixel-wise motion between two images by analyzing local intensity variations and tracking how small neighborhoods shift from the reference to the target image. The ILK method operates by minimizing the difference between the reference and the warped target image, iteratively refining the displacement field to achieve the best alignment. It is particularly well-suited for scenarios where the motion is relatively small and smooth, as it assumes that the displacement within each local window can be approximated linearly. The algorithm allows for customization of parameters such as the radius of the local window, the number of warping iterations, and the use of Gaussian smoothing or prefiltering, enabling users to adapt the method to different noise levels and image characteristics. After computing the displacement vectors, the results are transformed according to the affine properties of the target image, producing output images that represent the horizontal and vertical components of the motion, as well as the overall displacement magnitude
- `skioftvl1.py`
  The `algorithms.SkiOpticalFlowTVL1` algorithm provides a Python interface to the TV-L1 optical flow method, as implemented in scikit-image’s `optical_flow_tvl1` function. This approach is based on a variational framework that seeks to estimate the dense motion field between two images by minimizing an energy functional composed of a data attachment term and a regularization term. The TV-L1 method is particularly robust to noise and outliers, thanks to its use of the L1 norm for the data term and total variation (TV) regularization, which encourages piecewise-smooth motion fields while preserving sharp motion boundaries. The algorithm iteratively refines the displacement field through a multi-scale, coarse-to-fine strategy, allowing it to capture both large and small motions. Users can adjust parameters such as the strength of the data and regularization terms, the number of warping and optimization iterations, and the use of prefiltering, making the method adaptable to a wide range of imaging conditions. After the optical flow is computed, the results are mapped to the affine space of the target image, producing output images for the horizontal and vertical components of the displacement, as well as the overall magnitude  
- `skipccv.py`
  The `algorithms.SkiPCC_Vector` algorithm implements a phase cross-correlation (PCC) approach for estimating local displacements between two images, leveraging the `phase_cross_correlation` function from scikit-image. Unlike traditional optical flow methods that rely on intensity gradients, this technique operates in the frequency domain. Since the base function `phase_cross_correlation` outputs a single displacement for two input arrays, this implementation provides an utility for splitting the two images into several sub-arrays in a rolling-window fashion (see the `stepped_rolling_window` help for further details), than `phase_cross_correlation` is performed for each pair of windows, and the results are collected in a dataframe-like structure where each record is associated with displacements in the two directions (fields `RSHIFT` and `CSHIFT` for row and column displacement respectively), the resultant displacement (`L2`), and the normalized root mean square deviation between analyzed moving windows (`NRMS`). By using phase normalization, the method enhances its sensitivity to translational differences while suppressing the influence of amplitude variations. The process can be further refined by adjusting the window size, step size, and upsampling factor, allowing for subpixel accuracy in the displacement estimates.
- `image_processing/`  
  Sub-package with modules for normalization, equalization, conversion, and advanced manipulation of raster images and arrays.
  It is developed with the dispatcher pattern in order to be further extended with new features.
  Each processing function in implemented specifically for each library imployed by `sensetrack`, so that, for instance, the clahe algortihm is implemented in both `image_processing.opencv.py` and `image_processing.ski.py` sub-modules. The correct function can be dinamically called by using the `dispacther.dispatch_process`. 

### Main Features
- Abstract interfaces for multi-band image management
- Band and image normalization and transformation
- Image equalization and conversion for analysis
- Optical flow calculation between raster images (OpenCV, scikit-image)
- Support for command-line execution (CLI)

### Usage Example
```python
from sensetrack.ot.interfaces import Image
from sensetrack.lib import (rasterio_open,
                            image_to_rasterio,
                            basic_pixel_coregistration)
from sensetrack.ot.algorithms import OpenCVOpticalFlow

# Get reference image
ref = Image(*rasterio_open("ref.tif", band = 1), nodata = -9999.)
# Align target image pixels with reference image
tar = basic_pixel_coregistration(infile = "tar.tif", match = "ref.tif")
# Optical flow (Gunnar Farneback)
OT = OpenCVOpticalFlow.from_dict({"pyr_scale":0.5, "levels":4, "winsize":16})
result = OT(reference=ref, target=tar)
# Output: x-displacements
image_to_rasterio(result['dxx'], "output-x.tif")
# Output: y-displacements
image_to_rasterio(result['dyy'], "output-y.tif")
# Output: resultant displacements
image_to_rasterio(result['res'], "output-res.tif")
```

### CLI
To launch the same analysis from terminal (coregistration is performed by default):
```powershell
python -m sensetrack.ot.opencvof --reference ref.tif --target tar.tif 
  --output output.tif --winsize 16 --levels 4
```
By default it outputs the resultant displacements raster, to change this behaviour
pass the `--resultant_displacement False` as input

### Implemented Algorithms

#### OpenCV Optical Flow (Farneback, Lucas-Kanade)
- Implemented in `opencvof.py`.
- Classical algorithms for dense and sparse optical flow.
- Configurable parameters: `pyr_scale`, `levels`, `winsize`, `iterations`, `poly_n`, `poly_sigma`, `flags`.
- Suitable for images with moderate intensity variations and continuous movements.

#### Scikit-Image ILK (Iterative Lucas-Kanade)
- Implemented in `skiofilk.py`.
- Iterative algorithm for optical flow, robust on scientific and remote sensing images.
- Parameters: `radius`, `num_warp`, `gaussian`, `prefilter`.
- Indicated for small displacements and low-noise images.

#### Scikit-Image TV-L1 (Total Variation L1)
- Implemented in `skioftvl1.py`.
- Robust algorithm for optical flow on noisy images or with intensity variations.
- Parameters: `attachment`, `tightness`, `num_warp`, `num_iter`, `tol`, `prefilter`.
- Ideal for SAR data and images with discontinuities.

#### Phase Cross-Correlation (PCC)
- Implemented in `skipccv.py`.
- Algorithm for sub-pixel displacement estimation via phase cross-correlation.
- Parameters: `winsize`, `step_size`, `phase_norm`, `upsmp_fac`.
- Useful for precise offset measurements between aligned images.

### Notes
- All algorithms are configurable via Python parameters or CLI.
- The module is designed to be extensible: new algorithms can be added by implementing the provided interfaces.
- Output can be saved in various formats, including raster and DataFrame.
---

## SNAP-GPT Submodule (snap_gpt)

### Overview
The `sensetrack.snap_gpt` subpackage provides tools and workflows for SAR (Synthetic Aperture Radar) data preprocessing through integration with ESA's SNAP-GPT (Graph Processing Tool). It enables automated management of processing graphs, definition of areas of interest (AOI), parameter configuration, and batch execution of SAR processes.

### Module Structure
- `lib.py`
  - Functions and classes for managing SNAP-GPT workflows, including creation, modification, and execution of processing graphs.
  - Management of SNAP-GPT calls from Python, log parsing, and error handling.
  - Utilities for command generation, temporary file management, and result verification.

### Main Features
- **SNAP-GPT workflow management**: automates the creation and execution of SNAP graphs for SAR preprocessing (e.g., Sentinel-1, COSMO-SkyMed).
- **Flexible configuration**: parameters, paths, and AOIs are managed through YAML and GeoPackage files.
- **Multi-platform support**: predefined workflows for different platforms (Sentinel-1/2, COSMO-SkyMed) via XML graph files.
- **Batch processing**: execution of processes on multiple SAR files automatically and repeatably.
- **AOI management**: selection and application of areas of interest for image subsetting.
- **Logging and error handling**: SNAP-GPT log parsing, exception handling, and reporting.

### Usage Example
```python
import geopandas as gpd
from sensetrack.snap_gpt.lib import SARPreprocessing, AOI

# Select area of interest and process
subset = gpd.read_file("./aoi.shp")
process = SARPreprocessing.S1_SLC_DEFAULT
sarfile = "path/to/sarfile.zip"

# Create and launch preprocessing
preproc = SARPreprocessing(subset, process)
preproc.run(sarfile)
```

### Description of SNAP-GPT XML Graphs

Below is a summary of the main nodes present in the XML workflows provided with the module:

#### `s1_slc_default.xml`
- **Read**: Reading of SLC input file.
- **ApplyOrbit**: Application of precise orbit.
- **ThermalNoiseRemoval**: Thermal noise removal.
- **Calibration**: Radiometric calibration.
- **TOPSAR-Deburst**: Burst image reconstruction.
- **Multilook**: Spatial resolution reduction via multilooking.
- **Speckle-Filter**: Filter for speckle noise reduction.
- **TerrainCorrection**: Orthorectification and topographic correction.
- **Subset**: Extraction of a region of interest.
- **Write**: Writing result to GeoTIFF file.

#### `s1_slc_default_noSF.xml`
- Same as above, but without speckle filter.

#### `s1_slc_default+b3.xml`
- Same as `s1_slc_default.xml` up to TerrainCorrection.
- **BandMaths**: Calculation of additional band (e.g., polarization ratio).
- **BandMerge**: Merging original and calculated bands.
- **Write**: Final output.

#### `s1_slc_default+b3noSF.xml`
- Same as `s1_slc_default+b3.xml` but without speckle filter.

#### `s1_grd_default.xml`
- **Read**: Reading GRD product.
- **ApplyOrbit**: Orbit application.
- **ThermalNoiseRemoval**: Thermal noise removal.
- **TerrainCorrection**: Orthorectification.
- **Write**: Final output.

#### `s2_l2a_default.xml`
- **Read**: Reading Sentinel-2 product.
- **Resample**: Resampling to 10m.
- **BandSelect**: Selection of main bands.
- **Subset**: AOI extraction.
- **Write**: Final output.

#### `cosmo_scs-b_default.xml`
- **Read**: Reading Cosmo-SkyMed product.
- **LinearToFromdB**: Conversion between linear and dB scale.
- **Multilook**: Resolution reduction (automatically estimated parameters).
- **Terrain-Correction**: Orthorectification.
- **Subset**: AOI extraction with optional reprojection.
- **Write**: Final output.

### Notes
- SNAP graphs (XML files) are customizable and can be extended for new workflows.
- The module is designed to be integrated into broader SAR data processing pipelines.
- SNAP-GPT usage requires SNAP to be installed and accessible from the system.

---

## Sentinel-1 SAR Preprocessing Submodule (sentinel)

### Overview
The `sensetrack.sentinel` subpackage provides tools and classes for preprocessing SAR data from Sentinel-1 mission. It enables automated management of processing workflows, manifest reading and manipulation, and integration with broader analysis pipelines.

### Module Structure
- `preprocessing.py`
  - Main class: `S1Preprocessor`
  - Manages the entire preprocessing workflow for Sentinel-1 SLC and GRD data.
  - Allows parameter configuration, AOI selection, and SNAP-GPT integration.
  - Supports batch execution on multiple SAR files.
- `manifest_file.py`
  - Functions and classes for reading, parsing, and manipulating Sentinel-1 XML manifest files.
  - Extracts metadata, acquisition parameters, geometries, and orbit information useful for preprocessing and analysis.

### Main Features
- **Sentinel-1 SAR Preprocessing**: automates the processing chain (calibration, orthorectification, filtering, AOI subset, etc.).
- **Manifest Management**: advanced parsing of XML manifest files to extract metadata and scene parameters.
- **Flexible Configuration**: parameters and workflows customizable via configuration files and Python classes.
- **SNAP-GPT Integration**: use of XML graphs to launch SNAP processes directly from Python.
- **Batch Support**: processing of multiple scenes in automatic sequence.

### Usage Example
```python
from sensetrack.s1.preprocessing import S1Preprocessor
from sensetrack.snap_gpt.lib import GPTSubsetter

aoi = GPTSubsetter.get_subset("path/to/aoi.shp")

# Create a preprocessor specifying AOI and workflow
preprocessor = S1Preprocessor(subset=aoi, process="S1_SLC_DEFAULT") 

# Execute preprocessing
preprocessor.run("path/to/sarfile.zip")
```

### Notes
- The module requires SNAP to be installed and accessible from the system.
- Workflows can be extended to include additional processing steps (e.g., speckle filtering, coregistration, etc.).
- Manifest parsing enables automated parameter selection and data quality verification.

---

## COSMO-SkyMed Submodule (cosmo)

### Overview
The `sensetrack.cosmo` subpackage provides advanced tools for managing, preprocessing, and analyzing SAR data from first (CSK) and second (CSG) generation COSMO-SkyMed satellites. It includes classes for metadata decoding, HDF5 file manipulation, footprint extraction, quicklook conversion, and SNAP-GPT workflow integration.

### Module Structure
- `lib.py`: 
  - Defines main classes for representing and parsing COSMO-SkyMed products (CSK/CSG), including enumerations for polarization, orbit, product type, and squint.
  - Classes:
    - `CSKProduct` and `CSGProduct`: advanced file name parsing, metadata extraction, naming convention management.
    - `CSKFile`: extends `h5py.File` to provide direct access to quicklook, bands, footprint, and metadata.
    - `Pols`: specialized dictionary for polarization management.
- `utils.py`: 
  - Utility functions for:
    - HDF5 file parsing and manipulation (reading attributes, structure exploration, shape and footprint extraction).
    - Quicklook conversion to images, batch export, footprint GeoDataFrame creation.
    - Mean incidence angle calculation.
- `preprocessing.py`:
  - Classes `CSKPreprocessor` and `CSGPreprocessor`: manage CSK/CSG product preprocessing via SNAP-GPT workflows, with automatic multilook parameter estimation, AOI selection, custom projection (CRS), and georeferenced output generation.
  - Unified CLI for processing CSK/CSG HDF5 files specifying product type, workflow, and AOI.

### Main Features
- Advanced parsing of CSK/CSG file names and decoding of main metadata.
- Easy access to quicklook, bands, and metadata through Python classes.
- Extraction and conversion of footprints to Shapely geometries and GeoDataFrame.
- Calculation and conversion of quicklooks to images (JPEG, PNG, etc.).
- Automatic preprocessing via SNAP-GPT with support for AOI, estimated multilook parameters, and custom projection.
- Batch processing utilities and integration in analysis pipelines.

### Usage Example
```python
from sensetrack.cosmo.lib import CSKFile
from sensetrack.cosmo.utils import csk_footprint
from sensetrack.cosmo.preprocessing import CSKPreprocessor

# Load a CSK HDF5 file
csk = CSKFile('path/to/file.h5')
print(csk)

# Extract footprint as polygon
footprint = csk.footprint_polygon

# Export quicklook as image
csk.qlk_to_image.save('quicklook.jpg')

# Automatic preprocessing (example)
preproc = CSKPreprocessor(subset, process)
preproc.run('path/to/file.h5', CRS="EPSG:32632")
```

### Command Line Execution
```powershell
python -m sensetrack.cosmo.preprocessing --product_type CSK
  --file path/to/file.h5 --workflow CSK_HIMAGE_SLC --aoi path/to/aoi.shp
```

### Notes
- The module requires libraries: `h5py`, `numpy`, `shapely`, `Pillow`, `geopandas`.
- For preprocessing via SNAP-GPT, SNAP must be installed and accessible from the system.
- Classes and functions are designed to be integrated into multi-sensor SAR processing pipelines.

---

## PRISMA Post-Processing and Conversion Submodule (prisma)

### Overview
The `sensetrack.prisma` subpackage provides tools for converting PRISMA data, facilitating integration with GIS software and analysis pipelines.

### Module Structure
- `extract.py`
  - Functions for:
    - Reading PRISMA HDF5 files.
    - Extracting product metadata and information (`get_prisma_info`).
    - Extracting PRISMA bands (pan, swir, vnir) to `Image` objects (`get_prisma_image`).
    - Parsing command line arguments for GeoTiff conversion and information extraction.

### Main Features
- **Multi-band support**: processing of pan, SWIR, and VNIR bands
- **Georeferencing**: maintenance of spatial information
- **Batch processing**: processing of multiple PRISMA files
- **PRISMA metadata extraction**: prints key information (product ID, processing level, cloud coverage, EPSG, bandwidth data) from PRISMA HDF5 files.
- **GeoTIFF conversion**: exports panchromatic, SWIR, and VNIR bands from PRISMA HDF5 to georeferenced GeoTIFF files, maintaining projection and bounding box.
- **CLI**: allows launching conversions and info directly from terminal with dedicated options.

### Usage Example
#### From Python
```python
from sensetrack.ot.lib import image_to_rasterio
from sensetrack.prs.extract import get_prisma_info, get_prisma_image

# Extract info from a PRISMA file
get_prisma_info('path/to/file.h5')

# Convert SWIR band to GeoTIFF
swir_23 = get_prisma_image('path/to/file.h5', datacube='swir', band=23)
image_to_rasterio(swir_23, "swir_23.tif")

# Convert PAN band to GeoTIFF
pan = get_prisma_image('path/to/file.h5', datacube='pan')
image_to_rasterio(pan, "pan.tif")
```

#### From Terminal
```powershell
python sensetrack/prs/convert.py -f path/to/file.h5 --datacube vnir --band 34
python sensetrack/prs/convert.py -f path/to/file.h5 --datacube swir --band 125
```

### Notes
- Supported bands for conversion are: 'pan', 'swir', 'vnir'.
- GeoTIFF output maintains georeferencing and EPSG projection of the PRISMA product.
- The module can be extended to support other formats or post-processing functions.

For details on parameters and available options, consult the inline documentation in the source files.
