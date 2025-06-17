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
  Implements classes and functions for calculating optical flow and phase cross-correlation between images. Provides wrappers for OpenCV and scikit-image algorithms, along with utilities for converting results to DataFrame or GeoDataFrame.
- `cli.py`  
  Implements the Command Line Interface for launching optical flow processes, normalization, and other operations directly from the terminal.
- `helpmsg.py`  
  Contains help messages and text documentation for CLI and main module functions.
- `interfaces.py`  
  Defines classes for image representation, band management, and abstract interfaces for tracking algorithms.
- `lib.py`  
  Support functions and common utilities for image manipulation, format conversions, and recurring mathematical operations.
- `methods.py`  
  Factory for creating instances of optical flow algorithms. Allows dynamic selection and configuration of the desired algorithm via name or parameters.
- `opencvof.py`  
  Implementation of OpenCV-based optical flow algorithms (e.g., Farneback, Lucas-Kanade). Enables detailed parameter configuration and integration with processing pipelines.
- `skiofilk.py`  
  Implementation of the ILK (Iterative Lucas-Kanade) algorithm via scikit-image.
- `skioftvl1.py`  
  Implementation of the TV-L1 (Total Variation L1) algorithm via scikit-image, robust for noisy images and intensity variations.
- `skipccv.py`  
  Implementation of Phase Cross-Correlation (PCC Vector) for sub-pixel displacement estimation between images. Unlike other algorithms that return displacement maps in raster format, `skipccv` will return a georeferenced vector file where each point represents the center of the search window, associated with displacements in the two main directions (fields `RSHIFT` and `CSHIFT`), the resulting displacement (`L2`), and the normalized root mean square deviation between analyzed moving windows (`NRMS`).
- `image_processing/`  
  Sub-package with modules for normalization, equalization, conversion, and advanced manipulation of raster images and arrays.

### Main Features
- Abstract interfaces for multi-band image management
- Band and image normalization and transformation
- Image equalization and conversion for analysis
- Optical flow calculation between raster images (OpenCV, scikit-image)
- Support for command-line execution (CLI)

### Usage Example
```python
from sensetrack.ot.metodi import get_algorithm
from sensetrack.ot.interfaces import Image
from sensetrack.lib import (rasterio_open,
                            image_to_rasterio,
                            basic_pixel_coregistration)
from sensetrack.ot.algoritmi import OpenCVOpticalFlow

# Align target image pixels with reference image
basic_pixel_coregistration(infile = "tar.tif", match = "ref.tif",
                           outfile = "tar_coreg.tif")
ref = Image(**rasterio_open("ref.tif", band = 1), nodata = -9999.)
tar = Image(**rasterio_open("tar_coreg.tif", band = 1), nodata = -9999.)
# Optical flow (Gunnar Farneback)
OT = OpenCVOpticalFlow.from_dict({"pyr_scale":0.5, "levels":4, "winsize":16})
result = OT(reference=ref.get_band("B1"), target=tar.get_band("B1"))
image_to_rasterio(result, "output.tif")
```

### CLI
To launch the same analysis from terminal (coregistration is still performed):
```powershell
python -m sensetrack.ot.cli --algname OPENCVOF
  --reference ref.tif --target tar.tif 
  --output output.tif --winsize 16 --levels 4
```

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
- **Multi-sensor support**: predefined workflows for different sensors (S1, COSMO, S2) via XML graph files.
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
The `sensetrack.sentinel` subpackage provides tools and classes for preprocessing SAR data from the Sentinel-1 satellite. It enables automated management of processing workflows, manifest reading and manipulation, and integration with broader analysis pipelines.

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

# Create a preprocessor specifying AOI and workflow
preprocessor = S1Preprocessor(subset="path/to/aoi.shp",
                              process="S1_SLC_DEFAULT") 
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
- `convert.py`
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
from sensetrack.prs.convert import get_prisma_info, get_prisma_image

# Extract info from a PRISMA file
get_prisma_info('path/to/file.h5')

# Convert SWIR band to GeoTIFF
get_prisma_image('path/to/file.h5', datacube='swir', band=23)
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
