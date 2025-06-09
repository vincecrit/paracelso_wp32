"""
SNAP Graph Processing Tool (GPT) library for SAR data processing.

This module provides classes and utilities for preprocessing SAR data using SNAP GPT,
including multilook parameter estimation, subsetting, and workflow execution for
different SAR sensors (Sentinel-1, COSMO-SkyMed, CSG).

Classes:
    - SARResolutions: Abstract base class for SAR sensor resolutions
    - S1_IW_SLC, CSK_HIMAGE_SLC, CSG_HIMAGE_SLC: Concrete resolution classes
    - SARPreprocessing: Enum of available preprocessing workflows
    - Graphs: Enum mapping workflows to XML graph files
    - GPTSubsetter: Utility for extracting subset areas
    - SARPreprocessor: Abstract base class for SAR preprocessing
"""
"""
SNAP Graph Processing Tool (GPT) library for SAR data processing.

This module provides classes and utilities for preprocessing SAR data using SNAP GPT,
including multilook parameter estimation, subsetting, and workflow execution for
different SAR sensors (Sentinel-1, COSMO-SkyMed, CSG).

Classes:
    - SARResolutions: Abstract base class for SAR sensor resolutions
    - S1_IW_SLC, CSK_HIMAGE_SLC, CSG_HIMAGE_SLC: Concrete resolution classes
    - SARPreprocessing: Enum of available preprocessing workflows
    - Graphs: Enum mapping workflows to XML graph files
    - GPTSubsetter: Utility for extracting subset areas
    - SARPreprocessor: Abstract base class for SAR preprocessing
"""
import math
from abc import ABC
from collections import namedtuple
from enum import Enum, unique
from pathlib import Path
from pathlib import Path

import geopandas as gpd

from sensetrack import GRAPHS_WD
from sensetrack.cosmo.utils import (csg_mean_incidence_angle_rad,
                                    csk_mean_incidence_angle_rad)
from sensetrack.log import setup_logger
from sensetrack.sentinel.lib import s1_mean_incidence_angle_rad

logger = setup_logger(__name__)


MultiLook = namedtuple("MultiLook", ["Num_Range_LOOKS", "Num_Azimuth_LOOKS",
                       "Estimated_RangeResolution", "Estimated_AzimuthResolution"])
"""
Named tuple for multilook parameters and resulting resolutions.

Fields:
    Num_Range_LOOKS (int): Number of looks in range direction
    Num_Azimuth_LOOKS (int): Number of looks in azimuth direction
    Estimated_RangeResolution (float): Resulting ground range resolution in meters
    Estimated_AzimuthResolution (float): Resulting azimuth resolution in meters
"""
"""
Named tuple for multilook parameters and resulting resolutions.

Fields:
    Num_Range_LOOKS (int): Number of looks in range direction
    Num_Azimuth_LOOKS (int): Number of looks in azimuth direction
    Estimated_RangeResolution (float): Resulting ground range resolution in meters
    Estimated_AzimuthResolution (float): Resulting azimuth resolution in meters
"""


class SARResolutions(ABC):
    """
    Abstract base class defining SAR sensor resolution parameters.

    Attributes:
        RRES (float): Range resolution in meters
        ARES (float): Azimuth resolution in meters
    """
    """
    Abstract base class defining SAR sensor resolution parameters.

    Attributes:
        RRES (float): Range resolution in meters
        ARES (float): Azimuth resolution in meters
    """
    RRES = ...  # RANGE RESOLUTION
    ARES = ...  # AZIMUTH RESOLUTION


class S1_IW_SLC(SARResolutions):
    """Sentinel-1 IW SLC product resolutions."""
    """Sentinel-1 IW SLC product resolutions."""
    RRES = 2.30  # RANGE RESOLUTION
    ARES = 14.10  # AZIMUTH RESOLUTION


class CSK_HIMAGE_SLC(SARResolutions):
    """COSMO-SkyMed HIMAGE SLC product resolutions."""
    """COSMO-SkyMed HIMAGE SLC product resolutions."""
    RRES = 3.0  # RANGE RESOLUTION (AFAIK)
    ARES = 3.0  # AZIMUTH RESOLUTION (AFAIK)


class CSG_HIMAGE_SLC(SARResolutions):
    """COSMO-SkyMed Second Generation HIMAGE SLC product resolutions."""
    """COSMO-SkyMed Second Generation HIMAGE SLC product resolutions."""
    RRES = 2.6488857702529085  # RANGE RESOLUTION (AFAIK)
    ARES = 2.6488857702529085  # AZIMUTH RESOLUTION (AFAIK)


@unique
class SARPreprocessing(Enum):
    """
    Enumeration mapping preprocessing workflows to their XML graph files.

    Each member corresponds to a specific XML file in the GRAPHS_WD directory
    that defines the processing graph for SNAP GPT.
    """
    """
    Enumeration mapping preprocessing workflows to their XML graph files.

    Each member corresponds to a specific XML file in the GRAPHS_WD directory
    that defines the processing graph for SNAP GPT.
    """
    S2_L2A_DFLT = GRAPHS_WD / "s2_l2a_default.xml"
    S1_IW_GRD_DFLT = GRAPHS_WD / "s1_grd_default.xml"
    S1_IW_SLC_DFLT = GRAPHS_WD / "s1_slc_default.xml"
    S1_IW_SLC_DFLT_B3 = GRAPHS_WD / "s1_slc_default+b3.xml"
    S1_IW_SLC_DFLT_NOSF = GRAPHS_WD / "s1_slc_default_noSF.xml"
    S1_IW_SLC_DFLT_B3_NOSF = GRAPHS_WD / "s1_slc_default+b3noSF.xml"
    COSMO_HIMAGE_SLCB_DFLT = GRAPHS_WD / "cosmo_scs-b_default.xml"

    @classmethod
    def validate(cls, name: str) -> Exception | None:
        members = [member.name for member in cls]
        graph_path = Path(eval(f"cls.{name.upper()}").value)

        if name not in members:
            raise ValueError(f"Unrecognized graph name {name}.")
        elif not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}.")
        else:
            return graph_path


Subset = namedtuple("Subset", ["name", "geometry"])
"""
Named tuple representing a geographic subset for processing.

Fields:
    name (str): Name of the subset (e.g., layer name from vector file)
    geometry: Geometry object defining the subset area
"""
"""
Named tuple representing a geographic subset for processing.

Fields:
    name (str): Name of the subset (e.g., layer name from vector file)
    geometry: Geometry object defining the subset area
"""


class GPTSubsetter:
    """
    Utility class for extracting subset areas for SNAP GPT graphs.

    This class handles the loading and preparation of geographic subsets
    from various vector formats (Shapefile, GeoPackage) for use in
    SNAP GPT processing graphs.
    Utility class for extracting subset areas for SNAP GPT graphs.

    This class handles the loading and preparation of geographic subsets
    from various vector formats (Shapefile, GeoPackage) for use in
    SNAP GPT processing graphs.
    """

    @classmethod
    def get_subset(self, aoi: str) -> Subset:
        """
        Create a Subset from a vector file.

        Args:
            aoi (str): Path to vector file, optionally with layer name
                      Format: "/path/to/file.gpkg|layername" or "/path/to/file.shp"

        Returns:
            Subset: Named tuple with subset name and geometry

        Raises:
            ValueError: If layer name is empty when using layered format
            Various geopandas errors if file cannot be read
        """
        """
        Create a Subset from a vector file.

        Args:
            aoi (str): Path to vector file, optionally with layer name
                      Format: "/path/to/file.gpkg|layername" or "/path/to/file.shp"

        Returns:
            Subset: Named tuple with subset name and geometry

        Raises:
            ValueError: If layer name is empty when using layered format
            Various geopandas errors if file cannot be read
        """
        # Caso di shapefile (o gpkg) qualunque
        # "/percorso/a/file.gpkg|layername" oppure "/percorso/a/shapefile.shp"
        if "|" in aoi:
            file, layername = aoi.split("|")
            if not layername:
                raise ValueError("Layer name cannot be empty")

        else:
            file, layername = aoi, None

        geometry = gpd.read_file(file, layer=layername).to_crs(
            "EPSG:4326").geometry[0]
        
        return Subset(layername, geometry)


class SARPreprocessor(ABC):
    """
    Abstract base class for SAR data preprocessing.

    This class provides common functionality for preprocessing SAR data,
    including multilook parameter estimation and graph execution.
    """

    def __init__(self, SUBSET: GPTSubsetter, PROCESS: str) -> None:
        """
        Initialize the preprocessor.

        Args:
            SUBSET (GPTSubsetter): Subsetter object defining the area of interest
            PROCESS (SARPreprocessing): Preprocessing workflow to apply

        Raises:
            ValueError: If PROCESS do not match a recognized graph file
            FileNotFoundError: If the graph file is not found
        """

        self.GRAPH = SARPreprocessing.validate(PROCESS)
        self.SUBSET = SUBSET
        self.PROCESS = PROCESS

    def estimate_multilook_parms(self, filename: str,
                                 native_resolution: SARResolutions,
                                 n_az_looks: int = 1):
        """
        Estimate optimal multilook parameters for a SAR product.

        This method calculates the number of range looks needed to achieve
        approximately square pixels, given the native sensor resolution and
        desired number of azimuth looks.

        Args:
            filename (str): Path to SAR product file
            native_resolution (SARResolutions): Class defining sensor resolutions
            n_az_looks (int, optional): Number of azimuth looks. Defaults to 1

        Returns:
            MultiLook: Named tuple containing:
                - Number of range and azimuth looks
                - Estimated ground range and azimuth resolutions

        Raises:
            NotImplementedError: If the sensor type is not supported
        """
        """
        Estimate optimal multilook parameters for a SAR product.

        This method calculates the number of range looks needed to achieve
        approximately square pixels, given the native sensor resolution and
        desired number of azimuth looks.

        Args:
            filename (str): Path to SAR product file
            native_resolution (SARResolutions): Class defining sensor resolutions
            n_az_looks (int, optional): Number of azimuth looks. Defaults to 1

        Returns:
            MultiLook: Named tuple containing:
                - Number of range and azimuth looks
                - Estimated ground range and azimuth resolutions

        Raises:
            NotImplementedError: If the sensor type is not supported
        """
        if isinstance(native_resolution, S1_IW_SLC):
            incidence_angle = s1_mean_incidence_angle_rad(filename)

        elif isinstance(native_resolution, CSK_HIMAGE_SLC):
            incidence_angle = csk_mean_incidence_angle_rad(filename)

        elif isinstance(native_resolution, CSG_HIMAGE_SLC):
            incidence_angle = csg_mean_incidence_angle_rad(filename)

        else:
            raise NotImplementedError(
                f"{native_resolution.__class__.__name__} does not exists.")
            raise NotImplementedError(
                f"{native_resolution.__class__.__name__} does not exists.")

        candidate_n = range(1, int(n_az_looks*5))
        az_rg_res = list()
        diffs = list()

        for n in candidate_n:
            rg_res = native_resolution.RRES * n / math.sin(incidence_angle)
            az_res = n_az_looks * native_resolution.ARES
            az_rg_res.append([rg_res, az_res])
            diffs.append(math.sqrt((rg_res - az_res)**2))

        index = diffs.index(min(diffs))
        n_rg_looks = candidate_n[index]
        rg_res, az_res = az_rg_res[index]
        return MultiLook(n_rg_looks, n_az_looks, rg_res, az_res)

    def run(self, *args, **kwargs): ...
