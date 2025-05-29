import math
from abc import ABC
from collections import namedtuple
from enum import Enum, unique

import geopandas as gpd

from sensetrack import GRAPHS_WD
from sensetrack.cosmo.utils import (csg_mean_incidence_angle_rad,
                                    csk_mean_incidence_angle_rad)
from sensetrack.log import setup_logger
from sensetrack.sentinel.utils import s1_mean_incidence_angle_rad

logger = setup_logger(__name__)


MultiLook = namedtuple("MultiLook", ["Num_Range_LOOKS", "Num_Azimuth_LOOKS",
                       "Estimated_RangeResolution", "Estimated_AzimuthResolution"])


class SARResolutions:
    RRES = None  # RANGE RESOLUTION
    ARES = None  # AZIMUTH RESOLUTION


class S1_IW_SLC(SARResolutions):
    RRES = 2.30  # RANGE RESOLUTION
    ARES = 14.10  # AZIMUTH RESOLUTION


class CSK_HIMAGE_SLC(SARResolutions):
    RRES = 3.0  # RANGE RESOLUTION
    ARES = 3.0  # AZIMUTH RESOLUTION


class CSG_HIMAGE_SLC(SARResolutions):
    RRES = 2.6488857702529085  # RANGE RESOLUTION
    ARES = 2.6488857702529085  # AZIMUTH RESOLUTION


@unique
class SARPreprocessing(Enum):
    S2_L2A_DFLT = "S2_L2A_DFLT"
    S1_IW_GRD_DFLT = "S1_IW_GRD_DFLT"
    S1_IW_SLC_DFLT = "S1_IW_SLC_DFLT"
    S1_IW_SLC_DFLT_B3 = "S1_IW_SLC_DFLT_B3"
    S1_IW_SLC_DFLT_NOSF = "S1_IW_SLC_DFLT_NOSF"
    S1_IW_SLC_DFLT_B3_NOSF = "S1_IW_SLC_DFLT_B3_NOSF"
    COSMO_HIMAGE_SLCB_DFLT = "COSMO_HIMAGE_SLCB_DFLT"


@unique
class Graphs(Enum):
    S2_L2A_DFLT = GRAPHS_WD / "s2_l2a_default.xml"
    S1_IW_GRD_DFLT = GRAPHS_WD / "s1_grd_default.xml"
    S1_IW_SLC_DFLT = GRAPHS_WD / "s1_slc_default.xml"
    S1_IW_SLC_DFLT_B3 = GRAPHS_WD / "s1_slc_default+b3.xml"
    S1_IW_SLC_DFLT_NOSF = GRAPHS_WD / "s1_slc_default_noSF.xml"
    S1_IW_SLC_DFLT_B3_NOSF = GRAPHS_WD / "s1_slc_default+b3noSF.xml"
    COSMO_HIMAGE_SLCB_DFLT = GRAPHS_WD / "cosmo_scs-b_default.xml"


Subset = namedtuple("Subset", ["name", "geometry"])


class GPTSubsetter:
    """
    Utility per estrazioni delle aree di `subset` per graphs di SNAP-GPT.
    """

    @classmethod
    def get_subset(self, aoi: str) -> Subset:
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
    def __init__(self, SUBSET: GPTSubsetter, PROCESS: SARPreprocessing) -> None:
        if not isinstance(PROCESS, SARPreprocessing):
            raise TypeError(
                f"PROCESS must be SARPreprocessing enum, got {type(PROCESS)}")

        self._PROCESS = PROCESS.value
        self.SUBSET = SUBSET
        GRAPH_PATH = Graphs._member_map_[PROCESS.value].value

        if not GRAPH_PATH.is_file():
            raise FileNotFoundError(f"Graph file not found: {GRAPH_PATH}")

        self.GRAPH = GRAPH_PATH

    def estimate_multilook_parms(self, filename: str, native_resolution: SARResolutions, n_az_looks: int = 1):

        if isinstance(native_resolution, S1_IW_SLC):
            incidence_angle = s1_mean_incidence_angle_rad(filename)

        elif isinstance(native_resolution, CSK_HIMAGE_SLC):
            incidence_angle = csk_mean_incidence_angle_rad(filename)

        elif isinstance(native_resolution, CSG_HIMAGE_SLC):
            incidence_angle = csg_mean_incidence_angle_rad(filename)

        else:
            raise NotImplementedError

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
