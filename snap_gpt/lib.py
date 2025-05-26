from collections import namedtuple
from enum import Enum, unique
from pathlib import Path

import geopandas as gpd
import shapely

from snap_gpt.config import GRAPHS_WD


@unique
class SARPreprocessing(Enum):
    CSG_SLC_B_DEFAULT = "GSG_SLCB_DFLT"
    S1_SLC_DEFAULT = "S1SLC_DFLT"
    S1_SLC_DEFAULT_BAND3 = "S1SLC_DFLTB3"
    S1_SLC_NO_SPECKLE_FILTER = "S1SLC_NOSF"
    S1_SLC_NO_SPECKLE_FILTER_BAND3 = "S1SLC_NOSFB3"


@unique
class Graphs(Enum):
    GSG_SLCB_DFLT = GRAPHS_WD / "cosmo_scs-b_default.xml"
    S1SLC_DFLT = GRAPHS_WD / "s1_slc_processing_2bands.xml"
    S1SLC_DFLTB3 = GRAPHS_WD / "s1_slc_processing_3bands.xml"
    S1SLC_NOSF = GRAPHS_WD / "s1_slc_noSpeckleFilter.xml"
    S1SLC_NOSFB3 = GRAPHS_WD / "s1_slc_noSpeckleFilter+band3.xml"


@unique
class AOI(Enum):
    CALITA = "CL"
    SASSINERI = "SN"
    VALORIA = "VL"
    BALDIOLA = "BL"
    BOCCASSUOLO = "BS"


Subset = namedtuple("Subset", ["name", "geometry"])


class GPTSubsetter:
    """
    Utility per estrazioni delle aree di `subset` per graphs di SNAP-GPT.
    """

    @classmethod
    def get_subset(self, case_study: str) -> Subset:
        # Caso di shapefile (o gpkg) qualunque
        # "/percorso/a/file.gpkg|layername" oppure "/percorso/a/shapefile.shp"
        if "|" in case_study:
            file, layername = case_study.split("|")
            if not layername:
                raise ValueError("Layer name cannot be empty")

        else:
            file, layername = case_study, None

        geometry = gpd.read_file(file, layer=layername).to_crs(
            "EPSG:4326").geometry[0]
        return Subset(layername, geometry)
