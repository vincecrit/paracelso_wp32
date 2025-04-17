from collections import namedtuple
from enum import Enum, unique
from pathlib import Path

import geopandas as gpd
import shapely

from snap_gpt.config import GRAPHS_WD, AOI_GPKG


@unique
class SARPreprocessing(Enum):
    CSG_SLC_B_DEFAULT = "GSG_SLCB_DFLT"
    S1_SLC_DEFAULT = "S1SLC_DFLT"
    S1_SLC_DEFAULT_BAND3 = "S1SLC_DFLTB3"
    S1_SLC_NO_SPECKLE_FILTER = "S1SLC_NOSF"
    S1_SLC_NO_SPECKLE_FILTER_BAND3 = "S1SLC_NOSFB3"


@unique
class Graphs(Enum):
    GSG_SLCB_DFLT = GRAPHS_WD / "csg_scs-b_default.xml"
    S1SLC_DFLT = GRAPHS_WD / "s1_slc_default.xml"
    S1SLC_DFLTB3 = GRAPHS_WD / "s1_slc_default+band3.xml"
    S1SLC_NOSF = GRAPHS_WD / "s1_slc_noSpeckleFilter.xml"
    S1SLC_NOSFB3 = GRAPHS_WD / "s1_slc_noSpeckleFilter+band3.xml"


@unique
class AOI(Enum):
    CALITA = "CL"
    SASSINERI = "SN"
    VALORIA = "VL"
    BALDIOLA = "BL"


Subset = namedtuple("Subset", ["name", "geometry"])


class GPTSubsetter:
    """
    Utility per estrazioni delle aree di `subset` per graphs di SNAP-GPT.
    """

    def __init__(self, gpkg: str | Path | None = None):
        self._gpkg = None
        if gpkg:
            self.gpkg = gpkg  # Usa il setter per validazione
        else:
            self._gpkg = AOI_GPKG
            if not Path(self._gpkg).is_file():
                raise FileNotFoundError(f"Default GPKG file not found: {self._gpkg}")

    @property
    def gpkg(self): return self._gpkg

    @gpkg.setter
    def gpkg(self, value: str | Path) -> None:
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"GPKG file not found: {path}")
        self._gpkg = path

    def get_aoi(self, case_study: AOI | str) -> Subset:

        match case_study:

            # Casi studio UNIMORE x praticità
            case AOI.CALITA:  # calita
                geometry = gpd.read_file(self.gpkg, layer="aoi_calita").to_crs(
                    "EPSG:4326").geometry[0]
                return Subset(AOI.CALITA.value, geometry)

            case AOI.VALORIA:  # valoria
                geometry = gpd.read_file(self.gpkg, layer="aoi_valoria").to_crs(
                    "EPSG:4326").geometry[0]
                return Subset(AOI.VALORIA.value, geometry)

            case AOI.SASSINERI:  # sassi neri
                geometry = gpd.read_file(self.gpkg, layer="aoi_sassineri").to_crs(
                    "EPSG:4326").geometry[0]
                return Subset(AOI.SASSINERI.value, geometry)

            case AOI.BALDIOLA:  # baldiola
                geometry = gpd.read_file(self.gpkg, layer="aoi_baldiola").to_crs(
                    "EPSG:4326").geometry[0]
                return Subset(AOI.BALDIOLA.value, geometry)

            case _:  # Caso di shapefile (o gpkg) qualunque
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
