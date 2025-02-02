from collections import namedtuple
from enum import Enum, unique
from pathlib import Path

import geopandas as gpd
import shapely

from s1.config import GRAPHS_WD, AOI_GPKG


@unique
class SARPreprocessing(Enum):
    S1_SLC_DEFAULT = "S1SLC_DFLT"
    S1_SLC_DEFAULT_BAND3 = "S1SLC_DFLTB3"
    S1_SLC_NO_SPECKLE_FILTER = "S1SLC_NOSF"


@unique
class Graphs(Enum):
    S1SLC_DFLT = GRAPHS_WD / "prep_s1_slc_default.xml"
    S1SLC_DFLTB3 = GRAPHS_WD / "prep_s1_slc_default+band3.xml"
    S1SLC_NOSF = GRAPHS_WD / "prep_s1_slc_noSpeckleFilter.xml"


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
        self._gpkg = gpkg or AOI_GPKG

    @property
    def gpkg(self): return self._gpkg

    @gpkg.setter
    def gpkg(self, value: str | Path) -> None:
        assert Path(value).is_file()

        self._gpkg = Path(value)

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

                else:
                    file, layername = case_study, None

                geometry = gpd.read_file(file, layer=layername).to_crs(
                    "EPSG:4326").geometry[0]
                return Subset(layername, geometry)
