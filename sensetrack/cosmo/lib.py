
from collections import namedtuple
from enum import Enum, unique

import h5py
import numpy as np
import shapely
from PIL import Image

import sensetrack.cosmo.utils as utils


@unique
class Polarization(Enum):

    HH = "Horizontal Tx/Horizontal Rx"
    VV = "Vertical Tx/ Vertical Rx"
    HV = "Horizontal Tx/ Vertical Rx"
    VH = "Vertical Tx/ Horizontal Rx"
    CO = "Co-polar acquisition (HH/VV)"
    CH = "Cross polar acquisition (HH/HV) with Horizontal Tx polarization"
    CV = "Cross polar acquisition (VV/VH) with Vertical Tx polarization"


@unique
class Orbit(Enum):
    A = 'Ascending'
    D = 'Descending'


@unique
class CosmoProduct(Enum):
    CSG = "2nd generation"
    CSK = "1st generation"


@unique
class Squint(Enum):
    N = "Not squinted data"
    F = "Forward squint"
    B = "Backward squint"


class Product:

    _MISSION = None

    _s = dict(L='Left', R='Right')

    _o = dict(A='Ascending', D='Descending')

    @classmethod
    def _is_2ndgen(cls, value: str) -> bool:

        if value == cls._MISSION:
            return True
        else:
            return False


CSKInfo = namedtuple("CSKInfo", [
    "SatelliteID",
    "ProductType",
    "InstrumentMode",
    "Swath",
    "Polarization",
    "LookSide",
    "OrbitDirection",
    "DeliveryMode",
    "SelectiveAvaialability",
    "SensingStartTime",
    "SensingStopTime"
])


CSGInfo = namedtuple("CSGInfo", [
    'SatelliteID',
    'ProductType',
    'NumRangeLooks',
    'NumAzimuthLooks',
    'InstrumentMode',
    'Swath',
    'Polarization',
    'LookSide',
    'OrbitDirection',
    'OrbitalDataQuality',
    'SensingStartTime',
    'SensingStopTime',
    'FileSequenceID',
    'ProductCoverage',
    'LatitudeSceneCenter',
    'Hemisphere',
    'EastDirectionLocation',
    'SquintAngleSceneCenter'
])


class CSKProduct(Product):

    _MISSION = "CSK"

    _MM = dict(
        HI='Himage',
        PP='PingPong',
        WR='WideRegion',
        HR='HugeRegion',
        S2='Spotlight 2'
    )

    _PP = dict(
        HH="Horizontal Tx/Horizontal Rx",
        VV="Vertical Tx/ Vertical Rx",
        HV="Horizontal Tx/ Vertical Rx",
        VH="Vertical Tx/ Horizontal Rx",
        CO="Co-polar acquisition (HH/VV)",
        CH="Cross polar acquisition (HH/HV) with Horizontal Tx polarization",
        CV="Cross polar acquisition (VV/VH) with Vertical Tx polarization"
    )

    _D = dict(F="Fast delivery mode", S="Standard delivery mode")

    _G = dict(N='ON', F='OFF')

    @classmethod
    def parse_filename(cls, filename: str):

        NAME = utils.StrChopper(filename)
        mission = NAME.chop(3)

        if not cls._is_2ndgen(mission):
            raise ValueError("Non è un file CSK (I generazione)")

        i = NAME.chop(2)
        _ = NAME.chop(1)
        YYY_Z = NAME.chop(5)
        _ = NAME.chop(1)
        MM = NAME.chop(2)
        _ = NAME.chop(1)
        SS = NAME.chop(2)
        _ = NAME.chop(1)
        PP = NAME.chop(2)
        _ = NAME.chop(1)
        s = NAME.chop(1)
        o = NAME.chop(1)
        _ = NAME.chop(1)
        D = NAME.chop(1)
        G = NAME.chop(1)
        _ = NAME.chop(1)
        start = utils.str2dt(NAME.chop(14))
        _ = NAME.chop(1)
        end = utils.str2dt(NAME.chop(14))

        msg = f"""
        {filename}
        COSMO-SkyMed (I Generation)

        Satellite ID:            {i}
        Product Type:            {YYY_Z}
        Instrument mode:         {cls._MM[MM]}
        Swath:                   {SS}
        Polarization:            {cls._PP[PP]}
        Look side:               {cls._s[s]}
        Orbit Direction:         {cls._o[o]}
        Delivery mode:           {cls._D[D]}
        Selective Avaialability: {cls._G[G]}
        Sensing start time:      {start}
        Sensing stop time:       {end}
        """

        print(msg)

        return CSKInfo(i, YYY_Z, cls._MM[MM], SS, cls._PP[PP], cls._s[s],
                       cls._o[o], cls._D[D], cls._G[G], start, end)


class CSGProduct(Product):

    _MISSION = "CSG"

    _MMM = dict(
        S2A='Spotlight 2A',
        S2B='Spotlight 2B',
        S2C='Spotlight 2C',
        D2R='Spotlight 1 optimized resolution',
        D2S='Spotlight 2 optimized swath',
        D2J='Spotlight 1 joined',
        OQR='Spotlight 1 operational QuadPol optimized resolution',
        OQS='Spotlight 2 operational QuadPol optimized swath',
        STR='Stripmap',
        SC1='ScanSAR 1',
        SC2='ScanSAR 2',
        PPS='PingPong',
        QPS='QuadPol'
    )

    _PP = dict(
        HH="Horizontal Tx/Horizontal Rx",
        VV="Vertical Tx/ Vertical Rx",
        HV="Horizontal Tx/ Vertical Rx",
        VH="Vertical Tx/ Horizontal Rx"
    )

    _Q = dict(
        D='Downlinked',
        P='Predicted',
        F='Filtered',
        R='Restituted'
    )

    _H = dict(
        N='North',
        S='South'
    )

    _S = dict(
        F='Full standard size',
        C='Cropped product'
    )

    def _LL(s: str):
        assert s[0] == 'Z'

        s = s[1:]
        if int(s) in [*range(1, 61)]:
            descr = f'UTM Zone {s}'
        elif s == '00':
            descr = f'South Pole area'
        elif s == '61':
            descr = f'North Pole area'

        return descr

    def _AAA(s: str):
        if s[0] == 'N':
            descr = 'Not squinted data'
        elif s[0] == 'F':
            a = s[1]
            descr = f"Forward squint ({a}°)"
        elif s[0] == 'B':
            a = s[1]
            descr = f"Backward squint ({a}°)"
        return descr

    @classmethod
    def parse_filename(cls, filename: str):

        NAME = utils.StrChopper(filename)
        mission = NAME.chop(3)

        if not cls._is_2ndgen(mission):
            raise ValueError("Non è un file CSK (I generazione)")

        _ = NAME.chop(1)
        i = NAME.chop(5)
        _ = NAME.chop(1)
        YYY_Z = NAME.chop(5)
        _ = NAME.chop(1)
        RR = NAME.chop(2)
        AA = NAME.chop(2)
        _ = NAME.chop(1)
        MMM = NAME.chop(3)
        _ = NAME.chop(1)
        SSS = NAME.chop(3)
        _ = NAME.chop(1)
        PP = NAME.chop(2)
        _ = NAME.chop(1)
        s = NAME.chop(1)
        o = NAME.chop(1)
        _ = NAME.chop(1)
        Q = NAME.chop(1)
        _ = NAME.chop(1)
        start = utils.str2dt(NAME.chop(14))
        _ = NAME.chop(1)
        end = utils.str2dt(NAME.chop(14))
        _ = NAME.chop(1)
        j = NAME.chop(1)
        _ = NAME.chop(1)
        S = NAME.chop(1)
        _ = NAME.chop(1)
        ll = NAME.chop(2)
        H = NAME.chop(1)
        _ = NAME.chop(1)
        ZLL = NAME.chop(3)
        _ = NAME.chop(1)
        AAA = NAME.chop(3)

        msg = f"""
        {filename}
        COSMO-SkyMed (II Generation)

        Satellite ID:              {i}
        Product Type:              {YYY_Z}
        Number of range looks:     {RR}
        Number of azimuth looks:   {RR}
        Instrument mode:           {cls._MMM[MMM]}
        Swath:                     {SSS}
        Polarization:              {cls._PP[PP]}
        Look side:                 {cls._s[s]}
        Orbit Direction:           {cls._o[o]}
        Orbital data quality:      {cls._Q[Q]}
        Sensing start time:        {start}
        Sensing stop time:         {end}
        File sequence ID:          {j}
        Product coverage:          {cls._S[S]}
        Latitude scene center:     {int(ll)}°
        Hemisphere:                {cls._H[H]}
        East-direction location:   {cls._LL(ZLL)}
        Squint angle scene center: {cls._AAA(AAA)}
        """

        print(msg)

        return CSGInfo(i, YYY_Z, RR, RR, cls._MMM[MMM], SSS, cls._PP[PP],
                       cls._s[s], cls._o[o], cls._Q[Q], start, end, j, cls._S[S],
                       int(ll), cls._H[H], cls._LL(ZLL), cls._AAA(AAA))


class CSKFile(h5py.File):
    def __init__(self,
                 name, mode='r', driver=None, libver=None, userblock_size=None,
                 swmr=False, rdcc_nslots=None, rdcc_nbytes=None, rdcc_w0=None,
                 track_order=None, fs_strategy=None, fs_persist=False,
                 fs_threshold=1, fs_page_size=None, page_buf_size=None,
                 min_meta_keep=0, min_raw_keep=0, locking=None,
                 alignment_threshold=1, alignment_interval=1,
                 meta_block_size=None, **kwds) -> None:

        super().__init__(name, mode, driver, libver, userblock_size, swmr,
                         rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order,
                         fs_strategy, fs_persist, fs_threshold, fs_page_size,
                         page_buf_size, min_meta_keep, min_raw_keep, locking,
                         alignment_threshold, alignment_interval,
                         meta_block_size, **kwds)

    def __str__(self):
        return CSKProduct.parse_filename(
            self.attrs['Product Filename'].decode()).__str__()

    def __repr__(self):
        return self.attrs['Product Filename'].decode()

    @property
    def qkl_to_numpy(self): return np.array(self['/S01/QLK'])

    @property
    def SBI(self): return np.array(self['/S01/SBI'])

    @property
    def qlk_to_image(self): return Image.fromarray(self.qkl_to_numpy)

    @property
    def _etlc(self):
        'Estimated Top-Left Corner'
        return self['/'].attrs['Estimated Top Left Geodetic Coordinates']

    @property
    def _eblc(self):
        'Estimated Bottom-Left Corner'
        return self['/'].attrs['Estimated Bottom Left Geodetic Coordinates']

    @property
    def _etrc(self):
        'Estimated Top-Right Corner'
        return self['/'].attrs['Estimated Top Right Geodetic Coordinates']

    @property
    def _ebrc(self):
        'Estimated Bottom-Right Corner'
        return self['/'].attrs['Estimated Bottom Right Geodetic Coordinates']

    @property
    def estimated_corner_coordinated(self) -> tuple:
        return self._etlc, self._eblc, self._ebrc, self._etrc

    @property
    def footprint_polygon(self) -> shapely.Geometry:
        y, x, _ = np.array(self.estimated_corner_coordinated).T
        return shapely.Polygon(np.c_[x, y])


class Pols(dict):
    def __init__(self) -> None:
        return super().__init__()
