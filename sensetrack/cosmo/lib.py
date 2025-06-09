'''
sensetrack.cosmo.lib

This module provides classes and utilities for handling COSMO-SkyMed (CSK, 1st generation) and COSMO Second Generation (CSG, 2nd generation) satellite product metadata and files. It includes enumerations for product attributes, parsing utilities for product filenames, and wrappers for HDF5 product files.

Classes:

    - Polarization: Enum representing SAR polarization modes.
    - Orbit: Enum for orbit direction (Ascending/Descending).
    - CosmoProduct: Enum for COSMO product generations.
    - Squint: Enum for squint angle types.
    - Product: Base class for COSMO products, with mission and attribute mappings.
    - CSKInfo: Named tuple for CSK product metadata.
    - CSGInfo: Named tuple for CSG product metadata.
    - CSKProduct: Parser and metadata handler for CSK filenames.
    - CSGProduct: Parser and metadata handler for CSG filenames.
    - CSKFile: HDF5 file wrapper for CSK products, providing convenient access to image data and geolocation attributes.
    - Pols: Dictionary subclass for handling polarization data.

Functions:

    - CSKProduct.parse_filename(filename): Parses a CSK filename and returns a CSKInfo named tuple.
    - CSGProduct.parse_filename(filename): Parses a CSG filename and returns a CSGInfo named tuple.

Dependencies:

    - h5py: For reading HDF5 product files.
    - numpy: For array operations.
    - shapely: For geometric operations (footprint polygons).
    - PIL.Image: For image conversion.
    - sensetrack.cosmo.utils: Utility functions for string chopping and date parsing.

Usage:

This module is intended for internal use within the sensetrack.cosmo package to facilitate the reading, parsing, and interpretation of COSMO-SkyMed product files and their metadata.
'''

from pathlib import Path
import re
from collections import namedtuple
from datetime import datetime
from enum import Enum, unique

import h5py
import numpy as np
import shapely
from PIL import Image

import sensetrack.cosmo.utils as utils


class Product:

    _MISSION = None

    _s = dict(L='Left', R='Right')

    _o = dict(A='Ascending', D='Descending')


CSKInfo = namedtuple("CSKInfo", [
    "Mission",
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
    "Mission",
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
    def parse_filename_regex(cls, stem: str):
        """
        Parse Cosmo Product metadata from filename.

        Args:
            filename (str): The filename to parse.

        Returns:
            CSKInfo: A named tuple containing all the parsed information.

        Raises:
            ValueError: If the filename doesn't match the expected CSG format.
        """

        pattern = r"""
        CSK
        (\w{2})_ # satellite ID
        (\w{5})_ # product type
        (\w{2})_ # instrument mode
        (\w{2})_ # swath
        (\w{2})_ # polarization
        (\w) # look side
        (\w)_ # oribit direction
        (\w) # delivery mode
        (\w)_ # selective avaialability
        (\w{14})_ # Sensing start time
        (\w{14}) # Sensing end time
        """

        match = re.match(pattern, stem, re.VERBOSE)

        if not match:
            raise ValueError(f"Invalid CSK filename format: {stem}")

        (i, YYY_Z, MM, SS, PP, s, o,
         D, G, start_str, end_str) = match.groups()

        start = utils.str2dt(start_str)
        end = utils.str2dt(end_str)

        msg = f"""
        {stem}
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

        return CSKInfo(cls._MISSION, i, YYY_Z, cls._MM[MM], SS, cls._PP[PP], cls._s[s],
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
    def parse_filename_regex(cls, stem: str):
        """
        Parse Cosmo Product metadata from filename.

        Args:
            filename (str): The filename to parse.

        Returns:
            CSGInfo: A named tuple containing all the parsed information.

        Raises:
            ValueError: If the filename doesn't match the expected CSG format.
        """

        pattern = r"""
            CSG_    # Mission name
            (\w{5})_ # Satellite ID
            (\w{5})_ # Product Type
            (\d{2})  # Range looks
            (\d{2})_ # Azimuth looks
            (\w{3})_ # Instrument mode
            (\w{3})_ # Swath
            (\w{2})_ # Polarization
            ([LR])   # Look side
            ([AD])_  # Orbit Direction
            ([DPFR])_# Orbital data quality
            (\d{14})_# Start time
            (\d{14})_# End time
            (\w)_    # File sequence ID
            ([FC])_  # Product coverage
            (\d{2})  # Latitude
            ([NS])_  # Hemisphere
            (Z\d{2})_# East location
            ([NFB]\d{2})  # Squint angle
        """

        match = re.match(pattern, stem, re.VERBOSE)
        if not match:
            raise ValueError(f"Invalid CSG filename format: {stem}")

        (i, YYY_Z, RR, AA, MMM, SSS, PP, s, o, Q,
         start_str, end_str, j, S, ll, H, ZLL, AAA) = match.groups()

        start = utils.str2dt(start_str)
        end = utils.str2dt(end_str)

        msg = f"""
        {stem}
        COSMO-SkyMed (II Generation)

        Satellite ID:              {i}
        Product Type:              {YYY_Z}
        Number of range looks:     {RR}
        Number of azimuth looks:   {AA}
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

        return CSGInfo(cls._MISSION, i, YYY_Z, RR, AA, cls._MMM[MMM], SSS, cls._PP[PP],
                       cls._s[s], cls._o[o], cls._Q[Q], start, end, j, cls._S[S],
                       int(ll), cls._H[H], cls._LL(ZLL), cls._AAA(AAA))


class CosmoFilenameParser:
    """
    A factory class for parsing COSMO-SkyMed filenames.

    This class implements the factory pattern to create CSKInfo or CSGInfo objects
    based on the filename format.
    """

    @staticmethod
    def create_from_filename(filename: str) -> CSKInfo | CSGInfo:
        """
        Factory method to create the appropriate info object based on the filename.

        Args:
            filename (str): The filename to parse

        Returns:
            Union[CSKInfo, CSGInfo]: The parsed information in the appropriate format

        Raises:
            ValueError: If the filename format is not recognized
        """
        filename = Path(filename)

        if not filename.exists():
            raise FileNotFoundError(f"{filename} does not exists.")

        elif not filename.suffix == ".h5":
            raise ValueError(
                f"Invalid Cosmo product format. Expected `.h5` got {filename.suffix}")

        stem = filename.stem

        if stem.startswith('CSK'):
            return CSKProduct.parse_filename_regex(filename)
        
        elif stem.startswith('CSG'):
            return CSGProduct.parse_filename_regex(filename)
        
        else:
            raise ValueError(f"Unrecognized COSMO-SkyMed product: {filename}")

    @staticmethod
    def parse_metadata(filename: str) -> dict:
        """
        Parses the filename and returns a dictionary of metadata.

        Args:
            filename (str): The filename to parse

        Returns:
            dict: A dictionary containing all parsed metadata
        """
        info = CosmoFilenameParser.create_from_filename(filename)
        return info._asdict()  # Convert named tuple to dictionary


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
