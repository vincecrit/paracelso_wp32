"""
sensetrack.sentinel.lib
Utility functions for processing Sentinel SAR data.
This module provides helper functions for extracting orbit properties and calculating
mean incidence angles from Sentinel SAR product archives (SAFE format, typically zipped).

It includes:

- `reformat_node_time`: Reformatting ISO8601 node time strings to a compact format.
- `read_orbit_properties`: Extracting orbit pass, node time, and relative orbit number from a zipped SAFE product.
- `s1_mean_incidence_angle_rad`: Computing the mean incidence angle (in radians) for a given polarization from annotation XML files inside a zipped SAFE product.

Dependencies:

    - math
    - zipfile
    - collections.namedtuple
    - datetime
    - pathlib.Path
    - bs4.BeautifulSoup
    
Typical usage example:
    props = read_orbit_properties("path/to/S1_product.zip")
    mean_angle = s1_mean_incidence_angle_rad("path/to/S1_product.zip", polarization="vv")
"""
import math
import re
import zipfile
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from zipfile import BadZipFile

from bs4 import BeautifulSoup

S1Info = namedtuple(
    "S1Info", ['ORBIT_PASS', 'NODE_TIME', 'RELATIVE_ORBIT',
               'PLATFORM', 'MODE', 'PRODUCT', 'RES',
               'LEVEL', 'CLASS', 'ABSOLUTEORB',
               'DATAID', 'ID'])
"""
Named tuple containing Sentinel-1 orbit information.

Fields:
    ORBIT_PASS (str): Orbit pass direction ('ASCENDING' or 'DESCENDING')
    NODE_TIME (str): Node time in format 'YYYYMMDDTHHmmssSSS'
    RELATIVE_ORBIT (str): Relative orbit number
    PLATFORM (str): Satellite name
    MODE (str): Acquisition mode
    RES (str): Level of resolution
    LEVEL (str): Processing level
    CLASS (str): Product class
    ABSOLUTEORB (str): Absolute orbit number
    DATAID (str): Data take id
    ID (str): Unique product identifier
"""


def reformat_node_time(NODE_TIME: str) -> str:
    """
    Convert ISO8601 node time to compact format.

    Args:
        NODE_TIME (str): Node time in ISO8601 format

    Returns:
        str: Node time in format 'YYYYMMDDTHHmmssSSS'
    """
    return datetime.fromisoformat(NODE_TIME).strftime("%Y%m%dT%H%M%S%f")


class S1ManifestParser:
    """
    A factory class for parsing relevant information from Sentinel-1's manifest file
    """

    @classmethod
    def s1slc_orbit_properties(cls, S1FILE: str | Path) -> tuple[str, str, str]:
        """
        Extract minimal orbit properties from a Sentinel-1 SLC product archive.

        This function reads the `manifest.safe` file inside the zip archive to extract
        orbit pass direction, node time, and relative orbit number.

        Args:
            SARFILE (str | Path): Path to the Sentinel-1 product zip file

        Returns:
            OrbitProperties: Named tuple containing:
                - ORBIT_PASS: Orbit pass direction ('ASCENDING' or 'DESCENDING')
                - NODE_TIME: Node time in format 'YYYYMMDDTHHmmssSSS'
                - RELORBIT: Relative orbit number

        Raises:
            zipfile.BadZipFile: If the input is not a valid zip file
            IndexError: If the manifest.safe file is not found
        """
        S1FILE = Path(S1FILE)

        if not S1FILE.suffix == '.zip':
            raise ValueError("SAR file must be a zip archive")

        with zipfile.ZipFile(S1FILE, "r") as zf:
            xml_name = [e for e in zf.namelist() if e.endswith(".safe")][0]
            xmlf = zf.read(xml_name)

            soup = BeautifulSoup(xmlf, 'xml')

            # Relative orbit
            relorbit_elem = soup.select_one("relativeOrbitNumber")

            if not relorbit_elem:
                raise ValueError("Missing relative orbit number")
            RELORBIT = relorbit_elem.text

            # Orbit properties
            properties = soup.select_one("orbitProperties")

            if properties is None:
                raise ValueError("Error in parsing manifest file")

            # Orbit pass
            orbit_pass_elem = properties.find("pass")

            if not orbit_pass_elem or not orbit_pass_elem.text:
                raise ValueError("Missing ascending node time")
            else:
                ORBIT_PASS = orbit_pass_elem.text

            # Node time
            node_time_elem = properties.select_one("ascendingNodeTime")

            if not node_time_elem or not node_time_elem.text:
                raise ValueError("Missing ascending node time")
            NODE_TIME = node_time_elem.text

            return ORBIT_PASS, reformat_node_time(NODE_TIME), RELORBIT

    @classmethod
    def parse_filename_regex(cls, filename: str | Path) -> tuple:

        filename = Path(filename)

        pattern = r"""
        (S1[A|B])_ # platform
        (\w{2})_ # imaging mode
        (\w{3}) # product type
        (\w{1})_ # resolution
        (\w{1}) # processing level
        (\w{1}) # product class
        (\w{2})_ # polarization
        (\w{15})_ # product start time
        (\w{15})_ # product end time
        (\w{6})_ # absolute orbit number
        (\w{6})_ # mission data take id
        (\w{4}). # unique identifier
        """

        match_obj = re.match(pattern, filename.stem)

        if not match_obj:
            raise ValueError(f"Invalid Sentinel-1 file name: {filename}")

        (PLATFORM, MODE, PRODUCT, RES, LEVEL, CLASS,
         STARTTIME, ENDTIME, ABSOLUTEORB, DATAID, ID) = match_obj.groups()

        start_time = datetime.strptime(STARTTIME, "%Y%m%dT%H%M%S")
        end_time = datetime.strptime(ENDTIME, "%Y%m%dT%H%M%S")

        return (PLATFORM, MODE, PRODUCT, RES, LEVEL, CLASS,
                start_time, end_time, ABSOLUTEORB, DATAID, ID)

    @classmethod
    def create_from_filename(cls, filename: str | Path):

        ORBIT_PASS, NODE_TIME, RELATIVE_ORBIT = cls.s1slc_orbit_properties(
            filename)
        
        (PLATFORM, MODE, PRODUCT, RES, LEVEL, CLASS,
         _, _, ABSOLUTEORB, DATAID, ID) = cls.parse_filename_regex(filename)
        
        return S1Info(ORBIT_PASS, NODE_TIME, RELATIVE_ORBIT,
                      PLATFORM, MODE, PRODUCT, RES, LEVEL,
                      CLASS, ABSOLUTEORB, DATAID, ID)

    @classmethod
    def parse_metadata(cls, filename: str) -> dict:
        """
        Parses the filename and returns a dictionary of metadata.

        Args:
            filename (str): The filename to parse

        Returns:
            dict: A dictionary containing all parsed metadata
        """
        info = S1ManifestParser.create_from_filename(filename)
        return info._asdict()  # Convert named tuple to dictionary


def s1_mean_incidence_angle_rad(zip_path, polarization="vv"):
    """
    Calculate mean incidence angle from Sentinel-1 annotation files.

    This function extracts incidence angles from all annotation XML files
    for a given polarization and calculates their mean value in radians.

    Args:
        zip_path: Path to the Sentinel-1 product zip file
        polarization (str, optional): Polarization to process ('vv' or 'vh'). 
                                    Defaults to "vv"

    Returns:
        float: Mean incidence angle in radians

    Raises:
        ValueError: If no incidence angle values are found in the XML files
        zipfile.BadZipFile: If the input is not a valid zip file
    """
    all_angles = []

    with zipfile.ZipFile(zip_path, 'r') as archive:

        annotation_files = [
            f for f in archive.namelist()
            if "annotation" in f and f"slc-{polarization.lower()}" in f.lower() and f.endswith(".xml")
        ]

        for xml_file in annotation_files:
            with archive.open(xml_file) as file:
                soup = BeautifulSoup(file.read(), "xml")

            angle_tags = soup.find_all("incidenceAngle")

            try:
                angles = [float(tag.text.strip()) for tag in angle_tags]
                all_angles.extend(angles)

            except ValueError:
                _angles = [[float(e) for e in tag.text.split()]
                           for tag in angle_tags]
                angles = [sum(e)/len(e) for e in _angles]
                all_angles.extend(angles)

    if not all_angles:
        raise ValueError(
            "No incidence angle values found in XML files.")

    avg_angle = sum(all_angles) / len(all_angles)

    return (avg_angle * math.pi) / 180.
