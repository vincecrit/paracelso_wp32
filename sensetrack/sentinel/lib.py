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
    "OrbitProperties", ['ORBIT_PASS', 'NODE_TIME', 'RELORBIT'])
"""
Named tuple containing Sentinel-1 orbit information.

Fields:
    ORBIT_PASS (str): Orbit pass direction ('ASCENDING' or 'DESCENDING')
    NODE_TIME (str): Node time in format 'YYYYMMDDTHHmmssSSS'
    RELORBIT (str): Relative orbit number
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
    def s1slc_orbit_properties(cls, S1FILE: str | Path) -> S1Info:
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
            properties = soup.find_all("orbitProperties")[0]
            ORBIT_PASS = properties.find("pass").text
            NODE_TIME = properties.find("ascendingNodeTime").text
            RELORBIT = soup.find_all("relativeOrbitNumber")[0].text

            return ORBIT_PASS, reformat_node_time(NODE_TIME), RELORBIT
    
    def create_from_filename(cls, filename: str | Path):
        pattern = r"""
        (S1[A|B])_ # platform
        (\w{2})_ # imaging mode
        (\w{3}) # product type
        (\w{1})_ # resolution
        (\w{1}) # processing level
        (\w{1}) # product class
        (\w{2})_ # polarization
        (\w{4}) # product start time
        (\w{2})
        (\w{2})T
        (\w{2}) 
        (\w{2})
        (\w{2})_
        (\w{4}) # product end time
        (\w{2})
        (\w{2})T
        (\w{2}) 
        (\w{2})
        (\w{2})_
        (\w{6})_ # absolute orbit number
        (\w{6})_ # mission data take id
        (\w{4}). # unique identifier
        """


    @classmethod
    def parse_metadata(filename: str) -> dict:
        """
        Parses the filename and returns a dictionary of metadata.

        Args:
            filename (str): The filename to parse

        Returns:
            dict: A dictionary containing all parsed metadata
        """
        info = S1ManifestParser.s1slc_orbit_properties(filename)
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
