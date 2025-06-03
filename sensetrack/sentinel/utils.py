"""
utils.py
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
import zipfile
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

OrbitProperties = namedtuple(
    "OrbitProperties", ['ORBIT_PASS', 'NODE_TIME', 'RELORBIT'])


def reformat_node_time(NODE_TIME: str) -> str:
    return datetime.fromisoformat(NODE_TIME).strftime("%Y%m%dT%H%M%S%f")


def read_orbit_properties(SARFILE: str | Path) -> OrbitProperties:
    SARFILE = Path(SARFILE)

    assert SARFILE.is_file()

    with zipfile.ZipFile(SARFILE, "r") as zf:
        xml_name = [e for e in zf.namelist() if e.endswith(".safe")][0]
        xmlf = zf.read(xml_name)

        soup = BeautifulSoup(xmlf, 'xml')
        properties = soup.find_all("orbitProperties")[0]
        ORBIT_PASS = properties.find("pass").text
        NODE_TIME = properties.find("ascendingNodeTime").text
        RELORBIT = soup.find_all("relativeOrbitNumber")[0].text

        return OrbitProperties(ORBIT_PASS, reformat_node_time(NODE_TIME), RELORBIT)


def s1_mean_incidence_angle_rad(zip_path, polarization="vv"):
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
            "Nessun valore di angolo di incidenza trovato nei file XML.")

    avg_angle = sum(all_angles) / len(all_angles)

    return (avg_angle * math.pi) / 180.
