import zipfile
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

OrbitProperties = namedtuple("OrbitProperties", ['ORBIT_PASS', 'NODE_TIME', 'RELORBIT'])


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
