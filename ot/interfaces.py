"""
Module: interfaces
This module provides classes and functions for handling and processing images with multiple bands. It includes an enumeration for image bands, a class for representing images, and an abstract base class for optical tracking algorithms.
Classes:
    - Band: Enumeration for image bands.
    - Image: Class representing an image with multiple bands.
    - OTAlgorithm: Abstract base class for optical tracking algorithms.
Functions:
    - Image.__init__: Initialize an Image instance.
    - Image.__new__: Create a new instance of Image.
    - Image.__repr__: Return a string representation of the Image instance.
    - Image.__len__: Return the number of bands in the image.
    - Image.__iter__: Return an iterator over the band arrays.
    - Image.__getitem__: Get a specific band by index.
    - Image.__get_bandnames: Returns a tuple of band names.
    - Image.bandnames: Get a tuple of the band names.
    - Image.split_channels: Split the image into its individual channels.
    - Image.is_coregistered: Check if the image is coregistered with another image.
    - Image.get_band: Get a specific band from the image.
    - OTAlgorithm.library: libreria di appartenenza.
    - OTAlgorithm.from_dict: Create an instance from a dictionary.
    - OTAlgorithm.from_JSON: Create an instance from a JSON file.
    - OTAlgorithm.from_YAML: Create an instance from a YAML file.
    - OTAlgorithm.toJSON: Convert the instance to a JSON string and save to a file.
    - OTAlgorithm.show_name: Return the name of the algorithm class.
    - OTAlgorithm._to_displacements: Convert pixel offsets to displacements.
    - OTAlgorithm.__call__: Call the algorithm.
"""
import json
import logging
from abc import ABC
from inspect import signature
from pathlib import Path

import cv2
import numpy as np
from rasterio import CRS, Affine

from log import setup_logger

logger = setup_logger(__name__)


class PreprocessDispatcher:
    def __init__(self):
        self.processes = dict()

    def register(self, name: str, process):
        if name not in self.processes:
            self.processes[name] = list()
        self.processes[name].append(process)

    def dispatch_process(self, name: str, **kwargs):
        if not name in self.processes:
            logger.critical(
                f"Il metodo {name.upper()} non è tra quelli registrati: " +
                f"{self.processes.keys()}")
            exit(0)
        else:
            for process in self.processes[name]:
                return process(**kwargs)


class Image:
    """Class representing an image with multiple bands."""
    _image = None
    _affine = None
    _crs = None
    _nodata = None

    def __init__(self, image: np.ndarray, affine: Affine,
                 crs: CRS, nodata: int | float = None):
        """
        Initialize an Image instance.

        :param image: The image array.
        :param affine: The affine transformation matrix.
        :param crs: The coordinate reference system.
        :param nodata: The nodata value.
        """
        if image.ndim < 2:
            raise ValueError("Image array must have at least 2 dimensions.")

        self.image = image
        splitted = cv2.split(image)
        self.bandnames = self.__get_bandnames(len(splitted))

        for bandname, band_array in zip(self.bandnames, splitted):
            setattr(self, bandname, band_array)

        self.affine = affine
        self.crs = crs
        self.nodata = nodata

    def __new__(cls, image, affine, crs, nodata=None):
        obj = super().__new__(cls)
        obj.image = image
        obj.affine = affine
        obj.crs = crs
        obj.nodata = nodata
        return obj

    def __repr__(self): return f"{self.image}\n{self.affine}\n{self.crs}"
    def __len__(self): return self.count

    def __iter__(self):
        for band in self.bandnames:
            yield self.__getattribute__(band)

    def __getitem__(self, __index):
        name = list(self.bandnames)[__index]
        return self.__getattribute__(name)

    @classmethod
    def __get_bandnames(cls, n): return tuple(f"B{i+1}" for i in range(n))

    @property
    def nodata(self): return self._nodata

    @nodata.setter
    def nodata(self, value): self._nodata = value

    @property
    def image(self): return self._image

    @image.setter
    def image(self, value): self._image = value

    @property
    def affine(self): return self._affine

    @affine.setter
    def affine(self, value): self._affine = value

    @property
    def crs(self): return self._crs

    @crs.setter
    def crs(self, value): self._crs = value

    @property
    def n_channels(self): return self.image.ndim

    @property
    def count(self):
        if self.n_channels == 2:
            return 1
        else:
            return self.image.shape[self.n_channels-1]

    @property
    def mask(self): return np.equal(self.image, self.nodata)

    @property
    def shape(self): return self.get_band(0).image.shape

    @property
    def height(self): return self.get_band(0).image.shape[0]

    @property
    def width(self): return self.get_band(0).image.shape[1]

    def split_channels(self):
        """Split the image into its individual channels."""
        imgs = list()

        for e in cv2.split(self.image):
            imgs.append(Image(e, self.affine, self.crs, self.nodata))

        return imgs

    def is_coregistered(self, __other) -> bool:
        """
        Check if the image is coregistered with another image.
        """
        if all([self.affine is not None, __other.affine is not None]):
            return self.affine == __other.affine
        elif any([self.affine is not None, __other.affine is not None]):
            raise ValueError(
                "Una delle due immagini non possiede georeferenziazione")
        else:
            Warning(
                "Il controllo di coregistrazione per immagini non raster è molto approssimativo.")
            return self.shape == __other.shape

    def get_band(self, n: str | int | None = None):
        """
        Get a specific band from the image.
        """
        if n in self.bandnames:
            band = self.__getattribute__(n)
            return Image(band, self.affine, self.crs, self.nodata)

        elif isinstance(n, int):
            band = self[n]
            return Image(band, self.affine, self.crs, self.nodata)

        elif n is None:
            return self


class OTAlgorithm(ABC):
    """Abstract base class for optical tracking algorithms."""

    R, G, B = 0.21250, 0.71540, 0.07210  # ???

    @classmethod
    def from_dict(cls, __d: dict):
        """Create an instance from a dictionary."""
        keys = list(signature(cls.__init__).parameters.keys())

        kw = {key: value for key, value in __d.items()
              if key in keys and value is not None}

        return cls(**kw)

    @classmethod
    def from_JSON(cls, __json: Path | str):
        """Create an instance from a JSON file."""
        path = Path(__json)

        if not path.is_file():
            raise FileNotFoundError(f"File {__json} does not exist")

        __d = json.loads(path.read_text())

        return cls.from_dict(__d)

    @classmethod
    def from_YAML(cls, __yaml: Path | str):
        """Create an instance from a YAML file."""
        import yaml
        path = Path(__yaml)

        if not path.is_file():
            raise FileNotFoundError(f"File {__yaml} does not exist")

        __d = yaml.safe_load(path.read_text())

        return cls.from_dict(__d)

    def toJSON(self, file: str | Path = None) -> None:
        """Convert the instance to a JSON string and save to a file."""
        parms = self.__dict__

        if file is None:
            file = f"{self.__class__.__name__}_parms.json"

        logger.info(f"Esporto parametri su file: {file}")

        Path(file).write_text(json.dumps(parms, indent=4))

    def _to_displacements(self, transform, pixel_offsets) -> np.ndarray:
        """Convert pixel offsets to displacements."""
        if transform is not None:
            pxt = transform.a
            pyt = -transform.e
        else:
            pxt = pyt = 1

        px, py = pixel_offsets.T
        dxx, dyy = px.T * pxt, py.T * pyt

        return np.linalg.norm([dxx, dyy], axis=0)

    def __call__(self, *args, **kwargs) -> None: ...
