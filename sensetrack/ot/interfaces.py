"""
Module: interfaces

This module defines core classes and utilities for handling multi-band images and implementing optical tracking algorithms.

Classes:
    - Image: Represents an image with multiple bands, including georeferencing and nodata handling.
    - OTAlgorithm: Abstract base class for optical tracking algorithms, providing serialization and utility methods.

Functions and Methods:
    - Image.__init__: Initializes an Image instance with image data, affine transform, CRS, and nodata value.
    - Image.__new__: Allocates a new Image instance.
    - Image.__repr__: Returns a string representation of the Image.
    - Image.__len__: Returns the number of bands in the image.
    - Image.__iter__: Iterates over the image bands.
    - Image.__getitem__: Retrieves a specific band by index.
    - Image.__get_bandnames: Returns a tuple of band names.
    - Image.split_channels: Splits the image into its individual channels.
    - Image.is_coregistered: Checks if another image is coregistered with this one.
    - Image.get_band: Retrieves a specific band as a new Image instance.
    - OTAlgorithm: Abstract class for algorithm implementation.
    - OTAlgorithm.from_dict: Instantiates an algorithm from a dictionary.
    - OTAlgorithm.from_JSON: Instantiates an algorithm from a JSON file.
    - OTAlgorithm.from_YAML: Instantiates an algorithm from a YAML file.
    - OTAlgorithm.toJSON: Serializes the algorithm parameters to a JSON file.
    - OTAlgorithm._to_displacements: Converts pixel offsets to physical displacements.
    - OTAlgorithm.__call__: Abstract method for running the algorithm.
"""
import json
from abc import ABC
from inspect import signature
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rasterio import Affine
from rasterio import CRS

from sensetrack.log import setup_logger

logger = setup_logger(__name__)


class Image:
    """
    Class representing an image with multiple bands.
    
    This class provides functionality for handling multi-band images with georeferencing 
    information and nodata value handling. It supports operations like band splitting, 
    coregistration checking, and individual band access.

    Attributes:
        image (np.ndarray): The underlying image data
        affine (Affine): The affine transformation matrix for georeferencing
        crs (CRS): The coordinate reference system
        nodata (int | float): Value used to represent no data in the image
        bandnames (tuple): Names of the image bands
    """
    def __init__(self, image: np.ndarray, affine: Affine,
                 crs: str, nodata: float | None = None) -> None:
        """
        Initialize an Image instance.

        Args:
            image (np.ndarray): The image array, must have at least 2 dimensions
            affine (Affine): The affine transformation matrix for georeferencing
            crs (CRS): The coordinate reference system
            nodata (int | float, optional): Value used to represent no data. Defaults to None

        Raises:
            ValueError: If the image array has less than 2 dimensions
        """
        if image.ndim < 2:
            raise ValueError("Image array must have at least 2 dimensions.")

        self._image = image
        splitted = cv2.split(image)
        self.bandnames = self.__get_bandnames(len(splitted))

        for bandname, band_array in zip(self.bandnames, splitted):
            setattr(self, bandname, band_array)

        self._affine = affine
        self._crs = crs

        if nodata is None:
            image[image < 0] = 0.
            self._nodata = 0.
            logger.debug(f"Inferring/setting nodata values: {self._nodata = :.1f}")
        else:
            self._nodata = nodata

    def __new__(cls, image: np.ndarray, affine: Affine, crs: str, nodata: float = -9999.):
        """
        Allocate a new Image instance.

        Args:
            image: The image array
            affine: The affine transformation matrix
            crs: The coordinate reference system
            nodata: The nodata value. Defaults to None

        Returns:
            Image: A new Image instance
        """
        obj = super().__new__(cls)
        obj.image = image
        obj.affine = affine
        obj.crs = crs
        obj.nodata = nodata
        return obj

    def __repr__(self):
        return f"{self.image}\n{self.affine}\n{self.crs}"

    def __len__(self):
        return self.count

    def __iter__(self):
        for band in self.bandnames:
            yield self.__getattribute__(band)

    def __getitem__(self, __index):
        name = list(self.bandnames)[__index]
        return self.__getattribute__(name)

    @classmethod
    def __get_bandnames(cls, n):
        """
        Generate band names for the image.

        Args:
            n (int): Number of bands

        Returns:
            tuple: Band names in the format ('B1', 'B2', ...)
        """
        return tuple(f"B{i+1}" for i in range(n))

    @property
    def nodata(self):
        return self._nodata

    @nodata.setter
    def nodata(self, value):
        self._nodata = value

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value

    @property
    def affine(self):
        return self._affine

    @affine.setter
    def affine(self, value: Affine):
        self._affine = value

    @property
    def crs(self):
        return self._crs

    @crs.setter
    def crs(self, value):
        self._crs = value

    @property
    def n_channels(self) -> int:
        """
        Get the number of channels in the image.

        Returns:
            int: Number of dimensions in the image array
        """
        return self.image.ndim

    @property
    def count(self):
        """
        Get the number of bands in the image.

        Returns:
            int: 1 for 2D images, number of bands for multi-band images
        """
        if self.n_channels == 2:
            return 1
        elif self.image is not None:
            return self.image.shape[self.n_channels-1]

    @property
    def mask(self) -> np.ndarray:
        """
        Get a mask of nodata values.

        Returns:
            np.ndarray: Boolean mask where True indicates nodata values
        """
        return np.equal(self.image, self.nodata)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the first band.

        Returns:
            tuple: Shape of the first band (height, width)
        """
        return self.image.shape

    @property
    def height(self):
        """
        Get the height of the image.

        Returns:
            int: Height in pixels
        """
        return self.shape[0]

    @property
    def width(self):
        """
        Get the width of the image.

        Returns:
            int: Width in pixels
        """
        return self.shape[1]

    def split_channels(self):
        """
        Split the image into its individual channels.

        Returns:
            list[Image]: List of single-band images
        """
        imgs = list()

        for e in cv2.split(self.image):
            imgs.append(Image(e, self.affine, self.crs, self.nodata))

        return imgs

    def is_coregistered(self, __other) -> bool:
        """
        Check if the image is coregistered with another image.

        Args:
            __other (Image): Another image to check coregistration with

        Returns:
            bool: True if images are coregistered

        Raises:
            ValueError: If only one of the images has georeferencing
        """
        if all([self.affine is not None, __other.affine is not None]):
            return self.affine == __other.affine
        elif any([self.affine is not None, __other.affine is not None]):
            raise ValueError(
                "One of the images lacks georeferencing information")
        else:
            Warning(
                "Coregistration check for non-raster images is approximate.")
            return self.shape == __other.shape


def get_band(image: Image, n: str | int = 0) -> Image | None:
    """
    Get a specific band from the image.

    Args:
        n (str | int | None): Band identifier. Can be band name (str), 
                                index (int), or None for entire image

    Returns:
        Image: New Image instance containing the requested band
    """
    if isinstance(n, str):
        band = image.__getattribute__(n)
        return Image(band, image.affine, image.crs, image.nodata)

    elif isinstance(n, int):
        band = image[n]
        return Image(band, image.affine, image.crs, image.nodata)

    else:
        return


class OTAlgorithm(ABC):
    """
    Abstract base class for optical tracking algorithms.

    This class provides common functionality for optical tracking algorithms,
    including parameter serialization and deserialization, and utilities for
    converting pixel offsets to physical displacements.
    """

    library: str = 'Undefined'

    @classmethod
    def from_dict(cls, __d: dict):
        """
        Create an instance from a dictionary of parameters.

        Args:
            __d (dict): Dictionary containing algorithm parameters

        Returns:
            OTAlgorithm: New instance initialized with the parameters
        """
        keys = list(signature(cls.__init__).parameters.keys())

        kw = {key: value for key, value in __d.items()
              if key in keys and value is not None}

        return cls(**kw)

    @classmethod
    def from_JSON(cls, __json: Path | str):
        """
        Create an instance from parameters stored in a JSON file.

        Args:
            __json (Path | str): Path to the JSON file

        Returns:
            OTAlgorithm: New instance initialized with the parameters

        Raises:
            FileNotFoundError: If the JSON file does not exist
        """
        path = Path(__json)

        if not path.is_file():
            raise FileNotFoundError(f"File {__json} does not exist")

        __d = json.loads(path.read_text())

        return cls.from_dict(__d)

    @classmethod
    def from_YAML(cls, __yaml: Path | str):
        """
        Create an instance from parameters stored in a YAML file.

        Args:
            __yaml (Path | str): Path to the YAML file

        Returns:
            OTAlgorithm: New instance initialized with the parameters

        Raises:
            FileNotFoundError: If the YAML file does not exist
        """
        import yaml
        path = Path(__yaml)

        if not path.is_file():
            raise FileNotFoundError(f"File {__yaml} does not exist")

        __d = yaml.safe_load(path.read_text())

        return cls.from_dict(__d)

    def toJSON(self, file: str | Path | None = None) -> None:
        """
        Save algorithm parameters to a JSON file.

        Args:
            file (str | Path, optional): Path where to save the JSON file.
                If None, uses the class name with '_parms.json' suffix
        """
        parms = self.__dict__

        if file is None:
            file = f"{self.__class__.__name__}_parms.json"

        logger.info(f"Esporto parametri su file: {file}")

        Path(file).write_text(json.dumps(parms, indent=4))

    def _to_displacements(self, transform, pixel_offsets) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert pixel offsets to physical displacements.

        Args:
            transform: Affine transform for pixel to physical coordinate conversion
            pixel_offsets: Array of pixel offsets

        Returns:
            np.ndarray: Array of physical displacement magnitudes
        """
        if transform is not None:
            pxt = transform.a
            pyt = -transform.e
        else:
            pxt = pyt = 1

        px, py = pixel_offsets.T
        dxx, dyy = px.T * pxt, py.T * pyt

        return dxx, dyy

    def __call__(self, *args, **kwargs) -> Any: ...
