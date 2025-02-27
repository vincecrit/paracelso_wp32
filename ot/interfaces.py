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
    - Image.from_file: Create an Image instance from a file.
    - Image.split_channels: Split the image into its individual channels.
    - Image.is_coregistered: Check if the image is coregistered with another image.
    - Image.minmax_norm: Normalize the image bands using min-max normalization.
    - Image.zscore_norm: Normalize the image bands using z-score normalization.
    - Image.log_norm: Normalize the image bands using logarithmic transformation.
    - Image._normalize: Normalize the image using OpenCV normalization.
    - Image.to_single_band: Convert an RGB image to grayscale using specified coefficients.
    - Image.get_band: Get a specific band from the image.
    - Image._to_gdal: Save the image as a GDAL file.
    - Image._to_image: Save the image as a standard image file.
    - Image.to_file: Save the image to a file.
    - OTAlgorithm.from_dict: Create an instance from a dictionary.
    - OTAlgorithm.from_JSON: Create an instance from a JSON file.
    - OTAlgorithm.from_YAML: Create an instance from a YAML file.
    - OTAlgorithm.toJSON: Convert the instance to a JSON string and save to a file.
    - OTAlgorithm.show_name: Return the name of the algorithm class.
    - OTAlgorithm._to_displacements: Convert pixel offsets to displacements.
    - OTAlgorithm.__call__: Call the algorithm.
"""
import json
from abc import ABC
from enum import Enum, unique
from inspect import signature
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio import CRS, Affine

from ot.image_processing import cv2_clahe, norm_log, norm_minmax, norm_zscore


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

    @classmethod
    def from_file(cls, source: str, nodata: int | float | None = None) -> None:
        """Create an Image instance from a file."""
        assert Path(source).exists(), f"File {source} does not exist"
        assert Path(source).is_file(), f"File {source} is not a file"

        source = Path(source)
        suffix = source.suffix

        if (suffix in ['.tiff', '.tif']):
            with rasterio.open(str(source)) as src:
                bands = list()

                bandsindex = tuple(range(1, src.meta["count"] + 1))
                for bandname, bandnum in zip(cls.__get_bandnames(len(bandsindex)), bandsindex):
                    band_array = src.read(bandnum)
                    setattr(cls, bandname, band_array)
                    bands.append(band_array)

                image = cv2.merge(bands)
                affine = src.meta['transform']
                crs = src.meta['crs']
                nodata = src.meta['nodata'] or nodata

        elif suffix in ['.jpg', '.jpeg', '.png']:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)

            bandsindex = tuple(range(image.shape[-1]))
            for bandname, bandnum in zip(cls.__get_bandnames(len(bandsindex)), bandsindex):
                setattr(cls, bandname, image[:, :, bandnum])

            affine = None
            crs = None
            nodata = None

        return cls(image, affine, crs, nodata)

    def split_channels(self):
        """Split the image into its individual channels."""
        imgs = list()

        for e in cv2.split(self.image):
            imgs.append(Image(e, self.affine, self.crs, self.nodata))

        return imgs

    def is_coregistered(self, __other) -> bool:
        """Check if the image is coregistered with another image."""
        assert all([self.affine is not None, __other.affine is not None])
        return self.affine == __other.affine

    def minmax_norm(self):
        """Normalize the image bands using min-max normalization."""
        bands = list()
        for band in self:
            bands.append(norm_minmax(band))

        return Image(cv2.merge(bands), self.affine, self.crs)

    def zscore_norm(self):
        """Normalize the image bands using z-score normalization."""
        bands = list()
        for band in self:
            bands.append(norm_zscore(band))

        return Image(cv2.merge(bands), self.affine, self.crs)

    def clahe(self):
        """Normalize the image bands using z-score normalization."""
        bands = list()
        for band in self:
            bands.append(cv2_clahe(band))

        return Image(cv2.merge(bands), self.affine, self.crs)

    def log_norm(self):
        """Normalize the image bands using logarithmic transformation."""
        bands = list()
        for band in self:
            bands.append(norm_log(band))

        return Image(cv2.merge(bands), self.affine, self.crs)
    # magari la elimino

    def _normalize(self, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U):
        """Normalize the image using OpenCV normalization."""
        new_image = cv2.normalize(
            self.image, dst=None, alpha=alpha, beta=beta, norm_type=norm_type, dtype=dtype)
        return Image(new_image, self.affine, self.crs)

    def to_single_band(self, coeff: list | np.ndarray | None = None, dtype: int | None = None):
        """Convert an RGB image to grayscale using specified coefficients."""
        '''
        Converte un'immagine RGB in scala di grigi utilizzando la formula:
        Y709 = 0.2125*R + 0.7154*G + 0.0721*B
        '''
        if not self.n_channels == 3:
            raise ValueError("Input image must be a 3D array")

        coeff = np.asarray(coeff) or np.array([0.21250, 0.71540, 0.07210])
        band = self.image[:, :, :3].dot(coeff)
        return Image(band, self.affine, self.crs, self.nodata)

    def get_band(self, n: str | int | None = None):
        """Get a specific band from the image."""
        if n in self.bandnames:
            band = self.__getattribute__(n)
            return Image(band, self.affine, self.crs, self.nodata)

        elif isinstance(n, int):
            band = self[n]
            return Image(band, self.affine, self.crs, self.nodata)

        elif n is None:
            return self

    def _to_gdal(self, filename, driver: str = 'GTiff') -> None:
        """Save the image as a GDAL file."""
        height, width = self.shape

        with rasterio.open(filename, 'w', nodata=self.nodata,
                           transform=self.affine, width=width, height=height,
                           crs=self.crs, count=self.count, driver=driver,
                           dtype=self.image.dtype.__str__()) as ds:

            for n, band in enumerate(self):
                ds.write(band, n+1)

        print(filename, " saved.")

    def _to_image(self, filename) -> None:
        """Save the image as a standard image file."""
        raise NotImplementedError

    def to_file(self, filename) -> None:
        """Save the image to a file."""
        if all([self.affine, self.crs]):
            self._to_gdal(filename)
        else:
            self._to_image(filename)


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
