import json
from abc import ABC
from enum import Enum, unique
from pathlib import Path

import cv2
import numpy as np
import rasterio
from ot.normalize import _log_band, _normalize_band, _zscore_band
from rasterio import CRS, Affine


@unique
class Band(Enum):
    Red: int = 0
    Green: int = 1
    Blue: int = 2


class Image:
    _image = None
    _affine = None
    _crs = None
    _nodata = None

    def __init__(self, image: np.ndarray, affine: Affine,
                 crs: CRS, nodata: int | float = None):

        self.image = image

        for bandname, band_array in zip(list(Band.__members__.keys()), cv2.split(image)):
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
    def __iter__(self): return iter(list(Band.__members__.keys()))
    def __getitem__(self, key): return self.__getattribute__(key)

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
    def mask(self): return np.equal(self.image, self.nodata).any(axis=-1)

    @property
    def shape(self):
        return self.Red.shape

    @classmethod
    def from_file(cls, source: str, nodata: int | float | None = None) -> None:

        assert Path(source).exists(), f"File {source} does not exist"
        assert Path(source).is_file(), f"File {source} is not a file"

        source = Path(source)
        suffix = source.suffix

        if suffix in ['.tiff', '.tif']:
            with rasterio.open(str(source)) as src:
                bands = list()

                for bandname, bandnum in zip(list(Band.__members__.keys()),
                                             range(1, src.meta["count"] + 1)):
                    band_array = src.read(bandnum)
                    setattr(cls, bandname, band_array)
                    bands.append(band_array)

                image = cv2.merge(bands)
                affine = src.meta['transform']
                crs = src.meta['crs']
                nodata = src.meta['nodata'] or nodata

        elif suffix in ['.jpg', '.jpeg', '.png']:
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)

            for bandname, bandnum in zip(list(Band.__members__.keys()),
                                         range(image.shape[-1])):
                setattr(cls, bandname, image[:, :, bandnum])

            affine = None
            crs = None
            nodata = None

        return cls(image, affine, crs, nodata)

    def split_channels(self):
        imgs = list()

        for e in cv2.split(self.image):
            imgs.append(Image(e, self.affine, self.crs, self.nodata))

        return imgs

    def is_coregistered(self, __other) -> bool:

        assert all([self.affine is not None, __other.affine is not None])
        return self.affine == __other.affine

    def minmax_norm(self):
        bands = list()
        for bandname in list(Band.__members__.keys()):
            bands.append(_normalize_band(self[bandname], mask=self.mask))

        return Image(cv2.merge(bands), self.affine, self.crs)

    def zscore_norm(self, n: int | float = 1., dtype=cv2.CV_8U):
        bands = list()
        for bandname, _ in zip(iter(self), range(self.count)):
            bands.append(_zscore_band(self[bandname], mask=self.mask))

        return Image(cv2.merge(bands), self.affine, self.crs)

    def log_norm(self, n: int | float = 1., dtype=cv2.CV_8U):
        bands = list()
        for bandname in list(Band.__members__.keys()):
            bands.append(_log_band(self[bandname], mask=self.mask))

        return Image(cv2.merge(bands), self.affine, self.crs)

    def _normalize(self, *args, **kwargs):
        new_image = cv2.normalize(self.image, *args, **kwargs)
        return Image(new_image, self.affine, self.crs)

    def to_single_band(self, coeff: list | np.ndarray | None = None, dtype: int | None = None):
        '''
        Converte un'immagine RGB in scala di grigi utilizzando la formula:
        Y709 = 0.2125*R + 0.7154*G + 0.0721*B
        '''

        if not self.n_channels == 3:
            raise ValueError("Input image must be a 3D array")

        coeff = np.asarray(coeff) or np.array([0.21250, 0.71540, 0.07210])

        return Image(self.image[:, :, :3].dot(coeff), self.affine, self.crs, self.nodata)

    def get_band(self, bandname: str | None = None):
        if bandname is None:
            return self
        else:
            return Image(self[bandname], self.affine, self.crs, self.nodata)

    def _to_gdal(self, filename, driver: str = 'GTiff') -> None:
        height, width = self['Red'].shape

        with rasterio.open(filename, 'w', nodata=self.nodata,
                           transform=self.affine, width=width, height=height,
                           crs=self.crs, count=self.count, driver=driver,
                           dtype=self.image.dtype.__str__()) as ds:

            for n, bandname in zip(range(self.count), list(Band._member_map_.keys())):
                ds.write(self[bandname], n+1)

        print(filename, " saved.")

    def _to_image(self, filename) -> None:
        raise NotImplementedError

    def to_file(self, filename) -> None:
        if all([self.affine, self.crs]):
            self._to_gdal(filename)
        else:
            self._to_image(filename)


class OTAlgorithm(ABC):

    R, G, B = 0.21250, 0.71540, 0.07210

    def from_dict(__d: dict): ...

    @staticmethod
    def from_JSON(__json: Path | str):
        __d = json.loads(Path(__json).read_text())

        return OTAlgorithm.from_dict(__d)

    @staticmethod
    def from_YAML(__yaml: Path | str):
        import yaml

        __d = yaml.safe_load(Path(__yaml).read_text())

        return OTAlgorithm.from_dict(__d)

    def toJSON(self, parms: dict, file: str | Path = None):

        try:
            # non esiste una serializzazione per la classe `OPTFLOW_Flags`,
            # e non ho voglia di crearla al momento
            parms['flags'] = parms['flags'].value
        except KeyError:
            pass

        if file is None:
            file = f"{self.__class__.__name__}_parms.json"

        Path(file).write_text(json.dumps(parms, indent=4))

    @staticmethod
    def show_name(self): return self.__class__.__name__

    def _to_displacements(self, transform, pixel_offsets) -> np.ndarray:

        if transform is not None:
            pxt = transform.a
            pyt = -transform.e
        else:
            pxt = pyt = 1

        px, py = pixel_offsets.T
        dxx, dyy = px.T * pxt, py.T * pyt

        return np.linalg.norm([dxx, dyy], axis=0)

    def __call__(self, *args, **kwargs) -> None:
        self.toJSON(self.__dict__, f"{self.__class__.__name__}_parms.json")
