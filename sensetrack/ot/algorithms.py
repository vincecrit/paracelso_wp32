"""
# Offset-tracking module

This module provides classes and functions to estimate displacement between reference and target images by optical flow and phase cross-correlation.
The classes offer interfaces to use optical flow algorithms from OpenCV and scikit-image.

## Classes

    - OpenCVOpticalFlow: Wrapper for OpenCV's `calcOpticalFlowFarneback` function.
    - SkiOpticalFlowILK: Wrapper for scikit-image's `optical_flow_ilk` function.
    - SkiOpticalFlowTVL1: Wrapper for scikit-image's `optical_flow_tvl1` function.
    - SkiPCC_Vector: Wrapper for scikit-image's `phase_cross_correlation` function.

## Functions

    - stepped_rolling_window: Generates sliding windows over a 2D array.
    - xcorr_to_frame: Performs image cross-correlation and returns results as a GeoDataFrame or DataFrame.
"""

from dataclasses import dataclass

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import points_from_xy
from rasterio.transform import AffineTransformer
from skimage.registration import (
    optical_flow_ilk,
    optical_flow_tvl1,
    phase_cross_correlation,
)
from skimage.util import view_as_windows
from tqdm import tqdm

from sensetrack.log import setup_logger
from sensetrack.ot.interfaces import Image, OTAlgorithm

logger = setup_logger(__name__)


def normL2(a1, a2):
    return np.linalg.norm([a1, a2], axis=0)


def stepped_rolling_window(
    array_2d: np.ndarray, window_shape: tuple[int, int], step: int
):
    """
    Returns the sliding windows of the input array using the
    `skimage.utils.view_as_windows` function and the indices corresponding to
    the center of each window.

    Arguments:
        array_2d (np.ndarray): Input 2D array.
        window_shape (tuple): Window size (height, width).
        step (int): Sampling step, assumed to be the same in both directions.

    Returns:
        windows (np.ndarray): Sliding windows (NxM, *window_shape)
        centers (np.ndarray): Indices corresponding to the centers of the windows (NxM, 2).
    """
    windows = view_as_windows(arr_in=array_2d, window_shape=window_shape, step=step)

    # Calcolo dell'offset per il centro della finestra
    center_offset_row = window_shape[0] // 2
    center_offset_col = window_shape[1] // 2

    centers = []
    for i in range(windows.shape[0]):
        for j in range(windows.shape[1]):
            center_row = i * step + center_offset_row
            center_col = j * step + center_offset_col
            centers.append((center_row, center_col))

    return windows.reshape(-1, *window_shape), np.array(centers)


def xcorr_to_frame(
    ref: Image,
    tar: Image,
    win_size: tuple[int, int] | int,
    step_size: int,
    normalization: str = "phase",
    disambiguate=True,
    upsample_factor: int = 1,
) -> gpd.GeoDataFrame | pd.DataFrame:
    """
    Image cross-correlation. By default, images are normalized
    using FFT, thus performing phase cross-correlation (PCC);
    setting `normalization` to `None` performs a
    NON-normalized cross-correlation.

    The two input images are processed using the same window.

    Arguments:

        ref (Image): reference image
        tar (Image): target image
        win_size (tuple[int] | int): moving window size. If an integer (n) is
            provided, a square moving window (nxn) will be used.
        step_size (tuple[int] | int): sampling interval. The image will be
            split into several moving windows centered on image points spaced
            by `step_size` pixels.
        normalization (str | None): normalization type, `phase` for PCC,
            `None` for non-normalized CC. Default = `phase`
        upsample_factor (int | float): oversampling ratio.
            The images will be registered with a pixel size of `1 / upsample_factor`.
            Useful for identifying sub-pixel scale shifts.
            Strongly affects the computational load.

    Returns:
        frame (geopandas.GeoDataFrame | pandas.DataFrame):
            output `geopandas.GeoDataFrame` or `pandas.DataFrame`, depending on
            the definition of the Image.affine property

            The resulting displacement attributes (L2), row shift (RSHIFT),
                column
            shift (CSHIFT), and normalized root mean square error (NRMS) are
                stored.

            All displacements are expressed in the unit of measure of the
            input images. If no affine transform is defined in the input images,
            displacements are returned in terms of pixel displacements
    """

    logger.info("Eseguo algoritmo skimage.registration.phase_cross_correlation")
    logger.debug(f"{ref.shape=}, {ref.image.dtype=}")
    logger.debug(f"{tar.shape=}, {tar.image.dtype=}")

    if ref.crs is None:
        CRS = None
    else:
        CRS = ref.crs

    if isinstance(win_size, int):
        win_size = win_size, win_size

    _ref, index = stepped_rolling_window(ref.image, win_size, step_size)
    assert index.shape[0] == _ref.shape[0], f"{index.shape[0] =} {_ref.shape[0] =}"
    _tar, _ = stepped_rolling_window(tar.image, win_size, step_size)

    offset_record = list()
    for r, t in tqdm(
        iterable=zip(_ref, _tar), desc=f"IMGCORR", total=index.shape[0], ncols=150
    ):
        (sr, sc), error, _ = phase_cross_correlation(
            r,
            t,
            normalization=normalization,
            upsample_factor=upsample_factor,
            disambiguate=disambiguate,
        )

        L2 = np.sqrt(sr**2 + sc**2)
        row = L2, float(sr), float(sc), float(error)
        offset_record.append(row)

    columns = ["L2", "RSHIFT", "CSHIFT", "NRMS"]
    df = pd.DataFrame(offset_record, columns=columns)

    if ref.affine is not None:  # caso del raster di input
        transfomer = AffineTransformer(ref.affine)
        df *= ref.affine.a  # pixel -> metri
        coords = np.c_[transfomer.xy(*index.T)]
        geom = points_from_xy(*zip(*coords), crs=CRS)

        return gpd.GeoDataFrame(df, geometry=geom)

    else:  # caso di immagini *.jpeg ecc..
        x, y = index.T
        df.insert(0, "Y", y)
        df.insert(0, "X", x)
        return df


@dataclass
class RasterOutput:
    dxx: Image
    dyy: Image
    res: Image


class OPTFLOW_Flags:
    """Flags for OpenCV optical flow."""

    OPTFLOW_DEFAULT = None
    OPTFLOW_USE_INITIAL_FLOW = 4  # cv2.OPTFLOW_USE_INITIAL_FLOW
    OPTFLOW_FARNEBACK_GAUSSIAN = 256  # cv2.OPTFLOW_FARNEBACK_GAUSSIAN


class OpenCVOpticalFlow(OTAlgorithm):
    """
    Wrapper for the OpenCV `calcOpticalFlowFarneback` function.

    Returns the optical flow between two images.

    ## Arguments
        flow (np.ndarray, optional): Computed flow image that has the same size
            as prev and type CV_32FC2. Defaults to False.
        pyr_scale (float, optional): Specifies the image scale to build pyramids
            for each image. A pyr_scale = 0.5 means a classical pyramid, where
            each next layer is twice smaller than the previous one.
            Defaults to 0.5.
        levels (int, optional): Number of pyramid layers including the initial
            image. Setting levels=1 means that no extra layers are created and
            only the original images are used.
            Defaults to 4.
        winsize (int, optional): Averaging window size. Larger values increase
            the algorithm robustness to image noise and give more chances for
            fast motion detection, but yield more blurred motion field.
            Defaults to 16.
        iterations (int, optional): Number of iterations the algorithm does at
            each pyramid level.
            Defaults to 5.
        poly_n (int, optional): Size of the pixel neighborhood used to find
            polynomial expansion in each pixel. Larger values mean that the image
            will be approximated with smoother surfaces, yielding a more robust
            algorithm and a more blurred motion field.
            Typically poly_n=5 or 7.
            Defaults to 5.
        poly_sigma (float, optional): Standard deviation of the Gaussian that is
            used to smooth derivatives used as a basis for the polynomial expansion.
            For poly_n=5, you can set poly_sigma=1.1; for poly_n=7 a good value
            would be poly_sigma=1.5.
            Defaults to 1.1.
        flags (typing.Any, optional): Operation flags (see `OPTFLOW_Flags` class)
            that can be one of the following:
            - `OPTFLOW_Flags.OPTFLOW_USE_INITIAL_FLOW` (4) uses the input flow as
            an initial flow approximation.
            - `OPTFLOW_Flags.OPTFLOW_FARNEBACK_GAUSSIAN` (256) uses the Gaussian
                filter instead of a box filter of the same size for optical flow
                estimation. Usually, this option gives more accurate flow than with
                a box filter, at the cost of lower speed.
                Defaults to OPTFLOW_Flags.OPTFLOW_FARNEBACK_GAUSSIAN.
    """

    library = "OpenCV"

    def __init__(
        self,
        flow: np.ndarray | None = None,
        pyr_scale: float = 0.5,
        levels: int = 4,
        winsize: int = 16,
        iterations: int = 5,
        poly_n: int = 5,
        poly_sigma: float = 1.1,
        flags: int | None = OPTFLOW_Flags.OPTFLOW_DEFAULT,
    ) -> None:

        self.flow = flow
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

    def __call__(self, reference: Image, target: Image) -> dict:
        """
        Calculate displacement between reference and target images.

        ## Arguments
            reference : Image
                The reference image (source) for optical flow calculation.
            target : Image
                The target image (destination) for optical flow calculation.

        ## Returns
            dict[str, Image] : a dictionary containing the Image objects representing
            the row/column displacement (`dxx` and `dyy` keys) and the resultant displacement
            between the reference to the target image.

        ### Notes

        This method uses the `calcOpticalFlowFarneback` function from `OpenCV`
        to estimate the pixel-wise motion between two images.
        The resulting displacement field is transformed according to the target
        image's affine transformation. If no affine transform is defined in the input images,
        displacements are returned in terms of pixel displacements
        """
        if reference.shape != target.shape:
            raise ValueError(
                "Reference and target images must have the same dimensions."
            )

        logger.info("Eseguo algoritmo cv2.calcOpticalFlowFarneback")
        logger.debug(f"{reference.shape=}, {reference.image.dtype=}")
        logger.debug(f"{target.shape=}, {target.image.dtype=}")

        pixel_offsets = cv2.calcOpticalFlowFarneback(
            prev=reference.image,
            next=target.image,
            flow=self.flow,  # type: ignore
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.winsize,
            iterations=self.iterations,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self.flags,  # type: ignore
        )

        dxx, dyy = self._to_displacements(target.affine, pixel_offsets)

        dxx_img = Image(dxx, target.affine, target.crs, target.nodata)
        dyy_img = Image(dyy, target.affine, target.crs, target.nodata)
        res = Image(normL2(dxx, dyy), target.affine, target.crs, target.nodata)

        return dict(dxx=dxx_img, dyy=dyy_img, res=res)


class SkiOpticalFlowILK(OTAlgorithm):
    """
    Compute the optical flow between two images using the ILK
    (Inverse Lucas-Kanade) method from scikit-image.

    ## Arguments
        radius : int
            Radius of the window considered around each pixel.
        num_warp : int, optional
            Number of times moving_image is warped.
        gaussian : bool, optional
            If True, a Gaussian kernel is used for the local
            integration. Otherwise, a uniform kernel is used.
        prefilter : bool, optional
            Whether to prefilter the estimated optical flow before each
            image warp. When True, a median filter with window size 3
            along each axis is applied. This helps to remove potential
            outliers.
        dtype : dtype, optional
            Output data type: must be floating point. Single precision
            provides good results and saves memory usage and computation
            time compared to double precision.
    """

    library = "skimage"

    def __init__(self, radius=7, num_warp=10, gaussian=False, prefilter=False) -> None:

        self.radius = radius
        self.num_warp = num_warp
        self.gaussian = gaussian
        self.prefilter = prefilter

    def __call__(self, reference: Image, target: Image) -> dict[str, Image]:
        """
        Calculate displacement between reference and target images.

        ## Arguments

            reference : Image
                The reference image (source) for optical flow calculation.
            target : Image
                The target image (destination) for optical flow calculation.

        ## Returns
            dict[str, Image]: a dictionary containing the Image objects representing
            the row/column displacement (`dxx` and `dyy` keys) and the resultant displacement
            between the reference to the target image.

        ### Notes

        This method uses the `optical_flow_ilk` function from `skimage.registration`
        to estimate the pixel-wise motion between two images.
        The resulting displacement field is transformed according to the target
        image's affine transformation. If no affine transform is defined in the input images,
        displacements are returned in terms of pixel displacements.
        """
        logger.info("Eseguo algoritmo skimage.registration.optical_flow_ilk")
        logger.debug(f"{reference.shape=}, {reference.image.dtype=}")
        logger.debug(f"{target.shape=}, {target.image.dtype=}")

        pixel_offsets = optical_flow_ilk(
            reference.image,
            target.image,
            radius=self.radius,
            num_warp=self.num_warp,
            gaussian=self.gaussian,
            prefilter=self.prefilter,
        )

        logger.debug(f"Numero output: {len(pixel_offsets)}")
        logger.debug(f"Tipo output: {[type(e) for e in pixel_offsets]}")
        logger.debug(f"Shape output: {[e.shape for e in pixel_offsets]}")

        a, b = pixel_offsets
        dxx, dyy = self._to_displacements(target.affine, cv2.merge([a, b]))

        dxx_img = Image(dxx, target.affine, target.crs, target.nodata)
        dyy_img = Image(dyy, target.affine, target.crs, target.nodata)
        res = Image(normL2(dxx, dyy), target.affine, target.crs, target.nodata)

        return dict(dxx=dxx_img, dyy=dyy_img, res=res)


class SkiOpticalFlowTVL1(OTAlgorithm):
    """
    Compute the optical flow between two images using the TV-L1 algorithm from scikit-image.
    This class wraps the `skimage.registration.optical_flow_tvl1` function, providing a convenient interface
    for estimating the pixel-wise displacement field (optical flow) between a reference and a target image.

    ## Arguments
        attachment (float, optional): Attachment parameter for the TV-L1 algorithm, controlling the data term weight (default: 15)
        tightness (float, optional): Tightness parameter for the TV-L1 algorithm, controlling the smoothness term weight (default: 0.3)
        num_warp (int, optional): Number of warping iterations per pyramid level (default: 5)
        num_iter (int, optional): Number of iterations at each pyramid level (default: 10)
        tol (float, optional): Tolerance for the stopping criterion (default: 1e-4)
        prefilter (bool, optional): Whether to prefilter the images before computing optical flow (default: False)
    """

    library = "skimage"

    def __init__(
        self,
        attachment=15,
        tightness=0.3,
        num_warp=5,
        num_iter=10,
        tol=1e-4,
        prefilter=False,
    ) -> None:

        self.attachment = attachment
        self.tightness = tightness
        self.num_warp = num_warp
        self.num_iter = num_iter
        self.tol = tol
        self.prefilter = prefilter

    def __call__(self, reference: Image, target: Image) -> dict:
        """
        Calculate displacement between reference and target images.

        ## Arguments
            reference : Image
                The reference image (source) for optical flow calculation.
            target : Image
                The target image (destination) for optical flow calculation.

        ## Returns
            dict[str, Image]: a dictionary containing the Image objects representing
            the row/column displacement (`dxx` and `dyy` keys) and the resultant displacement
            between the reference to the target image.

        ### Notes

        This method uses the `optical_flow_tvl1` function from `skimage.registration`
        to estimate the pixel-wise motion between two images.
        The resulting displacement field is transformed according to the target
        image's affine transformation. If no affine transform is defined in the input images,
        displacements are returned in terms of pixel displacements.
        """
        logger.info("Running skimage.registration.optical_flow_tvl1")
        logger.debug(f"{reference.image.shape=}, {reference.image.dtype=}")
        logger.debug(f"{target.image.shape=}, {target.image.dtype=}")

        pixel_offsets = optical_flow_tvl1(
            reference.image,
            target.image,
            attachment=self.attachment,
            tightness=self.tightness,
            num_warp=self.num_warp,
            num_iter=self.num_iter,
            tol=self.tol,
            prefilter=self.prefilter,
        )

        logger.debug(f"Numero output: {len(pixel_offsets)}")
        logger.debug(f"Tipo output: {[type(e) for e in pixel_offsets]}")
        logger.debug(f"Shape output: {[e.shape for e in pixel_offsets]}")

        a, b = pixel_offsets
        dxx, dyy = self._to_displacements(target.affine, cv2.merge([a, b]))

        dxx_img = Image(dxx, target.affine, target.crs, target.nodata)
        dyy_img = Image(dyy, target.affine, target.crs, target.nodata)
        res = Image(normL2(dxx, dyy), target.affine, target.crs, target.nodata)

        return dict(dxx=dxx_img, dyy=dyy_img, res=res)


class SkiPCC_Vector(OTAlgorithm):
    """
    Compute phase cross-correlation vectors between reference and target images.
    This class serves as a wrapper for phase cross-correlation (PCC) vector calculation,
    typically used for image registration or displacement estimation. It leverages the
    skimage library to perform the computation.

    ## Arguments
        winsize : tuple[int] or int
            The size of the window used for local cross-correlation. Can be a single integer
            or a tuple specifying the window size in each dimension.
        step_size : tuple[int] or int
            The step size for moving the window across the image. Can be a single integer
            or a tuple specifying the step size in each dimension.
        phase_norm : bool, optional
            If True, apply phase normalization to the cross-correlation (default is True).
        upsmp_fac : int or float, optional
            Upsampling factor for subpixel accuracy in the cross-correlation (default is 1.0).
    """

    library = "skimage"

    def __init__(
        self,
        winsize: tuple[int, int] | int,
        step_size: int,
        phase_norm: bool = True,
        upsmp_fac: int = 1,
    ) -> None:

        if phase_norm:
            self.normalization = "phase"
        else:
            self.normalization = None

        self.winsize = winsize
        self.step_size = step_size
        self.upsmp_fac = upsmp_fac

    def __call__(
        self, reference: Image, target: Image
    ) -> gpd.GeoDataFrame | pd.DataFrame:
        """
        Calculate displacement between reference and target images.

        ## Arguments
            reference : Image
                The reference image (source) for optical flow calculation.
            target : Image
                The target image (destination) for optical flow calculation.

        ## Returns
        geopandas.GeoDataFrame | pandas.DataFrame: tabular data containing the displacement vectors
            between the reference and target images, along with their corresponding coordinates.
            If the input images have an affine transformation defined, a GeoDataFrame is returned;
            otherwise, a standard DataFrame is returned.

        ### Notes

        This method uses the `optical_flow_ilk` function from `skimage.registration`
        to estimate the pixel-wise motion between two images.
        The resulting displacement field is transformed according to the target
        image's affine transformation. If no affine transform is defined in the input images,
        displacements are returned in terms of pixel displacements.
        """
        return xcorr_to_frame(
            ref=reference,
            tar=target,
            win_size=self.winsize,
            step_size=self.step_size,
            normalization=self.normalization,  # type: ignore (type-linting is missing in `phase_cross_correlation` for `None`)
            upsample_factor=self.upsmp_fac,
        )
