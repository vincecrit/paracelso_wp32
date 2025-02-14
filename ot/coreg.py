'''
Modulo (una funzione) per coregistrazione di immagini
'''
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject


def basic_pixel_coregistration(infile: str, match: str, outfile: str) -> None:
    """Align pixels between a target image (infile) and a reference image (match).
    Optionally, reproject to the same CRS as 'match'."""
    with rasterio.open(infile) as src:
        src_transform = src.transform
        nodata = src.meta['nodata']

        with rasterio.open(match) as match:
            dst_crs = match.crs

            # calcolo trasformazione affine di output
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,
                dst_crs,
                match.width,
                match.height,
                *match.bounds,  # (left, bottom, right, top)
            )

        # imposta metadati per output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": nodata})

        print("Coregistrato a dimensioni:", dst_height,
              dst_width, '\n Affine\n', dst_transform)

        # Output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # itero tutte le bande di 'infile'
            for i in range(1, src.count + 1):
                reproject(source=rasterio.band(src, i),
                          destination=rasterio.band(dst, i),
                          src_transform=src.transform,
                          src_crs=src.crs,
                          dst_transform=dst_transform,
                          dst_crs=dst_crs,
                          resampling=Resampling.bilinear)