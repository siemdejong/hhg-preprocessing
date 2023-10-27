"""Create foreground tissue masks on higher harmonic generation images."""

import click
from typing import Union, Tuple
from pathlib import Path

from monai.data import WSIReader
import numpy as np
import numpy.typing as npt
import skimage
import scipy
from PIL import Image

from hhg_preprocess.masker.config import default as CFG


def _connected_components(
    image: npt.NDArray,
    connectivity: Union[int, None] = None,
    num_largest_components: Union[int, None] = None,
) -> np.ndarray:
    """
    Connected component analysis used by entropy_masker.

    Parameters
    ----------
    image : np.ndarray
        2D RGB image, with color channels at the last axis.
    connectivity : int, default=None
        Connectivity of the connected components analysis.
        If None, pick all components.
    num_largest_components : int, default=None
        Pick the `num_largest_components` largest components.
        If None, pick all components.

    Returns
    -------
    np.ndarray
        Binary mask of connected components.
    """
    # Pick the largest image component using connected components.
    labels, num = skimage.morphology.label(image, connectivity=connectivity, return_num=True)

    # Finding the largest component is not needed if there is only one component.
    if num == 1:
        return image

    label_props: list[skimage.measure._regionprops.RegionProperties] = skimage.measure.regionprops(labels)
    areas = [label_prop.area for label_prop in label_props]
    if num_largest_components is None or num_largest_components < 1:
        num_largest_components = len(areas)
    else:
        num_largest_components = min(len(areas), num_largest_components)

    # Get the indices of the num_largest_components largest components.
    max_area_indices = np.argpartition(areas, -num_largest_components)[-num_largest_components:]

    # Background is labelled 0
    largest_component_mask = np.isin(labels, max_area_indices + 1)

    return largest_component_mask


def entropy_masker(
    image: npt.NDArray,
    footprint: Union[np.ndarray, None] = None,
    bins: int = 30,
    threshold_bounds: Tuple[int, int] = (1, 4),
    connectivity: Union[int, None] = None,
    num_largest_components: Union[int, None] = None,
) -> np.ndarray:
    """
    Extract foreground from background from H&E WSIs and HHG images using the EntropyMasker algorithm [1].
    Connected component analysis to select the largest component(s) is optional.

    Parameters
    ----------
    image : np.ndarray
        2D RGB image, with color channels at the last axis.
    footprint : np.ndarray, default=`skimage.morphology.disk(5)`
        Footprint to use with `skimage.filters.rank.entropy`.
    bins : int
        Number of bins for the entropy histogram.
    threshold_bounds : tuple[int, int], default=(1, 4)
        Lower and upper threshold for the binned entropy minimum.
    connectivity : int, default=None
        Connectivity of the connected components analysis.
        If None, pick all components.
        `num_largest_components` must to be specified.
    num_largest_components : int, default=None
        Pick the `num_largest_components` largest components.
        If None or <1, pick all components.
        `connectivity` must be specified.

    Returns
    -------
    np.ndarray
        Tissue mask

    References
    ----------
    .. [1] https://doi.org/10.1038/s41598-023-29638-1
    """
    gray = skimage.color.rgb2gray(image)

    if footprint is None:
        footprint = skimage.morphology.disk(5)

    # Tissue has structural information that background does not have.
    # The entropy in tissue is assumed to be bigger.
    ent = skimage.filters.rank.entropy(gray, footprint)

    # Get the local minimum of the entropy histogram.
    # Corresponding entropy is the threshold below which
    # data is considered background.
    hist, bin_edges = np.histogram(ent, bins)
    minimum = scipy.signal.argrelmin(hist)
    for threshold in bin_edges[minimum]:
        if threshold_bounds[0] < threshold < threshold_bounds[1]:
            mask = ent > threshold
            break
    else:
        # If no threshold was within bounds,
        # return full image.
        mask = np.ones_like(gray, dtype=np.bool_)

    if connectivity and num_largest_components:
        return _connected_components(mask, connectivity, num_largest_components)

    return mask


@click.command()
@click.option('--data', help='txt file containing paths to data to create tissue masks for.')
@click.option('--output_dir', help='output directory.')
@click.option('--overwrite', type=bool, default=False, help='overwrite target files.')
def main(data: str, output_dir: str, overwrite: bool):
    """Main entry point for entropy_masker algorithm CLI."""
    with open(data, "r") as f:
        for file in f.readlines():
            path = Path(file[:-1])
            output_dir: Path = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            export_path = output_dir / (path.stem + "_tissue_mask.jpg")
            if export_path.exists() and not overwrite:
                continue
            reader = WSIReader(backend=CFG.backend, level=CFG.level)
            wsi = reader.read(path)
            dims = wsi.level_dimensions[CFG.level]
            hhg_image = np.array(wsi.get_thumbnail(dims))
            tissue_mask = entropy_masker(hhg_image, bins=CFG.bins, threshold_bounds=CFG.threshold_bounds, connectivity=CFG.connectivity, num_largest_components=CFG.num_largest_components)
            export_img = Image.fromarray(tissue_mask.astype(np.uint8) * 255, mode="L")
            export_img.save(export_path)


if __name__ == "__main__":
    main()