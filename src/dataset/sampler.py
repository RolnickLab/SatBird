# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo samplers."""

import abc
import random
from typing import Iterator, Optional, Tuple, Union

from rtree.index import Index
from torch.utils.data import Sampler

from .utils import BoundingBox


# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Sampler.__module__ = "torch.utils.data"


def _to_tuple(value: Union[Tuple[float, float], float]) -> Tuple[float, float]:
    """Convert value to a tuple if it is not already a tuple.
    Args:
        value: input value
    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, (float, int)):
        return (value, value)
    else:
        return value


def get_random_bounding_box(
    bounds: BoundingBox, size: Union[Tuple[float, float], float]
) -> BoundingBox:
    """Returns a random bounding box within a given bounding box.
    The ``size`` argument can either be:
        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension
    Args:
        bounds: the larger bounding box to sample from
        size: the size of the bounding box to sample
    Returns:
        randomly sampled bounding box from the extent of the input
    """
    t_size: Tuple[float, float] = _to_tuple(size)

    minx = random.uniform(bounds.minx, bounds.maxx - t_size[1])
    maxx = minx + t_size[1]

    miny = random.uniform(bounds.miny, bounds.maxy - t_size[0])
    maxy = miny + t_size[0]

    mint = bounds.mint
    maxt = bounds.maxt

    return BoundingBox(minx, maxx, miny, maxy, mint, maxt)

class GeoSampler(Sampler[BoundingBox], abc.ABC):
    """Abstract base class for sampling from :class:`~torchgeo.datasets.GeoDataset`.

    Unlike PyTorch's :class:`~torch.utils.data.Sampler`, :class:`GeoSampler`
    returns enough geospatial information to uniquely index any
    :class:`~torchgeo.datasets.GeoDataset`. This includes things like latitude,
    longitude, height, width, projection, coordinate system, and time.
    """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """


class RandomGeoSampler(GeoSampler):
    """Samples elements from a region of interest randomly.

    This is particularly useful during training when you want to maximize the size of
    the dataset and return as many random :term:`chips <chip>` as possible.

    This sampler is not recommended for use with tile-based datasets. Use
    :class:`RandomBatchGeoSampler` instead.
    """

    def __init__(
        self,
        index: Index,
        size: Union[Tuple[float, float], float],
        length: int,
        roi: Optional[BoundingBox] = None,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            index: index of a :class:`~torchgeo.datasets.GeoDataset`
            size: dimensions of each :term:`patch` in units of CRS
            length: number of random samples to draw per epoch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
                (defaults to the bounds of ``index``)
        """
        self.index = index
        self.size = _to_tuple(size)
        self.length = length
        if roi is None:
            roi = BoundingBox(*index.bounds)
        self.roi = roi
        self.hits = list(index.intersection(roi, objects=True))

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        for _ in range(len(self)):
            # Choose a random tile
            hit = random.choice(self.hits)
            bounds = BoundingBox(*hit.bounds)

            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size)

            yield bounding_box

    def __len__(self) -> int:
        """Return the number of samples in a single epoch.

        Returns:
            length of the epoch
        """
        return self.length


class GridGeoSampler(GeoSampler):
    """Samples elements in a grid-like fashion.

    This is particularly useful during evaluation when you want to make predictions for
    an entire region of interest. You want to minimize the amount of redundant
    computation by minimizing overlap between :term:`chips <chip>`.

    Usually the stride should be slightly smaller than the chip size such that each chip
    has some small overlap with surrounding chips. This is used to prevent `stitching
    artifacts <https://arxiv.org/abs/1805.12219>`_ when combining each prediction patch.
    The overlap between each chip (``chip_size - stride``) should be approximately equal
    to the `receptive field <https://distill.pub/2019/computing-receptive-fields/>`_ of
    the CNN.

    When sampling from :class:`~torchgeo.datasets.ZipDataset`, the ``index`` should come
    from a non-tile-based dataset if possible.
    """

    def __init__(
        self,
        index: Index,
        size: Union[Tuple[float, float], float],
        stride: Union[Tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
    ) -> None:
        """Initialize a new Sampler instance.

        The ``size`` and ``stride`` arguments can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

        Args:
            index: index of a :class:`~torchgeo.datasets.GeoDataset`
            size: dimensions of each :term:`patch` in units of CRS
            stride: distance to skip between each patch
            roi: region of interest to sample from (minx, maxx, miny, maxy, mint, maxt)
        """
        self.index = index
        self.size = _to_tuple(size)
        self.stride = _to_tuple(stride)
        if roi is None:
            roi = BoundingBox(*index.bounds)
        self.roi = roi
        self.hits = list(index.intersection(roi, objects=True))

        self.length: int = 0
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)

            rows = int((bounds.maxy - bounds.miny - self.size[0]) // self.stride[0]) + 1
            cols = int((bounds.maxx - bounds.minx - self.size[1]) // self.stride[1]) + 1
            self.length += rows * cols

    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits:
            bounds = BoundingBox(*hit.bounds)

            rows = int((bounds.maxy - bounds.miny - self.size[0]) // self.stride[0]) + 1
            cols = int((bounds.maxx - bounds.minx - self.size[1]) // self.stride[1]) + 1

            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]

                    yield BoundingBox(minx, maxx, miny, maxy, mint, maxt)

    def __len__(self) -> int:
        """Return the number of samples over the ROI.

        Returns:
            number of patches that will be sampled
        """
        return self.length


