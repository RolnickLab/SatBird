import os
import sys
from typing import Tuple
from datetime import datetime, timedelta



class BoundingBox(Tuple[float, float, float, float, float, float]):
    """Data class for indexing spatiotemporal data.
    Attributes:
        minx (float): western boundary
        maxx (float): eastern boundary
        miny (float): southern boundary
        maxy (float): northern boundary
        mint (float): earliest boundary
        maxt (float): latest boundary
    """

    def __new__(
        cls,
        minx: float,
        maxx: float,
        miny: float,
        maxy: float,
        mint: float,
        maxt: float,
    ) -> "BoundingBox":
        """Create a new instance of BoundingBox.
        Args:
            minx: western boundary
            maxx: eastern boundary
            miny: southern boundary
            maxy: northern boundary
            mint: earliest boundary
            maxt: latest boundary
        Raises:
            ValueError: if bounding box is invalid
                (minx > maxx, miny > maxy, or mint > maxt)
        """
        if minx > maxx:
            raise ValueError(f"Bounding box is invalid: 'minx={minx}' > 'maxx={maxx}'")
        if miny > maxy:
            raise ValueError(f"Bounding box is invalid: 'miny={miny}' > 'maxy={maxy}'")
        if mint > maxt:
            raise ValueError(f"Bounding box is invalid: 'mint={mint}' > 'maxt={maxt}'")

        # Using super() doesn't work with mypy, see:
        # https://stackoverflow.com/q/60611012/5828163
        return tuple.__new__(cls, [minx, maxx, miny, maxy, mint, maxt])

    def __init__(
        self,
        minx: float,
        maxx: float,
        miny: float,
        maxy: float,
        mint: float,
        maxt: float,
    ) -> None:
        """Initialize a new instance of BoundingBox.
        Args:
            minx: western boundary
            maxx: eastern boundary
            miny: southern boundary
            maxy: northern boundary
            mint: earliest boundary
            maxt: latest boundary
        """
        self.minx = minx
        self.maxx = maxx
        self.miny = miny
        self.maxy = maxy
        self.mint = mint
        self.maxt = maxt

    def __getnewargs__(self) -> Tuple[float, float, float, float, float, float]:
        """Values passed to the ``__new__()`` method upon unpickling.
        Returns:
            tuple of bounds
        """
        return self.minx, self.maxx, self.miny, self.maxy, self.mint, self.maxt

    def __repr__(self) -> str:
        """Return the formal string representation of the object.
        Returns:
            formal string representation
        """
        return (
            f"{self.__class__.__name__}(minx={self.minx}, maxx={self.maxx}, "
            f"miny={self.miny}, maxy={self.maxy}, mint={self.mint}, maxt={self.maxt})"
        )

    def intersects(self, other: "BoundingBox") -> bool:
        """Whether or not two bounding boxes intersect.
        Args:
            other: another bounding box
        Returns:
            True if bounding boxes intersect, else False
        """
        return (
            self.minx <= other.maxx
            and self.maxx >= other.minx
            and self.miny <= other.maxy
            and self.maxy >= other.miny
            and self.mint <= other.maxt
            and self.maxt >= other.mint
        )


def disambiguate_timestamp(date_str: str, format: str) -> Tuple[float, float]:
    """Disambiguate partial timestamps.
    TorchGeo stores the timestamp of each file in a spatiotemporal R-tree. If the full
    timestamp isn't known, a file could represent a range of time. For example, in the
    CDL dataset, each mask spans an entire year. This method returns the maximum
    possible range of timestamps that ``date_str`` could belong to. It does this by
    parsing ``format`` to determine the level of precision of ``date_str``.
    Args:
        date_str: string representing date and time of a data point
        format: format codes accepted by :meth:`datetime.datetime.strptime`
    Returns:
        (mint, maxt) tuple for indexing
    """
    mint = datetime.strptime(date_str, format)

    # TODO: This doesn't correctly handle literal `%%` characters in format
    # TODO: May have issues with time zones, UTC vs. local time, and DST
    # TODO: This is really tedious, is there a better way to do this?

    if not any([f"%{c}" in format for c in "yYcxG"]):
        # No temporal info
        return 0, sys.maxsize
    elif not any([f"%{c}" in format for c in "bBmjUWcxV"]):
        # Year resolution
        maxt = datetime(mint.year + 1, 1, 1)
    elif not any([f"%{c}" in format for c in "aAwdjcxV"]):
        # Month resolution
        if mint.month == 12:
            maxt = datetime(mint.year + 1, 1, 1)
        else:
            maxt = datetime(mint.year, mint.month + 1, 1)
    elif not any([f"%{c}" in format for c in "HIcX"]):
        # Day resolution
        maxt = mint + timedelta(days=1)
    elif not any([f"%{c}" in format for c in "McX"]):
        # Hour resolution
        maxt = mint + timedelta(hours=1)
    elif not any([f"%{c}" in format for c in "ScX"]):
        # Minute resolution
        maxt = mint + timedelta(minutes=1)
    elif not any([f"%{c}" in format for c in "f"]):
        # Second resolution
        maxt = mint + timedelta(seconds=1)
    else:
        # Microsecond resolution
        maxt = mint + timedelta(microseconds=1)

    mint -= timedelta(microseconds=1)
    maxt -= timedelta(microseconds=1)

    return mint.timestamp(), maxt.timestamp()