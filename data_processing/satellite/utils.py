import os
import numpy as np
from shapely.geometry import Polygon
import pandas as pd
from math import cos, radians

#from https://github.com/nasaharvest/cropharvest/blob/main/cropharvest/eo/eo.py
#utils for computing bounding box centred of specified lat-lon

def metre_per_degree(lat: float):
    # https://gis.stackexchange.com/questions/75528/understanding-terms-in
    # -length-of-degree-formula
    # see the link above to explain the magic numbers
    m_per_degree_lat = (
        111132.954
        + (-559.822 * cos(radians(2.0 * lat)))
        + (1.175 * cos(radians(4.0 * lat)))
        + (-0.0023 * cos(radians(6 * lat)))
    )
    m_per_degree_lon = (
        (111412.84 * cos(radians(lat)))
        + (-93.5 * cos(radians(3 * lat)))
        + (0.118 * cos(radians(5 * lat)))
    )

    return m_per_degree_lat, m_per_degree_lon

def bounding_box_from_centre(
    mid_lat, mid_lon, surrounding_metres):

    m_per_deg_lat, m_per_deg_lon = metre_per_degree(mid_lat)

    if isinstance(surrounding_metres, int):
        surrounding_metres = (surrounding_metres, surrounding_metres)

    surrounding_lat, surrounding_lon = surrounding_metres

    deg_lat = surrounding_lat / m_per_deg_lat
    deg_lon = surrounding_lon / m_per_deg_lon

    max_lat, min_lat = mid_lat + deg_lat, mid_lat - deg_lat
    max_lon, min_lon = mid_lon + deg_lon, mid_lon - deg_lon

    return Polygon([[min_lon, min_lat], [min_lon, max_lat], [max_lon, max_lat], [max_lon, min_lat]]
    )
