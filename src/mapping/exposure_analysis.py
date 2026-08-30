"""
src/mapping/exposure_analysis.py
---------------------------------------------------------------------------
GIS-based flood exposure analysis (Section 15): overlays the modeled flood
extent with land-use/land-cover and administrative boundaries to compute
exposed area. Does NOT estimate economic losses (unsupported without
validated damage/asset-value data — see docs/methodology.md Limitations).
---------------------------------------------------------------------------
"""
from typing import Dict

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape


def flood_raster_to_polygons(flood_map: np.ndarray, transform, crs) -> gpd.GeoDataFrame:
    """Vectorize a binary flood raster into polygons for overlay analysis."""
    mask = flood_map == 1
    polygons = [
        {"geometry": shape(geom), "value": val}
        for geom, val in shapes(flood_map, mask=mask, transform=transform)
    ]
    if not polygons:
        return gpd.GeoDataFrame(columns=["geometry", "value"], crs=crs)
    return gpd.GeoDataFrame(polygons, crs=crs)


def exposure_by_landuse(flood_gdf: gpd.GeoDataFrame, landuse_gdf: gpd.GeoDataFrame,
                         class_field: str = "lulc_class") -> Dict[str, float]:
    """Intersect flood polygons with an LULC layer and sum exposed area (km2) per class."""
    if flood_gdf.empty:
        return {}
    joined = gpd.overlay(landuse_gdf, flood_gdf, how="intersection")
    joined["area_km2"] = joined.geometry.area / 1e6
    result = joined.groupby(class_field)["area_km2"].sum().round(3).to_dict()
    return result


def exposure_by_admin_unit(flood_gdf: gpd.GeoDataFrame, admin_gdf: gpd.GeoDataFrame,
                            admin_field: str = "ADM3_NAME") -> Dict[str, float]:
    """Intersect flood polygons with administrative boundaries (e.g. upazila)
    and sum exposed area (km2) per unit.
    """
    if flood_gdf.empty:
        return {}
    joined = gpd.overlay(admin_gdf, flood_gdf, how="intersection")
    joined["area_km2"] = joined.geometry.area / 1e6
    result = joined.groupby(admin_field)["area_km2"].sum().round(3).to_dict()
    return result


def exposure_near_infrastructure(flood_gdf: gpd.GeoDataFrame, roads_gdf: gpd.GeoDataFrame,
                                  buffer_m: float = 50.0) -> float:
    """Compute km of road network within the flooded extent (buffered slightly
    to account for vectorization/rasterization edge effects).
    """
    if flood_gdf.empty or roads_gdf.empty:
        return 0.0
    flood_union = flood_gdf.geometry.unary_union.buffer(buffer_m)
    clipped = roads_gdf.geometry.intersection(flood_union)
    length_km = clipped.length.sum() / 1000
    return round(length_km, 2)
