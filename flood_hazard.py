#!/usr/bin/env python3
"""
Archeve — flood-hazard raster service (Deltares Global Flood Maps, CC BY 4.0).

Serves modeled COASTAL flood depth (GTSM surge+tide extreme sea levels, extended
overland with a bathtub-plus-attenuation model on NASADEM) for:
  rp        : 2 5 10 25 50 100 250     (return period, years)
  scenario  : today (2018) | 2050      (sea-level-rise year)
…as point depth, zonal exposure, a clipped GeoTIFF, and (in server.py) map tiles.
Coverage: India + the Gulf (UAE, KSA, Jordan, Oman, Kuwait) coastlines.

Rasters are pre-clipped ~1 km COGs baked into the image under data/ — no runtime
download, instant boot, no external bucket dependency. EPSG:4326.
Screening-grade orientation — not a hydraulic model or an approval basis.
"""
import os
import math

import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from shapely.geometry import shape, mapping

DATA_DIR = os.environ.get(
    "FLOOD_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
)
NODATA = -9999.0
MAX_DEG = 35.0
EXP_MAX_PX = 1400

RPS = [2, 5, 10, 25, 50, 100, 250]
YEARS = [2018, 2050]
SOURCE = "Deltares Global Flood Maps — coastal, NASADEM ~1 km (CC BY 4.0)"


def scenario_year(scenario) -> int:
    """Map a scenario id to a sea-level-rise year. Anything mentioning 2050/slr
    resolves to 2050; everything else is present-day (2018)."""
    s = str(scenario).lower()
    return 2050 if ("2050" in s or "slr" in s) else 2018


def layer_filename(ftype="coastal", rp=100, scenario="today") -> str:
    rp = int(rp)
    if rp not in RPS:
        raise ValueError("unsupported return period (use %s)" % RPS)
    return f"deltares_{scenario_year(scenario)}_rp{rp:04d}.tif"


def raster_path(ftype="coastal", rp=100, scenario="today") -> str:
    """Resolve the local COG path for a layer. `ftype` is accepted for API
    compatibility but ignored — this dataset is coastal only."""
    path = os.path.join(DATA_DIR, layer_filename(ftype, rp, scenario))
    if not os.path.exists(path):
        raise FileNotFoundError("layer not available: " + os.path.basename(path))
    return path


def ready() -> bool:
    """True once the baked-in rasters are present (they always should be)."""
    try:
        return os.path.exists(raster_path("coastal", 100, "today"))
    except Exception:
        return False


def _open(ftype="coastal", rp=100, scenario="today"):
    return rasterio.open(raster_path(ftype, rp, scenario))


def depth_at(lat, lon, ftype="coastal", rp=100, scenario="today", radius_cells=2):
    """Coastal flood depth (m) near a point — exact ~1 km cell plus the max within
    ~2 km (the inundated fringe can fall in a neighbouring cell). 0.0 = dry here."""
    year = scenario_year(scenario)
    with _open(ftype, rp, scenario) as ds:
        try:
            row, col = ds.index(lon, lat)
        except Exception:
            return {"ok": False, "error": "point outside raster"}
        r0, c0 = max(row - radius_cells, 0), max(col - radius_cells, 0)
        win = Window(c0, r0, radius_cells * 2 + 1, radius_cells * 2 + 1)
        a = ds.read(1, window=win, boundless=True, fill_value=NODATA).astype("float32")
        valid = a[(a != NODATA) & np.isfinite(a) & (a >= 0)]
        local_max = float(valid.max()) if valid.size else 0.0
        try:
            pv = float(a[row - r0, col - c0])
            point = 0.0 if (pv == NODATA or not np.isfinite(pv) or pv < 0) else pv
        except Exception:
            point = 0.0
        return {"ok": True, "lat": lat, "lon": lon, "type": "coastal", "rp": int(rp),
                "scenario": str(year), "year": year,
                "depth_m": round(local_max, 2), "point_depth_m": round(point, 2),
                "within_km": round((radius_cells + 0.5), 1), "flooded": local_max > 0,
                "source": SOURCE}


def _guard(w, s, e, n):
    if (e - w) > MAX_DEG or (n - s) > MAX_DEG or e <= w or n <= s:
        raise ValueError("bbox too large or invalid (max %d deg)" % MAX_DEG)


def exposure(geom=None, bbox=None, ftype="coastal", rp=100, scenario="today"):
    g = shape(geom) if geom is not None else None
    year = scenario_year(scenario)
    with _open(ftype, rp, scenario) as ds:
        w, s, e, n = g.bounds if g is not None else bbox
        _guard(w, s, e, n)
        win = from_bounds(w, s, e, n, ds.transform)
        if g is not None:
            # full-res window read + polygon mask using the native window transform —
            # robust across rasterio versions / overviews (downsampled masking was not).
            band = ds.read(1, window=win).astype("float32")
            from rasterio.features import geometry_mask
            inside = geometry_mask([mapping(g)], out_shape=band.shape,
                                   transform=ds.window_transform(win), invert=True)
            band = np.where(inside, band, NODATA)
            scale = 1.0
        else:
            h, wd = int(round(win.height)), int(round(win.width))
            scale = max(h, wd) / float(EXP_MAX_PX)
            if scale > 1:
                band = ds.read(1, window=win, out_shape=(max(1, int(h / scale)), max(1, int(wd / scale))))
            else:
                scale, band = 1.0, ds.read(1, window=win)
        valid = band[band != NODATA]
        flooded = valid[valid > 0]
        if valid.size == 0:
            return {"ok": False, "error": "no cells in area"}
        meanlat = (s + n) / 2.0
        cell_km2 = (ds.res[0] * 111.32 * math.cos(math.radians(meanlat))) * (ds.res[1] * 110.57) * (scale * scale)
        return {"ok": True, "type": "coastal", "rp": int(rp), "scenario": str(year), "year": year,
                "flooded_fraction": round(float(flooded.size) / valid.size, 4),
                "flooded_area_km2": round(flooded.size * cell_km2, 1),
                "mean_depth_m": round(float(flooded.mean()), 2) if flooded.size else 0.0,
                "max_depth_m": round(float(flooded.max()), 2) if flooded.size else 0.0,
                "source": SOURCE}


def clip_geotiff(w, s, e, n, out_path, ftype="coastal", rp=100, scenario="today"):
    _guard(w, s, e, n)
    with _open(ftype, rp, scenario) as ds:
        win = from_bounds(w, s, e, n, ds.transform)
        data = ds.read(1, window=win)
        profile = ds.profile.copy()
        profile.update(height=data.shape[0], width=data.shape[1],
                       transform=ds.window_transform(win), compress="deflate", driver="GTiff")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
    return out_path
