#!/usr/bin/env python3
"""
Archeve — flood-hazard raster service (WRI Aqueduct Floods, riverine RP100).

Serves the 1% annual-chance (100-year) river flood-depth layer for any location —
point depth, zonal exposure, a clipped GeoTIFF download, and (in server.py) map
tiles. Global coverage, so India / UAE / KSA / Jordan / Oman / Kuwait are all in.

Source: WRI Aqueduct Floods v2, `inunriver_historical_..._rp00100.tif` (CC BY 4.0).
The 88 MB raster is NOT bundled — it boot-downloads from the public S3 bucket
(same pattern as the GCN250 CN service). EPSG:4326, ~1 km (30") cells, depth in m.
Screening-grade orientation — not a hydraulic model.
"""
import os
import math

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape, box, mapping

FLOOD_PATH = os.environ.get("FLOOD_PATH", "/tmp/aqueduct_rp100.tif")
FLOOD_URL = os.environ.get(
    "FLOOD_URL",
    "https://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2/"
    "inunriver_historical_000000000WATCH_1980_rp00100.tif",
)
NODATA = -9999.0
MAX_DEG = 35.0  # country-scale guard (covers India / KSA)
EXP_MAX_PX = 1400  # cap exposure read dimension; downsample larger countries

_ensured = False


def ensure_raster():
    """Download the Aqueduct RP100 raster if absent (streams to .part then renames)."""
    global _ensured
    if _ensured or os.path.exists(FLOOD_PATH):
        _ensured = True
        return os.path.exists(FLOOD_PATH)
    if not FLOOD_URL:
        return False
    import urllib.request
    import shutil
    tmp = FLOOD_PATH + ".part"
    os.makedirs(os.path.dirname(FLOOD_PATH) or ".", exist_ok=True)
    req = urllib.request.Request(FLOOD_URL, headers={"User-Agent": "archeve-flood/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    os.replace(tmp, FLOOD_PATH)
    # build overviews once so map tiles render fast at country zoom (best-effort)
    try:
        from rasterio.enums import Resampling
        with rasterio.open(FLOOD_PATH, "r+") as ds:
            if not ds.overviews(1):
                ds.build_overviews([2, 4, 8, 16, 32], Resampling.average)
    except Exception:
        pass
    _ensured = True
    return True


def _open():
    ensure_raster()
    if not os.path.exists(FLOOD_PATH):
        raise FileNotFoundError("flood raster unavailable (set FLOOD_PATH or FLOOD_URL)")
    return rasterio.open(FLOOD_PATH)


def depth_at(lat, lon, radius_cells=2):
    """100-year river flood depth (m) near a point. Reports the depth in the exact
    ~1 km cell AND the max within ~2 km — because on a coarse riverine grid the river
    channel can fall in a neighbouring cell, so the local max is the useful screening
    answer for 'is there flood hazard here'. 0.0 = no modeled floodplain nearby."""
    from rasterio.windows import Window
    with _open() as ds:
        try:
            row, col = ds.index(lon, lat)
        except Exception:
            return {"ok": False, "error": "point outside raster"}
        r0 = max(row - radius_cells, 0)
        c0 = max(col - radius_cells, 0)
        win = Window(c0, r0, radius_cells * 2 + 1, radius_cells * 2 + 1)
        a = ds.read(1, window=win, boundless=True, fill_value=NODATA).astype("float32")
        valid = a[(a != NODATA) & np.isfinite(a) & (a >= 0)]
        local_max = float(valid.max()) if valid.size else 0.0
        try:
            pv = float(a[row - r0, col - c0])
            point = 0.0 if (pv == NODATA or not np.isfinite(pv) or pv < 0) else pv
        except Exception:
            point = 0.0
        return {"ok": True, "lat": lat, "lon": lon,
                "depth_m": round(local_max, 2), "point_depth_m": round(point, 2),
                "within_km": round((radius_cells + 0.5) * 1.0, 1),
                "return_period_yr": 100, "flooded": local_max > 0,
                "source": "WRI Aqueduct Floods v2, riverine RP100 (CC BY 4.0)"}


def _bbox_guard(w, s, e, n):
    if (e - w) > MAX_DEG or (n - s) > MAX_DEG or e <= w or n <= s:
        raise ValueError("bbox too large or invalid (max %d°)" % MAX_DEG)


def exposure(geom=None, bbox=None):
    """Flooded-area fraction, mean & max depth over a polygon or bbox."""
    with _open() as ds:
        if geom is not None:
            g = shape(geom)
            w, s, e, n = g.bounds
            _bbox_guard(w, s, e, n)
            arr, _ = rio_mask(ds, [mapping(g)], crop=True, nodata=NODATA, filled=True)
            band = arr[0]
            scale = 1.0
        else:
            w, s, e, n = bbox
            _bbox_guard(w, s, e, n)
            win = from_bounds(w, s, e, n, ds.transform)
            h, wd = int(round(win.height)), int(round(win.width))
            scale = max(h, wd) / float(EXP_MAX_PX)
            if scale > 1:  # downsample large countries so the read stays fast
                band = ds.read(1, window=win, out_shape=(max(1, int(h / scale)), max(1, int(wd / scale))))
            else:
                scale = 1.0
                band = ds.read(1, window=win)
        valid = band[band != NODATA]
        flooded = valid[valid > 0]
        total = int(valid.size)
        if total == 0:
            return {"ok": False, "error": "no land cells in area"}
        meanlat = (s + n) / 2.0
        # each (possibly downsampled) cell represents scale² original ~1 km cells
        cell_km2 = (ds.res[0] * 111.32 * math.cos(math.radians(meanlat))) * (ds.res[1] * 110.57) * (scale * scale)
        return {
            "ok": True,
            "flooded_fraction": round(float(flooded.size) / total, 4),
            "flooded_area_km2": round(flooded.size * cell_km2, 1),
            "mean_depth_m": round(float(flooded.mean()), 2) if flooded.size else 0.0,
            "max_depth_m": round(float(flooded.max()), 2) if flooded.size else 0.0,
            "return_period_yr": 100,
            "source": "WRI Aqueduct Floods v2, riverine RP100 (CC BY 4.0)",
        }


def clip_geotiff(w, s, e, n, out_path):
    """Clip the flood-depth raster to a bbox → GeoTIFF (for download)."""
    _bbox_guard(w, s, e, n)
    with _open() as ds:
        win = from_bounds(w, s, e, n, ds.transform)
        data = ds.read(1, window=win)
        transform = ds.window_transform(win)
        profile = ds.profile.copy()
        profile.update(height=data.shape[0], width=data.shape[1], transform=transform,
                       compress="deflate", driver="GTiff")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
    return out_path
