#!/usr/bin/env python3
"""
Archeve — flood-hazard raster service (WRI Aqueduct Floods, CC BY 4.0).

Serves modeled flood depth for ANY of:
  type      : riverine | coastal
  rp        : 2 5 10 25 50 100 250 500 1000   (return period, years)
  scenario  : historical | 2050 | 2080        (riverine future = RCP8.5)
…as point depth, zonal exposure, a clipped GeoTIFF, and (in server.py) map tiles.
Global coverage → India / UAE / KSA / Jordan / Oman / Kuwait all included.

Each layer's 88 MB GeoTIFF boot-downloads on first use from the public S3 bucket
and is cached in /tmp (overviews built once for fast tiles). EPSG:4326, ~1 km.
Screening-grade orientation — not a hydraulic model.
"""
import os
import math

import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.mask import mask as rio_mask
from shapely.geometry import shape, mapping

BASE_URL = "https://wri-projects.s3.amazonaws.com/AqueductFloodTool/download/v2"
CACHE_DIR = os.environ.get("FLOOD_DIR", "/tmp")
NODATA = -9999.0
MAX_DEG = 35.0
EXP_MAX_PX = 1400

RIVERINE_RP = {2: "00002", 5: "00005", 10: "00010", 25: "00025", 50: "00050",
               100: "00100", 250: "00250", 500: "00500", 1000: "01000"}
COASTAL_RP = {2: "0002", 5: "0005", 10: "0010", 25: "0025", 50: "0050",
              100: "0100", 250: "0250", 500: "0500", 1000: "1000"}
RPS = sorted(RIVERINE_RP.keys())

_ensured = set()


def layer_filename(ftype="riverine", rp=100, scenario="historical"):
    rp = int(rp)
    if ftype == "coastal":
        if rp not in COASTAL_RP:
            raise ValueError("unsupported coastal return period")
        return f"inuncoast_historical_nosub_hist_rp{COASTAL_RP[rp]}_0.tif"
    if rp not in RIVERINE_RP:
        raise ValueError("unsupported return period")
    tag = RIVERINE_RP[rp]
    if scenario in ("2050", "2080"):
        return f"inunriver_rcp8p5_00000NorESM1-M_{scenario}_rp{tag}.tif"
    return f"inunriver_historical_000000000WATCH_1980_rp{tag}.tif"


def raster_path(ftype="riverine", rp=100, scenario="historical"):
    """Resolve, download (once) and return the local path for a layer."""
    fname = layer_filename(ftype, rp, scenario)
    path = os.path.join(CACHE_DIR, fname)
    if path in _ensured or os.path.exists(path):
        _ensured.add(path)
        return path
    import urllib.request
    import shutil
    tmp = path + ".part"
    os.makedirs(CACHE_DIR, exist_ok=True)
    req = urllib.request.Request(f"{BASE_URL}/{fname}", headers={"User-Agent": "archeve-flood/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    os.replace(tmp, path)
    try:
        from rasterio.enums import Resampling
        with rasterio.open(path, "r+") as ds:
            if not ds.overviews(1):
                ds.build_overviews([2, 4, 8, 16, 32], Resampling.average)
    except Exception:
        pass
    _ensured.add(path)
    return path


def _open(ftype="riverine", rp=100, scenario="historical"):
    return rasterio.open(raster_path(ftype, rp, scenario))


def depth_at(lat, lon, ftype="riverine", rp=100, scenario="historical", radius_cells=2):
    """Flood depth (m) near a point — exact ~1 km cell + max within ~2 km (the river
    channel can fall in a neighbouring coarse cell). 0.0 = no modeled floodplain near."""
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
        return {"ok": True, "lat": lat, "lon": lon, "type": ftype, "rp": int(rp), "scenario": scenario,
                "depth_m": round(local_max, 2), "point_depth_m": round(point, 2),
                "within_km": round((radius_cells + 0.5), 1), "flooded": local_max > 0,
                "source": "WRI Aqueduct Floods v2 (CC BY 4.0)"}


def _guard(w, s, e, n):
    if (e - w) > MAX_DEG or (n - s) > MAX_DEG or e <= w or n <= s:
        raise ValueError("bbox too large or invalid (max %d deg)" % MAX_DEG)


def exposure(geom=None, bbox=None, ftype="riverine", rp=100, scenario="historical"):
    from rasterio import Affine
    g = shape(geom) if geom is not None else None
    with _open(ftype, rp, scenario) as ds:
        w, s, e, n = g.bounds if g is not None else bbox
        _guard(w, s, e, n)
        win = from_bounds(w, s, e, n, ds.transform)
        h, wd = int(round(win.height)), int(round(win.width))
        scale = max(h, wd) / float(EXP_MAX_PX)
        if scale > 1:  # downsample large extents so the read+mask stays fast on free CPU
            oh, ow = max(1, int(h / scale)), max(1, int(wd / scale))
        else:
            scale, oh, ow = 1.0, max(1, h), max(1, wd)
        band = ds.read(1, window=win, out_shape=(oh, ow)).astype("float32")
        if g is not None:  # mask to the polygon at the downsampled resolution
            from rasterio.features import geometry_mask
            dt = ds.window_transform(win) * Affine.scale(win.width / ow, win.height / oh)
            inside = geometry_mask([mapping(g)], out_shape=(oh, ow), transform=dt, invert=True)
            band = np.where(inside, band, NODATA)
        valid = band[band != NODATA]
        flooded = valid[valid > 0]
        if valid.size == 0:
            return {"ok": False, "error": "no land cells in area"}
        meanlat = (s + n) / 2.0
        cell_km2 = (ds.res[0] * 111.32 * math.cos(math.radians(meanlat))) * (ds.res[1] * 110.57) * (scale * scale)
        return {"ok": True, "type": ftype, "rp": int(rp), "scenario": scenario,
                "flooded_fraction": round(float(flooded.size) / valid.size, 4),
                "flooded_area_km2": round(flooded.size * cell_km2, 1),
                "mean_depth_m": round(float(flooded.mean()), 2) if flooded.size else 0.0,
                "max_depth_m": round(float(flooded.max()), 2) if flooded.size else 0.0,
                "source": "WRI Aqueduct Floods v2 (CC BY 4.0)"}


def clip_geotiff(w, s, e, n, out_path, ftype="riverine", rp=100, scenario="historical"):
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
