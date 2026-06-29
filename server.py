#!/usr/bin/env python3
"""
Archeve — flood-hazard API (FastAPI). 100-year river flood depth, anywhere.

  GET  /health
  GET  /depth?lat=&lon=                 -> modeled flood depth (m) at a point
  GET  /exposure?bbox=w,s,e,n           -> flooded fraction, mean/max depth
  POST /exposure   {geometry}           -> same, over a polygon
  GET  /download?bbox=w,s,e,n           -> clipped GeoTIFF (download)
  GET  /tiles/{z}/{x}/{y}.png           -> flood-depth map tiles (shallow→deep blue)

Run:  pip install -r requirements.txt fastapi "uvicorn[standard]"
      python3 server.py   # :8820
"""
import io
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flood_hazard as fh  # noqa: E402

import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import Response, FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from typing import Optional  # noqa: E402

app = FastAPI(title="Archeve Flood Hazard (Aqueduct RP100)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    r"|https://([a-z0-9-]+\.)*archeve\.in"
    r"|https://[a-z0-9-]+\.(vercel\.app|netlify\.app|pages\.dev)",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

DEPTH_MAX_M = 6.0          # colour ramp saturates here
RAMP = ((168, 212, 255), (8, 48, 107))  # light → deep blue


@app.on_event("startup")
def _prefetch():
    import threading
    if not os.path.exists(fh.FLOOD_PATH) and fh.FLOOD_URL:
        threading.Thread(target=lambda: fh.ensure_raster(), daemon=True).start()


@app.get("/health")
def health():
    ok = os.path.exists(fh.FLOOD_PATH)
    return {"status": "ok" if ok else "fetching_raster", "raster_present": ok, "flood_url": fh.FLOOD_URL}


@app.get("/depth")
def depth(lat: float = Query(...), lon: float = Query(...)):
    try:
        return fh.depth_at(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


def _parse_bbox(bbox: str):
    try:
        w, s, e, n = [float(x) for x in bbox.split(",")]
        return w, s, e, n
    except Exception:
        raise HTTPException(status_code=400, detail="bbox must be 'w,s,e,n'")


@app.get("/exposure")
def exposure_bbox(bbox: str = Query(..., description="w,s,e,n")):
    try:
        res = fh.exposure(bbox=_parse_bbox(bbox))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not res.get("ok"):
        raise HTTPException(status_code=422, detail=res.get("error"))
    return res


class GeomReq(BaseModel):
    geometry: Optional[dict] = None
    type: Optional[str] = None
    features: Optional[list] = None


@app.post("/exposure")
def exposure_geom(req: GeomReq):
    geom = req.geometry
    if not geom and req.type == "FeatureCollection" and req.features:
        geom = req.features[0].get("geometry")
    if not geom:
        raise HTTPException(status_code=400, detail="no geometry")
    try:
        res = fh.exposure(geom=geom)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not res.get("ok"):
        raise HTTPException(status_code=422, detail=res.get("error"))
    return res


@app.get("/download")
def download(bbox: str = Query(..., description="w,s,e,n"), name: str = "flood_rp100"):
    w, s, e, n = _parse_bbox(bbox)
    out = os.path.join(tempfile.gettempdir(), f"{name}.tif")
    try:
        fh.clip_geotiff(w, s, e, n, out)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return FileResponse(out, media_type="image/tiff", filename=f"{name}_rp100_aqueduct.tif")


_EMPTY_PNG = None


def _empty_tile():
    global _EMPTY_PNG
    if _EMPTY_PNG is None:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGBA", (256, 256), (0, 0, 0, 0)).save(buf, "PNG")
        _EMPTY_PNG = buf.getvalue()
    return _EMPTY_PNG


@app.get("/tiles/{z}/{x}/{y}.png")
def tiles(z: int, x: int, y: int):
    from rio_tiler.io import Reader
    from rio_tiler.errors import TileOutsideBounds
    from PIL import Image
    try:
        with Reader(fh.FLOOD_PATH) as r:
            img = r.tile(x, y, z)
    except TileOutsideBounds:
        return Response(_empty_tile(), media_type="image/png")
    except Exception:
        return Response(_empty_tile(), media_type="image/png")

    d = img.data[0].astype("float32")
    valid = (d > 0) & (d != fh.NODATA) & np.isfinite(d)
    t = np.clip(d / DEPTH_MAX_M, 0, 1)
    (r0, g0, b0), (r1, g1, b1) = RAMP
    R = (r0 + (r1 - r0) * t)
    G = (g0 + (g1 - g0) * t)
    B = (b0 + (b1 - b0) * t)
    A = np.where(valid, 210, 0)
    rgba = np.dstack([R, G, B, A]).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8820)))
