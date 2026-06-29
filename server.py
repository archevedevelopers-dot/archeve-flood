#!/usr/bin/env python3
"""
Archeve — flood-hazard API (FastAPI). Modeled flood depth, anywhere, any scenario.

  GET  /health
  GET  /layers                                       -> available types/RPs/scenarios
  GET  /depth?lat=&lon=[&type=&rp=&scenario=]        -> depth (m) at a point
  GET  /exposure?bbox=w,s,e,n[&type=&rp=&scenario=]  -> flooded fraction, mean/max
  POST /exposure  {geometry, type?, rp?, scenario?}  -> same, over a polygon
  GET  /download?bbox=w,s,e,n[&...]                  -> clipped GeoTIFF
  GET  /tiles/{z}/{x}/{y}.png?[type=&rp=&scenario=]  -> depth tiles (shallow→deep)
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

app = FastAPI(title="Archeve Flood Hazard (Aqueduct)", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?"
    r"|https://([a-z0-9-]+\.)*archeve\.in"
    r"|https://[a-z0-9-]+\.(vercel\.app|netlify\.app|pages\.dev)",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

DEPTH_MAX_M = 6.0
RAMP = ((168, 212, 255), (8, 48, 107))


def _default_path():
    return fh.raster_path("riverine", 100, "historical")


@app.on_event("startup")
def _prefetch():
    import threading
    threading.Thread(target=lambda: fh.raster_path("riverine", 100, "historical"), daemon=True).start()


@app.get("/health")
def health():
    ok = os.path.exists(os.path.join(fh.CACHE_DIR, fh.layer_filename("riverine", 100, "historical")))
    return {"status": "ok" if ok else "fetching_raster", "raster_present": ok}


@app.get("/layers")
def layers():
    return {
        "types": ["riverine", "coastal"],
        "return_periods": fh.RPS,
        "scenarios": ["historical", "2050", "2080"],
        "scenario_note": "future = RCP8.5 (riverine only); coastal is historical",
        "source": "WRI Aqueduct Floods v2 (CC BY 4.0), ~1 km",
    }


@app.get("/depth")
def depth(lat: float = Query(...), lon: float = Query(...),
          type: str = "riverine", rp: int = 100, scenario: str = "historical"):
    try:
        return fh.depth_at(lat, lon, ftype=type, rp=rp, scenario=scenario)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


def _bbox(b):
    try:
        w, s, e, n = [float(x) for x in b.split(",")]
        return w, s, e, n
    except Exception:
        raise HTTPException(status_code=400, detail="bbox must be 'w,s,e,n'")


@app.get("/exposure")
def exposure_bbox(bbox: str = Query(...), type: str = "riverine", rp: int = 100, scenario: str = "historical"):
    try:
        res = fh.exposure(bbox=_bbox(bbox), ftype=type, rp=rp, scenario=scenario)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not res.get("ok"):
        raise HTTPException(status_code=422, detail=res.get("error"))
    return res


class GeomReq(BaseModel):
    geometry: Optional[dict] = None
    type: Optional[str] = "riverine"
    rp: Optional[int] = 100
    scenario: Optional[str] = "historical"
    features: Optional[list] = None


@app.post("/exposure")
def exposure_geom(req: GeomReq):
    geom = req.geometry
    if not geom and req.features:
        geom = req.features[0].get("geometry")
    if not geom:
        raise HTTPException(status_code=400, detail="no geometry")
    try:
        res = fh.exposure(geom=geom, ftype=req.type or "riverine", rp=req.rp or 100, scenario=req.scenario or "historical")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    if not res.get("ok"):
        raise HTTPException(status_code=422, detail=res.get("error"))
    return res


@app.get("/download")
def download(bbox: str = Query(...), name: str = "flood", type: str = "riverine", rp: int = 100, scenario: str = "historical"):
    w, s, e, n = _bbox(bbox)
    out = os.path.join(tempfile.gettempdir(), f"{name}_{type}_rp{rp}_{scenario}.tif")
    try:
        fh.clip_geotiff(w, s, e, n, out, ftype=type, rp=rp, scenario=scenario)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return FileResponse(out, media_type="image/tiff", filename=os.path.basename(out))


_EMPTY = None


def _empty_tile():
    global _EMPTY
    if _EMPTY is None:
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGBA", (256, 256), (0, 0, 0, 0)).save(buf, "PNG")
        _EMPTY = buf.getvalue()
    return _EMPTY


@app.get("/tiles/{z}/{x}/{y}.png")
def tiles(z: int, x: int, y: int, type: str = "riverine", rp: int = 100, scenario: str = "historical"):
    from rio_tiler.io import Reader
    from rio_tiler.errors import TileOutsideBounds
    from PIL import Image
    try:
        path = fh.raster_path(type, rp, scenario)
        with Reader(path) as r:
            img = r.tile(x, y, z)
    except (TileOutsideBounds, ValueError):
        return Response(_empty_tile(), media_type="image/png")
    except Exception:
        return Response(_empty_tile(), media_type="image/png")
    d = img.data[0].astype("float32")
    valid = (d > 0) & (d != fh.NODATA) & np.isfinite(d)
    t = np.clip(d / DEPTH_MAX_M, 0, 1)
    (r0, g0, b0), (r1, g1, b1) = RAMP
    rgba = np.dstack([r0 + (r1 - r0) * t, g0 + (g1 - g0) * t, b0 + (b1 - b0) * t,
                      np.where(valid, 210, 0)]).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8820)))
