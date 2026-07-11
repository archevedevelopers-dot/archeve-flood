# Archeve flood-hazard API — container image. Build context = this flood-svc/ dir.
# Flood-depth rasters are pre-clipped ~1 km COGs baked in under data/ (Deltares
# Global Flood Maps, coastal, CC BY 4.0) — no runtime download, instant boot.
FROM python:3.11-slim

WORKDIR /app

# rasterio's bundled GDAL needs libexpat at runtime (not in slim)
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi "uvicorn[standard]"

COPY flood_hazard.py server.py ./
COPY data ./data

ENV PORT=8820
EXPOSE 8820

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8820}"]
