# Archeve flood-hazard API — container image. Build context = this flood-svc/ dir.
# The 88 MB Aqueduct RP100 raster boot-downloads from public S3 (FLOOD_URL).
FROM python:3.11-slim

WORKDIR /app

# rasterio's bundled GDAL needs libexpat at runtime (not in slim)
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi "uvicorn[standard]"

COPY flood_hazard.py server.py ./

ENV PORT=8820
ENV FLOOD_DIR=/tmp
EXPOSE 8820

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8820}"]
