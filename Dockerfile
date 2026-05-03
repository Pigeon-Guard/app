ARG HAILORT_VERSION
ARG PYTHON_VERSION
ARG IMAGE_VARIANT

FROM ghcr.io/pigeon-guard/hailort:${HAILORT_VERSION}-python${PYTHON_VERSION} AS builder
ARG IMAGE_VARIANT

WORKDIR /build
COPY requirements-${IMAGE_VARIANT}.txt .
RUN python -m pip install --no-cache-dir --prefix=/install -r requirements-${IMAGE_VARIANT}.txt

FROM ghcr.io/pigeon-guard/hailort:${HAILORT_VERSION}-python${PYTHON_VERSION}

WORKDIR /app
COPY --from=builder /install /usr/local

COPY app/ ./app/
COPY app.py .

RUN mkdir -p /app/detections /app/logs /app/models

ENV PGUARD_DETECTION_FOLDER=/app/detections
ENV PGUARD_LOG_FILE=/app/logs/pguard.log
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "app.py"]