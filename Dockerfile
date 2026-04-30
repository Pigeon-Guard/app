FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/* # delete package index metadata files

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY app.py .

# Create directories for detections, logs and models
RUN mkdir -p /app/detections /app/logs /app/models

# Set environment variables
ENV PGUARD_DETECTION_FOLDER=/app/detections
ENV PGUARD_LOG_FILE=/app/logs/pguard.log

# Don't buffer stdout and stderr in Python; write them immediately
ENV PYTHONUNBUFFERED=1

# Default entrypoint
ENTRYPOINT ["python3", "app.py"]
