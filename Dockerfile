# Dockerfile for Mental Health MLOps API

# Use Python 3.13 slim image (smaller than full image)
FROM python:3.13-slim

# Set working directory in container
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# - build-essential: For compiling Python packages
# - curl: For health checks
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
# If requirements don't change, Docker reuses this layer
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY data/processed/reference_data.csv ./data/processed/

# Create necessary directories
RUN mkdir -p /app/logs

# Expose port 8000
EXPOSE 8000

# Health check (Docker will periodically check if app is healthy)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
