# Use lightweight Python image
FROM python:3.9-slim

# Working directory
WORKDIR /app

# Prevent Python from writing .pyc files & enable logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies (exclude pywin32 — Windows-only, not available on Linux)
RUN grep -v "^pywin32" requirements.txt > /tmp/requirements_linux.txt && \
    pip install --no-cache-dir -r /tmp/requirements_linux.txt

# Copy project files
COPY . .

# Run as non-root user
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Run API with uvicorn
CMD ["uvicorn", "src.app.app:app", "--host", "0.0.0.0", "--port", "8000"]
