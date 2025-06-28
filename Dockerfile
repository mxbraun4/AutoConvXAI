# Multi-stage Dockerfile for TalkToModel
# Build with: docker build -t ttm-gpt4 .                    (light version, ~2 min build)
# Build with: docker build -t ttm-gpt4-full --target full . (full version, ~8 min build)

FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies in a single layer for better caching
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables early for better layer caching
ENV PYTHONPATH=/app
ENV FLASK_APP=flask_app_generalized.py
ENV PYTHONUNBUFFERED=1

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with optimization
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# === LIGHT VERSION (default) ===
FROM base as light

# Copy data files (these change less frequently)
COPY data/ data/
COPY templates/ templates/
COPY static/ static/

# Copy source code (copy most stable parts first for better caching)
COPY explain/ explain/
COPY flask_app_generalized.py .
COPY simple_autogen_app.py .

# Copy tests last (they change most frequently)
COPY tests/ tests/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 4455

# Use exec form for better signal handling
CMD ["python3", "flask_app_generalized.py"]

# === FULL VERSION (with torch and heavy dependencies) ===
FROM base as full

# Install heavy dependencies
COPY requirements-heavy.txt .
RUN pip install --no-cache-dir -r requirements-heavy.txt

# Copy data files (these change less frequently)
COPY data/ data/
COPY templates/ templates/
COPY static/ static/

# Copy source code (copy most stable parts first for better caching)
COPY explain/ explain/
COPY flask_app_generalized.py .
COPY simple_autogen_app.py .

# Copy tests last (they change most frequently)
COPY tests/ tests/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 4455

# Use exec form for better signal handling
CMD ["python3", "flask_app_generalized.py"]

# Default to light version
FROM light
