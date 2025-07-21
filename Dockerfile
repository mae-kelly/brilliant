# Multi-stage build for optimized production image
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    redis-server \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional ML libraries for production
RUN pip install --no-cache-dir \
    tensorflow==2.13.0 \
    onnx==1.14.0 \
    onnx-tf==1.10.0 \
    scipy==1.11.0 \
    scikit-optimize==0.9.0 \
    psutil==5.9.0

# Production image
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    redis-server \
    sqlite3 \
    supervisor \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set Python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Create app user for security
RUN useradd -m -u 1000 -s /bin/bash appuser

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . /app/

# Create necessary directories
RUN mkdir -p /app/{logs,data,models,tests} && \
    chown -R appuser:appuser /app

# Copy configuration files
COPY docker/supervisord.conf /etc/supervisor/conf.d/
COPY docker/nginx.conf /etc/nginx/sites-available/default
COPY docker/redis.conf /etc/redis/

# Expose ports
EXPOSE 8000 8001 6379

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to app user
USER appuser

# Start services with supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]
