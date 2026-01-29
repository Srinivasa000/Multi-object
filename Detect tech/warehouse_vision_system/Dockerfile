# Multi-stage Dockerfile for Warehouse Vision System
# Optimized for NVIDIA Jetson deployment

# Build stage
FROM nvcr.io/nvidia/l4t-base:r32.7.1 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    wget \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install DeepStream SDK
RUN apt-get update && apt-get install -y \
    deepstream-6.0 \
    deepstream-6.0-plugins \
    deepstream-6.0-samples \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Download pre-trained models
RUN mkdir -p /app/models && \
    cd /app/models && \
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt && \
    wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# Production stage
FROM nvcr.io/nvidia/l4t-base:r32.7.1

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libgtk-3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install DeepStream runtime
RUN apt-get update && apt-get install -y \
    deepstream-6.0 \
    deepstream-6.0-plugins \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.8/dist-packages/ /usr/local/lib/python3.8/dist-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Create application directory
WORKDIR /app

# Copy application code
COPY src/ /app/src/
COPY config/ /app/config/
COPY scripts/ /app/scripts/
COPY models/ /app/models/

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/cache

# Set permissions
RUN chmod +x /app/scripts/*.sh

# Create non-root user
RUN useradd -m -u 1000 warehouse && \
    chown -R warehouse:warehouse /app

# Switch to non-root user
USER warehouse

# Expose ports
EXPOSE 8000 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/metrics || exit 1

# Default command
CMD ["python3", "-m", "src.main"]
