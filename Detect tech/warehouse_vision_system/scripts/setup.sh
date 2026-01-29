#!/bin/bash

# Warehouse Vision System Setup Script
# Sets up the environment and dependencies for the system

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Jetson
check_jetson() {
    print_status "Checking if running on NVIDIA Jetson..."
    
    if [ -f /etc/nv_tegra_release ]; then
        print_status "NVIDIA Jetson detected"
        cat /etc/nv_tegra_release
    else
        print_warning "Not running on Jetson. Some features may not work."
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    sudo apt-get update
    sudo apt-get install -y \
        python3-pip \
        python3-dev \
        python3-venv \
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
        docker.io \
        docker-compose \
        redis-tools \
        htop \
        iotop
    
    print_status "System dependencies installed"
}

# Setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_status "Python environment setup complete"
}

# Install DeepStream SDK (Jetson only)
install_deepstream() {
    if [ -f /etc/nv_tegra_release ]; then
        print_status "Installing DeepStream SDK..."
        
        # Check if DeepStream is already installed
        if ! command -v deepstream-app &> /dev/null; then
            print_status "DeepStream not found. Installing..."
            
            # Install DeepStream from NVIDIA repository
            sudo apt-get update
            sudo apt-get install -y \
                deepstream-6.0 \
                deepstream-6.0-plugins \
                deepstream-6.0-samples \
                deepstream-6.0-docs
            
            print_status "DeepStream SDK installed"
        else
            print_status "DeepStream SDK already installed"
        fi
    else
        print_warning "Skipping DeepStream installation (not on Jetson)"
    fi
}

# Download pre-trained models
download_models() {
    print_status "Downloading pre-trained models..."
    
    mkdir -p models
    
    # Download YOLOv8 models
    if [ ! -f "models/yolov8n.pt" ]; then
        wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
    fi
    
    if [ ! -f "models/yolov8s.pt" ]; then
        wget -O models/yolov8s.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
    fi
    
    # Create model labels file
    cat > models/labels.txt << EOF
forklift
pallet
worker
package
obstacle
safety_zone
EOF
    
    # Create segmentation labels file
    cat > models/segmentation_labels.txt << EOF
floor
obstacle
restricted_area
safe_path
unknown
EOF
    
    print_status "Models downloaded and configured"
}

# Setup configuration files
setup_config() {
    print_status "Setting up configuration files..."
    
    # Create DeepStream configuration
    mkdir -p config
    
    # Create basic DeepStream config
    cat > config/deepstream_config.yml << EOF
application:
  enable-perf-measurement: 1
  perf-measurement-interval-sec: 5

tiled-display:
  width: 1920
  height: 1080
  rows: 2
  columns: 2
  width: 1280
  height: 720
  gpu-id: 0
  nvbuf-memory-type: 0

source:
  enable: 1
  type: 4
  uri: file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4
  num-sources: 1
  gpu-id: 0
  cudadec-memtype: 0

osd:
  enable: 1
  gpu-id: 0
  border-width: 1
  text-size: 15
  text-color: 1;1;1;1;
  text-bg-color: 0.3;0.3;0.3;1
  font: Serif
  show-clock: 0
  clock-x-offset: 800
  clock-y-offset: 820
  clock-text-size: 12
  clock-color: 1;0;0;0
  nvbuf-memory-type: 0

sgie:
  enable: 1
  gpu-id: 0
  batch-size: 4
  bbox-border-color0: 1;0;0;1
  bbox-border-color1: 0;1;1;1
  bbox-border-color2: 0;0;1;1
  bbox-border-color3: 0;1;0;1
  nvbuf-memory-type: 0
  interval: 0
  gie-unique-id: 2
  model-engine-file: models/primary_detector.etlt
  labelfile-path: models/labels.txt
  config-file: config/config_infer_primary.txt
EOF
    
    print_status "Configuration files created"
}

# Setup monitoring
setup_monitoring() {
    print_status "Setting up monitoring..."
    
    mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
    
    # Create Prometheus configuration
    cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'warehouse-vision'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
    metrics_path: /metrics
    
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
EOF
    
    # Create Grafana datasource configuration
    cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF
    
    print_status "Monitoring setup complete"
}

# Create systemd service
create_systemd_service() {
    print_status "Creating systemd service..."
    
    sudo tee /etc/systemd/system/warehouse-vision.service > /dev/null << EOF
[Unit]
Description=Warehouse Vision System
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$(pwd)
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable warehouse-vision.service
    
    print_status "Systemd service created and enabled"
}

# Setup permissions
setup_permissions() {
    print_status "Setting up permissions..."
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Set executable permissions on scripts
    chmod +x scripts/*.sh
    
    # Create log directory
    mkdir -p logs
    chmod 755 logs
    
    print_status "Permissions configured"
}

# Run system checks
run_system_checks() {
    print_status "Running system checks..."
    
    # Check Docker
    if command -v docker &> /dev/null; then
        print_status "Docker is installed"
        docker --version
    else
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if command -v docker-compose &> /dev/null; then
        print_status "Docker Compose is installed"
        docker-compose --version
    else
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        print_status "Python 3 is installed"
        python3 --version
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check CUDA (if on Jetson)
    if [ -f /etc/nv_tegra_release ]; then
        if command -v nvcc &> /dev/null; then
            print_status "CUDA is available"
            nvcc --version
        else
            print_warning "CUDA is not available"
        fi
    fi
    
    print_status "System checks completed"
}

# Main setup function
main() {
    print_status "Starting Warehouse Vision System setup..."
    
    check_jetson
    install_system_deps
    setup_python_env
    install_deepstream
    download_models
    setup_config
    setup_monitoring
    create_systemd_service
    setup_permissions
    run_system_checks
    
    print_status "Setup completed successfully!"
    print_warning "Please log out and log back in to apply Docker group changes."
    print_status "To start the system, run: sudo systemctl start warehouse-vision"
    print_status "To check status, run: sudo systemctl status warehouse-vision"
    print_status "Web interface will be available at: http://localhost:3000 (Grafana)"
    print_status "Metrics will be available at: http://localhost:9090 (Prometheus)"
}

# Run main function
main "$@"
