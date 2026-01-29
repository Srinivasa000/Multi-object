# 🚀 Installation Guide

## 📋 Prerequisites

### **System Requirements**
- **OS**: Linux (Ubuntu 18.04+), Windows 10+, macOS 10.15+
- **Python**: 3.9 or higher
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: Minimum 10GB free space
- **GPU**: NVIDIA GPU (optional, for full production features)

### **For Production Deployment (NVIDIA Jetson)**
- **Hardware**: NVIDIA Jetson Nano/Xavier/Orin
- **JetPack**: 4.6+ or 5.0+
- **CUDA**: 10.2+ or 11.0+
- **Docker**: 20.10+
- **DeepStream SDK**: 6.0+

## 🎯 Quick Start (Demo Mode)

### **Step 1: Clone Repository**
```bash
git clone <repository-url>
cd warehouse_vision_system
```

### **Step 2: Setup Python Environment**
```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **Step 3: Run Demo**
```bash
python src/demo_main.py
```

### **Step 4: Access Dashboard**
- **Main Dashboard**: http://localhost:8000
- **Metrics**: http://localhost:8000/metrics
- **Health Check**: http://localhost:8000/health

## 🏭 Full Production Installation

### **Option 1: Docker Deployment (Recommended)**

#### **Step 1: Install Docker**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### **Step 2: Setup Project**
```bash
git clone <repository-url>
cd warehouse_vision_system

# Make setup script executable
chmod +x scripts/setup.sh

# Run setup script
./scripts/setup.sh
```

#### **Step 3: Deploy System**
```bash
# Deploy all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

#### **Step 4: Access Services**
- **Main System**: http://localhost:8000
- **Grafana Dashboard**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **System Monitor**: `./scripts/monitor.sh`

### **Option 2: Native Installation**

#### **Step 1: Install System Dependencies**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-dev python3-venv \
    build-essential cmake pkg-config \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev \
    libgtk-3-dev libatlas-base-dev gfortran \
    wget git curl

# CentOS/RHEL
sudo yum install -y \
    python3 python3-pip python3-devel \
    gcc gcc-c++ cmake pkgconfig \
    libjpeg-turbo-devel libpng-devel libtiff-devel \
    ffmpeg-devel libv4l-devel \
    gtk3-devel atlas-devel \
    wget git curl
```

#### **Step 2: Install NVIDIA Software (Jetson)**
```bash
# Install DeepStream SDK
sudo apt-get update
sudo apt-get install -y \
    deepstream-6.0 \
    deepstream-6.0-plugins \
    deepstream-6.0-samples

# Verify installation
deepstream-app --version
```

#### **Step 3: Setup Python Environment**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install additional dependencies for production
pip install -r requirements.production.txt
```

#### **Step 4: Download Models**
```bash
# Create models directory
mkdir -p models

# Download pre-trained models
cd models
wget -O yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget -O yolov8s.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt

# Convert to TensorRT (if using NVIDIA GPU)
python -c "
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov8n', pretrained=True)
model.export(format='engine', device=0)  # TensorRT
"
```

#### **Step 5: Configure System**
```bash
# Copy configuration files
cp config/system_config.yaml.example config/system_config.yaml

# Edit configuration
nano config/system_config.yaml

# Set camera URLs, model paths, etc.
```

#### **Step 6: Run System**
```bash
# Run main application
python src/main.py

# Or run with specific configuration
python src/main.py --config config/system_config.yaml
```

## 🔧 Configuration

### **System Configuration**
Edit `config/system_config.yaml`:

```yaml
# Camera Configuration
cameras:
  total_cameras: 8
  resolution: "1080p"
  fps: 30
  protocol: "rtsp"

# Model Configuration
models:
  object_detection:
    name: "yolov8n"
    confidence_threshold: 0.5
    classes: ["forklift", "pallet", "worker", "package"]

# Performance Configuration
performance:
  max_latency_ms: 100
  target_fps: 30
  max_memory_gb: 6
```

### **Camera Setup**
```yaml
# Add camera URLs
cameras:
  streams:
    - id: 0
      url: "rtsp://192.168.1.100/stream"
      name: "camera_entrance"
    - id: 1
      url: "rtsp://192.168.1.101/stream"
      name: "camera_loading"
```

### **Model Paths**
```yaml
models:
  object_detection:
    model_path: "models/yolov8n.engine"
    labels_path: "models/labels.txt"
  semantic_segmentation:
    model_path: "models/bisenetv2.engine"
    labels_path: "models/segmentation_labels.txt"
```

## 📊 Monitoring Setup

### **Prometheus Configuration**
Edit `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'warehouse-vision'
    static_configs:
      - targets: ['localhost:8000']
    scrape_interval: 5s
```

### **Grafana Dashboard**
1. Access Grafana: http://localhost:3000
2. Login with admin/admin123
3. Add Prometheus data source
4. Import dashboard from `monitoring/grafana/dashboards/`

## 🧪 Testing Installation

### **Run Tests**
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-asyncio

# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/

# Run specific tests
python -m pytest tests/test_analytics.py
```

### **System Health Check**
```bash
# Check system status
curl http://localhost:8000/health

# Check metrics
curl http://localhost:8000/metrics

# Run monitoring script
./scripts/monitor.sh health
```

## 🚨 Troubleshooting

### **Common Issues**

#### **Docker Issues**
```bash
# Check Docker status
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker

# Check Docker logs
docker-compose logs
```

#### **Python Environment Issues**
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### **NVIDIA/Docker Issues**
```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```

#### **Performance Issues**
```bash
# Check system resources
htop
nvidia-smi
iotop

# Monitor system
./scripts/monitor.sh performance
```

### **Log Files**
```bash
# Application logs
tail -f logs/application.log

# System logs
journalctl -u docker

# Docker logs
docker-compose logs -f warehouse-vision
```

## 📞 Support

### **Getting Help**
- **Documentation**: Check all documentation files
- **Issues**: Create GitHub issue for bugs
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainer for urgent issues

### **System Information**
When reporting issues, include:
- **OS**: `uname -a`
- **Python**: `python --version`
- **Docker**: `docker --version`
- **GPU**: `nvidia-smi` (if applicable)
- **Logs**: Relevant log files

---

**🎉 Congratulations! Your Warehouse Vision System is now installed and ready to use!**
