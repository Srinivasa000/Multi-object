#!/bin/bash

# Warehouse Vision System Monitoring Script
# Provides real-time monitoring and health checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CONTAINER_NAME="warehouse-vision-system"
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
METRICS_PORT=8000

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

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check if container is running
check_container_status() {
    print_header "Container Status"
    
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q $CONTAINER_NAME; then
        print_status "Container is running"
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        print_error "Container is not running"
        return 1
    fi
}

# Check system resources
check_system_resources() {
    print_header "System Resources"
    
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | awk '{print "  " $2}'
    
    echo "Memory Usage:"
    free -h | grep "Mem:" | awk '{print "  " $3 "/" $2 " (" int($3/$2 * 100) "%)"}'
    
    echo "Disk Usage:"
    df -h / | tail -1 | awk '{print "  " $3 "/" $2 " (" $5 ")"}'
    
    # GPU usage if available
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Usage:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | head -1 | awk '{print "  GPU: " $1 "%, Memory: " $2 "MB/" $3 "MB, Temp: " $4 "°C"}'
    fi
}

# Check container logs for errors
check_container_logs() {
    print_header "Recent Container Logs (Last 50 lines)"
    
    if docker ps --filter "name=$CONTAINER_NAME" -q | grep -q .; then
        docker logs --tail 50 $CONTAINER_NAME 2>&1 | grep -E "(ERROR|WARN|CRITICAL)" || print_status "No errors or warnings in recent logs"
    else
        print_warning "Container not running, cannot check logs"
    fi
}

# Check service endpoints
check_service_endpoints() {
    print_header "Service Endpoints"
    
    # Check metrics endpoint
    if curl -s http://localhost:$METRICS_PORT/metrics > /dev/null 2>&1; then
        print_status "Metrics endpoint is accessible (http://localhost:$METRICS_PORT/metrics)"
    else
        print_warning "Metrics endpoint is not accessible"
    fi
    
    # Check Prometheus
    if curl -s http://localhost:$PROMETHEUS_PORT/api/v1/targets > /dev/null 2>&1; then
        print_status "Prometheus is accessible (http://localhost:$PROMETHEUS_PORT)"
    else
        print_warning "Prometheus is not accessible"
    fi
    
    # Check Grafana
    if curl -s http://localhost:$GRAFANA_PORT/api/health > /dev/null 2>&1; then
        print_status "Grafana is accessible (http://localhost:$GRAFANA_PORT)"
    else
        print_warning "Grafana is not accessible"
    fi
}

# Get key metrics from Prometheus
get_prometheus_metrics() {
    print_header "Key Metrics from Prometheus"
    
    if curl -s http://localhost:$PROMETHEUS_PORT/api/v1/query > /dev/null 2>&1; then
        # CPU usage
        cpu=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=warehouse_cpu_percent" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
        echo "CPU Usage: ${cpu}%"
        
        # Memory usage
        memory=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=warehouse_memory_percent" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
        echo "Memory Usage: ${memory}%"
        
        # Processing FPS
        fps=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=warehouse_processing_fps" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
        echo "Processing FPS: ${fps}"
        
        # Active tracks
        tracks=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=warehouse_active_tracks" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
        echo "Active Tracks: ${tracks}"
        
        # GPU utilization
        gpu=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=warehouse_gpu_utilization" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
        echo "GPU Utilization: ${gpu}%"
        
        # Total frames processed
        frames=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=warehouse_frames_processed_total" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
        echo "Total Frames Processed: ${frames}"
        
    else
        print_warning "Cannot connect to Prometheus"
    fi
}

# Check camera connections
check_cameras() {
    print_header "Camera Status"
    
    # Check if camera devices are available
    if ls /dev/video* 1> /dev/null 2>&1; then
        print_status "Camera devices found:"
        ls -la /dev/video*
    else
        print_warning "No camera devices found"
    fi
    
    # Check RTSP connections if configured
    if docker ps --filter "name=$CONTAINER_NAME" -q | grep -q .; then
        # Check for camera connection logs
        camera_logs=$(docker logs $CONTAINER_NAME 2>&1 | grep -i "camera\|rtsp" | tail -5)
        if [ -n "$camera_logs" ]; then
            echo "Recent camera logs:"
            echo "$camera_logs"
        else
            print_status "No recent camera connection logs"
        fi
    fi
}

# Performance analysis
performance_analysis() {
    print_header "Performance Analysis"
    
    # Get container resource usage
    if docker ps --filter "name=$CONTAINER_NAME" -q | grep -q .; then
        echo "Container Resource Usage:"
        docker stats $CONTAINER_NAME --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    else
        print_warning "Container not running"
    fi
    
    # Check for performance bottlenecks
    if curl -s http://localhost:$PROMETHEUS_PORT/api/v1/query > /dev/null 2>&1; then
        # Check processing latency
        latency=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=histogram_quantile(0.95,rate(warehouse_processing_latency_bucket[5m]))" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
        echo "95th Percentile Latency: ${latency}s"
        
        # Check error rate
        errors=$(curl -s "http://localhost:$PROMETHEUS_PORT/api/v1/query?query=rate(warehouse_errors_total[5m])" | jq -r '.data.result[0].value[1]' 2>/dev/null || echo "N/A")
        echo "Error Rate (5m): ${errors} errors/sec"
    fi
}

# Health check
health_check() {
    print_header "System Health Check"
    
    health_status=0
    
    # Check container
    if ! docker ps --filter "name=$CONTAINER_NAME" -q | grep -q .; then
        print_error "Container is not running"
        health_status=1
    else
        print_status "Container is running"
    fi
    
    # Check critical services
    if ! curl -s http://localhost:$METRICS_PORT/metrics > /dev/null 2>&1; then
        print_error "Metrics service is not responding"
        health_status=1
    else
        print_status "Metrics service is responding"
    fi
    
    # Check system resources
    cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        print_warning "High CPU usage: ${cpu_usage}%"
    fi
    
    memory_usage=$(free | grep Mem | awk '{printf "%.2f", $3/$2 * 100.0}')
    if (( $(echo "$memory_usage > 85" | bc -l) )); then
        print_warning "High memory usage: ${memory_usage}%"
    fi
    
    if [ $health_status -eq 0 ]; then
        print_status "System health check: PASSED"
    else
        print_error "System health check: FAILED"
    fi
    
    return $health_status
}

# Generate report
generate_report() {
    print_header "System Status Report"
    
    echo "Generated at: $(date)"
    echo "================================"
    
    check_container_status
    echo
    check_system_resources
    echo
    get_prometheus_metrics
    echo
    health_check
    
    echo
    echo "================================"
    echo "Report completed"
}

# Continuous monitoring
continuous_monitor() {
    print_status "Starting continuous monitoring (Press Ctrl+C to stop)"
    
    while true; do
        clear
        generate_report
        sleep 10
    done
}

# Show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo
    echo "Commands:"
    echo "  status      - Show container and service status"
    echo "  resources    - Show system resource usage"
    echo "  logs        - Show recent container logs"
    echo "  metrics      - Show key metrics from Prometheus"
    echo "  cameras      - Check camera status"
    echo "  performance  - Performance analysis"
    echo "  health       - Run health check"
    echo "  report       - Generate full status report"
    echo "  continuous  - Continuous monitoring mode"
    echo "  help         - Show this help message"
}

# Main function
main() {
    case "${1:-help}" in
        "status")
            check_container_status
            check_service_endpoints
            ;;
        "resources")
            check_system_resources
            ;;
        "logs")
            check_container_logs
            ;;
        "metrics")
            get_prometheus_metrics
            ;;
        "cameras")
            check_cameras
            ;;
        "performance")
            performance_analysis
            ;;
        "health")
            health_check
            ;;
        "report")
            generate_report
            ;;
        "continuous")
            continuous_monitor
            ;;
        "help"|*)
            show_usage
            ;;
    esac
}

# Check dependencies
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
    exit 1
fi

if ! command -v curl &> /dev/null; then
    print_error "curl is not installed"
    exit 1
fi

# Run main function
main "$@"
