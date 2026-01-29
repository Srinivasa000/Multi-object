"""
System Monitor for Warehouse Vision System
Provides comprehensive monitoring, metrics collection, and health checks
"""

import asyncio
import logging
import time
import psutil
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server

logger = structlog.get_logger()

@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_utilization: float
    gpu_memory_used_gb: float
    gpu_temperature: float
    disk_usage_percent: float
    network_io: Dict[str, float]
    processing_fps: float
    active_tracks: int
    detection_count: int
    reconstruction_confidence: float
    system_latency: float

class SystemMonitor:
    """Comprehensive system monitoring with Prometheus metrics"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        self.metrics_history = []
        self.alert_thresholds = {}
        self.alert_handlers = []
        
        # Prometheus metrics
        self._setup_prometheus_metrics()
        
        # Monitoring configuration
        self.metrics_port = config.get('monitoring.metrics_port', 8000)
        self.health_check_interval = config.get('monitoring.health_check_interval', 30)
        self.max_history_size = 1000
        
        # Initialize alert thresholds
        self._initialize_alert_thresholds()
        
        # Performance tracking
        self.frame_times = []
        self.last_metrics_time = time.time()
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # System metrics
        self.cpu_gauge = Gauge('warehouse_cpu_percent', 'CPU usage percentage')
        self.memory_gauge = Gauge('warehouse_memory_percent', 'Memory usage percentage')
        self.gpu_gauge = Gauge('warehouse_gpu_utilization', 'GPU utilization percentage')
        self.gpu_memory_gauge = Gauge('warehouse_gpu_memory_gb', 'GPU memory usage in GB')
        self.gpu_temp_gauge = Gauge('warehouse_gpu_temperature', 'GPU temperature in Celsius')
        
        # Application metrics
        self.fps_gauge = Gauge('warehouse_processing_fps', 'Processing frames per second')
        self.tracks_gauge = Gauge('warehouse_active_tracks', 'Number of active tracks')
        self.detections_gauge = Gauge('warehouse_detection_count', 'Number of detections')
        self.confidence_gauge = Gauge('warehouse_reconstruction_confidence', '3D reconstruction confidence')
        self.latency_histogram = Histogram('warehouse_processing_latency', 'Processing latency in seconds')
        
        # Counters
        self.frames_processed = Counter('warehouse_frames_processed_total', 'Total frames processed')
        self.errors_total = Counter('warehouse_errors_total', 'Total errors encountered', ['component'])
        self.alerts_total = Counter('warehouse_alerts_total', 'Total alerts triggered', ['severity'])
    
    def _initialize_alert_thresholds(self):
        """Initialize alert thresholds"""
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'gpu_utilization': 90.0,
            'gpu_temperature': 75.0,
            'processing_fps': 15.0,
            'system_latency': 0.2,
            'active_tracks': 100,
            'reconstruction_confidence': 0.3
        }
    
    async def initialize(self):
        """Initialize monitoring system"""
        try:
            logger.info("Initializing system monitor")
            
            # Start Prometheus HTTP server
            start_http_server(self.metrics_port)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
            
            # Initialize GPU monitoring
            await self._initialize_gpu_monitoring()
            
            logger.info("System monitor initialized")
            
        except Exception as e:
            logger.error(f"Monitor initialization failed: {e}")
            raise
    
    async def _initialize_gpu_monitoring(self):
        """Initialize GPU monitoring"""
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_available = True
            logger.info("GPU monitoring initialized")
        except ImportError:
            logger.warning("pynvml not available, GPU monitoring disabled")
            self.gpu_available = False
        except Exception as e:
            logger.warning(f"GPU monitoring initialization failed: {e}")
            self.gpu_available = False
    
    async def start(self):
        """Start monitoring loop"""
        try:
            self.running = True
            logger.info("Starting system monitoring")
            
            # Start monitoring tasks
            await asyncio.gather(
                self._monitoring_loop(),
                self._health_check_loop(),
                self._metrics_cleanup_loop()
            )
            
        except Exception as e:
            logger.error(f"Monitoring start failed: {e}")
            raise
    
    async def stop(self):
        """Stop monitoring"""
        logger.info("Stopping system monitoring")
        self.running = False
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(metrics)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                # Store metrics
                self.metrics_history.append(metrics)
                
                # Limit history size
                if len(self.metrics_history) > self.max_history_size:
                    self.metrics_history = self.metrics_history[-self.max_history_size:]
                
                await asyncio.sleep(5)  # Collect metrics every 5 seconds
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(10)
    
    async def _health_check_loop(self):
        """Health check loop"""
        while self.running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _metrics_cleanup_loop(self):
        """Cleanup old metrics"""
        while self.running:
            try:
                await self._cleanup_old_metrics()
                await asyncio.sleep(300)  # Cleanup every 5 minutes
            except Exception as e:
                logger.error(f"Metrics cleanup error: {e}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system metrics"""
        current_time = time.time()
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_utilization = 0.0
        gpu_memory_used_gb = 0.0
        gpu_temperature = 0.0
        
        if self.gpu_available:
            try:
                import pynvml
                gpu_utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle).gpu
                gpu_memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_memory_used_gb = gpu_memory_info.used / (1024**3)
                gpu_temperature = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception as e:
                logger.warning(f"GPU metrics collection failed: {e}")
        
        # Disk usage
        disk_usage = psutil.disk_usage('/').percent
        
        # Network I/O
        network_io = psutil.net_io_counters()._asdict()
        
        # Calculate FPS
        fps = len(self.frame_times) / (current_time - self.last_metrics_time) if self.frame_times else 0
        
        return SystemMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            gpu_utilization=gpu_utilization,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_temperature=gpu_temperature,
            disk_usage_percent=disk_usage,
            network_io=network_io,
            processing_fps=fps,
            active_tracks=0,  # Will be updated via update_metrics
            detection_count=0,  # Will be updated via update_metrics
            reconstruction_confidence=0.0,  # Will be updated via update_metrics
            system_latency=0.0  # Will be updated via update_metrics
        )
    
    def _update_prometheus_metrics(self, metrics: SystemMetrics):
        """Update Prometheus metrics"""
        self.cpu_gauge.set(metrics.cpu_percent)
        self.memory_gauge.set(metrics.memory_percent)
        self.gpu_gauge.set(metrics.gpu_utilization)
        self.gpu_memory_gauge.set(metrics.gpu_memory_used_gb)
        self.gpu_temp_gauge.set(metrics.gpu_temperature)
        self.fps_gauge.set(metrics.processing_fps)
        self.tracks_gauge.set(metrics.active_tracks)
        self.detections_gauge.set(metrics.detection_count)
        self.confidence_gauge.set(metrics.reconstruction_confidence)
    
    async def _check_alerts(self, metrics: SystemMetrics):
        """Check for alert conditions"""
        alerts = []
        
        # Check CPU
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            alerts.append({
                'metric': 'cpu_percent',
                'value': metrics.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent'],
                'severity': 'warning' if metrics.cpu_percent < 95 else 'critical'
            })
        
        # Check memory
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            alerts.append({
                'metric': 'memory_percent',
                'value': metrics.memory_percent,
                'threshold': self.alert_thresholds['memory_percent'],
                'severity': 'warning' if metrics.memory_percent < 95 else 'critical'
            })
        
        # Check GPU temperature
        if metrics.gpu_temperature > self.alert_thresholds['gpu_temperature']:
            alerts.append({
                'metric': 'gpu_temperature',
                'value': metrics.gpu_temperature,
                'threshold': self.alert_thresholds['gpu_temperature'],
                'severity': 'warning' if metrics.gpu_temperature < 85 else 'critical'
            })
        
        # Check FPS
        if metrics.processing_fps < self.alert_thresholds['processing_fps']:
            alerts.append({
                'metric': 'processing_fps',
                'value': metrics.processing_fps,
                'threshold': self.alert_thresholds['processing_fps'],
                'severity': 'warning'
            })
        
        # Process alerts
        for alert in alerts:
            await self._handle_alert(alert)
    
    async def _handle_alert(self, alert: Dict[str, Any]):
        """Handle alert"""
        logger.warning(
            f"Alert triggered: {alert['metric']} = {alert['value']} (threshold: {alert['threshold']})",
            severity=alert['severity']
        )
        
        # Update Prometheus counter
        self.alerts_total.labels(severity=alert['severity']).inc()
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
    
    async def _perform_health_checks(self):
        """Perform system health checks"""
        health_status = {
            'timestamp': time.time(),
            'status': 'healthy',
            'checks': {}
        }
        
        # Check disk space
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 90:
            health_status['checks']['disk_space'] = 'critical'
            health_status['status'] = 'critical'
        elif disk_usage.percent > 80:
            health_status['checks']['disk_space'] = 'warning'
            if health_status['status'] == 'healthy':
                health_status['status'] = 'warning'
        else:
            health_status['checks']['disk_space'] = 'healthy'
        
        # Check GPU availability
        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                health_status['checks']['gpu'] = 'healthy'
            except:
                health_status['checks']['gpu'] = 'critical'
                health_status['status'] = 'critical'
        else:
            health_status['checks']['gpu'] = 'unavailable'
        
        logger.info(f"Health check completed: {health_status['status']}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics"""
        cutoff_time = time.time() - 3600  # Keep last hour
        self.metrics_history = [
            m for m in self.metrics_history 
            if m.timestamp > cutoff_time
        ]
        
        # Clean up frame times
        cutoff_time_frames = time.time() - 60  # Keep last minute
        self.frame_times = [
            t for t in self.frame_times 
            if t > cutoff_time_frames
        ]
    
    async def update_metrics(self, metrics_update: Dict[str, Any]):
        """Update metrics from application"""
        if 'frame_processed' in metrics_update:
            self.frames_processed.inc()
            self.frame_times.append(time.time())
        
        if 'analytics_latency' in metrics_update:
            self.latency_histogram.observe(metrics_update['analytics_latency'])
        
        # Update latest metrics if available
        if self.metrics_history:
            latest = self.metrics_history[-1]
            if 'detection_count' in metrics_update:
                latest.detection_count = metrics_update['detection_count']
                self.detections_gauge.set(metrics_update['detection_count'])
            
            if 'tracking_count' in metrics_update:
                latest.active_tracks = metrics_update['tracking_count']
                self.tracks_gauge.set(metrics_update['tracking_count'])
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration_minutes: int = 60) -> List[SystemMetrics]:
        """Get metrics history for specified duration"""
        cutoff_time = time.time() - (duration_minutes * 60)
        return [m for m in self.metrics_history if m.timestamp > cutoff_time]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {}
        
        recent_metrics = self.get_metrics_history(10)  # Last 10 minutes
        
        if not recent_metrics:
            return {}
        
        return {
            'avg_cpu': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'avg_gpu': sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics),
            'avg_fps': sum(m.processing_fps for m in recent_metrics) / len(recent_metrics),
            'max_latency': max(m.system_latency for m in recent_metrics),
            'total_frames': len(self.frame_times),
            'uptime': time.time() - (self.metrics_history[0].timestamp if self.metrics_history else time.time())
        }
    
    def add_alert_handler(self, handler):
        """Add alert handler callback"""
        self.alert_handlers.append(handler)
