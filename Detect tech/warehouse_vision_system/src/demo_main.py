#!/usr/bin/env python3
"""
Demo version of Warehouse Vision System for demonstration purposes
Runs without NVIDIA-specific dependencies for testing on any system
"""

import asyncio
import logging
import time
import json
import random
from datetime import datetime
from pathlib import Path

import structlog
from flask import Flask, jsonify
import threading
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

class DemoWarehouseVisionSystem:
    """Demo version of warehouse vision system"""
    
    def __init__(self):
        self.running = False
        self.frame_count = 0
        self.start_time = time.time()
        self.metrics = {
            'frames_processed': 0,
            'detections': 0,
            'active_tracks': 0,
            'fps': 0,
            'cpu_percent': 0,
            'memory_percent': 0,
            'gpu_utilization': 0,
            'system_latency': 0
        }
        
        # Flask app for metrics endpoint
        self.app = Flask(__name__)
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes for metrics and status"""
        
        @self.app.route('/metrics')
        def metrics():
            """Prometheus-style metrics endpoint"""
            metrics_text = f"""# HELP warehouse_frames_processed_total Total frames processed
# TYPE warehouse_frames_processed_total counter
warehouse_frames_processed_total {self.metrics['frames_processed']}

# HELP warehouse_detection_count Number of detections
# TYPE warehouse_detection_count gauge
warehouse_detection_count {self.metrics['detections']}

# HELP warehouse_active_tracks Number of active tracks
# TYPE warehouse_active_tracks gauge
warehouse_active_tracks {self.metrics['active_tracks']}

# HELP warehouse_processing_fps Processing frames per second
# TYPE warehouse_processing_fps gauge
warehouse_processing_fps {self.metrics['fps']}

# HELP warehouse_cpu_percent CPU usage percentage
# TYPE warehouse_cpu_percent gauge
warehouse_cpu_percent {self.metrics['cpu_percent']}

# HELP warehouse_memory_percent Memory usage percentage
# TYPE warehouse_memory_percent gauge
warehouse_memory_percent {self.metrics['memory_percent']}

# HELP warehouse_gpu_utilization GPU utilization percentage
# TYPE warehouse_gpu_utilization gauge
warehouse_gpu_utilization {self.metrics['gpu_utilization']}

# HELP warehouse_processing_latency Processing latency in seconds
# TYPE warehouse_processing_latency gauge
warehouse_processing_latency {self.metrics['system_latency']}
"""
            return metrics_text, 200, {'Content-Type': 'text/plain; charset=utf-8'}
        
        @self.app.route('/')
        def index():
            """Main dashboard"""
            return jsonify({
                'system': 'Warehouse Vision System - Demo Mode',
                'status': 'running',
                'uptime': time.time() - self.start_time,
                'metrics': self.metrics,
                'timestamp': datetime.now().isoformat()
            })
        
        @self.app.route('/health')
        def health():
            """Health check endpoint"""
            return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
    
    async def simulate_processing(self):
        """Simulate video processing and AI inference"""
        while self.running:
            try:
                # Simulate frame processing
                self.frame_count += 1
                self.metrics['frames_processed'] = self.frame_count
                
                # Simulate detections (random between 0-10)
                self.metrics['detections'] = random.randint(0, 10)
                
                # Simulate active tracks (random between 0-5)
                self.metrics['active_tracks'] = random.randint(0, 5)
                
                # Calculate FPS
                elapsed_time = time.time() - self.start_time
                self.metrics['fps'] = self.frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # Get system metrics
                self.metrics['cpu_percent'] = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                self.metrics['memory_percent'] = memory.percent
                
                # Simulate GPU utilization (random for demo)
                self.metrics['gpu_utilization'] = random.randint(20, 80)
                
                # Simulate processing latency
                self.metrics['system_latency'] = random.uniform(0.01, 0.1)
                
                # Log progress every 100 frames
                if self.frame_count % 100 == 0:
                    logger.info(f"Processed {self.frame_count} frames at {self.metrics['fps']:.1f} FPS")
                
                await asyncio.sleep(0.033)  # ~30 FPS simulation
                
            except Exception as e:
                logger.error(f"Processing simulation error: {e}")
                await asyncio.sleep(0.1)
    
    def run_flask_app(self):
        """Run Flask app in separate thread"""
        self.app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)
    
    async def start(self):
        """Start the demo system"""
        try:
            self.running = True
            logger.info("Starting Warehouse Vision System Demo")
            
            # Start Flask app in background thread
            flask_thread = threading.Thread(target=self.run_flask_app, daemon=True)
            flask_thread.start()
            
            # Start processing simulation
            await self.simulate_processing()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.running = False
            logger.info("Warehouse Vision System Demo stopped")
    
    def stop(self):
        """Stop the demo system"""
        self.running = False

async def main():
    """Main entry point"""
    system = DemoWarehouseVisionSystem()
    
    try:
        await system.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        system.stop()

if __name__ == "__main__":
    asyncio.run(main())
