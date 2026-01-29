#!/usr/bin/env python3
"""
Main entry point for the Warehouse Vision System
Real-time multi-camera intelligent vision system for industrial warehouse automation
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path

import yaml
import structlog

from src.pipeline.deepstream_pipeline import DeepStreamPipeline
from src.analytics.tracking_analytics import TrackingAnalytics
from src.reconstruction.reconstruction_engine import ReconstructionEngine
from src.monitoring.system_monitor import SystemMonitor
from src.utils.config_manager import ConfigManager
from src.utils.failure_handler import FailureHandler

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

class WarehouseVisionSystem:
    """Main system orchestrator for warehouse vision processing"""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config = ConfigManager(config_path)
        self.pipeline = None
        self.analytics = None
        self.reconstruction = None
        self.monitor = None
        self.failure_handler = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    async def initialize(self):
        """Initialize all system components"""
        try:
            logger.info("Initializing Warehouse Vision System")
            
            # Initialize failure handler first
            self.failure_handler = FailureHandler(self.config)
            
            # Initialize DeepStream pipeline
            self.pipeline = DeepStreamPipeline(self.config)
            await self.pipeline.initialize()
            
            # Initialize analytics engine
            self.analytics = TrackingAnalytics(self.config)
            await self.analytics.initialize()
            
            # Initialize 3D reconstruction engine
            self.reconstruction = ReconstructionEngine(self.config)
            await self.reconstruction.initialize()
            
            # Initialize system monitor
            self.monitor = SystemMonitor(self.config)
            await self.monitor.initialize()
            
            logger.info("System initialization completed successfully")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    async def start(self):
        """Start the main processing loop"""
        try:
            self.running = True
            logger.info("Starting Warehouse Vision System")
            
            # Start all components
            await asyncio.gather(
                self.pipeline.start(),
                self.analytics.start(),
                self.reconstruction.start(),
                self.monitor.start()
            )
            
            # Main processing loop
            while self.running:
                await self._process_frame()
                await asyncio.sleep(0.001)  # 1ms loop interval
                
        except Exception as e:
            logger.error(f"System runtime error: {e}")
            await self.failure_handler.handle_error(e)
    
    async def _process_frame(self):
        """Process a single frame through the pipeline"""
        try:
            # Get frame from pipeline
            frame_data = await self.pipeline.get_frame()
            if frame_data is None:
                return
            
            # Process through analytics
            analytics_result = await self.analytics.process_frame(frame_data)
            
            # Process through reconstruction if needed
            if self.config.get('reconstruction.enabled', True):
                reconstruction_result = await self.reconstruction.process_frame(frame_data)
            else:
                reconstruction_result = None
            
            # Update monitoring metrics
            await self.monitor.update_metrics({
                'frame_processed': True,
                'analytics_latency': analytics_result.get('latency', 0),
                'detection_count': len(analytics_result.get('detections', [])),
                'tracking_count': len(analytics_result.get('tracks', []))
            })
            
        except Exception as e:
            logger.warning(f"Frame processing error: {e}")
            await self.failure_handler.handle_frame_error(e)
    
    async def stop(self):
        """Gracefully stop the system"""
        logger.info("Stopping Warehouse Vision System")
        self.running = False
        
        # Stop all components
        if self.pipeline:
            await self.pipeline.stop()
        if self.analytics:
            await self.analytics.stop()
        if self.reconstruction:
            await self.reconstruction.stop()
        if self.monitor:
            await self.monitor.stop()
        
        logger.info("System stopped successfully")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown")
        asyncio.create_task(self.stop())

async def main():
    """Main entry point"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Warehouse Vision System")
    parser.add_argument("--config", default="config/system_config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and start system
    system = WarehouseVisionSystem(args.config)
    
    try:
        await system.initialize()
        await system.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        await system.stop()

if __name__ == "__main__":
    asyncio.run(main())
