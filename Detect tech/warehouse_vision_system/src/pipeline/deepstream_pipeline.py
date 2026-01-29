"""
DeepStream Pipeline Implementation for Warehouse Vision System
Handles multi-camera video processing with NVIDIA DeepStream SDK
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pyds
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib

import structlog
import numpy as np

logger = structlog.get_logger()

class DeepStreamPipeline:
    """DeepStream pipeline for multi-camera processing"""
    
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        self.loop = None
        self.bus = None
        self.frame_queue = asyncio.Queue(maxsize=100)
        self.camera_configs = []
        self.pgie_configs = {}
        
        # Pipeline components
        self.sources = []
        self.primary_gie = None
        self.tracker = None
        self.sgie = None  # Secondary GIE for segmentation
        self.analytics = None
        
    async def initialize(self):
        """Initialize DeepStream pipeline"""
        try:
            logger.info("Initializing DeepStream pipeline")
            
            # Initialize GStreamer
            Gst.init(None)
            
            # Setup camera configurations
            await self._setup_camera_configs()
            
            # Create pipeline
            await self._create_pipeline()
            
            # Setup message bus
            await self._setup_message_bus()
            
            logger.info("DeepStream pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            raise
    
    async def _setup_camera_configs(self):
        """Setup camera configurations"""
        camera_count = self.config.get('cameras.total_cameras', 16)
        cameras_per_device = self.config.get('edge_devices.cameras_per_device', 4)
        
        for i in range(camera_count):
            camera_config = {
                'id': i,
                'uri': f'rtsp://192.168.1.{100 + i}/stream',
                'name': f'camera_{i}',
                'width': 1920,
                'height': 1080,
                'fps': self.config.get('cameras.fps', 30)
            }
            self.camera_configs.append(camera_config)
    
    async def _create_pipeline(self):
        """Create the DeepStream pipeline"""
        # Create main pipeline
        pipeline_str = self._generate_pipeline_string()
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Get references to key elements
        self.primary_gie = self.pipeline.get_by_name("primary-gie")
        self.tracker = self.pipeline.get_by_name("tracker")
        self.sgie = self.pipeline.get_by_name("sgie")
        self.analytics = self.pipeline.get_by_name("analytics")
        
        # Configure elements
        await self._configure_elements()
        
    def _generate_pipeline_string(self) -> str:
        """Generate GStreamer pipeline string"""
        batch_size = self.config.get('deepstream.batch_size', 4)
        
        # Source elements
        source_elements = []
        for i, camera in enumerate(self.camera_configs):
            source_elements.append(f"""
uridecodebin name=source_{i} uri={camera['uri']} !
nvarguscamerasrc !
video/x-raw(memory:NVMM),width={camera['width']},height={camera['height']},framerate={camera['fps']}/1 !
nvstreammux name=muxer batch-size={batch_size} batched-push-timeout=40000 live-source=true
            """)
        
        # Main pipeline string
        pipeline_str = f"""
{' '.join(source_elements)}

muxer. ! 
nvvideoconvert ! 
video/x-raw(memory:NVMM),format=NV12 !
nvinfer name=primary-gie config-file-path=config/infer_config_yolo.txt !
nvtracker name=tracker ll-config-file=config/tracker_config.txt !
nvinfer name=sgie config-file-path=config/infer_config_segmentation.txt !
nvvideoconvert !
nvdsosd !
nvvideoconvert !
video/x-raw,format=BGRx !
videoconvert !
video/x-raw,format=BGR !
appsink name=appsink emit-signals=true max-buffers=1 drop=true
        """
        
        return pipeline_str
    
    async def _configure_elements(self):
        """Configure pipeline elements"""
        # Configure primary GIE (object detection)
        if self.primary_gie:
            model_config = self.config.get('models.object_detection', {})
            self.pgie_configs = {
                'model-engine-file': 'models/yolov8n.etlt',
                'labelfile-path': 'models/labels.txt',
                'int8-calib-file': 'models/calibration.bin',
                'network-mode': 1,  # FP16
                'output-tensor-meta': 1,
                'cluster-mode': 2,  # Group rectangles
                'maintain-aspect-ratio': 1,
                'symmetric-padding': 1
            }
            
            for key, value in self.pgie_configs.items():
                self.primary_gie.set_property(key, value)
        
        # Configure tracker
        if self.tracker:
            tracker_config = {
                'll-lib-file': '/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so',
                'll-config-file': 'config/tracker_config.txt',
                'tracker-width': 640,
                'tracker-height': 384,
                'gpu-id': 0,
                'display-tracking-id': 1
            }
            
            for key, value in tracker_config.items():
                self.tracker.set_property(key, value)
        
        # Configure secondary GIE (segmentation)
        if self.sgie:
            sgie_config = {
                'model-engine-file': 'models/bisenetv2.etlt',
                'labelfile-path': 'models/segmentation_labels.txt',
                'network-mode': 1,
                'output-tensor-meta': 1,
                'process-full-frame': 1,
                'operate-on-gie-id': 1,  # Operate on primary GIE output
                'operate-on-class-ids': '0'  # Operate on all classes
            }
            
            for key, value in sgie_config.items():
                self.sgie.set_property(key, value)
    
    async def _setup_message_bus(self):
        """Setup GStreamer message bus"""
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message::error", self._on_error)
        self.bus.connect("message::eos", self._on_eos)
        self.bus.connect("message::warning", self._on_warning)
        
        # Setup appsink for frame extraction
        appsink = self.pipeline.get_by_name("appsink")
        if appsink:
            appsink.connect("new-sample", self._on_new_sample)
    
    def _on_new_sample(self, sink):
        """Handle new frame sample from appsink"""
        try:
            sample = sink.emit("pull-sample")
            if sample is None:
                return Gst.FlowReturn.OK
            
            # Get buffer and info
            buf = sample.get_buffer()
            caps = sample.get_caps()
            
            # Convert to numpy array
            success, info = buf.map(Gst.MapFlags.READ)
            if not success:
                return Gst.FlowReturn.OK
            
            # Get frame dimensions
            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")
            
            # Convert to numpy array
            frame_data = np.frombuffer(info.data, dtype=np.uint8)
            frame = frame_data.reshape((height, width, 3))
            
            # Get metadata
            batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
            frame_data_with_meta = self._extract_metadata(batch_meta, frame)
            
            # Add to queue
            try:
                self.frame_queue.put_nowait(frame_data_with_meta)
            except asyncio.QueueFull:
                # Drop oldest frame if queue is full
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame_data_with_meta)
                except asyncio.QueueEmpty:
                    pass
            
            buf.unmap(info)
            return Gst.FlowReturn.OK
            
        except Exception as e:
            logger.warning(f"Frame processing error: {e}")
            return Gst.FlowReturn.OK
    
    def _extract_metadata(self, batch_meta, frame):
        """Extract metadata from DeepStream batch"""
        frame_data = {
            'frame': frame,
            'timestamp': batch_meta.ntp_timestamp,
            'detections': [],
            'tracks': [],
            'segmentation': None
        }
        
        # Iterate through frame metadata
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            
            # Extract object detections
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                
                detection = {
                    'class_id': obj_meta.class_id,
                    'class_name': self._get_class_name(obj_meta.class_id),
                    'confidence': obj_meta.confidence,
                    'bbox': {
                        'x': obj_meta.rect_params.left,
                        'y': obj_meta.rect_params.top,
                        'width': obj_meta.rect_params.width,
                        'height': obj_meta.rect_params.height
                    },
                    'track_id': obj_meta.object_id if obj_meta.object_id != 0xFFFFFFFF else None
                }
                
                frame_data['detections'].append(detection)
                
                # Add tracking information if available
                if obj_meta.object_id != 0xFFFFFFFF:
                    track = {
                        'track_id': obj_meta.object_id,
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'class_name': detection['class_name']
                    }
                    frame_data['tracks'].append(track)
                
                l_obj = l_obj.next
            
            l_frame = l_frame.next
        
        return frame_data
    
    def _get_class_name(self, class_id):
        """Get class name from class ID"""
        classes = self.config.get('models.object_detection.classes', [])
        if 0 <= class_id < len(classes):
            return classes[class_id]
        return "unknown"
    
    async def start(self):
        """Start the pipeline"""
        try:
            logger.info("Starting DeepStream pipeline")
            
            # Set pipeline to playing state
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Failed to start pipeline")
            
            logger.info("DeepStream pipeline started successfully")
            
        except Exception as e:
            logger.error(f"Pipeline start failed: {e}")
            raise
    
    async def stop(self):
        """Stop the pipeline"""
        try:
            logger.info("Stopping DeepStream pipeline")
            
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
            
            logger.info("DeepStream pipeline stopped")
            
        except Exception as e:
            logger.error(f"Pipeline stop failed: {e}")
    
    async def get_frame(self) -> Optional[Dict[str, Any]]:
        """Get processed frame from queue"""
        try:
            return await asyncio.wait_for(self.frame_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    def _on_error(self, bus, msg):
        """Handle error messages"""
        error, debug = msg.parse_error()
        logger.error(f"Pipeline error: {error}, Debug: {debug}")
    
    def _on_eos(self, bus, msg):
        """Handle end-of-stream messages"""
        logger.info("Pipeline reached end of stream")
    
    def _on_warning(self, bus, msg):
        """Handle warning messages"""
        warning, debug = msg.parse_warning()
        logger.warning(f"Pipeline warning: {warning}, Debug: {debug}")
