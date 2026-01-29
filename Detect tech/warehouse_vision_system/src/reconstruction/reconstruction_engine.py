"""
3D Reconstruction Engine for Warehouse Vision System
Handles multi-view stereo reconstruction and spatial mapping
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import cv2

import structlog
import open3d as o3d
from scipy.spatial.transform import Rotation

logger = structlog.get_logger()

@dataclass
class CameraParams:
    """Camera intrinsic and extrinsic parameters"""
    camera_id: int
    intrinsic: np.ndarray  # 3x3 camera matrix
    distortion: np.ndarray  # Distortion coefficients
    extrinsic: np.ndarray  # 4x4 transformation matrix
    width: int
    height: int

@dataclass
class ReconstructionResult:
    """3D reconstruction result"""
    point_cloud: Optional[np.ndarray]
    depth_map: Optional[np.ndarray]
    normals: Optional[np.ndarray]
    confidence: float
    timestamp: float
    processing_time: float

class ReconstructionEngine:
    """Multi-view 3D reconstruction engine"""
    
    def __init__(self, config):
        self.config = config
        self.cameras = {}  # camera_id -> CameraParams
        self.volume = None  # TSDF volume for fusion
        self.last_reconstruction_time = 0
        self.reconstruction_interval = 1.0  # seconds
        
        # Reconstruction parameters
        self.min_depth = config.get('reconstruction.min_depth', 0.1)
        self.max_depth = config.get('reconstruction.max_depth', 10.0)
        self.depth_scale = config.get('reconstruction.depth_scale', 1000.0)
        self.voxel_size = config.get('reconstruction.voxel_size', 0.05)
        
        # Initialize TSDF volume
        self._initialize_tsdf_volume()
        
    async def initialize(self):
        """Initialize reconstruction engine"""
        try:
            logger.info("Initializing 3D reconstruction engine")
            
            # Load camera calibration data
            await self._load_camera_calibration()
            
            # Initialize stereo matching
            await self._initialize_stereo_matching()
            
            logger.info("3D reconstruction engine initialized")
            
        except Exception as e:
            logger.error(f"Reconstruction initialization failed: {e}")
            raise
    
    def _initialize_tsdf_volume(self):
        """Initialize TSDF volume for point cloud fusion"""
        # Define volume bounds (adjust based on warehouse dimensions)
        volume_bounds = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=np.array([-10, -2, -2]),
            max_bound=np.array([10, 5, 5])
        )
        
        # Create TSDF volume
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    
    async def _load_camera_calibration(self):
        """Load camera calibration parameters"""
        # In production, this would load from calibration file
        # For now, create mock calibration data
        
        camera_count = self.config.get('cameras.total_cameras', 16)
        
        for i in range(camera_count):
            # Mock intrinsic parameters (adjust based on actual camera)
            fx, fy = 800, 800  # Focal length
            cx, cy = 960, 540  # Principal point (for 1920x1080)
            
            intrinsic = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ])
            
            distortion = np.zeros(5)  # Assume no distortion for simplicity
            
            # Mock extrinsic parameters (camera poses)
            # In production, these would come from calibration
            angle = 2 * np.pi * i / camera_count
            radius = 5.0
            
            # Camera position in a circle around the scene
            position = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                2.0
            ])
            
            # Camera looking at origin
            rotation = Rotation.from_euler('xyz', [0, -angle, 0]).as_matrix()
            
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rotation
            extrinsic[:3, 3] = rotation @ (-position)
            
            camera_params = CameraParams(
                camera_id=i,
                intrinsic=intrinsic,
                distortion=distortion,
                extrinsic=extrinsic,
                width=1920,
                height=1080
            )
            
            self.cameras[i] = camera_params
        
        logger.info(f"Loaded calibration for {len(self.cameras)} cameras")
    
    async def _initialize_stereo_matching(self):
        """Initialize stereo matching algorithms"""
        # Initialize stereo matcher for multi-view stereo
        self.stereo_matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=256,
            blockSize=11,
            P1=8 * 3 * 11 ** 2,
            P2=32 * 3 * 11 ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
    
    async def process_frame(self, frame_data: Dict[str, Any]) -> Optional[ReconstructionResult]:
        """Process frame for 3D reconstruction"""
        start_time = time.time()
        
        try:
            current_time = frame_data.get('timestamp', time.time())
            
            # Check if reconstruction is needed
            if current_time - self.last_reconstruction_time < self.reconstruction_interval:
                return None
            
            # Get frame and detections
            frame = frame_data.get('frame')
            detections = frame_data.get('detections', [])
            
            if frame is None:
                return None
            
            # Perform depth estimation
            depth_map = await self._estimate_depth(frame)
            
            if depth_map is None:
                return None
            
            # Generate point cloud
            point_cloud = await self._generate_point_cloud(frame, depth_map)
            
            # Update TSDF volume
            await self._update_tsdf_volume(frame, depth_map)
            
            # Extract mesh from TSDF volume
            mesh = await self._extract_mesh()
            
            # Calculate confidence
            confidence = await self._calculate_reconstruction_confidence(point_cloud, detections)
            
            self.last_reconstruction_time = current_time
            
            result = ReconstructionResult(
                point_cloud=point_cloud,
                depth_map=depth_map,
                normals=None,  # Could compute normals if needed
                confidence=confidence,
                timestamp=current_time,
                processing_time=time.time() - start_time
            )
            
            logger.debug(f"Reconstruction completed in {result.processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Reconstruction processing error: {e}")
            return None
    
    async def _estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimate depth from single frame using monocular depth estimation"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use simple depth estimation based on gradient and texture
            # In production, this would use a trained monocular depth model
            
            # Create depth map based on vertical position (simple heuristic)
            height, width = gray.shape
            depth_map = np.zeros((height, width), dtype=np.float32)
            
            # Simple depth estimation: objects higher in image are closer
            for y in range(height):
                depth_value = (y / height) * self.max_depth
                depth_map[y, :] = depth_value
            
            # Add some texture-based variation
            gradient = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_magnitude = np.abs(gradient)
            
            # Normalize and add to depth
            gradient_normalized = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
            depth_map = depth_map * (1.0 + 0.2 * gradient_normalized)
            
            # Clip to valid range
            depth_map = np.clip(depth_map, self.min_depth, self.max_depth)
            
            return depth_map
            
        except Exception as e:
            logger.error(f"Depth estimation failed: {e}")
            return None
    
    async def _generate_point_cloud(self, frame: np.ndarray, depth_map: np.ndarray) -> Optional[np.ndarray]:
        """Generate point cloud from frame and depth map"""
        try:
            height, width = depth_map.shape
            
            # Get camera parameters (use first camera as reference)
            if len(self.cameras) == 0:
                return None
            
            camera = list(self.cameras.values())[0]
            intrinsic = camera.intrinsic
            
            # Create coordinate grids
            u, v = np.meshgrid(np.arange(width), np.arange(height))
            
            # Convert depth to 3D points
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
            
            # Back-project to 3D
            z = depth_map.flatten()
            valid = (z > self.min_depth) & (z < self.max_depth)
            
            x = (u.flatten() - cx) * z / fx
            y = (v.flatten() - cy) * z / fy
            
            # Create point cloud
            points = np.column_stack([x[valid], y[valid], z[valid]])
            colors = frame.reshape(-1, 3)[valid] / 255.0
            
            # Combine points and colors
            point_cloud = np.column_stack([points, colors])
            
            return point_cloud
            
        except Exception as e:
            logger.error(f"Point cloud generation failed: {e}")
            return None
    
    async def _update_tsdf_volume(self, frame: np.ndarray, depth_map: np.ndarray):
        """Update TSDF volume with new depth data"""
        try:
            # Create Open3D images
            color_image = o3d.geometry.Image(frame)
            depth_image = o3d.geometry.Image((depth_map * self.depth_scale).astype(np.uint16))
            
            # Get camera parameters
            if len(self.cameras) == 0:
                return
            
            camera = list(self.cameras.values())[0]
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=camera.width,
                height=camera.height,
                fx=camera.intrinsic[0, 0],
                fy=camera.intrinsic[1, 1],
                cx=camera.intrinsic[0, 2],
                cy=camera.intrinsic[1, 2]
            )
            
            # Create camera pose
            pose = np.linalg.inv(camera.extrinsic)
            
            # Integrate into TSDF volume
            self.volume.integrate(
                depth_image,
                color_image,
                intrinsic,
                pose
            )
            
        except Exception as e:
            logger.error(f"TSDF volume update failed: {e}")
    
    async def _extract_mesh(self) -> Optional[o3d.geometry.TriangleMesh]:
        """Extract mesh from TSDF volume"""
        try:
            mesh = self.volume.extract_triangle_mesh()
            return mesh
        except Exception as e:
            logger.error(f"Mesh extraction failed: {e}")
            return None
    
    async def _calculate_reconstruction_confidence(self, point_cloud: np.ndarray, detections: List[Dict]) -> float:
        """Calculate confidence score for reconstruction"""
        try:
            if point_cloud is None or len(point_cloud) == 0:
                return 0.0
            
            # Base confidence from point cloud density
            point_density = len(point_cloud) / (1920 * 1080)  # Points per pixel
            density_confidence = min(point_density / 0.1, 1.0)  # Normalize to [0, 1]
            
            # Confidence from detection consistency
            detection_confidence = 0.5  # Base confidence
            if detections:
                avg_detection_confidence = np.mean([d['confidence'] for d in detections])
                detection_confidence = avg_detection_confidence
            
            # Combined confidence
            total_confidence = 0.6 * density_confidence + 0.4 * detection_confidence
            
            return float(total_confidence)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def get_reconstruction_metrics(self) -> Dict[str, Any]:
        """Get reconstruction performance metrics"""
        return {
            'cameras_calibrated': len(self.cameras),
            'voxel_size': self.voxel_size,
            'depth_range': (self.min_depth, self.max_depth),
            'last_reconstruction_time': self.last_reconstruction_time,
            'volume_bounds': self.volume.get_axis_aligned_bounding_bound() if self.volume else None
        }
    
    async def reset_reconstruction(self):
        """Reset reconstruction volume"""
        try:
            self._initialize_tsdf_volume()
            logger.info("Reconstruction volume reset")
        except Exception as e:
            logger.error(f"Reconstruction reset failed: {e}")
    
    async def start(self):
        """Start reconstruction engine"""
        logger.info("Starting 3D reconstruction engine")
    
    async def stop(self):
        """Stop reconstruction engine"""
        logger.info("Stopping 3D reconstruction engine")
