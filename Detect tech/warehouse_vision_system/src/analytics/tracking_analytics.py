"""
Tracking and Analytics Engine for Warehouse Vision System
Handles multi-object tracking, counting, and anomaly detection
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import time

import structlog
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = structlog.get_logger()

@dataclass
class Track:
    """Track data structure"""
    track_id: int
    class_name: str
    bbox: Dict[str, float]
    confidence: float
    timestamp: float
    velocity: Optional[Tuple[float, float]] = None
    trajectory: deque = None
    
    def __post_init__(self):
        if self.trajectory is None:
            self.trajectory = deque(maxlen=30)  # Keep last 30 positions

@dataclass
class AnalyticsResult:
    """Analytics processing result"""
    detections: List[Dict]
    tracks: List[Dict]
    counts: Dict[str, int]
    anomalies: List[Dict]
    latency: float
    timestamp: float

class TrackingAnalytics:
    """Advanced tracking and analytics engine"""
    
    def __init__(self, config):
        self.config = config
        self.tracks = {}  # track_id -> Track
        self.track_counter = 0
        self.last_frame_time = time.time()
        
        # Analytics components
        self.counting_zones = self._setup_counting_zones()
        self.anomaly_detector = self._setup_anomaly_detector()
        self.clustering_model = self._setup_clustering_model()
        
        # Performance metrics
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        
    async def initialize(self):
        """Initialize analytics components"""
        try:
            logger.info("Initializing tracking analytics engine")
            
            # Load counting zones from config
            await self._load_counting_zones()
            
            # Initialize anomaly detection
            await self._initialize_anomaly_detection()
            
            # Initialize clustering models
            await self._initialize_clustering()
            
            logger.info("Tracking analytics engine initialized")
            
        except Exception as e:
            logger.error(f"Analytics initialization failed: {e}")
            raise
    
    def _setup_counting_zones(self) -> Dict[str, Dict]:
        """Setup counting zones for different areas"""
        return {
            'entrance': {
                'polygon': [(100, 100), (200, 100), (200, 200), (100, 200)],
                'direction': 'in',
                'count': 0
            },
            'exit': {
                'polygon': [(800, 100), (900, 100), (900, 200), (800, 200)],
                'direction': 'out',
                'count': 0
            },
            'loading_zone': {
                'polygon': [(400, 300), (600, 300), (600, 500), (400, 500)],
                'direction': 'both',
                'count': 0
            }
        }
    
    def _setup_anomaly_detector(self) -> IsolationForest:
        """Setup anomaly detection model"""
        return IsolationForest(
            contamination=self.config.get('analytics.anomaly_detection.contamination', 0.1),
            random_state=42,
            n_estimators=100
        )
    
    def _setup_clustering_model(self) -> KMeans:
        """Setup clustering model for pattern discovery"""
        return KMeans(
            n_clusters=self.config.get('analytics.clustering.n_clusters', 8),
            random_state=42,
            n_init=10
        )
    
    async def _load_counting_zones(self):
        """Load counting zones from configuration"""
        # This would load from config file or database
        pass
    
    async def _initialize_anomaly_detection(self):
        """Initialize anomaly detection with historical data"""
        # In production, this would load historical data
        # For now, initialize with empty data
        self.anomaly_data = []
        self.feature_scaler = StandardScaler()
    
    async def _initialize_clustering(self):
        """Initialize clustering for pattern discovery"""
        # Initialize trajectory clustering
        self.trajectory_data = []
        self.trajectory_clusters = {}
    
    async def process_frame(self, frame_data: Dict[str, Any]) -> AnalyticsResult:
        """Process frame through analytics pipeline"""
        start_time = time.time()
        
        try:
            # Update tracks
            updated_tracks = await self._update_tracks(frame_data)
            
            # Perform counting
            counts = await self._perform_counting(updated_tracks)
            
            # Detect anomalies
            anomalies = await self._detect_anomalies(updated_tracks)
            
            # Update clustering
            await self._update_clustering(updated_tracks)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Create result
            result = AnalyticsResult(
                detections=frame_data.get('detections', []),
                tracks=[self._track_to_dict(track) for track in updated_tracks.values()],
                counts=counts,
                anomalies=anomalies,
                latency=processing_time,
                timestamp=frame_data.get('timestamp', time.time())
            )
            
            self.frame_count += 1
            return result
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            # Return empty result on error
            return AnalyticsResult(
                detections=[],
                tracks=[],
                counts={},
                anomalies=[],
                latency=time.time() - start_time,
                timestamp=time.time()
            )
    
    async def _update_tracks(self, frame_data: Dict[str, Any]) -> Dict[int, Track]:
        """Update tracking with new detections"""
        detections = frame_data.get('detections', [])
        current_time = frame_data.get('timestamp', time.time())
        
        # Update existing tracks and create new ones
        updated_tracks = {}
        
        # Process detections with track IDs
        for detection in detections:
            track_id = detection.get('track_id')
            
            if track_id is not None and track_id in self.tracks:
                # Update existing track
                track = self.tracks[track_id]
                track.bbox = detection['bbox']
                track.confidence = detection['confidence']
                track.timestamp = current_time
                
                # Calculate velocity
                if len(track.trajectory) > 0:
                    last_pos = track.trajectory[-1]
                    current_pos = (detection['bbox']['x'], detection['bbox']['y'])
                    dt = current_time - track.timestamp
                    
                    if dt > 0:
                        velocity = (
                            (current_pos[0] - last_pos[0]) / dt,
                            (current_pos[1] - last_pos[1]) / dt
                        )
                        track.velocity = velocity
                
                track.trajectory.append((detection['bbox']['x'], detection['bbox']['y']))
                updated_tracks[track_id] = track
                
            else:
                # Create new track
                new_track_id = self._generate_track_id()
                track = Track(
                    track_id=new_track_id,
                    class_name=detection['class_name'],
                    bbox=detection['bbox'],
                    confidence=detection['confidence'],
                    timestamp=current_time
                )
                track.trajectory.append((detection['bbox']['x'], detection['bbox']['y']))
                
                self.tracks[new_track_id] = track
                updated_tracks[new_track_id] = track
        
        # Remove old tracks
        max_age = self.config.get('models.tracking.max_disappeared_frames', 30)
        tracks_to_remove = []
        
        for track_id, track in self.tracks.items():
            if current_time - track.timestamp > max_age / 30.0:  # Convert frames to seconds
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return updated_tracks
    
    def _generate_track_id(self) -> int:
        """Generate unique track ID"""
        self.track_counter += 1
        return self.track_counter
    
    async def _perform_counting(self, tracks: Dict[int, Track]) -> Dict[str, int]:
        """Perform object counting in defined zones"""
        counts = {}
        
        for zone_name, zone_config in self.counting_zones.items():
            zone_count = 0
            
            for track in tracks.values():
                if self._is_in_zone(track.bbox, zone_config['polygon']):
                    zone_count += 1
            
            counts[zone_name] = zone_count
            zone_config['count'] = zone_count
        
        return counts
    
    def _is_in_zone(self, bbox: Dict[str, float], polygon: List[Tuple[float, float]]) -> bool:
        """Check if bounding box center is in polygon"""
        center_x = bbox['x'] + bbox['width'] / 2
        center_y = bbox['y'] + bbox['height'] / 2
        
        # Simple point-in-polygon test
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > center_y) != (yj > center_y)) and \
               (center_x < (xj - xi) * (center_y - yi) / (yj - yi) + xi):
                inside = not inside
            
            j = i
        
        return inside
    
    async def _detect_anomalies(self, tracks: Dict[int, Track]) -> List[Dict]:
        """Detect anomalies in tracking data"""
        anomalies = []
        
        if len(tracks) < 2:
            return anomalies
        
        # Extract features for anomaly detection
        features = []
        track_ids = []
        
        for track_id, track in tracks.items():
            feature_vector = [
                track.bbox['x'],
                track.bbox['y'],
                track.bbox['width'],
                track.bbox['height'],
                track.confidence,
                track.velocity[0] if track.velocity else 0,
                track.velocity[1] if track.velocity else 0,
                len(track.trajectory)
            ]
            features.append(feature_vector)
            track_ids.append(track_id)
        
        # Detect anomalies
        if len(features) > 0:
            try:
                # Scale features
                features_scaled = self.feature_scaler.fit_transform(features)
                
                # Predict anomalies
                anomaly_labels = self.anomaly_detector.fit_predict(features_scaled)
                
                # Collect anomalies
                for i, (track_id, is_anomaly) in enumerate(zip(track_ids, anomaly_labels)):
                    if is_anomaly == -1:  # Anomaly detected
                        track = tracks[track_id]
                        anomaly = {
                            'type': 'tracking_anomaly',
                            'track_id': track_id,
                            'class_name': track.class_name,
                            'confidence': track.confidence,
                            'velocity': track.velocity,
                            'trajectory_length': len(track.trajectory),
                            'timestamp': track.timestamp
                        }
                        anomalies.append(anomaly)
                        
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    async def _update_clustering(self, tracks: Dict[int, Track]):
        """Update clustering for pattern discovery"""
        # Collect trajectory data
        trajectories = []
        
        for track in tracks.values():
            if len(track.trajectory) > 5:  # Only use tracks with sufficient history
                trajectory = list(track.trajectory)
                # Normalize trajectory to fixed length
                if len(trajectory) > 20:
                    # Sample trajectory points
                    indices = np.linspace(0, len(trajectory) - 1, 20, dtype=int)
                    trajectory = [trajectory[i] for i in indices]
                
                trajectories.append(trajectory)
        
        # Update clustering model periodically
        if len(trajectories) > 10 and self.frame_count % 100 == 0:
            try:
                # Flatten trajectories for clustering
                flattened_trajectories = []
                for traj in trajectories:
                    flattened = [coord for point in traj for coord in point]
                    # Pad to fixed length
                    while len(flattened) < 40:  # 20 points * 2 coordinates
                        flattened.extend([0, 0])
                    flattened_trajectories.append(flattened[:40])
                
                # Perform clustering
                cluster_labels = self.clustering_model.fit_predict(flattened_trajectories)
                
                # Update cluster assignments
                self.trajectory_clusters = {}
                for i, label in enumerate(cluster_labels):
                    if label not in self.trajectory_clusters:
                        self.trajectory_clusters[label] = []
                    self.trajectory_clusters[label].append(trajectories[i])
                
                logger.info(f"Updated trajectory clustering: {len(self.trajectory_clusters)} clusters")
                
            except Exception as e:
                logger.warning(f"Clustering update failed: {e}")
    
    def _track_to_dict(self, track: Track) -> Dict:
        """Convert track to dictionary"""
        return {
            'track_id': track.track_id,
            'class_name': track.class_name,
            'bbox': track.bbox,
            'confidence': track.confidence,
            'timestamp': track.timestamp,
            'velocity': track.velocity,
            'trajectory_length': len(track.trajectory)
        }
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics"""
        if len(self.processing_times) == 0:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': np.max(self.processing_times),
            'min_processing_time': np.min(self.processing_times),
            'fps': self.frame_count / (time.time() - self.last_frame_time) if self.frame_count > 0 else 0,
            'active_tracks': len(self.tracks),
            'total_tracks_processed': self.track_counter
        }
    
    async def start(self):
        """Start analytics engine"""
        logger.info("Starting tracking analytics engine")
    
    async def stop(self):
        """Stop analytics engine"""
        logger.info("Stopping tracking analytics engine")
