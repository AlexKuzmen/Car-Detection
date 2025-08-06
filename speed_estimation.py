import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
import math
from dataclasses import dataclass
from collections import defaultdict, deque

@dataclass
class VehicleTrack:
    """Data class to store vehicle tracking information"""
    track_id: int
    positions: deque = None  # Store recent positions (x, y, timestamp)
    speeds: deque = None    # Store recent speed estimates
    last_speed: float = 0.0
    confidence: float = 0.0
    max_history: int = 30  # Keep last 30 positions for smoothing
    
    def __post_init__(self):
        if self.positions is None:
            self.positions = deque(maxlen=self.max_history)
        if self.speeds is None:
            self.speeds = deque(maxlen=10)

class LaneDetector:
    """Detect lane markings for scale reference"""
    
    def __init__(self):
        self.lane_dash_length_meters = 3.048  # 10 feet in meters
        self.lane_gap_length_meters = 9.144   # 30 feet gap between dashes
        
    def detect_lane_markings(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect lane dash markings using edge detection and line detection
        Returns list of (x1, y1, x2, y2) line segments
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Line detection using Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is None:
            return []
        
        # Filter horizontal lines (lane markings)
        horizontal_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.atan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Filter for near-horizontal lines (within 20 degrees)
            if angle < 20 or angle > 160:
                horizontal_lines.append((x1, y1, x2, y2))
        
        return horizontal_lines
    
    def estimate_scale_from_lanes(self, frame: np.ndarray, 
                                 lane_lines: List[Tuple[int, int, int, int]]) -> float:
        """
        Estimate pixels per meter using lane dash markings
        Returns scale factor (pixels/meter)
        """
        if not lane_lines:
            return 1.0  # Default scale if no lanes detected
        
        # Find the most common y-coordinate (road level)
        y_coords = []
        for x1, y1, x2, y2 in lane_lines:
            y_coords.extend([y1, y2])
        
        if not y_coords:
            return 1.0
        
        # Use median y-coordinate as road level
        road_y = int(np.median(y_coords))
        
        # Find lane dashes near road level
        road_level_lines = []
        for x1, y1, x2, y2 in lane_lines:
            if abs(y1 - road_y) < 20 and abs(y2 - road_y) < 20:
                road_level_lines.append((x1, y1, x2, y2))
        
        if len(road_level_lines) < 2:
            return 1.0
        
        # Calculate average dash length in pixels
        dash_lengths = []
        for x1, y1, x2, y2 in road_level_lines:
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            dash_lengths.append(length)
        
        avg_dash_length_pixels = np.mean(dash_lengths)
        
        # Calculate scale (pixels per meter)
        scale = avg_dash_length_pixels / self.lane_dash_length_meters
        
        return scale

class HomographyEstimator:
    """Estimate homography for perspective correction"""
    
    def __init__(self):
        self.homography_matrix = None
        self.is_calibrated = False
    
    def estimate_homography_from_lanes(self, frame: np.ndarray, 
                                     lane_lines: List[Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Estimate homography matrix using lane markings
        Returns homography matrix for perspective correction
        """
        if len(lane_lines) < 4:
            # Use default homography (no correction)
            h, w = frame.shape[:2]
            src_points = np.float32([[0, h], [w, h], [0, 0], [w, 0]])
            dst_points = np.float32([[0, h], [w, h], [0, 0], [w, 0]])
            return cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Extract points from lane lines
        points = []
        for x1, y1, x2, y2 in lane_lines:
            points.extend([(x1, y1), (x2, y2)])
        
        # Find vanishing point (intersection of parallel lines)
        vanishing_point = self._find_vanishing_point(lane_lines)
        
        if vanishing_point is None:
            # Use default homography
            h, w = frame.shape[:2]
            src_points = np.float32([[0, h], [w, h], [0, 0], [w, 0]])
            dst_points = np.float32([[0, h], [w, h], [0, 0], [w, 0]])
            return cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Create homography matrix for perspective correction
        h, w = frame.shape[:2]
        vx, vy = vanishing_point
        
        # Source points (original image)
        src_points = np.float32([
            [0, h],      # Bottom left
            [w, h],      # Bottom right
            [vx, vy],    # Vanishing point
            [w//2, h//2] # Center point
        ])
        
        # Destination points (corrected image)
        dst_points = np.float32([
            [0, h],      # Bottom left
            [w, h],      # Bottom right
            [w//2, 0],   # Top center (vanishing point)
            [w//2, h//2] # Center point
        ])
        
        homography = cv2.getPerspectiveTransform(src_points, dst_points)
        self.homography_matrix = homography
        self.is_calibrated = True
        
        return homography
    
    def _find_vanishing_point(self, lane_lines: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int]]:
        """Find vanishing point from lane lines"""
        if len(lane_lines) < 2:
            return None
        
        intersections = []
        for i in range(len(lane_lines)):
            for j in range(i + 1, len(lane_lines)):
                x1, y1, x2, y2 = lane_lines[i]
                x3, y3, x4, y4 = lane_lines[j]
                
                # Calculate intersection
                intersection = self._line_intersection((x1, y1, x2, y2), (x3, y3, x4, y4))
                if intersection is not None:
                    intersections.append(intersection)
        
        if not intersections:
            return None
        
        # Return median intersection point
        x_coords = [p[0] for p in intersections]
        y_coords = [p[1] for p in intersections]
        return (int(np.median(x_coords)), int(np.median(y_coords)))
    
    def _line_intersection(self, line1: Tuple[int, int, int, int], 
                          line2: Tuple[int, int, int, int]) -> Optional[Tuple[int, int]]:
        """Calculate intersection of two lines"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (int(x), int(y))

class SpeedEstimator:
    """Estimate vehicle speeds using tracking and scale information"""
    
    def __init__(self, fps: float = 6.0):
        self.fps = fps
        self.frame_time = 1.0 / fps
        self.vehicle_tracks: Dict[int, VehicleTrack] = {}
        self.lane_detector = LaneDetector()
        self.homography_estimator = HomographyEstimator()
        self.scale_factor = 1.0
        self.homography_matrix = None
        
    def update_scale_and_homography(self, frame: np.ndarray):
        """Update scale factor and homography from lane detection"""
        # Detect lane markings
        lane_lines = self.lane_detector.detect_lane_markings(frame)
        
        # Estimate scale from lane dashes
        self.scale_factor = self.lane_detector.estimate_scale_from_lanes(frame, lane_lines)
        
        # Estimate homography for perspective correction
        self.homography_matrix = self.homography_estimator.estimate_homography_from_lanes(frame, lane_lines)
    
    def update_tracks(self, tracks: List, frame: np.ndarray, timestamp: float):
        """Update vehicle tracks and calculate speeds"""
        # Update scale and homography every 30 frames (5 seconds at 6 FPS)
        if int(timestamp * self.fps) % 30 == 0:
            self.update_scale_and_homography(frame)
        
        current_track_ids = set()
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            
            # Calculate center point
            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            
            current_track_ids.add(track_id)
            
            # Initialize track if new
            if track_id not in self.vehicle_tracks:
                self.vehicle_tracks[track_id] = VehicleTrack(track_id=track_id)
            
            # Add position to track history
            track_data = self.vehicle_tracks[track_id]
            track_data.positions.append((center_x, center_y, timestamp))
            
            # Calculate speed if we have enough history
            if len(track_data.positions) >= 3:
                speed = self._calculate_speed(track_data, timestamp)
                track_data.speeds.append(speed)
                track_data.last_speed = speed
                
                # Speeds are automatically limited by deque maxlen
        
        # Remove old tracks
        old_tracks = set(self.vehicle_tracks.keys()) - current_track_ids
        for track_id in old_tracks:
            del self.vehicle_tracks[track_id]
    
    def _calculate_speed(self, track: VehicleTrack, current_time: float) -> float:
        """Calculate speed in km/h using displacement and time"""
        if len(track.positions) < 3:
            return 0.0
        
        # Get recent positions
        positions = list(track.positions)
        
        # Calculate displacement over last few frames
        start_pos = positions[0]
        end_pos = positions[-1]
        
        start_x, start_y, start_time = start_pos
        end_x, end_y, end_time = end_pos
        
        # Apply perspective correction if available
        if self.homography_matrix is not None:
            start_corrected = self._apply_homography(start_x, start_y)
            end_corrected = self._apply_homography(end_x, end_y)
            start_x, start_y = start_corrected
            end_x, end_y = end_corrected
        
        # Calculate displacement in pixels
        displacement_pixels = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        
        # Convert to meters using scale factor
        displacement_meters = displacement_pixels / self.scale_factor
        
        # Calculate time difference
        time_diff = end_time - start_time
        
        if time_diff <= 0:
            return 0.0
        
        # Calculate speed in m/s
        speed_ms = displacement_meters / time_diff
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        return speed_kmh
    
    def _apply_homography(self, x: int, y: int) -> Tuple[int, int]:
        """Apply homography transformation to a point"""
        if self.homography_matrix is None:
            return (x, y)
        
        point = np.array([[x, y]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point.reshape(-1, 1, 2), self.homography_matrix)
        return (int(transformed[0][0][0]), int(transformed[0][0][1]))
    
    def get_vehicle_speed(self, track_id: int) -> float:
        """Get the current speed for a vehicle"""
        if track_id in self.vehicle_tracks:
            return self.vehicle_tracks[track_id].last_speed
        return 0.0
    
    def get_smoothed_speed(self, track_id: int) -> float:
        """Get smoothed speed using moving average"""
        if track_id in self.vehicle_tracks:
            speeds = list(self.vehicle_tracks[track_id].speeds)
            if speeds:
                return np.mean(speeds[-5:])  # Average of last 5 speed estimates
        return 0.0 