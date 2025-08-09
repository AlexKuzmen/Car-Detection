"""
Configuration file for vehicle speed estimation system
"""

# Camera configuration
CAMERA_CONFIG = {
    'fixed': {
        'fps': 6.0,
        'lane_dash_length_meters': 3.048,  # 10 feet
        'lane_gap_length_meters': 9.144,   # 30 feet
        'perspective_correction': True,
        'scale_update_interval': 30,  # frames
        'speed_smoothing_window': 5,
        'track_history_length': 30
    },
    'moving': {
        'fps': 6.0,
        'lane_dash_length_meters': 3.048,  # 10 feet
        'lane_gap_length_meters': 9.144,   # 30 feet
        'perspective_correction': True,
        'scale_update_interval': 15,  # More frequent updates for moving camera
        'speed_smoothing_window': 3,  # Less smoothing for moving camera
        'track_history_length': 20
    }
}

# Detection configuration
DETECTION_CONFIG = {
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'classes': [2],  # Car class in COCO dataset
    'model_sizes': {
        'yolov8n.pt': 'nano',
        'yolov8s.pt': 'small',
        'yolov8m.pt': 'medium',
        'yolov8l.pt': 'large'
    }
}

# Speed estimation configuration
SPEED_CONFIG = {
    'min_speed_threshold': 0.0,  # km/h
    'max_speed_threshold': 200.0,  # km/h
    'speed_color_thresholds': {
        'low': 80,    # Green
        'medium': 120, # Orange
        'high': 200   # Red
    },
    'calibration': {
        'min_lane_lines': 4,
        'max_angle_deviation': 20,  # degrees
        'road_level_tolerance': 20,  # pixels
        'vanishing_point_tolerance': 50  # pixels
    }
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    'colors': {
        'low_speed': (0, 255, 0),      # Green
        'medium_speed': (0, 165, 255), # Orange
        'high_speed': (0, 0, 255),     # Red
        'no_speed': (128, 128, 128),   # Gray
        'lane_markings': (255, 0, 0),  # Blue
        'text': (255, 255, 255),       # White
        'fps': (0, 0, 255),           # Red
        'frame_counter': (0, 255, 255), # Yellow
        'scale_factor': (255, 255, 0),  # Cyan
        'vehicle_count': (255, 0, 255), # Magenta
        'camera_mode': (255, 255, 255)  # White
    },
    'font_scale': {
        'speed': 0.6,
        'info': 0.7,
        'fps': 1.0,
        'frame_counter': 1.0
    },
    'thickness': {
        'bounding_box': 2,
        'text': 2,
        'lane_markings': 2
    }
}

# Lane detection configuration
LANE_CONFIG = {
    'edge_detection': {
        'gaussian_blur_kernel': (5, 5),
        'canny_low': 50,
        'canny_high': 150
    },
    'line_detection': {
        'rho': 1,
        'theta': 1,
        'threshold': 50,
        'min_line_length': 50,
        'max_line_gap': 10
    },
    'filtering': {
        'max_angle_deviation': 20,  # degrees from horizontal
        'min_line_count': 2
    }
}

# Homography configuration
HOMOGRAPHY_CONFIG = {
    'min_points': 4,
    'ransac_threshold': 3.0,
    'max_iterations': 2000,
    'confidence': 0.99
}

def get_config(camera_mode='fixed'):
    """Get configuration for specified camera mode"""
    if camera_mode not in CAMERA_CONFIG:
        raise ValueError(f"Unknown camera mode: {camera_mode}")
    
    return {
        'camera': CAMERA_CONFIG[camera_mode],
        'detection': DETECTION_CONFIG,
        'speed': SPEED_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'lane': LANE_CONFIG,
        'homography': HOMOGRAPHY_CONFIG
    }

def validate_config(config):
    """Validate configuration parameters"""
    camera = config['camera']
    
    # Validate camera parameters
    assert camera['fps'] > 0, "FPS must be positive"
    assert camera['lane_dash_length_meters'] > 0, "Lane dash length must be positive"
    assert camera['scale_update_interval'] > 0, "Scale update interval must be positive"
    
    # Validate speed parameters
    speed = config['speed']
    assert speed['min_speed_threshold'] >= 0, "Min speed threshold must be non-negative"
    assert speed['max_speed_threshold'] > speed['min_speed_threshold'], "Max speed must be greater than min speed"
    
    return True 