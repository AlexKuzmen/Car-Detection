# Vehicle Speed Estimation System

## Overview

This enhanced version of the Car-Detection repository adds real-world speed estimation capabilities to the existing vehicle detection and tracking system. The system can estimate vehicle speeds in km/h using lane markings as scale references, making it suitable for both fixed and moving camera scenarios.

## 🎯 Key Features

- **Real-time Vehicle Detection**: Using YOLOv8 models
- **Multi-Object Tracking**: DeepSORT for consistent vehicle ID tracking
- **Speed Estimation**: Real-world speed calculation in km/h
- **Lane Detection**: Automatic lane marking detection for scale calibration
- **Perspective Correction**: Homography estimation for accurate measurements
- **Multi-Camera Support**: Fixed and moving camera modes
- **Visual Speed Display**: Color-coded speed labels on vehicles

## 🏗️ Architecture

### Core Components

1. **Vehicle Detection Module** (`object_detection_speed_tracking.py`)
   - YOLOv8 object detection
   - DeepSORT tracking
   - Main processing pipeline

2. **Speed Estimation Module** (`speed_estimation.py`)
   - `LaneDetector`: Lane marking detection and scale estimation
   - `HomographyEstimator`: Perspective correction
   - `SpeedEstimator`: Vehicle speed calculation
   - `VehicleTrack`: Track data management

3. **Configuration System** (`config.py`)
   - Camera mode settings
   - Detection parameters
   - Visualization options

### Data Flow

```
Input Video → Frame Extraction → YOLO Detection → DeepSORT Tracking → 
Lane Detection → Scale Estimation → Homography Calculation → 
Speed Estimation → Visual Overlay → Output Video
```

## 🚀 Installation

### Prerequisites

- Python 3.7+
- OpenCV 4.8+
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   ```linux
   python3 -m venv env
   source env/bin/activate
   pip install -r requirements.txt
   ```

2. **Download YOLO models** (if not already present):
   ```bash
   # Models will be downloaded automatically on first use
   # Available models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt
   ```

## 📖 Usage

### Basic Speed Estimation

```bash
# Run with default settings (fixed camera, 6 FPS)
python object_detection_speed_tracking.py --input 1.mp4

# Specify output file
python object_detection_speed_tracking.py --input 1.mp4 --output my_output.mp4
```

### Advanced Options

```bash
# Moving camera mode
python object_detection_speed_tracking.py --input dashcam.mp4 --camera-mode moving

# Show lane markings and scale factor
python object_detection_speed_tracking.py --input 1.mp4 --show-lanes --show-scale

# Use different YOLO model
python object_detection_speed_tracking.py --input 1.mp4 --model yolov8n.pt

# Adjust confidence threshold
python object_detection_speed_tracking.py --input 1.mp4 --conf 0.7

# Custom FPS setting
python object_detection_speed_tracking.py --input 1.mp4 --fps 30.0
```

### Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--input, -i` | str | `1.mp4` | Input video file path |
| `--output, -o` | str | `auto` | Output video file path |
| `--fps` | float | `6.0` | Video frame rate |
| `--conf` | float | `0.5` | Detection confidence threshold |
| `--model` | str | `yolov8s.pt` | YOLO model to use |
| `--camera-mode` | str | `fixed` | Camera mode: fixed/moving |
| `--show-lanes` | flag | `False` | Show detected lane markings |
| `--show-scale` | flag | `False` | Show current scale factor |

## 🔧 Technical Details

### Speed Estimation Algorithm

1. **Lane Detection**:
   - Edge detection using Canny algorithm
   - Line detection using Hough Transform
   - Filtering for horizontal lane markings

2. **Scale Calibration**:
   - Measure lane dash length in pixels
   - Convert to real-world scale (pixels/meter)
   - Use standard lane dash length (3.048m = 10ft)

3. **Perspective Correction**:
   - Estimate vanishing point from lane lines
   - Calculate homography matrix
   - Apply perspective transformation

4. **Speed Calculation**:
   - Track vehicle positions across frames
   - Calculate displacement in corrected coordinates
   - Convert to real-world distance using scale
   - Compute speed: `speed = displacement / time`

### Camera Modes

#### Fixed Camera Mode
- **Use Case**: Traffic monitoring, parking lots
- **Scale Update**: Every 30 frames (5 seconds at 6 FPS)
- **Speed Smoothing**: 5-frame moving average
- **Track History**: 30 positions

#### Moving Camera Mode
- **Use Case**: Dashcam footage, mobile recording
- **Scale Update**: Every 15 frames (2.5 seconds at 6 FPS)
- **Speed Smoothing**: 3-frame moving average
- **Track History**: 20 positions

### Speed Color Coding

- **Green**: 0-80 km/h (normal speed)
- **Orange**: 80-120 km/h (moderate speed)
- **Red**: 120+ km/h (high speed)
- **Gray**: No speed data available

## 📊 Performance

### Processing Speed
- **YOLOv8n**: ~15-20 FPS (faster, less accurate)
- **YOLOv8s**: ~6-8 FPS (balanced)
- **YOLOv8m**: ~4-6 FPS (more accurate)
- **YOLOv8l**: ~2-4 FPS (most accurate)

### Accuracy Considerations
- **Lane Visibility**: Requires clear lane markings
- **Camera Stability**: Moving camera reduces accuracy
- **Distance**: Accuracy decreases with distance
- **Lighting**: Poor lighting affects lane detection

## 🛠️ Configuration

### Customizing Parameters

Edit `config.py` to modify:

```python
# Camera settings
CAMERA_CONFIG = {
    'fixed': {
        'fps': 6.0,
        'lane_dash_length_meters': 3.048,  # Adjust for different regions
        'scale_update_interval': 30,
        'speed_smoothing_window': 5
    }
}

# Speed thresholds
SPEED_CONFIG = {
    'speed_color_thresholds': {
        'low': 80,    # km/h
        'medium': 120, # km/h
        'high': 200   # km/h
    }
}
```

### Regional Adaptations

For different countries/regions, adjust lane marking parameters:

```python
# US Standard (10ft dashes, 30ft gaps)
'lane_dash_length_meters': 3.048

# European Standard (3m dashes, 9m gaps)
'lane_dash_length_meters': 3.0

# Custom measurements
'lane_dash_length_meters': 2.5  # Measure your local lane markings
```

## 🔍 Troubleshooting

### Common Issues

1. **No Speed Estimates**:
   - Check if lane markings are visible
   - Verify video has sufficient resolution
   - Ensure vehicles are tracked for multiple frames

2. **Inaccurate Speeds**:
   - Calibrate lane dash length for your region
   - Check camera stability
   - Verify FPS setting matches video

3. **Poor Lane Detection**:
   - Improve video lighting
   - Check for clear lane markings
   - Adjust edge detection parameters

4. **Low Processing Speed**:
   - Use smaller YOLO model (yolov8n.pt)
   - Reduce video resolution
   - Use GPU acceleration

### Debug Mode

Enable debug visualizations:

```bash
python object_detection_speed_tracking.py --input 1.mp4 --show-lanes --show-scale
```

## 📈 Future Enhancements

### Planned Features
- [ ] Multi-lane speed estimation
- [ ] Vehicle type classification
- [ ] Speed limit violation detection
- [ ] Traffic flow analysis
- [ ] Real-time web interface
- [ ] Database logging
- [ ] API integration

### Research Areas
- [ ] Deep learning-based lane detection
- [ ] Camera calibration from video
- [ ] Multi-camera fusion
- [ ] Weather-resistant detection
- [ ] Night vision support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project extends the original Car-Detection repository. Please refer to the original license terms.

## 🙏 Acknowledgments

- Original Car-Detection repository
- YOLOv8 by Ultralytics
- DeepSORT implementation
- OpenCV community
- Computer vision research community

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include video samples and error logs

---

**Note**: This system provides estimates and should not be used for legal speed enforcement without proper calibration and validation. 