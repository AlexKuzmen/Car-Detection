import numpy as np
import datetime
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from speed_estimation import SpeedEstimator
import argparse
import os

##---- Trained using COCO ----##
from helper import create_video_writer


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Vehicle Detection and Speed Estimation')
    parser.add_argument('--input', '-i', type=str, default='On_The_Road.mp4',
                       help='Input video file path')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output video file path')
    parser.add_argument('--fps', type=float, default=6.0,
                       help='Video frame rate (default: 6.0)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for detection (default: 0.5)')
    parser.add_argument('--model', type=str, default='yolov8s.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt'],
                       help='YOLO model to use (default: yolov8s.pt)')
    parser.add_argument('--camera-mode', type=str, default='fixed',
                       choices=['fixed', 'moving'],
                       help='Camera mode: fixed or moving (default: fixed)')
    parser.add_argument('--show-lanes', action='store_true',
                       help='Show detected lane markings')
    parser.add_argument('--show-scale', action='store_true',
                       help='Show current scale factor')

    return parser.parse_args()

def draw_speed_info(frame, track_id, speed, bbox, scale_factor=None):
    """Draw speed information on the frame"""
    xmin, ymin, xmax, ymax = bbox
    
    # Format speed text
    if speed > 0:
        speed_text = f"ID:{track_id} {speed:.1f} km/h"
        color = (0, 255, 0) if speed < 80 else (0, 165, 255) if speed < 120 else (0, 0, 255)
    else:
        speed_text = f"ID:{track_id} -- km/h"
        color = (128, 128, 128)
    
    # Draw bounding box
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    
    # Draw speed label background
    text_size = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    cv2.rectangle(frame, (xmin, ymin - 25), (xmin + text_size[0] + 10, ymin), color, -1)
    
    # Draw speed text
    cv2.putText(frame, speed_text, (xmin + 5, ymin - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def draw_lane_markings(frame, lane_lines):
    """Draw detected lane markings on frame"""
    for x1, y1, x2, y2 in lane_lines:
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

def process_video(input_path: str,
                  output_path: str | None = None,
                  fps: float = 6.0,
                  conf: float = 0.5,
                  model: str = 'yolov8s.pt',
                  camera_mode: str = 'fixed',
                  show_lanes: bool = False,
                  show_scale: bool = False,
                  display: bool = True) -> str:
    """Process a video and return the path to the processed file."""

    # Set up input/output paths
    input_video = input_path
    if output_path is None:
        base_name = os.path.splitext(input_video)[0]
        output_video = f"output_speed_{base_name}.mp4"
    else:
        output_video = output_path

    # Initialize video capture and writer
    video_cap = cv2.VideoCapture(input_video)
    writer = create_video_writer(video_cap, output_video)

    # Check if video opened successfully
    if not video_cap.isOpened():
        raise ValueError(f"Error: Could not open video file {input_video}")

    print(f"Processing {input_video} -> {output_video}")
    print(f"Camera mode: {camera_mode}")
    print(f"Video FPS: {fps}")
    print(f"Model: {model}")

    # Initialize models and estimators
    model_instance = YOLO(model)
    tracker = DeepSort(max_age=50)
    speed_estimator = SpeedEstimator(fps=fps)

    # Processing variables
    frame_count = 0
    start_time = datetime.datetime.now()

    print("Starting vehicle detection and speed estimation...")

    while True:
        # Measure processing time
        frame_start = datetime.datetime.now()

        # Read frame
        ret, frame = video_cap.read()
        if not ret:
            print(f"End of video file... Processed {frame_count} frames")
            break

        frame_count += 1
        current_time = frame_count / fps  # Current timestamp in seconds

        # Detect vehicles using YOLO
        results = model_instance(frame, classes=[2])  # Detect only 'car' class (class id 2)

        # Prepare detections for tracking
        detections_list = []
        for result in results:
            for data in result.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = data
                x = int(x1)
                y = int(y1)
                w = int(x2) - int(x1)
                h = int(y2) - int(y1)
                class_id = int(class_id)

                # Filter by confidence threshold
                if confidence > conf and class_id == 2:
                    detections_list.append([[x, y, w, h], confidence, class_id])

        # Update tracker
        tracks = tracker.update_tracks(detections_list, frame=frame)

        # Update speed estimator
        speed_estimator.update_tracks(tracks, frame, current_time)

        # Draw lane markings if requested
        if show_lanes:
            lane_lines = speed_estimator.lane_detector.detect_lane_markings(frame)
            draw_lane_markings(frame, lane_lines)

        # Process each track
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])

            # Get speed estimate
            speed = speed_estimator.get_smoothed_speed(track_id)

            # Draw speed information
            draw_speed_info(frame, track_id, speed, (xmin, ymin, xmax, ymax))

        # Draw additional information
        frame_end = datetime.datetime.now()
        processing_fps = 1 / (frame_end - frame_start).total_seconds()

        # FPS counter
        cv2.putText(frame, f"FPS: {processing_fps:.1f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Scale factor display
        if show_scale:
            scale_text = f"Scale: {speed_estimator.scale_factor:.2f} px/m"
            cv2.putText(frame, scale_text, (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Vehicle count
        active_vehicles = len([t for t in tracks if t.is_confirmed()])
        cv2.putText(frame, f"Vehicles: {active_vehicles}", (50, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Camera mode indicator
        mode_text = f"Mode: {camera_mode.upper()}"
        cv2.putText(frame, mode_text, (50, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show frame if requested
        if display:
            cv2.imshow("Vehicle Speed Detection", frame)
            if cv2.waitKey(1) == ord("q"):
                print("User interrupted processing")
                break

        # Write frame to output video
        writer.write(frame)

    # Cleanup
    video_cap.release()
    writer.release()
    if display:
        cv2.destroyAllWindows()

    # Print summary
    total_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\nProcessing Summary:")
    print(f"Total frames processed: {frame_count}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average FPS: {frame_count/total_time:.2f}")
    print(f"Output video saved as: {output_video}")

    return output_video


def main():
    """Main function for vehicle detection and speed estimation"""
    args = parse_arguments()
    process_video(input_path=args.input,
                  output_path=args.output,
                  fps=args.fps,
                  conf=args.conf,
                  model=args.model,
                  camera_mode=args.camera_mode,
                  show_lanes=args.show_lanes,
                  show_scale=args.show_scale,
                  display=True)


if __name__ == "__main__":
    main()
