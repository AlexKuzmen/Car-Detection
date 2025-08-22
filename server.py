from flask import Flask, request, send_file, after_this_request, jsonify
from flask_cors import CORS
import os
import tempfile
import json
import uuid
from datetime import datetime
from object_detection_speed_tracking import process_video

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create uploads directory if it doesn't exist
UPLOADS_DIR = 'uploads'
if not os.path.exists(UPLOADS_DIR):
    os.makedirs(UPLOADS_DIR)

# Create subdirectories for better organization
ORIGINAL_VIDEOS_DIR = os.path.join(UPLOADS_DIR, 'original_videos')
PROCESSED_VIDEOS_DIR = os.path.join(UPLOADS_DIR, 'processed_videos')
REPORTS_DIR = os.path.join(UPLOADS_DIR, 'reports')

for directory in [ORIGINAL_VIDEOS_DIR, PROCESSED_VIDEOS_DIR, REPORTS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# In-memory storage for reports (in production, use a database)
reports = []

@app.route('/process-video', methods=['POST'])
def process_video_endpoint():
    """Process video with YOLO speed detection"""
    if 'video' not in request.files:
        return {'error': 'No video file provided'}, 400

    file = request.files['video']
    if file.filename == '':
        return {'error': 'Empty filename'}, 400

    # Generate unique filename for the original video
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    original_filename = f"original_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
    original_path = os.path.join(ORIGINAL_VIDEOS_DIR, original_filename)
    
    # Save the original video file for manual review
    file.save(original_path)
    print(f"Original video saved for review: {original_path}")

    # Save input video to a temporary file for processing
    suffix = os.path.splitext(file.filename)[1] or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_temp:
        file.seek(0)  # Reset file pointer
        file.save(input_temp.name)
        input_path = input_temp.name

    # Create output temporary file path
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_temp.close()
    output_path = output_temp.name

    try:
        # Check if the video file is valid by checking its size
        file_size = os.path.getsize(input_path)
        if file_size < 100:  # If file is too small, it's probably not a valid video
            print(f"Warning: Video file is too small ({file_size} bytes), skipping YOLO processing")
            return jsonify({
                'status': 'warning',
                'message': 'Video file appears to be invalid or empty, skipping YOLO processing',
                'original_video_path': original_path,
                'processed_video_path': None
            }), 200
        
        # Process video with YOLO speed detection
        process_video(input_path, output_path, display=False)
        
        # Save processed video to processed_videos directory
        processed_filename = f"processed_{timestamp}_{uuid.uuid4().hex[:8]}.mp4"
        processed_path = os.path.join(PROCESSED_VIDEOS_DIR, processed_filename)
        
        # Copy the processed video to processed_videos directory
        import shutil
        shutil.copy2(output_path, processed_path)
        
        print(f"Video processed successfully: {processed_path}")
        
        return jsonify({
            'status': 'success',
            'message': 'Video processed successfully',
            'original_video_path': original_path,
            'processed_video_path': processed_path
        }), 200
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        # Return a warning instead of error, so the report submission can continue
        return jsonify({
            'status': 'warning',
            'message': f'Video processing failed: {str(e)}, but report submission will continue',
            'original_video_path': original_path,
            'processed_video_path': None
        }), 200
    finally:
        # Clean up temporary files
        try:
            os.remove(input_path)
            os.remove(output_path)
        except:
            pass

@app.route('/api/submit-report', methods=['POST'])
def submit_report():
    """Submit a speed report with processed video"""
    try:
        # Get the JSON data from form
        report_data = request.form.get('reportData')
        if not report_data:
            return jsonify({'error': 'No report data provided'}), 400
        
        # Parse the JSON data
        data = json.loads(report_data)
        
        # Generate unique report ID
        report_id = str(uuid.uuid4())
        
        # Create report object
        report = {
            'id': report_id,
            'licensePlate': data.get('licensePlate'),
            'currentSpeed': data.get('currentSpeed'),
            'speedLimit': data.get('speedLimit'),
            'location': data.get('location'),
            'timestamp': data.get('timestamp'),
            'videoUri': data.get('videoUri'),
            'status': 'pending',
            'submittedAt': datetime.now().isoformat()
        }
        
        # Handle video file upload if present
        if 'video' in request.files:
            video_file = request.files['video']
            if video_file:
                # Generate timestamp for filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                license_plate = data.get('licensePlate', 'UNKNOWN')
                
                # Save video file with descriptive filename
                video_filename = f"report_{timestamp}_{license_plate}_{report_id[:8]}.mp4"
                video_path = os.path.join(ORIGINAL_VIDEOS_DIR, video_filename)
                video_file.save(video_path)
                report['videoPath'] = video_path
                print(f"Report video saved: {video_path}")
                
                # Also save report data as JSON file for manual review
                report_filename = f"report_{timestamp}_{license_plate}_{report_id[:8]}.json"
                report_path = os.path.join(REPORTS_DIR, report_filename)
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"Report data saved: {report_path}")
        
        # Store report (in production, save to database)
        reports.append(report)
        
        print(f"Report submitted: {report}")
        
        return jsonify({
            'status': 'success',
            'message': 'Report submitted successfully',
            'reportId': report_id
        }), 200
        
    except Exception as e:
        print(f"Error submitting report: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports', methods=['GET'])
def get_reports():
    """Get all reports"""
    return jsonify({
        'status': 'success',
        'reports': reports
    }), 200

@app.route('/api/reports/<report_id>', methods=['GET'])
def get_report(report_id):
    """Get a specific report by ID"""
    report = next((r for r in reports if r['id'] == report_id), None)
    if not report:
        return jsonify({'error': 'Report not found'}), 404
    
    return jsonify({
        'status': 'success',
        'report': report
    }), 200

@app.route('/api/reports/<report_id>/confirm', methods=['POST'])
def confirm_report(report_id):
    """Confirm a report and calculate reward"""
    try:
        report = next((r for r in reports if r['id'] == report_id), None)
        if not report:
            return jsonify({'error': 'Report not found'}), 404
        
        # Calculate fine based on speed over limit
        speed_over_limit = report['currentSpeed'] - report['speedLimit']
        if speed_over_limit > 0:
            # Basic fine calculation (you can adjust this)
            fine_amount = speed_over_limit * 10  # $10 per km/h over limit
            reward = fine_amount * 0.02  # 2% reward
        else:
            fine_amount = 0
            reward = 0
        
        # Update report status
        report['status'] = 'confirmed'
        report['fineAmount'] = fine_amount
        report['reward'] = reward
        report['confirmedAt'] = datetime.now().isoformat()
        
        return jsonify({
            'status': 'success',
            'message': 'Report confirmed',
            'fineAmount': fine_amount,
            'reward': reward
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'reports_count': len(reports),
        'features': [
            'YOLO video processing',
            'Speed detection',
            'Report submission',
            'CORS enabled'
        ]
    }), 200

@app.route('/files', methods=['GET'])
def list_files():
    """List all saved files for manual review"""
    try:
        files_info = {
            'original_videos': [],
            'processed_videos': [],
            'reports': []
        }
        
        # List original videos
        if os.path.exists(ORIGINAL_VIDEOS_DIR):
            for filename in os.listdir(ORIGINAL_VIDEOS_DIR):
                file_path = os.path.join(ORIGINAL_VIDEOS_DIR, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files_info['original_videos'].append({
                        'filename': filename,
                        'size_bytes': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'path': file_path
                    })
        
        # List processed videos
        if os.path.exists(PROCESSED_VIDEOS_DIR):
            for filename in os.listdir(PROCESSED_VIDEOS_DIR):
                file_path = os.path.join(PROCESSED_VIDEOS_DIR, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files_info['processed_videos'].append({
                        'filename': filename,
                        'size_bytes': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'path': file_path
                    })
        
        # List report files
        if os.path.exists(REPORTS_DIR):
            for filename in os.listdir(REPORTS_DIR):
                file_path = os.path.join(REPORTS_DIR, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files_info['reports'].append({
                        'filename': filename,
                        'size_bytes': stat.st_size,
                        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                        'path': file_path
                    })
        
        return jsonify({
            'status': 'success',
            'files': files_info,
            'directories': {
                'original_videos': ORIGINAL_VIDEOS_DIR,
                'processed_videos': PROCESSED_VIDEOS_DIR,
                'reports': REPORTS_DIR
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Enhanced Speed Report API Server...")
    print("🤖 YOLO Video Processing Enabled")
    print("📱 Ready to receive reports from mobile app")
    print("🌐 Server running on http://localhost:5000")
    print("📊 Health check: http://localhost:5000/health")
    print("📁 Files endpoint: http://localhost:5000/files")
    print("📝 API endpoints:")
    print("   POST /process-video - Process video with YOLO")
    print("   POST /api/submit-report - Submit a new report")
    print("   GET  /api/reports - Get all reports")
    print("   GET  /api/reports/<id> - Get specific report")
    print("   POST /api/reports/<id>/confirm - Confirm report")
    print("   GET  /files - List all saved files")
    print("\n📂 File storage structure:")
    print(f"   Original videos: {ORIGINAL_VIDEOS_DIR}")
    print(f"   Processed videos: {PROCESSED_VIDEOS_DIR}")
    print(f"   Report data: {REPORTS_DIR}")
    
    app.run(debug=True, host='0.0.0.0', port=5000) 