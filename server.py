from flask import Flask, request, send_file, after_this_request
import os
import tempfile
from object_detection_speed_tracking import process_video

app = Flask(__name__)

@app.route('/process-video', methods=['POST'])
def process_video_endpoint():
    if 'video' not in request.files:
        return {'error': 'No video file provided'}, 400

    file = request.files['video']
    if file.filename == '':
        return {'error': 'Empty filename'}, 400

    # Save input video to a temporary file
    suffix = os.path.splitext(file.filename)[1] or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as input_temp:
        file.save(input_temp.name)
        input_path = input_temp.name

    # Create output temporary file path
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_temp.close()
    output_path = output_temp.name

    try:
        process_video(input_path, output_path, display=False)
    finally:
        os.remove(input_path)

    @after_this_request
    def remove_file(response):
        try:
            os.remove(output_path)
        except OSError:
            pass
        return response

    return send_file(output_path, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
