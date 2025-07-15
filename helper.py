import cv2

def create_video_writer(video_cap, output_filename):
    """
    Create a video writer object to save the output video.
    
    Args:
        video_cap: VideoCapture object
        output_filename: Name of the output video file
    
    Returns:
        VideoWriter object
    """
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))

    # initialize the FourCC and a video writer object
    # Use 'mp4v' codec which is more compatible
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))

    return writer