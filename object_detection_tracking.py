import numpy as np
import datetime
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

##---- Trained using COCO ----##
from helper import create_video_writer

conf_threshold = 0.5 #confidence score above this is taken into consideration

# Initialize the video capture and the video writer objects
video_cap = cv2.VideoCapture("1.mp4")
writer = create_video_writer(video_cap, "output.mp4")
# Initialize the YOLOv8 model using the default weights
model = YOLO("yolov8s.pt")

# Initialize the deep sort tracker
tracker = DeepSort(max_age=50)

##---- Read frames until end of video file ----##
# loop over the frames
while True:
    # starter time to computer the fps
    start = datetime.datetime.now()
    ret, frame = video_cap.read()
    # if there is no frame, we have reached the end of the video
    if not ret:
        print("End of the video file...")
        break
    ############################################################
    ### Detect the objects in the frame using the YOLO model ###
    ############################################################
    # run the YOLO model on the frame
    results = model(frame)

    # initialize the list of bounding boxes and confidences
    detections_list = []

##---- Adding boxes ----##
    # loop over the results
    for result in results:
        # loop over the detections
        for data in result.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = data
            x = int(x1)
            y = int(y1)
            w = int(x2) - int(x1)
            h = int(y2) - int(y1)
            class_id = int(class_id)
            # filter out weak predictions by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > conf_threshold:
                # add the bounding box (x, y, w, h), confidence and class id to the results list
                detections_list.append([[x, y, w, h], confidence, class_id])

    ##---- deep sort tracking ----##
    ############################################################
    ### Track the objects in the frame using DeepSort        ###
    ############################################################
    # update the tracker with the new detections
    tracks = tracker.update_tracks(detections_list, frame=frame)
    
    # loop over the tracks
    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue

        # get the track id and the bounding box
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        
        # draw the bounding box and the track id
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), (0, 255, 0), -1)
        cv2.putText(frame, str(track_id), (xmin + 5, ymin - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

##---- Post Overlay----##
    ############################################################
    ### Some post-processing to display the results          ###
    ############################################################
    # end time to compute the fps
    end = datetime.datetime.now()
    # calculate the frame per second and draw it on the frame
    fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
    cv2.putText(frame, fps, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)
    # show the output frame
    cv2.imshow("Output", frame)
    # write the frame to disk
    writer.write(frame)
    if cv2.waitKey(1) == ord("q"):
        break

# release the video capture, video writer, and close all windows
video_cap.release()
writer.release()
cv2.destroyAllWindows()