import numpy as np
import datetime
import cv2
from ultralytics import YOLO

##---- Trained using COCO ----##
from helper import create_video_writer

conf_threshold = 0.5 #confidence score above this is taken into consideration

# Initialize the video capture and the video writer objects
video_cap = cv2.VideoCapture("1.mp4")
writer = create_video_writer(video_cap, "output.mp4")
# Initialize the YOLOv8 model using the default weights
model = YOLO("yolov8s.pt")

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

# ##---- After running YOLOv8 model on frame: below are attributes ----##
# boxes: ultralytics.yolo.engine.results.Boxes object
# keypoints: None 
# keys: ['boxes']
# masks: None
# names: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
# orig_img: array([[[ 99, 145, 138],
#                   [103, 149, 142],
#                   [107, 153, 146],
#                   ...,
#                   [132, 150, 138],
#                   [132, 150, 138],
#                   [125, 143, 131]],

#                 ...,

#                 [[111, 164, 156],
#                   [105, 158, 150],
#                   [105, 158, 150],
#                   ...,
#                   [133, 138, 144],
#                   [133, 138, 144],
#                   [133, 138, 144]]], dtype=uint8)
# orig_shape: (720, 1280)
# path: 'image0.jpg'
# probs: None
# speed: {'preprocess': 0.5915164947509766, 'inference': 34.77835655212402, 'postprocess': 0.5271434783935547}

# ##---- Bounding Boxes INFORMATION ----##
# print(results[0].boxes)
# # output:
# boxes: tensor([[7.8548e+02, 5.1154e-01, 1.0214e+03, 6.2262e+02, 9.2543e-01, 0.0000e+00],
#         [5.0879e+02, 2.5563e+02, 6.3798e+02, 6.2519e+02, 8.5625e-01, 0.0000e+00],
#         [3.0231e+02, 3.6799e+02, 7.0716e+02, 6.3381e+02, 5.6319e-01, 1.3000e+01],
#         [3.0361e+02, 3.6963e+02, 5.5384e+02, 6.3172e+02, 3.0199e-01, 1.3000e+01]])
# cls: tensor([ 0.,  0., 13., 13.])
# conf: tensor([0.9254, 0.8562, 0.5632, 0.3020])
# data: tensor([[7.8548e+02, 5.1154e-01, 1.0214e+03, 6.2262e+02, 9.2543e-01, 0.0000e+00],
#         [5.0879e+02, 2.5563e+02, 6.3798e+02, 6.2519e+02, 8.5625e-01, 0.0000e+00],
#         [3.0231e+02, 3.6799e+02, 7.0716e+02, 6.3381e+02, 5.6319e-01, 1.3000e+01],
#         [3.0361e+02, 3.6963e+02, 5.5384e+02, 6.3172e+02, 3.0199e-01, 1.3000e+01]])
# id: None
# is_track: False
# orig_shape: tensor([ 720, 1280])
# shape: torch.Size([4, 6])
# xywh: tensor([[903.4377, 311.5681, 235.9163, 622.1130],
#               [573.3878, 440.4119, 129.1873, 369.5559],
#               [504.7360, 500.8981, 404.8489, 265.8228],
#               [428.7267, 500.6769, 250.2260, 262.0896]])
# xywhn: tensor([[0.7058, 0.4327, 0.1843, 0.8640],
#                [0.4480, 0.6117, 0.1009, 0.5133],
#                [0.3943, 0.6957, 0.3163, 0.3692],
#                [0.3349, 0.6954, 0.1955, 0.3640]])
# xyxy: tensor([[7.8548e+02, 5.1154e-01, 1.0214e+03, 6.2262e+02],
#               [5.0879e+02, 2.5563e+02, 6.3798e+02, 6.2519e+02],
#               [3.0231e+02, 3.6799e+02, 7.0716e+02, 6.3381e+02],
#               [3.0361e+02, 3.6963e+02, 5.5384e+02, 6.3172e+02]])
# xyxyn: tensor([[6.1366e-01, 7.1047e-04, 7.9797e-01, 8.6476e-01],
#                [3.9750e-01, 3.5505e-01, 4.9842e-01, 8.6832e-01],
#                [2.3618e-01, 5.1109e-01, 5.5247e-01, 8.8029e-01],
#                [2.3720e-01, 5.1338e-01, 4.3269e-01, 8.7739e-01]])

##---- Adding boxes ----##
    # loop over the results
    for result in results:
        # initialize the list of bounding boxes, confidences, and class IDs
        bboxes = []
        confidences = []
        class_ids = []
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
                bboxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

##---- Post ----##
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