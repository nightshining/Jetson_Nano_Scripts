import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time

global_width = 320
global_height = 240

def gstreamer_pipeline(
    capture_width=global_width,
    capture_height=global_width,
    display_width=global_width,
    display_height=global_height,
    framerate=24,
    flip_method=0,
):
    return (
        "v4l2src ! "
        "video/x-raw, width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGRx ! "
        "videoscale ! "
        "video/x-raw, width=(int)%d, height=(int)%d ! "
        "appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            display_width,
            display_height,
        )
    )

def apply_bayer_dithering(image, threshold=128):
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    bayer = np.array([
        [ 0, 8, 2, 10],
        [12, 4, 14, 6],
        [ 3, 11, 1, 9],
        [15, 7, 13, 5]
    ]) * 16 
    
    h, w = gray.shape
    bayer_tiled = np.tile(bayer, ((h + 3) // 4, (w + 3) // 4))
    bayer_tiled = bayer_tiled[:h, :w]
    
    dithered = np.zeros_like(gray)
    dithered[gray > bayer_tiled] = 255
    return dithered

def main():
    model = YOLO('models/yolov10n-face.pt')
        # Configuration for detection frequency
    DETECTION_INTERVAL = 0.5  # Perform detection every 0.5 seconds
    last_detection_time = 0
    last_boxes = []  # Store the last detected boxes
    
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("GStreamer pipeline failed, falling back to regular capture")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, global_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, global_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Couldn't read frame")
                break

            # Apply dithering to the frame
            dithered_frame = apply_bayer_dithering(frame)
            dithered_frame = cv2.cvtColor(dithered_frame, cv2.COLOR_GRAY2BGR)

            # Run detection (faces are class 0 in COCO dataset)
            results = model(frame, classes=[0])

            # Draw bounding boxes on the dithered frame
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf.item()
                    cv2.rectangle(dithered_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(dithered_frame, 
                        str(round(conf, 2)), 
                        (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 255),
                        2)

            cv2.imshow('Face Detection', dithered_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()