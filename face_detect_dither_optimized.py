import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Reduce global resolution for faster processing
global_width = 256
global_height = 144

def gstreamer_pipeline(
    capture_width=global_width,
    capture_height=global_height,
    display_width=global_width,
    display_height=global_height,
    framerate=30,  # Increased framerate
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
    # Use NumPy's faster vectorized operations
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
    bayer_tiled = np.tile(bayer, ((h + 3) // 4, (w + 3) // 4))[:h, :w]
    
    return (gray > bayer_tiled).astype(np.uint8) * 255

def main():
    # Use smaller, faster model
    model = YOLO('yolov8n.pt')
    
    # Configure camera for lower latency
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, global_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, global_height)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize frame buffering
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set explicit FPS

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    # Reduce detection frequency
    detection_interval = 3
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Apply dithering more efficiently
            dithered_frame = apply_bayer_dithering(frame)
            dithered_frame = cv2.cvtColor(dithered_frame, cv2.COLOR_GRAY2BGR)

            # Perform detection less frequently
            if frame_count % detection_interval == 0:
                results = model(frame, classes=[0], verbose=False, conf=0.5)

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = box.conf.item()
                        cv2.rectangle(dithered_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(dithered_frame, 
                            str(round(conf, 2)), 
                            (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (0, 255, 0),
                            2)

            cv2.imshow('Face Detection', dithered_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()