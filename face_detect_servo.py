import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
import smbus2

global_width = 256
global_height = 144

class PCA9685:
    def __init__(self, address=0x40, bus=1):
        self.bus = smbus2.SMBus(bus)
        self.address = address
        self.MODE1 = 0x00
        self.PRESCALE = 0xFE
        self.LED0_ON_L = 0x06
        
        self.bus.write_byte_data(self.address, self.MODE1, 0x00)
        prescale = int(25000000.0 / (4096 * 50.0) - 1)
        self.bus.write_byte_data(self.address, self.PRESCALE, prescale)
        self.bus.write_byte_data(self.address, self.MODE1, 0x80)
        time.sleep(0.005)

    def set_angle(self, channel, angle):
        pulse = int(150 + (angle * (600 - 150) / 180))
        base_addr = self.LED0_ON_L + 4 * channel
        self.bus.write_word_data(self.address, base_addr, 0)
        self.bus.write_word_data(self.address, base_addr + 2, pulse)

def gstreamer_pipeline(
    capture_width=global_width,
    capture_height=global_height,
    display_width=global_width,
    display_height=global_height,
    framerate=30,
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

def apply_bayer_dithering(image):
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

def update_servos(pwm, x, y, w, h):
    center_x = x + w//2
    center_y = y + h//2
    
    # Map coordinates to angles (adjust ranges as needed)
    yaw = int(np.interp(center_x, [0, global_width], [45, 135]))
    pitch = int(np.interp(center_y, [0, global_height], [135, 45]))
    
    # Update servos (assuming channels 0,1,2 for yaw,pitch,roll)
    pwm.set_angle(0, yaw)
    pwm.set_angle(1, pitch)
    pwm.set_angle(2, 90)  # Keep roll level

def main():
    model = YOLO('yolov8n.pt')
    pwm = PCA9685()
    
    # Initialize servos to center
    pwm.set_angle(0, 90)
    pwm.set_angle(1, 90)
    pwm.set_angle(2, 90)
    
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, global_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, global_height)
    
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    detection_interval = 2
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            dithered_frame = apply_bayer_dithering(frame)
            dithered_frame = cv2.cvtColor(dithered_frame, cv2.COLOR_GRAY2BGR)

            if frame_count % detection_interval == 0:
                results = model(frame, classes=[0], verbose=False, conf=0.5)

                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w = x2 - x1
                        h = y2 - y1
                        
                        update_servos(pwm, x1, y1, w, h)
                        
                        cv2.rectangle(dithered_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.circle(dithered_frame, (x1 + w//2, y1 + h//2), 4, (0, 0, 255), -1)

            cv2.imshow('Face Tracking', dithered_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Center servos before exit
        pwm.set_angle(0, 90)
        pwm.set_angle(1, 90)
        pwm.set_angle(2, 90)

if __name__ == '__main__':
    main()