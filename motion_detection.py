import cv2
import numpy as np
import time
import threading
import os

# --- 1. CONFIGURATION ---
RTSP_URLS = [
]

W, H = 640, 360  
IDLE_FPS = 1            
POWER_FPS = 10          
MIN_POWER_TIME = 1    # 10 Minutes
COOLDOWN_TIME = 1     
LEARNING_RATE = 0.02    

class CameraStream:
    def __init__(self, url, name):
        self.url = url
        self.name = name
        self.stream = cv2.VideoCapture(url)
        self.frame = None
        self.stopped = False
        self.static_back = None
        
        # State Management
        self.is_power_mode = False
        self.last_motion_time = 0
        self.power_mode_start_time = 0
        
        # Performance & Metrics
        self.proc_ms = 0
        self.actual_fps = 0
        self.brightness = 0
        self.prev_frame_time = time.perf_counter()
        self.fps_start_time = time.time()
        self.fps_counter = 0

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            
            if not ret:
                # RTSP Reconnection Logic
                time.sleep(2)
                self.stream.release()
                self.stream = cv2.VideoCapture(self.url)
                continue

            now = time.time()
            
            # --- STATE MACHINE ---
            time_since_last_motion = now - self.last_motion_time
            time_since_power_started = now - self.power_mode_start_time
            self.is_power_mode = (time_since_last_motion < COOLDOWN_TIME or 
                                 time_since_power_started < MIN_POWER_TIME)

            # --- FPS THROTTLING ---
            target_fps = POWER_FPS if self.is_power_mode else IDLE_FPS
            if (time.perf_counter() - self.prev_frame_time) < (1.0 / target_fps):
                continue

            # Actual FPS Calculation
            self.fps_counter += 1
            if (now - self.fps_start_time) > 1.0:
                self.actual_fps = self.fps_counter / (now - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = now

            start_proc = time.perf_counter()
            self.prev_frame_time = start_proc

            # --- PROCESSING ---
            res = cv2.resize(frame, (W, H))
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            
            # Brightness Meter for Night Detection
            self.brightness = np.mean(gray)
            is_night = self.brightness < 90  
            
            kernel_size = (7, 7) if is_night else (21, 21)
            sensitivity = 12 if is_night else 25
            min_area = 120 if is_night else 250
            
            blurred = cv2.GaussianBlur(gray, kernel_size, 0)

            if self.static_back is None:
                self.static_back = blurred.copy().astype("float")
                self.frame = res
                continue

            cv2.accumulateWeighted(blurred, self.static_back, LEARNING_RATE)
            diff = cv2.absdiff(cv2.convertScaleAbs(self.static_back), blurred)
            _, thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_in_frame = any(cv2.contourArea(c) > min_area for c in cnts)
            if motion_in_frame:
                if not self.is_power_mode: self.power_mode_start_time = now
                self.last_motion_time = now
                self.is_power_mode = True

            self.proc_ms = (time.perf_counter() - start_proc) * 1000

            # --- OVERLAYS ---
            mode_color = (0, 0, 255) if self.is_power_mode else (0, 255, 0)
            status = "NIGHT" if is_night else "DAY"
            timer = int(max(0, MIN_POWER_TIME - (now - self.power_mode_start_time)))
            
            cv2.putText(res, f"{self.name} | {status} (Bri:{int(self.brightness)})", (10, 30), 1, 1.2, mode_color, 2)
            cv2.putText(res, f"Lat: {self.proc_ms:.1f}ms | FPS: {self.actual_fps:.1f}", (10, H-15), 1, 1.0, (255,255,255), 1)
            
            if self.is_power_mode: cv2.rectangle(res, (0,0), (W-1, H-1), (0,0,255), 4)
            self.frame = res

    def stop(self):
        self.stopped = True
        self.stream.release()

# --- 2. 3x3 GRID EXECUTION ---
streams = [CameraStream(url, f"CAM {i+1}").start() for i, url in enumerate(RTSP_URLS)]

print("Initializing 9 RTSP streams... please wait.")

while True:
    # Build 3x3 Grid
    rows = []
    ready_count = 0
    
    for i in range(0, 9, 3):
        row_frames = []
        for j in range(3):
            s = streams[i+j]
            if s.frame is not None:
                row_frames.append(s.frame)
                ready_count += 1
            else:
                # Black frame if stream not ready
                black = np.zeros((H, W, 3), np.uint8)
                cv2.putText(black, f"Connecting {s.name}...", (150, H//2), 1, 1.5, (255,255,255), 2)
                row_frames.append(black)
        rows.append(np.hstack(row_frames))

    grid = np.vstack(rows)
    
    # Adaptive sizing: if 9 cams are too small on your screen, change 1280, 720 to something larger
    cv2.imshow("Wildlife 9-CAM Monitor", cv2.resize(grid, (1280, 720)))

    if cv2.waitKey(1) & 0xFF == ord('q'): break

for s in streams: s.stop()
cv2.destroyAllWindows()