# import eventlet
# eventlet.monkey_patch()

import os
import time
import json
import threading
import base64
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from dotenv import load_dotenv
import psutil
from sqlalchemy.orm.attributes import flag_modified

import requests
import jwt
import datetime
from functools import wraps
from flask_bcrypt import Bcrypt
from flask_bcrypt import Bcrypt
from database import get_db, Detection, User, Stream, Recording, SessionLocal
from recorder_service import recorder_service
from gcs_service import gcs_service

from stream_bridge import bridge_manager

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({'error': 'Token is missing!'}), 401
        
        try:
            token = auth_header.split(" ")[1] if " " in auth_header else auth_header
            data = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            db = get_db()
            user = db.query(User).filter(User.username == data['user']).first()
            if not user:
                return jsonify({'error': 'User not found!'}), 401
            kwargs['current_user'] = user
        except Exception as e:
            return jsonify({'error': f'Token is invalid: {e}'}), 401
        
        return f(*args, **kwargs)
    return decorated

# Load environment variables from .env file
load_dotenv()

# Force TCP for stability (User Optimization)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Lazy-loaded models and locks
_yolo_model = None
_detectron_predictor = None
_yolo_lock = threading.Lock()
_detectron_lock = threading.Lock()

# ========================
# Motion Detection Engine
# (Parameters from motion_detection.py)
# ========================
# Frame resolution for motion analysis
_MD_W, _MD_H = 640, 360
# FPS throttle: idle = 1fps scanning, power = 10fps when motion active
_MD_IDLE_FPS   = 1
_MD_POWER_FPS  = 10
# Minimum seconds to stay in power mode after last motion
_MD_MIN_POWER_TIME = 1
# Cooldown seconds; if motion seen within this window, stay in power mode
_MD_COOLDOWN_TIME  = 1
# Background model learning rate
_MD_LEARNING_RATE  = 0.02

class _MotionDetector:
    """
    Per-stream lightweight motion detector.
    Uses the same adaptive Gaussian background model and night/day
    sensitivity from motion_detection.py.
    """
    def __init__(self, stream_url: str):
        self.url = stream_url
        self.static_back = None
        self.is_power_mode = False
        self.last_motion_time = 0.0
        self.power_mode_start_time = 0.0
        self.prev_frame_time = time.perf_counter()
        self._lock = threading.Lock()
        self.motion_frame_count = 0
        self.yolo_cooldown_until = 0.0

    def check_motion_in_roi(self, frame_bgr, roi_config=None) -> bool:
        """
        Analyse frame for motion, optionally restricted to the ROI.
        Returns True if significant motion is detected.
        """
        with self._lock:
            now = time.time()

            # --- FPS throttle ---
            time_since_last_motion  = now - self.last_motion_time
            time_since_power_start  = now - self.power_mode_start_time
            self.is_power_mode = (
                time_since_last_motion < _MD_COOLDOWN_TIME or
                time_since_power_start < _MD_MIN_POWER_TIME
            )
            target_fps = _MD_POWER_FPS if self.is_power_mode else _MD_IDLE_FPS
            elapsed = time.perf_counter() - self.prev_frame_time
            if elapsed < (1.0 / target_fps):
                # Return last known state without re-processing this frame
                return self.is_power_mode
            self.prev_frame_time = time.perf_counter()

            # --- Resize & grayscale ---
            res  = cv2.resize(frame_bgr, (_MD_W, _MD_H))
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

            # Night/day adaptive thresholds
            brightness  = float(np.mean(gray))
            is_night    = brightness < 90
            kernel_size = (7, 7)  if is_night else (21, 21)
            sensitivity = 12      if is_night else 25
            min_area    = 120     if is_night else 250

            blurred = cv2.GaussianBlur(gray, kernel_size, 0)

            # Initialise background model on first frame
            if self.static_back is None:
                self.static_back = blurred.copy().astype("float")
                return False

            # Update background model
            cv2.accumulateWeighted(blurred, self.static_back, _MD_LEARNING_RATE)
            diff   = cv2.absdiff(cv2.convertScaleAbs(self.static_back), blurred)
            _, thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY)

            # --- Optionally mask to ROI ---
            if roi_config and roi_config.get("matrix"):
                roi_mask = self._build_roi_mask(thresh.shape, roi_config)
                thresh   = cv2.bitwise_and(thresh, thresh, mask=roi_mask)

            cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_detected = any(cv2.contourArea(c) > min_area for c in cnts)

            # Decay frame count if no motion, otherwise increment
            if motion_detected:
                if not self.is_power_mode:
                    self.power_mode_start_time = now
                self.last_motion_time = now
                self.is_power_mode = True
                self.motion_frame_count += 1
            else:
                self.motion_frame_count = max(0, self.motion_frame_count - 1)

            # Evaluate triggers based on 15 sustained motion frames
            trigger_recording = False
            trigger_yolo = False

            if self.motion_frame_count == 15:
                # Exactly 15 frames reached - trigger recording
                trigger_recording = True
                
                # Trigger YOLO only if cooldown expired
                if now > self.yolo_cooldown_until:
                    trigger_yolo = True
                    
                    # We do NOT set the cooldown here!
                    # The cooldown is ONLY set if YOLO detects nothing.
                    # That way, if it detects an animal, the normal alert/clip logic handles it,
                    # and it can detect another animal soon if needed.
                    # But the stream manager wrapper will execute exactly 1 inference frame
                    # and apply the cooldown if the YOLO result is empty.

            return trigger_recording, trigger_yolo

    @staticmethod
    def _build_roi_mask(frame_shape, roi_config):
        """Build a binary mask from the ROI grid matrix."""
        h, w      = frame_shape[:2]
        grid_size = roi_config.get("grid_size", 16)
        matrix    = roi_config["matrix"]
        mask      = np.zeros((h, w), dtype=np.uint8)
        cell_w    = w / grid_size
        cell_h    = h / grid_size
        for row_idx, row in enumerate(matrix):
            for col_idx, val in enumerate(row):
                if val == 1:
                    x1 = int(col_idx * cell_w)
                    y1 = int(row_idx * cell_h)
                    x2 = int((col_idx + 1) * cell_w)
                    y2 = int((row_idx + 1) * cell_h)
                    mask[y1:y2, x1:x2] = 255
        return mask


# Per-stream registry: stream_url -> _MotionDetector
_motion_detectors: dict = {}
_motion_detectors_lock = threading.Lock()

def _get_motion_detector(stream_url: str) -> _MotionDetector:
    """Return (and lazily create) the per-stream motion detector."""
    with _motion_detectors_lock:
        if stream_url not in _motion_detectors:
            _motion_detectors[stream_url] = _MotionDetector(stream_url)
            print(f"[MotionDetector] Created detector for: {stream_url}")
        return _motion_detectors[stream_url]

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BACKEND_DIR, "input")

# Load configuration from environment variables
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
PORT = int(os.getenv("PORT", "8000"))
RAW_MODE = os.getenv("RAW_MODE", "false").lower() in ("true", "1", "yes")

# Unused leftover — pause state is now stored in DB (Stream.motion_detection_enabled)
# and read each frame via get_stream_config() + _stream_configs cache.

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_COOLDOWN_SECONDS = 60  # Reduced to 60s for testing

print(f"[Config] BACKEND_URL: {BACKEND_URL}")
print(f"[Config] FRONTEND_URL: {FRONTEND_URL}")
print(f"[Config] PORT: {PORT}")
print(f"[Config] RAW_MODE: {RAW_MODE} - Use ?raw=true in requests to override")
if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    print(f"[Config] Telegram Alerts: ENABLED (Chat ID: {TELEGRAM_CHAT_ID})")
else:
    print(f"[Config] Telegram Alerts: DISABLED (Missing credentials)")



# Determine if we should start MediaMTX (only if main process)
# Flask reloader spawns a child, we want it only once.
# Using a simple check if we are in main execution flow or rely on bridge manager internal checks.
# Bridge manager has locks but separate processes might still race if not careful with ports.
# However, this app.py is likely run via python app.py or flask run.
# For simplicity, we create the app context to initialize things.

app = Flask(__name__)
bcrypt = Bcrypt(app)
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-key-change-this")

# Start MediaMTX
try:
    bridge_manager.start_mediamtx()
except Exception as e:
    print(f"[Init] Failed to start MediaMTX: {e}")

# Register cleanup on exit
import atexit
atexit.register(bridge_manager.cleanup)
# Configure CORS to allow frontend and website URLs
# Temporarily allowing all origins to debug "Failed to fetch"
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)
# Use threading mode for better stability on Windows/pymongo
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", logger=False, engineio_logger=False)

# --- DATABASE SESSION MANAGEMENT ---
@app.teardown_appcontext
def shutdown_session(exception=None):
    from database import SessionLocal
    SessionLocal.remove()

# --- GLOBAL JSON ERROR HANDLER ---
@app.errorhandler(Exception)
def handle_exception(e):
    """Ensure all errors return JSON instead of HTML"""
    print(f"[Internal Error] {e}")
    import traceback
    traceback.print_exc()
    return jsonify({
        "error": "Internal Server Error",
        "message": str(e)
    }), 500

@app.errorhandler(404)
def handle_404(e):
    return jsonify({"error": "Resource not found"}), 404

# Alerting State
_stream_configs = {} # Real-time config cache
_last_telegram_alert_time = 0
_last_dashboard_alert_time = {} # Key: source, Value: timestamp
_alert_lock = threading.Lock()
_alerts_history = []  # Store recent alerts in memory
_telegram_success_count = 0
_telegram_fail_count = 0
_system_events = [] # Audit log for admin dashboard

# Recording State
_detection_recording_cooldowns = {} # {stream_id: timestamp}  — short 30s clip on animal detection
_motion_recording_cooldowns   = {} # {stream_id: timestamp}  — 10-min clip on first motion seen

# Per-stream viewer reference count.
# When > 0 the MJPEG / Socket.IO path is serving at least one client,
# meaning that path already opens a VideoCapture AND runs inference.
# The background processor must back off to avoid double-opening RTSP.
_stream_viewers: dict = {}        # {stream_url_str: int}
_stream_viewers_lock = threading.Lock()

def _viewer_join(url: str):
    """Call when MJPEG or Socket.IO begins serving a client for this stream."""
    with _stream_viewers_lock:
        _stream_viewers[url] = _stream_viewers.get(url, 0) + 1

def _viewer_leave(url: str):
    """Call when MJPEG or Socket.IO stops serving a client for this stream."""
    with _stream_viewers_lock:
        _stream_viewers[url] = max(_stream_viewers.get(url, 1) - 1, 0)

def _viewer_count(url: str) -> int:
    return _stream_viewers.get(url, 0)

# Motion-triggered 10-min recording
def _handle_motion_recording(stream_id: int, stream_url: str):
    """
    On first motion detected for a stream, record a 10-minute clip and upload to GCS.
    Cooldown: 12 minutes (720 s) so recordings don't overlap.
    This is separate from the short animal-detection clip (_handle_detection_recording).
    """
    now      = time.time()
    last_rec = _motion_recording_cooldowns.get(stream_id, 0)
    if now - last_rec < 300:   # 5-minute cooldown (matches clip duration)
        return
    _motion_recording_cooldowns[stream_id] = now

    def _record_and_upload():
        print(f"[MotionRec] Starting 5-min motion clip for stream {stream_id}")
        filepath = recorder_service.record_clip_sync(stream_url, duration=300)  # 5 min
        if not filepath:
            print(f"[MotionRec] Recording failed for stream {stream_id}")
            return

        folder_name = f"Motion_Clips/Stream_{stream_id}"
        storage_url = gcs_service.upload_file(filepath, folder_name)

        db = SessionLocal()
        try:
            rec = Recording(
                stream_id=stream_id,
                type="motion",
                storage_url=storage_url,
                file_name=os.path.basename(filepath),
                duration_seconds=300
            )
            db.add(rec)
            db.commit()
            socketio.emit("NEW_RECORDING", {
                "stream_id":   stream_id,
                "storage_url": storage_url,
                "type":        "motion"
            })
            print(f"[MotionRec] Uploaded motion clip for stream {stream_id}: {storage_url}")
        except Exception as e:
            print(f"[MotionRec] DB error: {e}")
        finally:
            db.close()
            try:
                os.remove(filepath)
            except Exception:
                pass

    threading.Thread(target=_record_and_upload, daemon=True).start()

def get_stream_config(source: Union[str, int]):
    source_str = str(source)
    if source_str in _stream_configs:
         return _stream_configs[source_str]

    db = get_db()
    try:
        stream = db.query(Stream).filter(Stream.stream_url == source_str).first()
        if stream:
            config = {
                "id":                       stream.id,
                "owner_id":                 stream.client_id,
                "motion_detection_enabled": stream.motion_detection_enabled,
                "detection_region":         stream.detection_region,
            }
            _stream_configs[source_str] = config
            return config
    finally:
        from database import SessionLocal
        SessionLocal.remove()
    return {"motion_detection_enabled": True, "owner_id": None}

def _send_telegram_message(message: str, image_bytes: Optional[bytes] = None):
    """Send a message (and optional photo) to the configured Telegram chat."""
    global _telegram_success_count, _telegram_fail_count
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    try:
        if image_bytes:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
            files = {'photo': ('alert.jpg', image_bytes, 'image/jpeg')}
            data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message, 'parse_mode': 'Markdown'}
            response = requests.post(url, data=data, files=files, timeout=10)
        else:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown"
            }
            response = requests.post(url, json=payload, timeout=5)

        if response.status_code == 200:
            print(f"[Telegram] Alert sent: {message[:50]}...")
            _telegram_success_count += 1
        else:
            print(f"[Telegram] Failed to send alert: {response.text}")
            _telegram_fail_count += 1
    except Exception as e:
        print(f"[Telegram] Error sending alert: {e}")
        _telegram_fail_count += 1

def _handle_detection_recording(stream_id, stream_url):
    """
    Handle potential recording trigger on detection.
    1. Check cooldown (60s).
    2. Record 30s clip (async).
    3. Upload to Drive.
    """
    now = time.time()
    last_rec = _detection_recording_cooldowns.get(stream_id, 0)
    
    # 60 second cooldown
    if now - last_rec < 30:
        return None

    _detection_recording_cooldowns[stream_id] = now
    
    def _record_and_upload():
        # print(f"[Recording] Starting detection clip for Stream {stream_id}")
        filepath = recorder_service.record_clip_sync(stream_url, duration=30)
        
        storage_url = None
        if filepath:
            # Create folder structure: Stream_{id}_Detections/
            folder_name = f"Stream_{stream_id}_Detections"
            storage_url = gcs_service.upload_file(filepath, folder_name)
            
            # Save to DB - BUT we want to link this to the ALERTS if possible.
            # Independent recording entry:
            db = SessionLocal()
            try:
                rec = Recording(
                    stream_id=stream_id,
                    type="detection",
                    storage_url=storage_url,
                    file_name=os.path.basename(filepath),
                    duration_seconds=30
                )
                db.add(rec)
                db.commit()
                
                # Emit event so frontend can show "New Clip Available"
                socketio.emit('NEW_RECORDING', {
                    "stream_id": stream_id,
                    "storage_url": storage_url,
                    "type": "detection"
                })
                
            except Exception as e:
                print(f"[Recording] DB Error: {e}")
            finally:
                db.close()
                
            # Cleanup temp file
            try:
                os.remove(filepath)
            except:
                pass

    # Start in thread
    t = threading.Thread(target=_record_and_upload)
    t.start()
    return True

def _upload_snapshot(frame_bgr, stream_id):
    """
    Save frame to temp, upload to drive, return link.
    """
    if frame_bgr is None: return None
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{stream_id}_{timestamp}.jpg"
    filepath = os.path.join(recorder_service.output_dir, filename)
    cv2.imwrite(filepath, frame_bgr)
    
    folder_name = f"Stream_{stream_id}_Snapshots"
    link = gcs_service.upload_file(filepath, folder_name, mime_type='image/jpeg')
    
    try:
        os.remove(filepath)
    except:
        pass
        
    return link

def _check_and_send_alert(detections: list, source: str, frame=None):
    """
    Check detections and send alert.
    - Dashboard/DB Alerts: throttled 30 s per stream (keyed by DB stream ID, not camera name).
    - Telegram Alerts: strict global cooldown.
    - Detection Clips: 10-second video recorded on detection.
    """
    global _last_telegram_alert_time, _last_dashboard_alert_time
    
    if not detections:
        return

    # Filter for high confidence animals
    valid_detections = [d for d in detections if d['confidence'] > 0.4]
    
    if not valid_detections:
        return

    now = time.time()

    # --- Resolve camera name, owner and stream ID from Stream table ---
    # Do this BEFORE the lock so the DB call doesn't block other streams.
    db = get_db()
    stream_info   = None
    camera_name   = source          # Default to RTSP URL if no DB match
    display_name  = source          # Unique label: "owner / camera"
    owner_id      = None
    owner_username = None
    stream_db_id  = source          # Fallback cooldown key = RTSP URL

    try:
        stream_info = db.query(Stream).filter(Stream.stream_url == source).first()
        if not stream_info:
            stream_info = db.query(Stream).filter(Stream.stream_url.ilike(f"%{source}%")).first()
        if stream_info:
            camera_name    = stream_info.stream_name or f"Camera {stream_info.id}"
            owner_id       = stream_info.client_id
            stream_db_id   = stream_info.id       # Unique per-stream DB PK
            if stream_info.owner:
                owner_username = stream_info.owner.username
            # Build a human-readable, globally-unique display label
            if owner_username:
                display_name = f"{owner_username} / {camera_name}"
            else:
                display_name = camera_name
    except Exception as e:
        print(f"[Alert] Stream lookup failed: {e}")

    with _alert_lock:
        # --- Cooldown keyed by stream DB ID (not camera name or RTSP URL) ---
        # This ensures "Camera 1" from client A and "Camera 1" from client B
        # each get their own independent 30-second cooldown bucket.
        cooldown_key     = str(stream_db_id)
        last_source_time = _last_dashboard_alert_time.get(cooldown_key, 0)
        DASHBOARD_COOLDOWN = 30.0
        
        if now - last_source_time < DASHBOARD_COOLDOWN:
            return
            
        _last_dashboard_alert_time[cooldown_key] = now

        # Construct Alert Object
        animals = {}
        for d in valid_detections:
            label = d['label']
            animals[label] = animals.get(label, 0) + 1
        
        summary = ", ".join([f"{count} {label}(s)" for label, count in animals.items()])
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S')

        # --- Upload detection snapshot to GCS ---
        image_url = None
        if frame is not None:
            gcs_stream_id = stream_info.id if stream_info else "unknown"
            image_url = _upload_snapshot(frame, gcs_stream_id)

        alert_obj = {
            "id":        int(now * 1000),
            "timestamp": timestamp_str,
            # display_name is unique: "alice / Camera 1" vs "john / Camera 1"
            "source":    display_name,
            "message":   f"Detected {summary} at {display_name}",
            "type":      "critical" if any(d['label'] in ['Leopard', 'Elephant', 'Lion', 'Bear'] for d in valid_detections) else "warning",
            "animals":   animals,
            "image_url": image_url,
            "imageUrl":  image_url,
            # Extra fields for frontend filtering
            "stream_id": stream_db_id,
            "owner":     owner_username,
            "camera":    camera_name,
        }
        
        # Save to in-memory history
        _alerts_history.insert(0, alert_obj)
        if len(_alerts_history) > 50:
            _alerts_history.pop()

        # --- Save detection to DB ---
        detection_id = None
        try:
            new_detection = Detection(
                timestamp=datetime.datetime.now(),
                # source stored as unique "owner / camera" so Supabase records are unambiguous
                source=display_name,
                message=alert_obj['message'],
                type=alert_obj['type'],
                animals=animals,
                raw_timestamp=now,
                image_url=image_url,
                owner_id=owner_id
            )
            db.add(new_detection)
            db.commit()
            detection_id = new_detection.id

            # --- Broadcast to owner's private room only ---
            if stream_info and stream_info.owner:
                user = stream_info.owner
                socketio.emit('new_alert', alert_obj, room=user.username)
                print(f"[Alert] Broadcasted to private room: {user.username}")
            else:
                # Fallback: check all users (legacy)
                all_users = db.query(User).all()
                for user in all_users:
                    user_streams = user.assigned_streams or []
                    for s in user_streams:
                        if isinstance(s, dict) and s.get('url') == source:
                            socketio.emit('new_alert', alert_obj, room=user.username)
                            break
                        elif isinstance(s, str) and s == source:
                            socketio.emit('new_alert', alert_obj, room=user.username)
                            break
        except Exception as e:
            print(f"[Database] Alert save/broadcast failed: {e}")
            if db: db.rollback()
            detection_id = None
        finally:
            from database import SessionLocal
            SessionLocal.remove()
        
        # Broadcast to admin's private room only (not all clients)
        socketio.emit('live_alert_ticker', alert_obj, room='admin')
        
        # --- Telegram Alerts ---
        if now - _last_telegram_alert_time >= TELEGRAM_COOLDOWN_SECONDS:
            _last_telegram_alert_time = now
            telegram_msg = (
                f"🚨 *CRITICAL ALERT*\n"
                f"Client: `{owner_username or 'Unknown'}`\n"
                f"Camera: `{camera_name}`\n"
                f"Detected: {summary}\n"
                f"Time: {timestamp_str}"
            )
            image_bytes = _jpeg_bytes(frame) if frame is not None else None
            threading.Thread(target=_send_telegram_message, args=(telegram_msg, image_bytes), daemon=True).start()
            print(f"[Telegram] Cooldown passed, sending alert.")

        print(f"[Alert] Triggered for {display_name}: {summary}")


        # --- Record 10-second detection clip in background ---
        # Capture closure variables so the thread sees the right values
        _clip_source       = source
        _clip_owner        = owner_username or "unknown_client"
        _clip_camera       = camera_name
        _clip_detection_id = detection_id
        _clip_alert_obj    = alert_obj

        def _record_and_upload_clip():
            try:
                clip_path = recorder_service.record_clip_sync(_clip_source, duration=10)
                if clip_path and os.path.exists(clip_path):
                    # GCS folder: Detection_Clips/{owner}/{camera_name}
                    # Scoped per-client so "Camera 1" from alice and "Camera 1" from john
                    # are stored in separate folders and are easy to find.
                    safe_owner  = _clip_owner.replace(' ', '_')
                    safe_camera = _clip_camera.replace(' ', '_')
                    folder = f"Detection_Clips/{safe_owner}/{safe_camera}"
                    clip_gcs_url = gcs_service.upload_file(clip_path, folder, mime_type='video/mp4')
                    
                    # Clean up local file
                    try: os.remove(clip_path)
                    except: pass
                    
                    if clip_gcs_url:
                        # Update DB record with clip URL
                        if _clip_detection_id:
                            db2 = get_db()
                            try:
                                det = db2.query(Detection).filter(Detection.id == _clip_detection_id).first()
                                if det:
                                    det.clip_url = clip_gcs_url
                                    db2.commit()
                            except Exception as e:
                                print(f"[Clip] DB update failed: {e}")
                            finally:
                                from database import SessionLocal
                                SessionLocal.remove()
                        
                        # Notify connected clients about the clip
                        clip_update = {
                            "alert_id": _clip_alert_obj['id'],
                            "clip_url": clip_gcs_url
                        }
                        socketio.emit('alert_media_update', clip_update)
                        print(f"[Clip] Uploaded and broadcasted: {clip_gcs_url[:80]}...")
                    else:
                        print(f"[Clip] GCS upload failed for {clip_path}")
                else:
                    print(f"[Clip] Recording failed for {_clip_source}")
            except Exception as e:
                print(f"[Clip] Error recording/uploading clip: {e}")

        threading.Thread(target=_record_and_upload_clip, daemon=True).start()

def _ensure_video_capture(source: Union[str, int]) -> cv2.VideoCapture:
    """
    Create a VideoCapture object from various source types:
    - str ending with .mp4/.avi/etc: video file from INPUT_DIR
    - int (0, 1, 2, etc): camera index
    - str starting with rtsp:// or http://: camera stream URL
    """
    # Determine source type
    if isinstance(source, int):
        # Camera index
        print(f"[VideoCapture] Opening camera index: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera index: {source}")
        return cap
    
    elif isinstance(source, str):
        # Check if it's a URL (RTSP or HTTP stream)
        if source.startswith(("rtsp://", "http://", "https://")):
            print(f"[VideoCapture] Opening stream URL: {source}")
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise IOError(f"Cannot open stream URL: {source}")
            return cap
        
        # Otherwise treat as video file
        video_path = os.path.join(INPUT_DIR, source)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        print(f"[VideoCapture] Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        return cap
    
    else:
        raise ValueError(f"Invalid source type: {type(source)}. Expected str or int.")


def _looped_read(cap: cv2.VideoCapture, is_camera: bool = False) -> Optional[Tuple[bool, Optional[any]]]:
    """
    Read a frame from the video capture.
    For video files: loop back to start when reaching the end.
    For cameras: just read continuously without looping.
    """
    ret, frame = cap.read()
    if ret:
        return True, frame
    
    # For cameras, if read fails, it's a real error
    if is_camera:
        return False, None
    
    # For video files, loop back to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        return False, None
    return True, frame


def _jpeg_bytes(frame_bgr) -> Optional[bytes]:
    ret, jpg = cv2.imencode(".jpg", frame_bgr)
    if not ret:
        return None
    return jpg.tobytes()


# =========================
# Models (Lazy Load)
# =========================
def _load_yolo():
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            print("[YOLO] Loading YOLOv8n model...")
            from ultralytics import YOLO
            _yolo_model = YOLO("yolov8n.pt")
            print("[YOLO] Model loaded.")
    return _yolo_model


def _load_detectron():
    global _detectron_predictor
    with _detectron_lock:
        if _detectron_predictor is None:
            print("[Detectron2] Loading model...")
            # ... (Detectron2 loading logic omitted for brevity as it's unchanged) ...
            # Assuming existing logic is fine, just placeholder here if needed
            # But since we are replacing the top part, we don't need to touch this unless we deleted it.
            # Wait, I am replacing a large chunk. I need to be careful not to delete _load_detectron logic if it was in the chunk.
            # The previous view showed up to line 100, and _load_detectron was not there.
            # It was further down.
            pass 
    return _detectron_predictor

# ... (Rest of the file) ...

# Wait, I need to be careful with replace_file_content. 
# I should only replace the imports and setup part, and then insert the new functions.
# And then insert the calls in specific places.
# Replacing a huge block blindly is risky if I don't have the full content.

# Let's do it in chunks.

# Chunk 1: Imports and Config



def _ensure_video_capture(source: Union[str, int]) -> cv2.VideoCapture:
    """
    Create a VideoCapture object from various source types:
    - str ending with .mp4/.avi/etc: video file from INPUT_DIR
    - int (0, 1, 2, etc): camera index
    - str starting with rtsp:// or http://: camera stream URL
    """
    # Determine source type
    if isinstance(source, int):
        # Camera index
        print(f"[VideoCapture] Opening camera index: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise IOError(f"Cannot open camera index: {source}")
        return cap
    
    elif isinstance(source, str):
        # Check if it's a URL (RTSP or HTTP stream)
        if source.startswith(("rtsp://", "http://", "https://")):
            print(f"[VideoCapture] Opening stream URL: {source}")
            
            # Direct connection to the source URL (for detection logic)
            # The frontend now plays the stream directly.
            print(f"[VideoCapture] Opening stream URL: {source}")
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                # We log this but don't choke so hard since frontend handles the view now
                 print(f"[VideoCapture] WARNING: Could not open stream for detection: {source}")
                 raise IOError(f"Cannot open stream URL: {source}")
            
            # Set buffer to 1 to lower latency (User Optimization)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        
        # Otherwise treat as video file
        video_path = os.path.join(INPUT_DIR, source)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        print(f"[VideoCapture] Opening video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")
        return cap
    
    else:
        raise ValueError(f"Invalid source type: {type(source)}. Expected str or int.")


def _looped_read(cap: cv2.VideoCapture, is_camera: bool = False) -> Optional[Tuple[bool, Optional[any]]]:
    """
    Read a frame from the video capture.
    For video files: loop back to start when reaching the end.
    For cameras: just read continuously without looping.
    """
    ret, frame = cap.read()
    if ret:
        return True, frame
    
    # For cameras, if read fails, it's a real error
    if is_camera:
        return False, None
    
    # For video files, loop back to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if not ret:
        return False, None
    return True, frame


def _jpeg_bytes(image_bgr) -> bytes:
    success, encoded_image = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not success:
        return b""
    return encoded_image.tobytes()


def _load_yolo():
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            from ultralytics import YOLO
            import torch

            # Get model path from environment, defaulting to yolo11m.pt
            model_filename = os.getenv("YOLO_MODEL_PATH", "yolo11m.pt")
            
            candidate_paths = [
                os.path.join(BACKEND_DIR, model_filename),
                os.path.join(os.path.dirname(BACKEND_DIR), model_filename),
                model_filename,
                os.path.join(BACKEND_DIR, "yolov8n.pt"), # Fallback
            ]
            model_path = next((p for p in candidate_paths if os.path.exists(p)), model_filename)
            print(f"[YOLO] Loading model from: {model_path}")
            try:
                _yolo_model = YOLO(model_path)
                import numpy as np
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                try:
                    _yolo_model.predict(dummy_img, verbose=False, imgsz=640, conf=0.3)
                    print("[YOLO] Model pre-warmed successfully")
                except AttributeError as fuse_err:
                    if "'Conv' object has no attribute 'bn'" in str(fuse_err):
                        print("[YOLO] Fusion error on pre-warm, will handle at inference time")
                    else:
                        raise
                except Exception as e:
                    print(f"[YOLO] Pre-warm warning: {e}")
                print("[YOLO] Model loaded successfully")
            except Exception as e:
                print(f"[YOLO] Error loading model: {e}")
                try:
                    _yolo_model = YOLO(model_path, task='detect')
                    print("[YOLO] Model loaded with explicit task='detect'")
                except Exception as e2:
                    print(f"[YOLO] Failed to load model: {e2}")
                    raise
    return _yolo_model


def _load_detectron():
    global _detectron_predictor
    with _detectron_lock:
        if _detectron_predictor is None:
            try:
                from detectron2.config import get_cfg
                from detectron2.engine import DefaultPredictor
                from detectron2 import model_zoo
            except Exception:
                import sys

                sys.path.append(BACKEND_DIR)
                from detectron2.config import get_cfg  # type: ignore
                from detectron2.engine import DefaultPredictor  # type: ignore
                from detectron2 import model_zoo  # type: ignore

            try:
                import torch
            except Exception:
                torch = None

            model_weights_path = None
            try:
                from huggingface_hub import hf_hub_download

                model_weights_path = hf_hub_download(
                    "sandbox338/wildlife-detector-detectron2", filename="model_final.pth"
                )
            except Exception:
                model_weights_path = None

            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
            if model_weights_path:
                cfg.MODEL.WEIGHTS = model_weights_path
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
            if torch is not None:
                cfg.MODEL.DEVICE = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
            _detectron_predictor = DefaultPredictor(cfg)
    return _detectron_predictor


ANIMAL_CLASSES = {"elephant", "bear", "zebra", "giraffe"}
LEOPARD_LABEL = "Leopard"

def _is_detection_in_roi(bbox, roi_config, frame_width, frame_height):
    """
    Check if detection center is within an enabled ROI cell.
    bbox: [x1, y1, x2, y2]
    roi_config: {"grid_size": 16, "matrix": [[1...],...]}
    """
    if not roi_config or not roi_config.get("matrix"):
        return True
        
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    
    grid_size = roi_config.get("grid_size", 16)
    matrix = roi_config["matrix"]
    
    # Calculate cell
    cell_w = frame_width / grid_size
    cell_h = frame_height / grid_size
    
    col = int(cx / cell_w)
    row = int(cy / cell_h)
    
    # Clamp to grid
    col = max(0, min(col, grid_size - 1))
    row = max(0, min(row, grid_size - 1))
    
    # Check matrix
    try:
        return matrix[row][col] == 1
    except IndexError:
        return True # Default to allow if error

def _raw_infer_on_frame(frame_bgr, roi_config=None):
    """Raw frame passthrough - no AI processing, just return frame as-is"""
    return frame_bgr, []


def _animal_infer_on_frame(frame_bgr, roi_config=None):
    """
    Run best.pt multi-class animal detection on a frame.
    All classes from the model are accepted (no whitelist filter).
    Detections outside the ROI are discarded.
    """
    try:
        model = _load_yolo()
        results = model.predict(frame_bgr, verbose=False, imgsz=640, conf=0.3)
        if not results or len(results) == 0:
            return frame_bgr, []
        boxes = results[0].boxes
        detections = []
        for box in boxes:
            cls_id   = int(box.cls)
            cls_name = model.names[cls_id]
            conf     = float(box.conf)
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Filter by ROI grid
            if roi_config and not _is_detection_in_roi(
                [x1, y1, x2, y2], roi_config,
                frame_bgr.shape[1], frame_bgr.shape[0]
            ):
                continue

            detections.append({
                "label":      cls_name,
                "confidence": conf,
                "bbox":       [x1, y1, x2, y2]
            })
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                f"{cls_name} {conf:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        return frame_bgr, detections

    except AttributeError as e:
        if "'Conv' object has no attribute 'bn'" in str(e):
            print("[YOLO] Fusion error detected, reloading model...")
            global _yolo_model
            with _yolo_lock:
                _yolo_model = None
            try:
                model = _load_yolo()
                results = model.predict(frame_bgr, verbose=False, imgsz=640, conf=0.3)
                if not results or len(results) == 0:
                    return frame_bgr, []
                boxes = results[0].boxes
                detections = []
                for box in boxes:
                    cls_id   = int(box.cls)
                    cls_name = model.names[cls_id]
                    conf     = float(box.conf)
                    if conf < 0.3:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if roi_config and not _is_detection_in_roi(
                        [x1, y1, x2, y2], roi_config,
                        frame_bgr.shape[1], frame_bgr.shape[0]
                    ):
                        continue
                    detections.append({"label": cls_name, "confidence": conf, "bbox": [x1, y1, x2, y2]})
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, f"{cls_name} {conf:.2f}",
                                (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                return frame_bgr, detections
            except Exception as e2:
                print(f"[YOLO] Retry failed: {e2}")
                return frame_bgr, []
        else:
            print(f"[YOLO] AttributeError: {e}")
            return frame_bgr, []
    except Exception as e:
        print(f"[YOLO] Inference error: {e}")
        return frame_bgr, []


def _leopard_infer_on_frame(frame_bgr, roi_config=None):
    predictor = _load_detectron()
    outputs = predictor(frame_bgr)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else []
    scores = instances.scores if instances.has("scores") else []
    classes = instances.pred_classes if instances.has("pred_classes") else []

    class_names = ["Antelope", "Lion", "Elephant", "Zebra", "Gorilla", "Wolf", "Leopard", "Giraffe"]
    detections = []
    for box, score, cls in zip(boxes, scores, classes):
        label = class_names[cls.item()] if cls.item() < len(class_names) else str(cls.item())
        if label.lower() != LEOPARD_LABEL.lower() or float(score) < 0.5:
            continue
        x1, y1, x2, y2 = [int(x) for x in box]
        
        # Filter by ROI
        if roi_config and not _is_detection_in_roi([x1, y1, x2, y2], roi_config, frame_bgr.shape[1], frame_bgr.shape[0]):
            continue
            
        detections.append({"label": label, "confidence": float(score), "bbox": [x1, y1, x2, y2]})
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame_bgr,
            f"{label} {float(score):.2f}",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    cv2.putText(
        frame_bgr,
        f"Leopards detected: {sum(1 for d in detections if d['label']==LEOPARD_LABEL)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )
    return frame_bgr, detections


def _parse_source(source_param: str) -> Union[str, int]:
    """
    Parse the source parameter to determine if it's a camera index or video file.
    - "0", "1", "2" etc -> int (camera index)
    - "11.mp4", "video.avi" etc -> str (video file)
    - "rtsp://..." or "http://..." -> str (stream URL)
    """
    # Try to parse as integer (camera index)
    try:
        return int(source_param)
    except ValueError:
        # It's a string (video file or URL)
        return source_param


def _is_camera_source(source: Union[str, int]) -> bool:
    """Check if source is a camera (index or stream URL)"""
    if isinstance(source, int):
        return True
    if isinstance(source, str) and source.startswith(("rtsp://", "http://", "https://")):
        return True
    return False


def _get_placeholder_frame():
    """Generate a placeholder 'Stream Offline' frame"""
    import numpy as np
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Stream Offline", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return img

def _mjpeg_stream(infer_fn, source: Union[str, int], max_fps: Optional[float] = 15.0) -> Generator[bytes, None, None]:
    source_str = str(source)
    _viewer_join(source_str)  # Increment BEFORE ensuring video capture
    boundary = "frame"
    try:
        cap = _ensure_video_capture(source)
    except Exception as e:
        print(f"[MJPEG] Connection failed: {e}")
        _viewer_leave(source_str)
        # Yield a single placeholder frame and close
        frame = _get_placeholder_frame()
        jpg = _jpeg_bytes(frame)
        def gen_error():
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
            )
            time.sleep(1) # Prevent hot loop if client retries immediately
        return Response(gen_error(), mimetype=f"multipart/x-mixed-replace; boundary={boundary}")

    is_camera = _is_camera_source(source)
    
    def gen():
        nonlocal cap
        try:
            print(f"[MJPEG] Stream generator started for {source}")
            last_time = 0.0
            fail_count = 0
            max_fails = 10 # Allow some retries before giving up
            
            while True:
                ok, frame = _looped_read(cap, is_camera)
                if not ok:
                    fail_count += 1
                    print(f"[MJPEG] _looped_read failed for {source} (Attempt {fail_count}/{max_fails})")
                    
                    if is_camera and fail_count < max_fails:
                        # Try to reconnect
                        print(f"[MJPEG] Reconnecting to {source}...")
                        cap.release()
                        time.sleep(1) # Grace period
                        try:
                            cap = _ensure_video_capture(source)
                            print(f"[MJPEG] Reconnected successfully to {source}")
                            # No reset of fail_count here, want to see consecutive failures
                            continue
                        except Exception as e:
                            print(f"[MJPEG] Reconnection attempt failed: {e}")
                            continue
                    else:
                        print(f"[MJPEG] Giving up on {source} after {fail_count} failures.")
                        break
                
                # Reset fail count on successful read
                if fail_count > 0:
                    fail_count = 0

                now = time.time()
                if max_fps:
                    min_dt = 1.0 / max_fps
                    if now - last_time < min_dt:
                        continue
                    last_time = now
                
                try:
                    # Check stream config (motion detection, ROI)
                    config     = get_stream_config(source)
                    roi_config = config.get("detection_region")
                    stream_id  = config.get("id")

                    if config.get("motion_detection_enabled", True):
                        # --- Motion pre-filter ---
                        source_key  = str(source)
                        detector    = _get_motion_detector(source_key)
                        
                        try:
                            # Use new 15-frame motion detector logic
                            trigger_recording, trigger_yolo = detector.check_motion_in_roi(frame, roi_config=roi_config)
                        except TypeError:
                            # Fallback if the detector somehow returns a single bool
                            motion_seen = detector.check_motion_in_roi(frame, roi_config=roi_config)
                            trigger_yolo = motion_seen
                            trigger_recording = motion_seen

                        if trigger_recording and stream_id:
                            # Trigger motion recording
                            threading.Thread(
                                target=_handle_motion_recording,
                                args=(stream_id, str(source)),
                                daemon=True
                            ).start()

                        if trigger_yolo:
                            frame_bgr, detections = infer_fn(frame, roi_config=roi_config)
                            
                            # If we triggered YOLO but found no animals, apply the 5 min cooldown
                            if not detections:
                                detector.yolo_cooldown_until = time.time() + 300
                        else:
                            frame_bgr  = frame
                            detections = []
                    else:
                        # Detection disabled for this stream (or paused via API)
                        frame_bgr  = frame
                        detections = []
                except Exception as e:
                    print(f"[MJPEG] Inference error: {e}")
                    frame_bgr  = frame
                    detections = []


                # Trigger alerts if detections found
                if detections:
                    threading.Thread(
                        target=_check_and_send_alert,
                        args=(detections, str(source), frame_bgr.copy()),
                        daemon=True
                    ).start()

                jpg = _jpeg_bytes(frame_bgr)
                if not jpg:
                    continue
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n"
                )
        finally:
            cap.release()
            _viewer_leave(str(source))  # ← deregister viewer
    resp = Response(gen(), mimetype=f"multipart/x-mixed-replace; boundary={boundary}")
    # Headers to prevent buffering by Cloudflare/Nginx
    resp.headers["Cache-Control"] = "no-cache, no-transform"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

def _sse_events(infer_fn, source: Union[str, int], max_fps: Optional[float] = 5.0) -> Response:
    source_str = str(source)
    _viewer_join(source_str)
    try:
        cap = _ensure_video_capture(source)
    except Exception as e:
         print(f"[SSE] Connection failed: {e}")
         _viewer_leave(source_str)
         return Response("data: {\"error\": \"Stream offline\"}\n\n", mimetype="text/event-stream")

    is_camera = _is_camera_source(source)
    
    def gen():
        try:
            last_time = 0.0
            last_heartbeat = 0.0
            while True:
                ok, frame = _looped_read(cap, is_camera)
                if not ok:
                    break
                now = time.time()
                if max_fps:
                    min_dt = 1.0 / max_fps
                    if now - last_time < min_dt:
                        continue
                    last_time = now
                _, detections = infer_fn(frame.copy())
                payload = {
                    "ts": int(time.time() * 1000),
                    "source": str(source),
                    "count": len(detections),
                    "detections": detections,
                }
                
                yield f"data: {json.dumps(payload)}\n\n"
                # Periodic heartbeat to keep Cloudflare/Proxies from closing idle streams
                if now - last_heartbeat > 10.0:
                    last_heartbeat = now
                    yield "event: ping\ndata: {}\n\n"
        finally:
            cap.release()
            _viewer_leave(source_str)
    resp = Response(gen(), mimetype="text/event-stream")
    # Recommended SSE headers for proxies/CDNs
    resp.headers["Cache-Control"] = "no-cache, no-transform"
    resp.headers["Connection"] = "keep-alive"
    resp.headers["X-Accel-Buffering"] = "no"
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


def _get_infer_fn(mode: str, raw: bool = False):
    """Get the appropriate inference function based on mode and raw flag"""
    if raw or RAW_MODE:
        return _raw_infer_on_frame
    if mode == "animal":
        return _animal_infer_on_frame
    elif mode == "leopard":
        return _leopard_infer_on_frame
    else:
        return _raw_infer_on_frame


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "message": "Backend is reachable"})


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "WildTrack AI Flask backend is running",
        "config": {
            "raw_mode": RAW_MODE,
            "note": "Add ?raw=true to any endpoint to bypass AI models"
        },
        "endpoints": {
            "streams": ["/stream/animal", "/stream/leopard"],
            "events": ["/events/animal", "/events/leopard"],
            "socketio": "WebSocket on /socket.io/"
        },
        "usage": {
            "ai_mode": "/stream/animal?video=11.mp4",
            "raw_mode": "/stream/animal?video=11.mp4&raw=true"
        }
    }


@app.get("/stream/animal")
def stream_animal():
    source_param = request.args.get("video", default="11.mp4")
    source = _parse_source(source_param)
    raw = request.args.get("raw", "").lower() in ("true", "1", "yes")
    infer_fn = _get_infer_fn("animal", raw)
    if raw:
        print(f"[Stream] Animal stream (RAW mode): {source}")
    
    # Allow controlling FPS via query param (default 15)
    try:
        max_fps = float(request.args.get("fps", 15.0))
    except ValueError:
        max_fps = 15.0

    try:
        return _mjpeg_stream(infer_fn, source, max_fps=max_fps)
    except Exception as e:
        print(f"[Stream] Error opening stream {source}: {e}")
        # Return a placeholder stream or 503
        return Response("Stream Offline", status=503)


@app.get("/events/animal")
def events_animal():
    source_param = request.args.get("video", default="11.mp4")
    source = _parse_source(source_param)
    raw = request.args.get("raw", "").lower() in ("true", "1", "yes")
    infer_fn = _get_infer_fn("animal", raw)
    if raw:
        print(f"[Events] Animal events (RAW mode): {source}")
    
    # Allow controlling FPS via query param (default 5 for events)
    try:
        max_fps = float(request.args.get("fps", 5.0))
    except ValueError:
        max_fps = 5.0

    return _sse_events(infer_fn, source, max_fps=max_fps)


@app.get("/snapshot/animal")
def snapshot_animal():
    """Serve a single JPEG frame for the grid view to avoid browser connection limits."""
    source_param = request.args.get("video", default="11.mp4")
    source = _parse_source(source_param)
    raw = request.args.get("raw", "").lower() in ("true", "1", "yes")
    infer_fn = _get_infer_fn("animal", raw)
    
    try:
        cap = _ensure_video_capture(source)
        # Random seek for variety if it's a file? No, just read next frame or start.
        # Actually, just reading start is boring.
        # But for valid snapshot of "live", we just read.
        # Since we open efficient new capture for file every time, it will always be frame 0.
        # To make it look "live", we could store open captures? Too complex for now.
        # Workaround: read a few frames or just accept frame 0 for files?
        # A file loop implies time progression.
        # If we re-open file every snapshot, we always get 0:00.
        # We need a persistent capture manager or just use the random access.
        
        # IMPROVEMENT: Use the global StreamBridge local RTSP if available?
        # Or simple hack: cache the capture?
        # Let's just return frame 0 for now to ensure it works, or maybe seek based on Time.time()?
        
        # Seek based on time to simulate playback!
        if isinstance(source, str) and not source.startswith("rtsp"):
             # It is a file
             # Get total frames usually requires property check, expensive?
             # Let's just trust it opens.
             total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
             fps = cap.get(cv2.CAP_PROP_FPS) or 30
             duration = total_frames / fps
             
             # Calculate current position based on wall clock to sync all clients roughly
             now = time.time()
             loop_pos = now % duration
             target_frame = int(loop_pos * fps)
             cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        ok, frame = cap.read()
        cap.release()
        
        if not ok:
             return Response("Error reading frame", status=500)
             
        frame_bgr, detections = infer_fn(frame)
        
        # Trigger alerts if detections found (even in snapshot mode for grid monitoring)
        if detections:
            # We pass a copy because `_check_and_send_alert` might be threaded or async in future, 
            # though currently it's synchronous logic mostly.
            _check_and_send_alert(detections, str(source), frame_bgr.copy())

        jpg = _jpeg_bytes(frame_bgr)
        return Response(jpg, mimetype="image/jpeg")
        
    except Exception as e:
        print(f"[Snapshot] Error: {e}")
        return Response("Error", status=500)


@app.get("/stream/leopard")
def stream_leopard():
    source_param = request.args.get("video", default="18.mp4")
    source = _parse_source(source_param)
    raw = request.args.get("raw", "").lower() in ("true", "1", "yes")
    infer_fn = _get_infer_fn("leopard", raw)
    if raw:
        print(f"[Stream] Leopard stream (RAW mode): {source}")
    return _mjpeg_stream(infer_fn, source)


@app.get("/events/leopard")
def events_leopard():
    source_param = request.args.get("video", default="18.mp4")
    source = _parse_source(source_param)
    raw = request.args.get("raw", "").lower() in ("true", "1", "yes")
    infer_fn = _get_infer_fn("leopard", raw)
    if raw:
        print(f"[Events] Leopard events (RAW mode): {source}")
    return _sse_events(infer_fn, source)


# =========================
# Socket.IO: Realtime Feed
# =========================
def start_realtime_stream(source: Union[str, int], infer_fn, room: str, max_fps: Optional[float] = 15.0):
    _viewer_join(str(source))          # ← register active viewer
    cap = _ensure_video_capture(source)
    is_camera = _is_camera_source(source)
    source_key = str(source)
    detector = _get_motion_detector(source_key)

    try:
        last_time = 0.0
        while True:
            ok, frame = _looped_read(cap, is_camera)
            if not ok:
                break
            now = time.time()
            if max_fps:
                min_dt = 1.0 / max_fps
                if now - last_time < min_dt:
                    continue
                last_time = now

            # --- Motion pre-filter then YOLO ---
            try:
                config     = get_stream_config(source)
                roi_config = config.get("detection_region")
                stream_id  = config.get("id")

                if config.get("motion_detection_enabled", True):
                    try:
                        trigger_recording, trigger_yolo = detector.check_motion_in_roi(frame, roi_config=roi_config)
                    except TypeError:
                        motion_seen = detector.check_motion_in_roi(frame, roi_config=roi_config)
                        trigger_yolo = motion_seen
                        trigger_recording = motion_seen

                    if trigger_recording and stream_id:
                        threading.Thread(
                            target=_handle_motion_recording,
                            args=(stream_id, str(source)),
                            daemon=True
                        ).start()

                    if trigger_yolo:
                        frame_bgr, detections = infer_fn(frame, roi_config=roi_config)
                        if not detections:
                            detector.yolo_cooldown_until = time.time() + 300
                    else:
                        frame_bgr  = frame
                        detections = []
                else:
                    # Detection paused via API or disabled per stream
                    frame_bgr  = frame
                    detections = []
            except Exception as e:
                print(f"[SocketIO] Inference error for {source}: {e}")
                frame_bgr  = frame
                detections = []


            # Encode to base64 JPG
            success, jpg = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                continue
            b64_frame = base64.b64encode(jpg.tobytes()).decode("utf-8")
            payload = {
                "source":     str(source),
                "timestamp":  int(time.time() * 1000),
                "detections": detections,
                "frame":      b64_frame,
            }
            socketio.emit("frame_update", payload, room=room)
    finally:
        cap.release()
        _viewer_leave(str(source))     # ← deregister viewer


@socketio.on("connect")
def handle_connect():
    sid = request.sid
    print(f"[SocketIO] Client connected: {sid}")
    emit("connected", {"sid": sid, "status": "connected"})


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected:", request.sid)


@socketio.on("join_private")
def handle_join_private(data):
    """Allow a client to join their private notification room"""
    username = (data or {}).get("username")
    if username:
        join_room(username)
        print(f"[SocketIO] Client {request.sid} joined private room: {username}")
        emit("joined_private", {"room": username, "status": "success"})
def start_raw_stream(source: Union[str, int], room: str, max_fps: Optional[float] = 20.0):
    source_str = str(source)
    _viewer_join(source_str)
    try:
        cap = _ensure_video_capture(source)
        is_camera = _is_camera_source(source)
        
        try:
            last_time = 0.0
            while True:
                ok, frame = _looped_read(cap, is_camera)
                if not ok:
                    break

                now = time.time()
                if max_fps:
                    min_dt = 1.0 / max_fps
                    if now - last_time < min_dt:
                        continue
                    last_time = now

                success, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not success:
                    continue

                b64_frame = base64.b64encode(jpg.tobytes()).decode("utf-8")

                socketio.emit("frame_update", {
                    "source": str(source),
                    "timestamp": int(time.time() * 1000),
                    "frame": b64_frame,
                    "detections": [],
                    "raw": True
                }, room=room)

        finally:
            cap.release()
    finally:
        _viewer_leave(source_str)

@socketio.on("start_stream")
def handle_start_stream(data):
    source_param = (data or {}).get("video", "11.mp4")
    source = _parse_source(source_param)
    room = (data or {}).get("room", "animal")
    mode = (data or {}).get("mode", "ai")  # "raw" or "ai"
    
    # Respect global RAW_MODE if not explicitly set
    if RAW_MODE and mode == "ai":
        mode = "raw"
        print(f"[SocketIO] Global RAW_MODE enabled, switching to raw mode")

    join_room(room)

    if mode == "raw":
        print(f"[SocketIO] Starting raw stream: {source} in room {room}")
        threading.Thread(
            target=start_raw_stream,
            args=(source, room),
            daemon=True,
        ).start()
    else:
        print(f"[SocketIO] Starting AI stream: {source} in room {room}")
        threading.Thread(
            target=start_realtime_stream,
            args=(source, _animal_infer_on_frame, room),
            daemon=True,
        ).start()

    emit("stream_started", {"source": str(source), "room": room, "mode": mode})


@socketio.on("webcam_frame")
def handle_webcam_frame(data):
    """
    Handle webcam frames from browser.
    Receives base64-encoded JPEG frames, runs AI detection, and returns results.
    """
    try:
        import numpy as np
        
        frame_data = data.get("frame")
        if not frame_data:
            print("[Webcam] No frame data received")
            return
        
        print(f"[Webcam] Received frame, size: {len(frame_data)} bytes")
        
        # Decode base64 to image
        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            print("[Webcam] Failed to decode frame")
            return
        
        print(f"[Webcam] Decoded frame shape: {frame.shape}")
        
        # Run AI detection
        frame_with_boxes, detections = _animal_infer_on_frame(frame)
        
        print(f"[Webcam] Detections: {len(detections)} objects found")
        if detections:
            print(f"[Webcam] Detection details: {detections}")
            # Check for alerts
            _check_and_send_alert(detections, "Webcam", frame_with_boxes)
        
        # Emit detection results back to client
        emit("webcam_detection", {
            "timestamp": int(time.time() * 1000),
            "detections": detections,
            "count": len(detections)
        })
        
    except Exception as e:
        print(f"[Webcam] Error processing frame: {e}")
        import traceback
        traceback.print_exc()


# @socketio.on("start_stream")
# def handle_start_stream(data):
#     # Expected data: { "video": "11.mp4", "room": "animal" }
#     video = (data or {}).get("video", "11.mp4")
#     room = (data or {}).get("room", "animal")
#     join_room(room)
#     threading.Thread(
#         target=start_realtime_stream,
#         args=(video, _animal_infer_on_frame, room),
#         daemon=True,
#     ).start()
#     emit("stream_started", {"video": video, "room": room})

@app.get("/api/cameras")
@token_required
def list_cameras(current_user):
    """List available camera devices based on user assignments"""
    client_id = request.args.get("client_id", type=int)
    if client_id and current_user.username == 'admin':
        db = get_db()
        try:
            streams = db.query(Stream).filter(Stream.client_id == client_id).all()
            if not streams:
                # Fallback to check if user even exists or has legacy JSON
                target_user = db.query(User).filter(User.id == client_id).first()
                if not target_user:
                    return jsonify({"error": "Target client not found"}), 404
                # Seed from legacy if missing (emergency recovery)
                from migrate_streams import migrate_streams
                migrate_streams() 
                streams = db.query(Stream).filter(Stream.client_id == client_id).all()
        finally:
            from database import SessionLocal
            SessionLocal.remove()
    else:
        db = get_db()
        try:
            streams = db.query(Stream).filter(Stream.client_id == current_user.id).all()
        finally:
            from database import SessionLocal
            SessionLocal.remove()

    camera_list = []
    for s in streams:
        camera_list.append({
            "id": s.id,
            "url": s.stream_url,
            "name": s.stream_name,
            "type": "video" if s.stream_url.endswith(".mp4") else "camera",
            "motion_detection_enabled": s.motion_detection_enabled
        })
    return jsonify({"cameras": camera_list})

@app.post("/api/cameras/rename")
@token_required
def rename_camera(current_user):
    """Allow user to rename their assigned stream"""
    data = request.get_json()
    url = data.get("url")
    new_name = data.get("name")
    
    if not url or not new_name:
        return jsonify({"error": "Missing url or name"}), 400
        
    db = get_db()
    try:
        # Update in Stream table
        stream = db.query(Stream).filter(
            Stream.client_id == current_user.id,
            Stream.stream_url == url
        ).first()
        
        if not stream:
            return jsonify({"error": "Stream not found in your assignments"}), 404
            
        stream.stream_name = new_name
        db.commit()
        return jsonify({"message": "Stream renamed successfully", "new_name": new_name})
    finally:
        from database import SessionLocal
        SessionLocal.remove()


@app.get("/api/alerts")
@token_required
def get_alerts(current_user):
    """Get recent alerts history for the current user (strictly owner-scoped)."""
    client_id = request.args.get("client_id", type=int)
    is_admin   = current_user.username == 'admin'

    # Determine which owner_id to filter on
    if is_admin and client_id:
        filter_owner_id = client_id
    elif is_admin and not client_id:
        filter_owner_id = None   # Admin with no client_id param → see ALL
    else:
        filter_owner_id = current_user.id  # Regular client → own alerts only

    try:
        db = get_db()
        if db is not None:
            try:
                q = db.query(Detection).order_by(Detection.id.desc())
                if filter_owner_id is not None:
                    q = q.filter(Detection.owner_id == filter_owner_id)
                results = q.limit(50).all()

                alerts = []
                for r in results:
                    alerts.append({
                        "id":        int(r.raw_timestamp * 1000) if r.raw_timestamp else r.id,
                        "timestamp": r.timestamp.strftime('%Y-%m-%d %H:%M:%S') if r.timestamp else "",
                        "source":    r.source,
                        "message":   r.message,
                        "type":      r.type,
                        "animals":   r.animals,
                        "image_url": r.image_url,
                        "imageUrl":  r.image_url,
                        "clip_url":  r.clip_url,
                    })
                return jsonify({"alerts": alerts})
            finally:
                db.close()
        else:
            # Fallback: in-memory history, filtered by owner
            if is_admin and not client_id:
                return jsonify({"alerts": _alerts_history})
            username = current_user.username
            filtered = [a for a in _alerts_history if a.get("owner") == username]
            return jsonify({"alerts": filtered})
    except Exception as e:
        print(f"[API] Error fetching alerts: {e}")
        # Same scoped fallback on error
        if is_admin and not client_id:
            return jsonify({"alerts": _alerts_history})
        username = current_user.username
        filtered = [a for a in _alerts_history if a.get("owner") == username]
        return jsonify({"alerts": filtered})


# ============================================================
# Detection Pause / Resume  (persisted via motion_detection_enabled on Stream)
# ============================================================

@app.post("/api/detection/pause")
@token_required
def pause_detection(current_user):
    """
    Body: { "paused": true|false, "client_id": <int optional - admin only> }

    - Admin + no client_id  → toggle ALL streams in the system
    - Admin + client_id     → toggle only that client's streams
    - Regular client        → toggle only their own streams
    Persists to DB via Stream.motion_detection_enabled.
    """
    data      = request.get_json() or {}
    paused    = data.get("paused")
    if paused is None:
        return jsonify({"error": "Missing 'paused' field"}), 400
    paused    = bool(paused)
    client_id = data.get("client_id")
    is_admin  = current_user.username == "admin"

    db = SessionLocal()
    try:
        if is_admin and client_id is None:
            # Update ALL streams globally
            db.query(Stream).update({"motion_detection_enabled": not paused},
                                     synchronize_session=False)
            db.commit()
            # Clear the per-stream config cache so loops re-read immediately
            _stream_configs.clear()
            print(f"[Detection] Admin {'PAUSED' if paused else 'RESUMED'} ALL streams globally")
            # Count to report back
            total = db.query(Stream).count()
            return jsonify({"status": "ok", "global_paused": paused, "streams_affected": total})

        # Per-client
        target_id = int(client_id) if (is_admin and client_id is not None) else current_user.id
        updated = db.query(Stream).filter(Stream.client_id == target_id)\
                    .update({"motion_detection_enabled": not paused},
                             synchronize_session=False)
        db.commit()
        _stream_configs.clear()
        print(f"[Detection] owner_id={target_id} detection {'PAUSED' if paused else 'RESUMED'} ({updated} streams)")
        return jsonify({
            "status":           "ok",
            "owner_id":         target_id,
            "client_paused":    paused,
            "streams_affected": updated
        })
    except Exception as e:
        db.rollback()
        print(f"[Detection] DB error in pause_detection: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        SessionLocal.remove()


@app.get("/api/detection/status")
@token_required
def detection_status(current_user):
    """
    Returns current detection state.
    - paused=True  means ALL of the target's streams have motion_detection_enabled=False
    - paused=False means at least one stream is still enabled
    """
    client_id = request.args.get("client_id", type=int)
    is_admin  = current_user.username == "admin"
    target_id = int(client_id) if (is_admin and client_id) else current_user.id

    db = SessionLocal()
    try:
        if is_admin and client_id is None:
            total    = db.query(Stream).count()
            disabled = db.query(Stream).filter(Stream.motion_detection_enabled == False).count()
            paused   = (total > 0 and disabled == total)
            return jsonify({
                "global_paused":    paused,
                "client_paused":    False,
                "effective_paused": paused,
                "total_streams":    total,
                "disabled_streams": disabled
            })
        streams  = db.query(Stream).filter(Stream.client_id == target_id).all()
        total    = len(streams)
        disabled = sum(1 for s in streams if not s.motion_detection_enabled)
        paused   = (total > 0 and disabled == total)
        return jsonify({
            "global_paused":    False,
            "client_paused":    paused,
            "effective_paused": paused,
            "total_streams":    total,
            "disabled_streams": disabled
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        SessionLocal.remove()


@app.post("/api/login")
def login():
    """Authenticate user and return JWT token"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400
        
    username = data.get("username")
    password = data.get("password")
    
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    db = get_db()
    if db is None:
         return jsonify({"error": "Database unavailable"}), 500

    try:
        user = db.query(User).filter(User.username == username).first()
        if user and bcrypt.check_password_hash(user.password, password):
            # Generate Token
            token = jwt.encode({
                "user": username,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
            }, JWT_SECRET, algorithm="HS256")
            
            return jsonify({
            "token": token, 
            "username": username, 
            "role": user.role,
            "assigned_streams": user.assigned_streams,
            "status": "success"
        })
    finally:
        db.close()
    
    return jsonify({"error": "Invalid credentials"}), 401


@app.get("/api/admin/overview-metrics")
@token_required
def get_admin_metrics(current_user):
    if current_user.username != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    
    db = get_db()
    try:
        active_deployments = db.query(User).filter(User.role == 'client').count()
        users = db.query(User).filter(User.role == 'client').all()
        
        # Unique cameras assigned across all users
        all_assigned_urls = set()
        for u in users:
            raw_streams = u.assigned_streams or []
            for s in raw_streams:
                if isinstance(s, dict):
                    all_assigned_urls.add(s.get('url'))
                else:
                    all_assigned_urls.add(s)
        
        total_unique_cameras = len(all_assigned_urls)
        
        # Calculate online/offline based on active bridge processes
        online_count = 0
        with bridge_manager.lock:
            for url in all_assigned_urls:
                if url in bridge_manager.ffmpeg_processes:
                    proc, _ = bridge_manager.ffmpeg_processes[url]
                    if proc.poll() is None:
                        online_count += 1
        
        offline_count = total_unique_cameras - online_count
        
        now = datetime.datetime.utcnow()
        day_ago = now - datetime.timedelta(days=1)
        week_ago = now - datetime.timedelta(days=7)
        
        alerts_24h = db.query(Detection).filter(Detection.timestamp >= day_ago).count()
        alerts_7d = db.query(Detection).filter(Detection.timestamp >= week_ago).count()
        critical_24h = db.query(Detection).filter(Detection.timestamp >= day_ago, Detection.type == 'critical').count()
        
        # Active critical events (in last 10 mins)
        ten_mins_ago = now - datetime.timedelta(minutes=10)
        active_critical = db.query(Detection).filter(Detection.timestamp >= ten_mins_ago, Detection.type == 'critical').count()

        total_telegram = _telegram_success_count + _telegram_fail_count
        success_rate = (_telegram_success_count / total_telegram * 100) if total_telegram > 0 else 100

        return jsonify({
            "activeDeployments": active_deployments,
            "totalCameras": total_unique_cameras,
            "onlineStreams": online_count,
            "offlineStreams": offline_count,
            "alerts24h": alerts_24h,
            "alerts7d": alerts_7d,
            "critical24h": critical_24h,
            "activeCriticalEvents": active_critical,
            "telegramSuccessRate": success_rate,
            "telegramSentCount": _telegram_success_count
        })
    finally:
        db.close()

@app.get("/api/admin/system-health")
@token_required
def get_system_health(current_user):
    if current_user.username != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    
    # Real system metrics
    try:
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
    except Exception as e:
        print(f"[Health] psutil error: {e}")
        cpu_usage = 0
        ram_usage = 0
    
    # Check if YOLO model is loaded
    ai_status = "Running" if _yolo_model is not None or _detectron_predictor is not None else "Standby"
    
    return jsonify({
        "websocketStatus": "Connected",
        "aiEngineStatus": ai_status,
        "cpuUsage": cpu_usage,
        "ramUsage": ram_usage,
        "gpuUsage": 0, # Placeholder unless we add torch.cuda logic
        "workerQueueSize": threading.active_count()
    })

@app.get("/api/admin/activity")
@token_required
def get_admin_activity(current_user):
    if current_user.username != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    return jsonify({"activities": _system_events})

def log_system_event(message):
    _system_events.insert(0, {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "message": message
    })
    if len(_system_events) > 50:
        _system_events.pop()

# --- ADMIN API ENDPOINTS ---

@app.get("/api/admin/clients")
@token_required
def admin_get_clients(current_user):
    if current_user.username != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    
    db = get_db()
    try:
        users = db.query(User).filter(User.role == 'client').all()
        client_list = []
        for u in users:
            raw_streams = u.assigned_streams or []
            normalized_streams = []
            online_count = 0
            
            for s in raw_streams:
                url = s.get('url') if isinstance(s, dict) else s
                name = s.get('name') if isinstance(s, dict) else f"Stream {s}"
                normalized_streams.append({"url": url, "name": name})
                
                # Check online status
                if url in bridge_manager.ffmpeg_processes:
                    proc, _ = bridge_manager.ffmpeg_processes[url]
                    if proc.poll() is None:
                        online_count += 1

            # Compute Status
            if online_count > 0:
                status = 'ACTIVE'
            else:
                status = 'OFFLINE'

            client_list.append({
                "id": u.id,
                "username": u.username,
                "total_streams": len(normalized_streams),
                "status": status,
                "assigned_streams": normalized_streams,
                "created_at": u.created_at.isoformat() if u.created_at else None
            })
        return jsonify({"clients": client_list})
    finally:
        db.close()

@app.post("/api/admin/clients")
@token_required
def admin_create_client(current_user):
    if current_user.username != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    
    data = request.get_json()
    username = data.get("username")
    password = data.get("password")
    streams = data.get("assigned_streams", []) # Processed on frontend or here
    
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    db = get_db()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            return jsonify({"error": "User already exists"}), 400
        
        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(
            username=username,
            password=hashed_pw,
            role="client",
            assigned_streams=streams
        )
        db.add(new_user)
        db.flush() # Get ID
        
        # Populate Stream table
        for s in streams:
            url = s.get('url') if isinstance(s, dict) else s
            name = s.get('name') if isinstance(s, dict) else f"Stream {s}"
            new_stream = Stream(
                client_id=new_user.id,
                stream_url=url,
                stream_name=name,
                motion_detection_enabled=True
            )
            db.add(new_stream)
            
        db.commit()
        log_system_event(f"New client unit provisioned: {username}")
        return jsonify({"message": "Client created successfully", "id": new_user.id})
    finally:
        db.close()

@app.put("/api/admin/clients/<int:client_id>")
@token_required
def admin_update_client(current_user, client_id):
    if current_user.username != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    
    data = request.get_json()
    db = get_db()
    try:
        user = db.query(User).filter(User.id == client_id).first()
        if not user:
            return jsonify({"error": "Client not found"}), 404
        
        if "assigned_streams" in data:
            streams = data["assigned_streams"]
            user.assigned_streams = streams
            # Sync Stream table
            db.query(Stream).filter(Stream.client_id == client_id).delete()
            for s in streams:
                url = s.get('url') if isinstance(s, dict) else s
                name = s.get('name') if isinstance(s, dict) else f"Stream {url}"
                new_stream = Stream(
                    client_id=user.id,
                    stream_url=url,
                    stream_name=name,
                    motion_detection_enabled=True
                )
                db.add(new_stream)
                
        if "password" in data and data["password"]:
            user.password = bcrypt.generate_password_hash(data["password"]).decode('utf-8')
        if "username" in data and data["username"]:
            user.username = data["username"]

        db.commit()
        log_system_event(f"Client unit configuration updated: {user.username}")
        return jsonify({"message": "Client updated successfully"})
    finally:
        db.close()

@app.delete("/api/admin/clients/<int:client_id>")
@token_required
def admin_delete_client(current_user, client_id):
    if current_user.username != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    
    db = get_db()
    try:
        user = db.query(User).filter(User.id == client_id).first()
        if not user:
            return jsonify({"error": "Client not found"}), 404
        
        if user.username == 'admin':
            return jsonify({"error": "Cannot delete admin user"}), 400
            
        username = user.username
        # Delete associated streams
        db.query(Stream).filter(Stream.client_id == client_id).delete()
        db.delete(user)
        db.commit()
        log_system_event(f"Client unit decommissioned: {username}")
        return jsonify({"message": "Client deleted successfully"})
    finally:
        db.close()

@app.get("/api/admin/clients/<int:client_id>/streams")
@token_required
def admin_get_client_streams(current_user, client_id):
    if current_user.username != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    
    db = get_db()
    try:
        user = db.query(User).filter(User.id == client_id).first()
        if not user:
            return jsonify({"error": "Client not found"}), 404
        
        streams = db.query(Stream).filter(Stream.client_id == client_id).all()
        return jsonify({
            "username": user.username,
            "streams": [
                {
                    "id": s.id,
                    "url": s.stream_url,
                    "name": s.stream_name,
                    "motion_detection_enabled": s.motion_detection_enabled
                } for s in streams
            ]
        })
    finally:
        db.close()

@app.get("/api/admin/streams/<int:stream_id>")
@token_required
def admin_get_stream(current_user, stream_id):
    if current_user.username != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    
    db = get_db()
    try:
        stream = db.query(Stream).filter(Stream.id == stream_id).first()
        if not stream:
            return jsonify({"error": "Stream not found"}), 404
            
        return jsonify({
            "id": stream.id,
            "client_id": stream.client_id,
            "stream_url": stream.stream_url,
            "stream_name": stream.stream_name,
            "stream_name": stream.stream_name,
            "motion_detection_enabled": stream.motion_detection_enabled,
            "detection_region": stream.detection_region,
            "recording_enabled": stream.recording_enabled
        })
    finally:
        from database import SessionLocal
        SessionLocal.remove()


@app.patch("/api/admin/streams/<int:stream_id>")
@token_required
def admin_update_stream(current_user, stream_id):
    if current_user.username != 'admin':
        return jsonify({"error": "Forbidden"}), 403
    
    data = request.get_json()
    if data is None:
        return jsonify({"error": "Missing body"}), 400
        
    db = get_db()
    try:
        stream = db.query(Stream).filter(Stream.id == stream_id).first()
        if not stream:
            return jsonify({"error": "Stream not found"}), 404
            
        if "motion_detection_enabled" in data:
            val = data["motion_detection_enabled"]
            stream.motion_detection_enabled = val
            # Update cache
            if stream.stream_url in _stream_configs:
                _stream_configs[stream.stream_url]["motion_detection_enabled"] = val
            else:
                _stream_configs[stream.stream_url] = {
                    "motion_detection_enabled": val,
                    "detection_region": stream.detection_region,
                    "recording_enabled": stream.recording_enabled
                }
        
        if "detection_region" in data:
            region = data["detection_region"]
            stream.detection_region = region
            flag_modified(stream, "detection_region")
            # Update cache
            if stream.stream_url in _stream_configs:
                _stream_configs[stream.stream_url]["detection_region"] = region
            else:
                _stream_configs[stream.stream_url] = {
                    "motion_detection_enabled": stream.motion_detection_enabled,
                    "detection_region": region,
                    "recording_enabled": stream.recording_enabled
                }
            
        if "stream_name" in data:
            stream.stream_name = data["stream_name"]

        if "recording_enabled" in data:
            rec_enabled = data["recording_enabled"]
            stream.recording_enabled = rec_enabled
            if stream.stream_url in _stream_configs:
                _stream_configs[stream.stream_url]["recording_enabled"] = rec_enabled
            else:
                _stream_configs[stream.stream_url] = {
                    "motion_detection_enabled": stream.motion_detection_enabled,
                    "detection_region": stream.detection_region,
                    "recording_enabled": rec_enabled
                }
            
        db.commit()
        
        socketio.emit('STREAM_CONFIG_UPDATED', {
            "stream_id": stream.id,
            "stream_url": stream.stream_url,
            "motion_detection_enabled": stream.motion_detection_enabled,
            "detection_region": stream.detection_region,
            "recording_enabled": stream.recording_enabled
        })
        
        return jsonify({
            "message": "Stream updated successfully",
            "motion_detection_enabled": stream.motion_detection_enabled,
            "detection_region": stream.detection_region,
            "recording_enabled": stream.recording_enabled
        })

    except Exception as e:
        db.rollback()
        return jsonify({"error": str(e)}), 500
    finally:
        db.close()


_active_recordings = {} # Key: stream_id, Value: { "proc": subprocess.Popen, "filepath": str, "start_time": float }

def _recording_worker():
    """Background worker to manage continuous recordings"""
    while True:
        try:
            # First, check database for all streams to ensure we have latest state
            db = SessionLocal()
            try:
                # Join with User to get client name (username)
                streams = db.query(Stream).all()
                current_time = time.time()
                
                for stream in streams:
                    s_id = stream.id
                    is_enabled = stream.recording_enabled
                    s_url = stream.stream_url
                    cam_name = stream.stream_name or f"Stream_{s_id}"
                    
                    # Fetch owner for client name
                    owner = db.query(User).filter(User.id == stream.client_id).first()
                    client_name = owner.username if owner else "unknown_client"
                    
                    # Manage state
                    if is_enabled:
                        # Check if already recording
                        if s_id not in _active_recordings:
                            # START RECORDING (10 minutes = 600s)
                            print(f"[RecordingWorker] Starting 10-min loop for {client_name}/{cam_name}")
                            proc, filepath = recorder_service.start_recording(s_id, s_url, duration=600, prefix="manual")
                            if proc:
                                _active_recordings[s_id] = {
                                    "proc": proc,
                                    "filepath": filepath,
                                    "start_time": current_time,
                                    "cam_name": cam_name,
                                    "client_name": client_name
                                }
                        else:
                            # CHECK ROTATION (10 mins = 600s)
                            rec_data = _active_recordings[s_id]
                            # If 10 mins passed OR process died unexpectedly
                            if current_time - rec_data["start_time"] > 600 or rec_data["proc"].poll() is not None:
                                print(f"[RecordingWorker] Rotating segment for {client_name}/{cam_name}")
                                
                                # Stop previous if still running
                                if rec_data["proc"].poll() is None:
                                    rec_data["proc"].terminate()
                                    try:
                                        rec_data["proc"].wait(timeout=2)
                                    except:
                                        rec_data["proc"].kill()
                                
                                # Upload previous Async
                                old_filepath = rec_data["filepath"]
                                old_cam_name = rec_data["cam_name"]
                                old_client_name = rec_data["client_name"]
                                start_ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(rec_data["start_time"]))

                                def _upload_bg(path, sid, cname, clname, ts):
                                    try:
                                        if os.path.exists(path):
                                            # New naming: folder=client_name, file=camera_name_timestamp.mp4
                                            new_filename = f"{cname}_{ts}.mp4"
                                            # We need to rename the local file before upload to get the right name in GCS if service uses basename
                                            # Or we can modify gcs_service.upload_file to take a destination name.
                                            # Looking at gcs_service.py might be good, but let's assume it uses basename.
                                            target_path = os.path.join(os.path.dirname(path), new_filename)
                                            os.rename(path, target_path)
                                            
                                            link = gcs_service.upload_file(target_path, clname)
                                            
                                            # Save to DB
                                            dbt = SessionLocal()
                                            try:
                                                rec = Recording(
                                                    stream_id=sid,
                                                    type="manual",
                                                    storage_url=link,
                                                    file_name=new_filename,
                                                    duration_seconds=600
                                                )
                                                dbt.add(rec)
                                                dbt.commit()
                                            finally:
                                                dbt.close()
                                            
                                            # Clean up local file
                                            if os.path.exists(target_path):
                                                os.remove(target_path)
                                    except Exception as e:
                                        print(f"[RecordingWorker] Upload failed for {clname}/{cname}: {e}")

                                threading.Thread(target=_upload_bg, args=(old_filepath, s_id, old_cam_name, old_client_name, start_ts)).start()
                                
                                # Start New Segment
                                proc, filepath = recorder_service.start_recording(s_id, s_url, duration=600, prefix="manual")
                                if proc:
                                    _active_recordings[s_id] = {
                                        "proc": proc,
                                        "filepath": filepath,
                                        "start_time": current_time,
                                        "cam_name": cam_name,
                                        "client_name": client_name
                                    }
                                else:
                                    del _active_recordings[s_id]

                    else:
                        # If disabled but active, STOP
                        if s_id in _active_recordings:
                            print(f"[RecordingWorker] Stopping recording for {client_name}/{cam_name}")
                            rec_data = _active_recordings[s_id]
                            if rec_data["proc"].poll() is None:
                                rec_data["proc"].terminate()
                            
                            # Upload partial
                            filepath = rec_data["filepath"]
                            c_name = rec_data["cam_name"]
                            cl_name = rec_data["client_name"]
                            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(rec_data["start_time"]))

                            def _upload_bg_stop(path, sid, cname, clname, timestamp):
                                try:
                                    if os.path.exists(path):
                                        new_filename = f"{cname}_{timestamp}.mp4"
                                        target_path = os.path.join(os.path.dirname(path), new_filename)
                                        os.rename(path, target_path)
                                        
                                        link = gcs_service.upload_file(target_path, clname)
                                        dbt = SessionLocal()
                                        try:
                                            rec = Recording(
                                                stream_id=sid, 
                                                type="manual", 
                                                storage_url=link, 
                                                file_name=new_filename
                                            )
                                            dbt.add(rec)
                                            dbt.commit()
                                        finally:
                                            dbt.close()
                                        if os.path.exists(target_path):
                                            os.remove(target_path)
                                except Exception as e:
                                    print(f"[RecordingWorker] Stop upload failed for {clname}/{cname}: {e}")
                            
                            threading.Thread(target=_upload_bg_stop, args=(filepath, s_id, c_name, cl_name, ts)).start()
                            
                            del _active_recordings[s_id]

            except Exception as e:
                print(f"[RecordingWorker] Error: {e}")
            finally:
                db.close()
        
        except Exception as outer_e:
            print(f"[RecordingWorker] Critical Error: {outer_e}")
        
        time.sleep(10)

def _background_health_broadcaster():
    """Periodically emit system health to all connected admins"""
    while True:
        try:
            try:
                cpu = psutil.cpu_percent()
                ram = psutil.virtual_memory().percent
            except:
                cpu, ram = 0, 0

            health_data = {
                "websocketStatus": "Connected",
                "aiEngineStatus":  "Running" if _yolo_model is not None else "Standby",
                "cpuUsage":        cpu,
                "ramUsage":        ram,
                "gpuUsage":        0,
                "workerQueueSize": threading.active_count()
            }
            socketio.emit('system_health_update', health_data)
        except Exception as e:
            print(f"[Health] Broadcast error: {e}")
        time.sleep(10)


# ============================================================
# Always-On Background Stream Supervisor
# ============================================================

# Track active background processor threads: stream_id → (Thread, stop_event)
_bg_processors: dict = {}   # {stream_id: (thread, stop_event)}


def _background_stream_processor(stream_id: int, stream_url: str, stop_event: threading.Event):
    """
    Permanently process one stream in the background:
      1. Open stream with OpenCV (auto-reconnects on drop).
      2. Run motion detection on every frame.
      3. On motion: run YOLO → trigger alerts & 10-min motion recording.
      4. Exit cleanly if stop_event is set OR motion_detection_enabled is toggled off.
    """
    infer_fn     = _get_infer_fn("animal", raw=False)
    detector     = _get_motion_detector(str(stream_url))
    is_camera    = _is_camera_source(stream_url)
    MAX_FAILS    = 20
    RETRY_WAIT   = 5        # seconds between reconnect attempts
    FPS_LIMIT    = 8.0      # background processing capped at 8 fps to save CPU
    min_dt       = 1.0 / FPS_LIMIT
    last_time    = 0.0

    print(f"[BGProcessor] Starting for stream {stream_id}: {stream_url}")

    while not stop_event.is_set():
        # Re-read config each outer loop to detect toggle-offs
        _stream_configs.pop(str(stream_url), None)   # force a fresh DB read
        config = get_stream_config(stream_url)
        if not config.get("motion_detection_enabled", True):
            print(f"[BGProcessor] Stream {stream_id} detection disabled — processor stopping")
            break

        cap = None
        fail_count = 0
        try:
            cap = _ensure_video_capture(stream_url)
        except Exception as e:
            print(f"[BGProcessor] Cannot open stream {stream_id}: {e}. Retrying in {RETRY_WAIT}s")
            stop_event.wait(RETRY_WAIT)
            continue

        print(f"[BGProcessor] Stream {stream_id} opened OK")

        while not stop_event.is_set():
            # If a client is actively viewing this stream, back off entirely:
            # the MJPEG/SocketIO path already has a cap open AND does inference.
            # Avoid double-opening the RTSP connection.
            if _viewer_count(stream_url) > 0:
                if cap is not None:
                    cap.release()
                    cap = None
                    print(f"[BGProcessor] Stream {stream_id} has active viewer — backing off")
                stop_event.wait(2.0)  # check again in 2 s
                continue

            # No active viewer → ensure we have our own cap
            if cap is None:
                try:
                    cap = _ensure_video_capture(stream_url)
                    print(f"[BGProcessor] Stream {stream_id} cap re-opened (viewer gone)")
                    fail_count = 0
                except Exception as e:
                    print(f"[BGProcessor] Re-open failed for stream {stream_id}: {e}")
                    stop_event.wait(RETRY_WAIT)
                    continue

            # FPS throttle
            now = time.time()
            if now - last_time < min_dt:
                time.sleep(min_dt - (now - last_time))
                continue
            last_time = time.time()

            ok, frame = _looped_read(cap, is_camera)
            if not ok:
                fail_count += 1
                if fail_count >= MAX_FAILS:
                    print(f"[BGProcessor] Stream {stream_id} lost after {MAX_FAILS} failures — reconnecting")
                    break   # break inner → reconnect via outer loop
                time.sleep(0.2)
                continue
            fail_count = 0  # reset on success

            # Re-check motion_detection_enabled periodically (every ~30 s of processing)
            config     = get_stream_config(stream_url)
            roi_config = config.get("detection_region")
            if not config.get("motion_detection_enabled", True):
                print(f"[BGProcessor] Stream {stream_id} detection disabled mid-run — stopping")
                stop_event.set()
                break

            # Motion pre-filter
            try:
                # Use new 15-frame motion detector logic
                trigger_recording, trigger_yolo = detector.check_motion_in_roi(frame, roi_config=roi_config)
            except TypeError:
                try:
                    motion_seen = detector.check_motion_in_roi(frame, roi_config=roi_config)
                    trigger_recording = motion_seen
                    trigger_yolo = motion_seen
                except Exception:
                    trigger_recording = False
                    trigger_yolo = False
            except Exception:
                trigger_recording = False
                trigger_yolo = False

            if trigger_recording:
                # Trigger 5-min motion recording (respects its own cooldown)
                threading.Thread(
                    target=_handle_motion_recording,
                    args=(stream_id, str(stream_url)),
                    daemon=True
                ).start()

            if trigger_yolo:
                # YOLO inference
                try:
                    frame_bgr, detections = infer_fn(frame, roi_config=roi_config)
                    if not detections:
                        detector.yolo_cooldown_until = time.time() + 300
                except Exception as e:
                    print(f"[BGProcessor] Inference error on stream {stream_id}: {e}")
                    frame_bgr  = frame
                    detections = []

                # Trigger animal-detection alert + 10-s detection clip
                if detections:
                    threading.Thread(
                        target=_check_and_send_alert,
                        args=(detections, str(stream_url), frame_bgr.copy()),
                        daemon=True
                    ).start()

        if cap is not None:
            cap.release()

        if not stop_event.is_set():
            # Reconnect wait before retry
            stop_event.wait(RETRY_WAIT)

    print(f"[BGProcessor] Stopped for stream {stream_id}")
    _bg_processors.pop(stream_id, None)


def _stream_supervisor():
    """
    Runs every 30 seconds.
    Starts a background processor for every stream that has motion_detection_enabled=True
    and has no live processor thread. Also stops processors for disabled streams.
    """
    print("[Supervisor] Stream supervisor started")
    while True:
        try:
            db = SessionLocal()
            active_streams = db.query(Stream).filter(
                Stream.motion_detection_enabled == True,
                Stream.stream_url.isnot(None),
                Stream.stream_url != ""
            ).all()
            SessionLocal.remove()

            active_ids = {s.id for s in active_streams}

            # Start processors for newly active streams
            for s in active_streams:
                if s.id not in _bg_processors or not _bg_processors[s.id][0].is_alive():
                    stop_event = threading.Event()
                    t = threading.Thread(
                        target=_background_stream_processor,
                        args=(s.id, s.stream_url, stop_event),
                        daemon=True,
                        name=f"BGStream-{s.id}"
                    )
                    t.start()
                    _bg_processors[s.id] = (t, stop_event)
                    print(f"[Supervisor] Started processor for stream {s.id}: {s.stream_url}")

            # Stop processors for streams that are now disabled / deleted
            for sid in list(_bg_processors.keys()):
                if sid not in active_ids:
                    t, ev = _bg_processors[sid]
                    ev.set()   # signal thread to stop
                    print(f"[Supervisor] Signalled stop for stream {sid}")
                    del _bg_processors[sid]

        except Exception as e:
            print(f"[Supervisor] Error: {e}")

        time.sleep(30)


# Start health broadcaster
threading.Thread(target=_background_health_broadcaster, daemon=True).start()


# --- STARTUP ---

def _start_stream_supervisor():
    """Launch the always-on stream supervisor at server startup."""
    t = threading.Thread(target=_stream_supervisor, daemon=True, name="StreamSupervisor")
    t.start()
    print("[Startup] Stream supervisor started — will poll DB every 30s for active streams.")

if __name__ == '__main__':
    print("=" * 30)
    print("WildTrack AI Surveillance System - Backend Starting")
    print(f"Server will start on http://0.0.0.0:{PORT}")
    print("=" * 30)
    
    try:
        # Start the always-on stream supervisor (auto-processes all enabled streams)
        _start_stream_supervisor()
        
        # Run Flask
        socketio.run(app, host='0.0.0.0', port=PORT, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nServer error: {e}")
        import traceback
        traceback.print_exc()
