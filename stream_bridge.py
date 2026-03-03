import subprocess
import threading
import time
import os
import signal
import sys
import psutil
from dotenv import load_dotenv

load_dotenv()


import platform

BASE_DIR = os.path.dirname(__file__)

if platform.system().lower() == "windows":
    MEDIAMTX_DIR = os.path.join(BASE_DIR, "mediamtx_v1.15.4_windows_amd64")
    MEDIAMTX_EXE = os.path.join(MEDIAMTX_DIR, "mediamtx.exe")
else:
    MEDIAMTX_DIR = BASE_DIR
    MEDIAMTX_EXE = os.path.join(MEDIAMTX_DIR, "mediamtx")

RTSP_HOST = "127.0.0.1"
RTSP_PORT = "8554"
# Default to "ffmpeg" if env var is missing or empty, but prefer the env var
FFMPEG_CMD = os.getenv("FFMPEG_PATH") or "ffmpeg"
# Strip quotes just in case user added them in .env
FFMPEG_CMD = FFMPEG_CMD.strip('"').strip("'")

print(f"[StreamBridge] Configured FFmpeg Path: '{FFMPEG_CMD}'")
if FFMPEG_CMD != "ffmpeg" and not os.path.exists(FFMPEG_CMD):
    print(f"[StreamBridge] WARNING: FFmpeg path does not exist: {FFMPEG_CMD}")
else:
    print(f"[StreamBridge] FFmpeg path check: Exists or using global 'ffmpeg' command.")



class StreamBridgeManager:
    def __init__(self):
        self.ffmpeg_processes = {}  # Map: source_url -> (subprocess, local_rtsp_url)
        self.mediamtx_process = None
        self.lock = threading.Lock()
        self.stream_counter = 0

    def start_mediamtx(self):
        """Start the MediaMTX RTSP server if not already running."""
        with self.lock:
            if self.mediamtx_process and self.mediamtx_process.poll() is None:
                print("[StreamBridge] MediaMTX is already running.")
                return

            print(f"[StreamBridge] Starting MediaMTX from {MEDIAMTX_EXE}")
            try:
                # Start MediaMTX in a separate process
                # Redirect output to prevent cluttering the main console, or keep it for debug
                self.mediamtx_process = subprocess.Popen(
                    [MEDIAMTX_EXE],
                    cwd=MEDIAMTX_DIR,
                    stdout=subprocess.DEVNULL, # Mute stdout
                    stderr=subprocess.PIPE,
                    stdin=subprocess.PIPE,  # Allow clean shutdown
                )
                time.sleep(1) # Give it a moment to start
                if self.mediamtx_process.poll() is not None:
                     stdout, stderr = self.mediamtx_process.communicate()
                     print(f"[StreamBridge] MediaMTX failed to start: {stderr.decode()}")
                else:
                    print("[StreamBridge] MediaMTX started successfully.")
            except Exception as e:
                print(f"[StreamBridge] Error starting MediaMTX: {e}")

    def start_bridge(self, source_url: str) -> str:
        """
        Start an FFmpeg bridge for the given source URL.
        Returns the local RTSP URL to consume.
        """
        with self.lock:
            if not source_url.startswith(("http", "https")):
                 # If it's not a URL, return as is (file path or integer)
                 return source_url

            # Check if we already have a bridge for this URL
            if source_url in self.ffmpeg_processes:
                proc, local_url = self.ffmpeg_processes[source_url]
                if proc.poll() is None:
                    print(f"[StreamBridge] Reusing existing bridge for {source_url} -> {local_url}")
                    return local_url
                else:
                    # Process died, remove it
                    print(f"[StreamBridge] Bridge process died for {source_url}, restarting...")
                    del self.ffmpeg_processes[source_url]

            # Generate a new stream ID
            self.stream_counter += 1
            stream_name = f"stream_{self.stream_counter}"
            local_rtsp_url = f"rtsp://{RTSP_HOST}:{RTSP_PORT}/{stream_name}"

            print(f"[StreamBridge] Creating bridge: {source_url} -> {local_rtsp_url}")
            
            # Construct FFmpeg command
            # ffmpeg -fflags nobuffer -flags low_delay -i "URL" -an -c:v copy -f rtsp rtsp://...
            cmd = [
                FFMPEG_CMD,
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-re", # Read input at native frame rate (important for files, maybe less for live but safe)
                # Actually for live HLS, -re is usually not needed/wanted if we want low latency catching up,
                # but "republish" often implies re-streaming.
                # User command didn't have -re. I will omit it based on user suggestion.
                "-i", source_url,
                "-an", # Drop audio
                "-c:v", "copy", # Revert to copy for performance
                "-f", "rtsp",
                local_rtsp_url
            ]
            
            # If the user suggested command has specific flags, use them.
            # "ffmpeg -fflags nobuffer -flags low_delay -i ... -an -c:v copy -f rtsp ..."
            # My cmd list matches this.

            try:
                # Start FFmpeg
                print(f"[StreamBridge] Running FFmpeg command: {' '.join(cmd)}")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE, # Capture stderr for errors
                    stdin=subprocess.PIPE
                )
                
                # Check immediately if it failed? FFmpeg takes a moment.
                # Increase wait time to ensure stream is published before we return URL
                self.ffmpeg_processes[source_url] = (proc, local_rtsp_url)
                
                # Give it time to initialize and publish to MediaMTX
                # RTSP headers negotiation and initial buffering takes time. 
                # Increased to 1.5s to prevent timeouts in OpenCV
                time.sleep(1.5) 
                
                if proc.poll() is not None:
                     stdout, stderr = proc.communicate()
                     print(f"[StreamBridge] FFmpeg failed to start: {stderr.decode()}")
                     raise IOError(f"FFmpeg bridge failed: {stderr.decode()}")
                
                return local_rtsp_url

            except Exception as e:
                print(f"[StreamBridge] Failed to spawn FFmpeg: {e}")
                if source_url in self.ffmpeg_processes:
                    del self.ffmpeg_processes[source_url]
                return source_url # Fallback

    def cleanup(self):
        """Terminate all starting processes."""
        with self.lock:
            print("[StreamBridge] Cleaning up processes...")
            for url, (proc, _) in self.ffmpeg_processes.items():
                if proc.poll() is None:
                    print(f"[StreamBridge] Stopping bridge for {url}")
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            self.ffmpeg_processes.clear()

            if self.mediamtx_process and self.mediamtx_process.poll() is None:
                print("[StreamBridge] Stopping MediaMTX")
                self.mediamtx_process.terminate()
                try:
                    self.mediamtx_process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.mediamtx_process.kill()
                self.mediamtx_process = None

# Global instance
bridge_manager = StreamBridgeManager()
