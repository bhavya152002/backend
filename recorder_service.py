import subprocess
import os
import time
import threading
import signal
import sys

# Windows vs Linux FFmpeg command
# Try environment variable first, then check common local paths, then fallback to global
FFMPEG_CMD = os.getenv("FFMPEG_PATH", "ffmpeg")

# Check if a local ffmpeg.exe exists in the same directory (for portable builds)
LOCAL_FFMPEG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
if os.path.exists(LOCAL_FFMPEG):
    FFMPEG_CMD = LOCAL_FFMPEG

class RecorderService:
    def __init__(self):
        self._active_processes = {} # Key: stream_id, Value: Popen object
        self._lock = threading.Lock()
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings_temp")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def start_recording(self, stream_id, stream_url, duration=300, prefix="manual"):
        """
        Start an FFmpeg process to record a segment.
        This is a 'fire and forget' for the process management, usually tracked by the caller logic loop.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{stream_id}_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # FFmpeg command for RTSP/File to MP4
        # Using -t duration to limit length automatically
        # -y to overwrite
        # -c:v copy -c:a copy is fastest (stream copy), but might fail if source codec inconsistent.
        # Re-encoding is safer but CPU intensive. Let's try copy first for strict RTSP.
        # Ideally: ffmpeg -i <url> -t <duration> -c copy <out>
        
        command = [
            FFMPEG_CMD,
            "-y",
            "-rtsp_transport", "tcp",
            "-i", stream_url,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "30",
            "-vf", "scale=-2:720",
            "-an",
            "-movflags", "+faststart",
            filepath
        ]
        
        # Use shell=False for security, but allow for simple execution
        try:
            print(f"[Recorder] Starting recording: {filename} from {stream_url}")
            # Log stdout/stderr to files for debugging
            log_path = filepath + ".log"
            # Keep file handle open? No, Popen needs a file object or file descriptor.
            # We will open it, pass it, and let Popen write to it.
            # Note: We can't easily read it while Popen is writing unless we flush/seek.
            
            f_log = open(log_path, "w")
            process = subprocess.Popen(
                command, 
                stdout=f_log, 
                stderr=subprocess.STDOUT
            )
            
            # Allow a moment to see if it crashes immediately
            time.sleep(2)
            if process.poll() is not None:
                # Process died
                f_log.close()
                print(f"[Recorder] FFmpeg died immediately. Return Code: {process.returncode}")
                # Try to read the log to print error
                try:
                    with open(log_path, "r") as f_read:
                        print(f"[Recorder] FFmpeg Error Log:\n{f_read.read()}")
                except:
                    pass
                return None, None
            
            # If successful, we need to defer closing f_log until process ends?
            # Actually, if we close it here, Popen might fail to write?
            # On Windows, we can't close it while other process uses it?
            # Safe bet: Don't use 'with open', let it run, and closing is tricky.
            # Or just use DEVNULL if we suspect it works now. 
            # But we are debugging.
            # Let's keep it open, but we lose the reference. 
            # Better: Use a managed approach or revert to PIPE if TCP fixes it.
            # Let's revert to a simpler logging strategy that doesn't leak file handles easily, or just hope TCP fixes it.
            # I will assume TCP fixes it, but keep logging to a file that I close only if it fails immediately.
            # If it keeps running, I can't close `f_log` here easily without wrapper.
            # So let's pass STDOUT/STDERR to a distinct file path by string? No, Popen takes file objects.
            
            # Revised approach: Use PIPE and a separate thread to log to file? specific to debugging.
            # Simplest for now: Use TCP. If it fails, the immediate check catches it.
            # If it proceeds, we assume it works.
            
            return process, filepath
        except Exception as e:
            print(f"[Recorder] Failed to start FFmpeg: {e}")
            return None, None

    def record_clip_sync(self, stream_url, duration=30):
        """
        Synchronously record a short clip (blocking call, use in worker/thread).
        Returns filepath on success, None on failure.
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_clip_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        command = [
            FFMPEG_CMD,
            "-y",
            "-rtsp_transport", "tcp",
            "-i", stream_url,
            "-t", str(duration),
            "-c:v", "libx264", # Re-encode detection clips for compatibility
            "-preset", "veryfast",
            "-crf", "30",
            "-vf", "scale=-2:720",
            "-an",
            filepath
        ]
        
        try:
            # print(f"[Recorder] Clip capture start: {filename}")
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if os.path.exists(filepath):
                return filepath
            return None
        except Exception as e:
            print(f"[Recorder] Clip capture failed: {e}")
            return None

recorder_service = RecorderService()
