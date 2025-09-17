# stable_audio_client_multi_os.py
# Multi-platform version (Windows, macOS, Linux)

import asyncio
import subprocess
import json
import time
import platform  # <-- NEW: To detect the OS
import re        # <-- NEW: For text parsing
import sys       # <-- NEW: For clean exit

# ==============================================================================
# NEW SECTION: MULTI-PLATFORM HANDLING
# ==============================================================================

def get_platform_config():
    """Returns the FFmpeg configuration specific to the current OS."""
    system = platform.system()
    if system == "Windows":
        return {
            "format": "dshow",
            "device_prefix": "audio=",
            "list_devices_cmd": ['ffmpeg', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy']
        }
    elif system == "Darwin":  # macOS
        return {
            "format": "avfoundation",
            "device_prefix": ":",
            "list_devices_cmd": ['ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', '""']
        }
    elif system == "Linux":
        return {
            "format": "alsa",
            "device_prefix": "",  # Device name is direct, e.g., "hw:0,0"
            "list_devices_cmd": ['arecord', '-l']
        }
    else:
        raise NotImplementedError(f"Unsupported operating system: {system}")

def list_audio_devices():
    """Lists the available audio input devices for the current OS."""
    print(f"🔍 Searching for audio devices for {platform.system()}...")
    try:
        config = get_platform_config()
        cmd = config['list_devices_cmd']
        
        print(f"   (Command being executed: {' '.join(cmd)})")
        
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')

        output = result.stdout + "\n" + result.stderr
        
        print("-" * 50)
        if platform.system() == "Windows":
            print("Found audio devices (dshow):")
            # Search for lines like "DirectShow audio device." then the next line
            devices = re.findall(r'\"(.*?)\" \(audio\)', output, re.IGNORECASE)
            if not devices:
                print("No devices found. Make sure FFmpeg is installed and accessible in your PATH.")
            for i, name in enumerate(devices):
                print(f"  ➡️  {name}")
        
        elif platform.system() == "Darwin": # macOS
            print("Found audio devices (AVFoundation):")
            devices = re.findall(r'\[AVFoundation indev @ .*?\] \[(\d+)\] (.*)', output)
            if not devices:
                print("No devices found. Make sure FFmpeg is installed and accessible in your PATH.")
            for index, name in devices:
                print(f"  ➡️  Index: {index} | Name: {name.strip()} (use ':{index}' as the device name)")
                
        elif platform.system() == "Linux":
            print("Found audio devices (ALSA) (use names like 'hw:0,0'):")
            if "no soundcards found" in output.lower():
                 print("No soundcards found by 'arecord'.")
            else:
                print(output)
        print("-" * 50)

    except FileNotFoundError:
        tool = "ffmpeg" if platform.system() != "Linux" else "arecord (from alsa-utils)"
        print(f"[ERROR] '{tool}' is not installed or not in your system's PATH.")
        print("Please install it to continue.")
    except Exception as e:
        print(f"[ERROR] An error occurred while searching for devices: {e}")

# ==============================================================================
# MAIN CLASS (MODIFIED)
# ==============================================================================

class StableAudioStreamer:
    """Audio streamer with robust connection management (now multi-OS)"""
    
    def __init__(self, device_name: str, ws_url: str):
        self.device_name = device_name
        self.ws_url = ws_url
        self.ffmpeg_process = None
        self.ws = None
        self.running = False
        self.connection_count = 0
        # NEW: Load OS-specific config on startup
        self.platform_config = get_platform_config()
        
    async def start_streaming(self):
        """Starts streaming with an automatic reconnection loop"""
        self.running = True
        
        while self.running:
            try:
                await self.connect_and_stream()
            except KeyboardInterrupt:
                print("\n[info] Shutdown requested by user")
                break
            except Exception as e:
                print(f"[error] Connection error: {e}")
                if self.running:
                    print("[info] Reconnecting in 2 seconds...")
                    await asyncio.sleep(2)
    
    async def connect_and_stream(self):
        """Optimized WebSocket connection"""
        self.connection_count += 1
        print(f"[info] Connection attempt #{self.connection_count} to {self.ws_url}")
        
        async with websockets.connect(
            self.ws_url, max_size=2**20, ping_interval=15, ping_timeout=8,
            close_timeout=3, compression=None
        ) as ws:
            self.ws = ws
            print(f"[info] ✅ Connection #{self.connection_count} established")
            await ws.send(json.dumps({"cmd": "hello", "client": "audio_streamer_multi_os"}))
            await self.start_ffmpeg_optimized()
            await self.stream_with_heartbeat()
    
    async def start_ffmpeg_optimized(self):
        """FFmpeg with optimized and multi-platform parameters"""
        
        # MODIFIED: Dynamically build the FFmpeg command
        device_full_name = f"{self.platform_config['device_prefix']}{self.device_name}"
        
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",
            "-f", self.platform_config['format'],
            "-i", device_full_name,
            "-ac", "1",
            "-ar", "16000",
            "-f", "s16le",
            "-flush_packets", "1",
            "pipe:1"
        ]
        
        # Add specific options if needed
        if self.platform_config['format'] == 'dshow':
            cmd.insert(5, "-audio_buffer_size")
            cmd.insert(6, "20")

        print(f"[info] Launching FFmpeg with command: {' '.join(cmd)}")

        try:
            self.ffmpeg_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=2**20
            )
            print("[info] ✅ FFmpeg started successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to start FFmpeg: {e}")
    
    async def stream_with_heartbeat(self):
        """Stream audio with a heartbeat to keep the connection alive"""
        chunk_size = 3200  # 0.1s of audio at 16kHz mono (smaller = less latency)
        last_heartbeat = time.time()
        heartbeat_interval = 10  # Heartbeat every 10s
        bytes_sent = 0
        
        try:
            while self.running and self.ffmpeg_process:
                try:
                    chunk = await asyncio.wait_for(
                        self.ffmpeg_process.stdout.read(chunk_size), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    if time.time() - last_heartbeat > heartbeat_interval:
                        await self.send_heartbeat()
                        last_heartbeat = time.time()
                    continue
                
                if not chunk:
                    stderr_output = await self.ffmpeg_process.stderr.read()
                    print(f"[warning] No more audio data from FFmpeg. FFmpeg errors: {stderr_output.decode(errors='ignore')}")
                    break
                
                await self.ws.send(chunk)
                bytes_sent += len(chunk)
                
                if time.time() - last_heartbeat > heartbeat_interval:
                    await self.send_heartbeat()
                    last_heartbeat = time.time()
                
                if bytes_sent > 0 and bytes_sent % (16000 * 2 * 30) == 0:  # Every 30s
                    print(f"[info] 📊 {bytes_sent // 1024}KB of audio sent")
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[warning] WebSocket connection closed: {e}")
        except Exception as e:
            print(f"[error] Streaming error: {e}")
        finally:
            await self.cleanup_ffmpeg()
    
    async def send_heartbeat(self):
        """Sends a heartbeat message to keep the connection alive"""
        try:
            heartbeat_msg = json.dumps({"cmd": "ping", "timestamp": time.time()})
            await self.ws.send(heartbeat_msg)
        except Exception:
            pass  # Ignore errors if connection is already closing
    
    async def cleanup_ffmpeg(self):
        """Cleans up the FFmpeg process properly"""
        if self.ffmpeg_process:
            try:
                if self.ffmpeg_process.returncode is None:
                    self.ffmpeg_process.terminate()
                    await asyncio.wait_for(self.ffmpeg_process.wait(), timeout=3)
                    print("[info] ✅ FFmpeg stopped cleanly")
            except asyncio.TimeoutError:
                self.ffmpeg_process.kill()
                print("[warning] ⚠️ FFmpeg was force-killed")
            except Exception:
                pass
            finally:
                self.ffmpeg_process = None
    
    def stop(self):
        """Requests the streaming to stop"""
        print("[info] 🛑 Streaming stop requested")
        self.running = False

async def run_stable_client(device_name: str, ws_url: str = "ws://127.0.0.1:8123/"):
    """Launches the stable audio client"""
    streamer = StableAudioStreamer(device_name, ws_url)
    try:
        await streamer.start_streaming()
    except KeyboardInterrupt:
        print("\n[info] Shutdown requested")
    finally:
        streamer.stop()
        await asyncio.sleep(1)  # Allow time for cleanup

# ==============================================================================
# SCRIPT ENTRY POINT (MODIFIED)
# ==============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stable Multi-OS Audio Client")
    # NEW: Argument to list devices
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Lists available audio input devices and exits."
    )
    parser.add_argument(
        "--device",
        help="Name of the audio device (use --list-devices to find it)"
    )
    parser.add_argument("--ws", default="ws://127.0.0.1:8123/", help="WebSocket URL")
    
    args = parser.parse_args()
    
    if args.list_devices:
        list_audio_devices()
        sys.exit(0) # Exit after listing devices

    if not args.device:
        print("[ERROR] The --device argument is required.")
        print("Use 'python stable_audio_client_multi_os.py --list-devices' to find your device name.")
        sys.exit(1)
    
    print("🎤 STABLE AUDIO CLIENT (MULTI-OS)")
    print("=" * 50)
    print(f"OS: {platform.system()}")
    print(f"Device: {args.device}")
    print(f"WebSocket: {args.ws}")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    asyncio.run(run_stable_client(args.device, args.ws))