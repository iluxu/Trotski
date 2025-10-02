# stable_audio_client_multi_os.py
# Multi-platform version (Windows, macOS, Linux) — fixed & hardened

import asyncio
import subprocess
import json
import time
import platform
import re
import sys
import websockets

# ==============================
# Platform helpers
# ==============================

def get_platform_config():
    """Return FFmpeg capture format and listing command per OS."""
    system = platform.system()
    if system == "Windows":
        # DirectShow (dshow). Device names must match exactly what ffmpeg lists.
        return {
            "format": "dshow",
            "device_prefix": "audio=",
            "list_devices_cmd": ["ffmpeg", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        }
    elif system == "Darwin":  # macOS
        # AVFoundation. For listing, ffmpeg expects -i "" (empty string).
        return {
            "format": "avfoundation",
            "device_prefix": ":",  # e.g. ":1" (audio index)
            "list_devices_cmd": ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
        }
    elif system == "Linux":
        # ALSA. Listing via arecord -l (alsa-utils). Capture device examples: "default", "hw:0,0"
        return {
            "format": "alsa",
            "device_prefix": "",  # pass the device name as-is
            "list_devices_cmd": ["arecord", "-l"],
        }
    else:
        raise NotImplementedError(f"Unsupported OS: {system}")


def list_audio_devices():
    """List available audio input devices depending on OS."""
    os_name = platform.system()
    print(f"🔍 Searching for audio devices on {os_name}...")
    try:
        cfg = get_platform_config()
        cmd = cfg["list_devices_cmd"]
        print(f"   (Command: {' '.join([str(x) for x in cmd])})")

        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
        output = (result.stdout or "") + "\n" + (result.stderr or "")

        print("-" * 60)
        if os_name == "Windows":
            print("DirectShow audio devices (use the name exactly as shown):")
            # ffmpeg dshow prints lines like:  "Microphone (XYZ)" (audio)
            devices = re.findall(r'"(.*?)"\s+\(audio\)', output, flags=re.IGNORECASE)
            if not devices:
                print(output.strip() or "No devices found. Ensure ffmpeg.exe is in PATH.")
            for d in devices:
                print(f"  ➡️  {d}")

        elif os_name == "Darwin":
            print("AVFoundation devices (use :<index> as --device for audio):")
            # ffmpeg avfoundation listing prints lines like: [AVFoundation indev @ ...] [1] Built-in Microphone
            # We extract [index] name
            lines = re.findall(r"\[(\d+)\]\s+(.+)", output)
            if not lines:
                print(output.strip() or "No devices found. Ensure ffmpeg is installed.")
            else:
                print("Audio devices often appear as a second list; try indexes you see here.")
                for idx, name in lines:
                    print(f"  ➡️  Index: {idx} | Name: {name.strip()} (use :{idx})")

        elif os_name == "Linux":
            print("ALSA capture devices (suggested names: 'default', 'hw:0,0'):")
            if "no soundcards found" in output.lower():
                print("No soundcards found by 'arecord -l'.")
            else:
                print(output.strip())

        print("-" * 60)

    except FileNotFoundError:
        tool = "ffmpeg" if os_name != "Linux" else "arecord (alsa-utils)"
        print(f"[ERROR] '{tool}' not found in PATH.")
    except Exception as e:
        print(f"[ERROR] Listing error: {e}")


# ==============================
# Streamer
# ==============================

class StableAudioStreamer:
    """Audio streamer with robust WS connection and FFmpeg capture."""

    def __init__(self, device_name: str, ws_url: str, sample_rate: int = 16000):
        self.device_name = device_name
        self.ws_url = ws_url
        self.sample_rate = sample_rate

        self.ffmpeg_process = None
        self.ws = None
        self.running = False
        self.connection_count = 0
        self.platform_config = get_platform_config()

    async def start_streaming(self):
        """Reconnect loop with backoff on connection errors."""
        self.running = True
        while self.running:
            try:
                await self.connect_and_stream()
            except KeyboardInterrupt:
                print("\n[info] Shutdown requested by user")
                break
            except Exception as e:
                # If FFmpeg failed with a fatal input error, we stop looping.
                msg = str(e)
                if "FATAL_INPUT" in msg:
                    print("[error] Fatal input error from FFmpeg. Stopping.")
                    self.running = False
                    break
                print(f"[error] Connection error: {e!r}")
                if self.running:
                    print("[info] Reconnecting in 2 seconds...")
                    await asyncio.sleep(2)

    async def connect_and_stream(self):
        """Connect once and stream until error/stop."""
        self.connection_count += 1
        print(f"[info] Connection attempt #{self.connection_count} to {self.ws_url}")

        async with websockets.connect(
            self.ws_url,
            max_size=2**20,
            ping_interval=15,
            ping_timeout=8,
            close_timeout=3,
            compression=None,
        ) as ws:
            self.ws = ws
            print(f"[info] ✅ Connection #{self.connection_count} established")
            await ws.send(json.dumps({"cmd": "hello", "client": "audio_streamer_multi_os"}))

            # Start FFmpeg
            await self.start_ffmpeg_optimized()
            # Stream
            await self.stream_with_heartbeat()

    def build_ffmpeg_cmd(self) -> list:
        """Build the correct FFmpeg command by OS (correct order!)."""
        fmt = self.platform_config["format"]
        prefix = self.platform_config["device_prefix"]
        device_full_name = f"{prefix}{self.device_name}"

        # Base
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-f", fmt]

        # Windows dshow needs audio buffer option BEFORE -i
        if fmt == "dshow":
            cmd += ["-audio_buffer_size", "20"]

        # Input
        cmd += ["-i", device_full_name]

        # Output format: mono 16kHz, 16-bit PCM to stdout
        cmd += ["-ac", "1", "-ar", str(self.sample_rate), "-f", "s16le", "-flush_packets", "1", "pipe:1"]
        return cmd

    async def start_ffmpeg_optimized(self):
        """Launch FFmpeg as a subprocess with stdout=audio stream."""
        cmd = self.build_ffmpeg_cmd()
        print(f"[info] Launching FFmpeg with command: {' '.join(cmd)}")
        try:
            self.ffmpeg_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=2**20,
            )
            print("[info] ✅ FFmpeg started successfully")
        except FileNotFoundError:
            raise RuntimeError("FFmpeg not found in PATH. Please place ffmpeg.exe next to your .exe or add to PATH.")
        except Exception as e:
            raise RuntimeError(f"Failed to start FFmpeg: {e}")

    async def stream_with_heartbeat(self):
        """Read PCM chunks from FFmpeg and send over WS. Stop on fatal FFmpeg errors."""
        CHUNK_SIZE = 3200  # 0.1s @ 16kHz mono (2 bytes per sample)
        last_heartbeat = time.time()
        heartbeat_interval = 10
        bytes_sent = 0

        try:
            while self.running and self.ffmpeg_process:
                try:
                    chunk = await asyncio.wait_for(self.ffmpeg_process.stdout.read(CHUNK_SIZE), timeout=1.0)
                except asyncio.TimeoutError:
                    # Periodic heartbeat
                    if time.time() - last_heartbeat > heartbeat_interval:
                        await self.send_heartbeat()
                        last_heartbeat = time.time()
                    continue

                if not chunk:
                    # No more data: check stderr to understand why
                    stderr_output = await self._read_ffmpeg_stderr()
                    print(f"[warning] No more audio data from FFmpeg.\n{stderr_output}")
                    # Detect fatal device errors and stop the reconnection loop
                    if ("Error opening input" in stderr_output) or ("No such file or directory" in stderr_output) \
                       or ("could not find" in stderr_output.lower()):
                        self.running = False
                        raise RuntimeError("FATAL_INPUT: FFmpeg cannot open the selected device.")
                    break

                await self.ws.send(chunk)
                bytes_sent += len(chunk)

                if time.time() - last_heartbeat > heartbeat_interval:
                    await self.send_heartbeat()
                    last_heartbeat = time.time()

                # Light progress log every ~30s
                if bytes_sent and (bytes_sent % (self.sample_rate * 2 * 30) == 0):
                    print(f"[info] 📊 {bytes_sent // 1024}KB of audio sent")

        except websockets.exceptions.ConnectionClosed as e:
            print(f"[warning] WebSocket connection closed: {e}")
        except Exception as e:
            # Reraise to let start_streaming decide whether to retry or stop
            raise
        finally:
            await self.cleanup_ffmpeg()

    async def _read_ffmpeg_stderr(self) -> str:
        try:
            if self.ffmpeg_process and self.ffmpeg_process.stderr:
                data = await self.ffmpeg_process.stderr.read()
                return (data or b"").decode(errors="ignore")
        except Exception:
            pass
        return ""

    async def send_heartbeat(self):
        """Send small ping message to keep WS alive."""
        try:
            await self.ws.send(json.dumps({"cmd": "ping", "timestamp": time.time()}))
        except Exception:
            pass

    async def cleanup_ffmpeg(self):
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
        print("[info] 🛑 Streaming stop requested")
        self.running = False


async def run_stable_client(device_name: str, ws_url: str = "ws://127.0.0.1:8123/"):
    streamer = StableAudioStreamer(device_name, ws_url)
    try:
        await streamer.start_streaming()
    except KeyboardInterrupt:
        print("\n[info] Shutdown requested")
    finally:
        streamer.stop()
        await asyncio.sleep(0.5)


# ==============================
# CLI
# ==============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stable Multi-OS Audio Client")
    parser.add_argument("--list-devices", action="store_true", help="List available audio input devices and exit.")
    parser.add_argument("--device", help="Device name (Windows: exact dshow name; macOS: use index like ':1'; Linux: 'default' or 'hw:0,0').")
    parser.add_argument("--ws", default="ws://127.0.0.1:8123/", help="WebSocket URL (default: ws://127.0.0.1:8123/)")
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    if not args.device:
        print("[ERROR] The --device argument is required.")
        print("Examples:")
        if platform.system() == "Windows":
            print('  python stable_audio_client_multi_os.py --list-devices')
            print('  python stable_audio_client_multi_os.py --device "Microphone (Your Device)"')
        elif platform.system() == "Darwin":
            print('  python stable_audio_client_multi_os.py --list-devices')
            print('  python stable_audio_client_multi_os.py --device :1')
        else:
            print("  python stable_audio_client_multi_os.py --device default")
        sys.exit(1)

    print("🎤 STABLE AUDIO CLIENT (MULTI-OS)")
    print("=" * 50)
    print(f"OS: {platform.system()}")
    print(f"Device: {args.device}")
    print(f"WebSocket: {args.ws}")
    print("Press Ctrl+C to stop")
    print("=" * 50)

    asyncio.run(run_stable_client(args.device, args.ws))
