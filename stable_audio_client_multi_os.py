# Patch pour améliorer la stabilité des connexions WebSocket

import asyncio
import websockets
import subprocess
import json
import time

class StableAudioStreamer:
    """Streamer audio avec gestion robuste des connexions"""
    
    def __init__(self, device_name: str, ws_url: str):
        self.device_name = device_name
        self.ws_url = ws_url
        self.ffmpeg_process = None
        self.ws = None
        self.running = False
        self.connection_count = 0
        
    async def start_streaming(self):
        """Démarre le streaming avec reconnexion automatique"""
        self.running = True
        
        while self.running:
            try:
                await self.connect_and_stream()
            except KeyboardInterrupt:
                print("\n[info] Arrêt demandé par l'utilisateur")
                break
            except Exception as e:
                print(f"[error] Erreur de connexion: {e}")
                if self.running:
                    print("[info] Reconnexion dans 2 secondes...")
                    await asyncio.sleep(2)
    
    async def connect_and_stream(self):
        """Connexion WebSocket optimisée"""
        self.connection_count += 1
        print(f"[info] Tentative de connexion #{self.connection_count} vers {self.ws_url}")
        
        # Paramètres WebSocket optimisés pour la stabilité
        async with websockets.connect(
            self.ws_url,
            max_size=2**20,        # 1MB max message
            ping_interval=15,      # Ping toutes les 15s (plus fréquent)
            ping_timeout=8,        # Timeout de 8s
            close_timeout=3,       # Fermeture rapide
            compression=None       # Pas de compression pour moins de latence
        ) as ws:
            self.ws = ws
            print(f"[info] ✅ Connexion #{self.connection_count} établie")
            
            # Envoyer un message initial pour confirmer la connexion
            await ws.send(json.dumps({"cmd": "hello", "client": "audio_streamer"}))
            
            # Démarrer FFmpeg
            await self.start_ffmpeg_optimized()
            
            # Streaming principal
            await self.stream_with_heartbeat()
    
    async def start_ffmpeg_optimized(self):
        """FFmpeg avec paramètres optimisés pour la stabilité"""
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "warning",  # Moins de logs
            "-f", "dshow",
            "-audio_buffer_size", "20",              # Buffer plus petit
            "-i", f"audio={self.device_name}",
            "-ac", "1",                              # Mono
            "-ar", "16000",                          # 16kHz
            "-f", "s16le",                           # PCM 16-bit
            "-flush_packets", "1",                   # Flush immédiat
            "pipe:1"
        ]
        
        try:
            self.ffmpeg_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=2**20  # Limit buffer size
            )
            print("[info] ✅ FFmpeg démarré avec paramètres optimisés")
        except Exception as e:
            raise RuntimeError(f"Impossible de démarrer FFmpeg: {e}")
    
    async def stream_with_heartbeat(self):
        """Streaming avec heartbeat pour maintenir la connexion"""
        chunk_size = 3200  # 0.1s d'audio à 16kHz mono (plus petit = moins de latence)
        last_heartbeat = time.time()
        heartbeat_interval = 10  # Heartbeat toutes les 10s
        bytes_sent = 0
        
        try:
            while self.running and self.ffmpeg_process:
                # Lire chunk audio
                try:
                    chunk = await asyncio.wait_for(
                        self.ffmpeg_process.stdout.read(chunk_size), 
                        timeout=1.0  # Timeout de lecture
                    )
                except asyncio.TimeoutError:
                    # Pas de données audio, envoyer un heartbeat si nécessaire
                    if time.time() - last_heartbeat > heartbeat_interval:
                        await self.send_heartbeat()
                        last_heartbeat = time.time()
                    continue
                
                if not chunk:
                    print("[warning] Plus de données audio de FFmpeg")
                    break
                
                # Envoyer chunk audio
                await self.ws.send(chunk)
                bytes_sent += len(chunk)
                
                # Heartbeat périodique
                if time.time() - last_heartbeat > heartbeat_interval:
                    await self.send_heartbeat()
                    last_heartbeat = time.time()
                
                # Log de progression (moins fréquent)
                if bytes_sent % (16000 * 2 * 30) == 0:  # Toutes les 30s
                    print(f"[info] 📊 {bytes_sent // 1024}KB audio envoyés")
                    
        except websockets.exceptions.ConnectionClosed as e:
            print(f"[warning] Connexion WebSocket fermée: {e}")
        except Exception as e:
            print(f"[error] Erreur de streaming: {e}")
        finally:
            await self.cleanup_ffmpeg()
    
    async def send_heartbeat(self):
        """Envoie un heartbeat pour maintenir la connexion"""
        try:
            heartbeat_msg = json.dumps({"cmd": "ping", "timestamp": time.time()})
            await self.ws.send(heartbeat_msg)
        except Exception as e:
            print(f"[warning] Erreur heartbeat: {e}")
    
    async def cleanup_ffmpeg(self):
        """Nettoyage propre de FFmpeg"""
        if self.ffmpeg_process:
            try:
                # Tentative d'arrêt propre
                self.ffmpeg_process.terminate()
                await asyncio.wait_for(self.ffmpeg_process.wait(), timeout=3)
                print("[info] ✅ FFmpeg arrêté proprement")
            except asyncio.TimeoutError:
                # Force kill si nécessaire
                self.ffmpeg_process.kill()
                print("[warning] ⚠️ FFmpeg forcé à s'arrêter")
            except Exception as e:
                print(f"[warning] Erreur cleanup FFmpeg: {e}")
            finally:
                self.ffmpeg_process = None
    
    def stop(self):
        """Arrête le streaming"""
        print("[info] 🛑 Arrêt du streaming demandé")
        self.running = False

# Serveur WebSocket avec gestion améliorée des heartbeats
class ImprovedWebSocketHandler:
    """Handler WebSocket amélioré pour gérer les heartbeats"""
    
    def __init__(self, original_handler):
        self.original_handler = original_handler
    
    async def enhanced_handler(self, ws, path):
        """Handler avec gestion des messages de contrôle"""
        print(f"[ws] 🔌 Nouvelle connexion depuis {ws.remote_address}")
        
        # Wrapper pour intercepter les messages JSON
        original_send = ws.send
        
        async def intercepted_handler():
            try:
                async for message in ws:
                    if isinstance(message, str):
                        # Message JSON - traiter les commandes de contrôle
                        try:
                            data = json.loads(message)
                            if data.get("cmd") == "ping":
                                # Répondre au ping
                                await ws.send(json.dumps({"cmd": "pong", "timestamp": time.time()}))
                                continue
                            elif data.get("cmd") == "hello":
                                # Message de bienvenue
                                print(f"[ws] 👋 Client identifié: {data.get('client', 'unknown')}")
                                continue
                        except json.JSONDecodeError:
                            pass
                    
                    # Rediriger vers le handler original pour les données audio
                    # (On simule en récréant le message via une queue)
                    if hasattr(ws, '_message_queue'):
                        await ws._message_queue.put(message)
            except Exception as e:
                print(f"[ws] ⚠️ Erreur dans le handler: {e}")
        
        # Lancer les deux handlers en parallèle
        await asyncio.gather(
            intercepted_handler(),
            self.original_handler(ws, path),
            return_exceptions=True
        )

# Fonction utilitaire pour appliquer le patch
def apply_stability_patches():
    """Applique les patches de stabilité"""
    
    # Patch 1: Paramètres WebSocket serveur optimisés
    def get_optimized_server_args():
        return {
            'max_size': 2**20,           # 1MB max
            'ping_interval': 10,         # Ping toutes les 15s
            'ping_timeout': 30,           # Timeout 8s
            'close_timeout': 3,          # Fermeture rapide
            'compression': None,         # Pas de compression
            'max_queue': 32,            # Queue plus petite
        }
    
    # Patch 2: Filtre pour les lignes de tirets
    def filter_transcription_output(text: str) -> str:
        """Filtre les artefacts de transcription"""
        # Supprimer les longues séquences de caractères répétés
        import re
        # Ligne avec plus de 20 caractères identiques consécutifs
        if re.search(r'(.)\1{20,}', text):
            return ""
        # Ligne trop courte ou que des espaces/caractères spéciaux
        if len(text.strip()) < 3 or not re.search(r'[a-zA-Z]', text):
            return ""
        return text
    
    return {
        'server_args': get_optimized_server_args(),
        'text_filter': filter_transcription_output
    }

# Usage example
async def run_stable_client(device_name: str, ws_url: str = "ws://127.0.0.1:8123/"):
    """Lance le client audio stable"""
    streamer = StableAudioStreamer(device_name, ws_url)
    
    try:
        await streamer.start_streaming()
    except KeyboardInterrupt:
        print("\n[info] Arrêt demandé")
    finally:
        streamer.stop()
        await asyncio.sleep(1)  # Laisser le temps pour le cleanup

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Client Audio Stable")
    parser.add_argument("--device", required=True, help="Nom du device audio")
    parser.add_argument("--ws", default="ws://127.0.0.1:8123/", help="URL WebSocket")
    
    args = parser.parse_args()
    
    print("🎤 CLIENT AUDIO STABLE")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"WebSocket: {args.ws}")
    print("Appuyez sur Ctrl+C pour arrêter")
    print("=" * 50)
    
    asyncio.run(run_stable_client(args.device, args.ws))
