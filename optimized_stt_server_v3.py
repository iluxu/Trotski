import asyncio, json, os, time, re, hashlib, shutil
from typing import Set, Optional, List, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path # For multi-OS compatibility

import numpy as np
import websockets
from faster_whisper import WhisperModel

# ========================
# Configuration
# ========================
HOST = os.getenv("STT_HOST", "127.0.0.1")
PORT = int(os.getenv("STT_PORT", "8123"))

SAMPLE_RATE = 16000
WINDOW_SECONDS = float(os.getenv("STT_WINDOW_SECONDS", "6"))
HOP_SECONDS = float(os.getenv("STT_HOP_SECONDS", "0.8"))
ENERGY_GATE = float(os.getenv("STT_ENERGY_GATE", "1e-4"))

MODEL_NAME = os.getenv("STT_MODEL", "small")
COMPUTE_TYPE = os.getenv("STT_COMPUTE", "int8")
FORCE_LANG = os.getenv("STT_LANG") or None
# Generic tech prompt for better transcription of jargon
INITIAL_PROMPT = os.getenv(
    "STT_INITIAL_PROMPT",
    "Software engineering, data structures, algorithms, system design, cloud computing, AWS, Azure, GCP, microservices, API, "
    "CI/CD, DevOps, machine learning, data science, Python, Java, JavaScript, SQL, NoSQL, product management, agile, scrum."
)

# LLM
LLM_ENABLED = os.getenv("STT_LLM_ENABLED", "1") not in ("0", "false", "False")
GPT_MODEL = os.getenv("STT_LLM_MODEL", "gpt-5-nano")
GPT_EFFORT = os.getenv("STT_LLM_EFFORT", "low")
# ⚠️ Do not hardcode a default key in source code
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# LLM Context Control
CONTEXT_WINDOW_CHARS = int(os.getenv("STT_CONTEXT_CHARS", "3000"))
MEMORY_SEGMENTS = int(os.getenv("STT_MEMORY_SEGMENTS", "15"))
LLM_MIN_GAP_SEC = float(os.getenv("STT_LLM_MIN_GAP_SEC", "1.0"))
ANSWERS_PER_MIN = int(os.getenv("STT_LLM_ANSWERS_PER_MIN", "8"))
SEEN_TTL_SEC = float(os.getenv("STT_SEEN_TTL_SEC", "60"))
MAX_CONCURRENT_LLM = int(os.getenv("STT_MAX_CONCURRENT_LLM", "2"))
MAX_OUTTOK = int(os.getenv("STT_MAX_OUTTOK", "512")) # Max output tokens for generation

LLM_INCLUDE_FULL_TRANSCRIPT = os.getenv("STT_LLM_INCLUDE_FULL_TRANSCRIPT", "1") not in ("0", "false", "False")
LLM_CONTEXT_MODE = os.getenv("STT_LLM_CONTEXT_MODE", "full").lower() # full | window | headtail
LLM_WINDOW_LINES = int(os.getenv("STT_LLM_WINDOW_LINES", "160"))
LLM_HEAD_LINES   = int(os.getenv("STT_LLM_HEAD_LINES",   "60"))
LLM_TAIL_LINES   = int(os.getenv("STT_LLM_TAIL_LINES",   "300"))
LLM_MAX_CONTEXT_CHARS = int(os.getenv("STT_LLM_MAX_CONTEXT_CHARS", "300000"))

PERSONA = os.getenv("STT_LLM_PERSONA", "candidate").lower()

# Debug
DEBUG = os.getenv("STT_DEBUG", "1") not in ("0", "false", "False")
VERBOSE_BUFFER = os.getenv("STT_VERBOSE_BUFFER", "0") not in ("0", "false", "False")

# ========================
# Small Helpers
# ========================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _qkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(s).lower())

def build_llm_context_text() -> str:
    """
    Constructs the context to send to the LLM based on LLM_CONTEXT_MODE,
    then truncates it from the end if necessary to stay under LLM_MAX_CONTEXT_CHARS.
    """
    mode = (LLM_CONTEXT_MODE or "window").lower()
    if mode == "full":
        txt = "\n".join(f"{i+1}. {line}" for i, line in enumerate(transcript_lines))
    elif mode == "headtail":
        txt = "\n".join(get_head_tail_lines(LLM_HEAD_LINES, LLM_TAIL_LINES))
    else:  # "window"
        txt = "\n".join(get_last_n_lines(LLM_WINDOW_LINES))

    if LLM_MAX_CONTEXT_CHARS and len(txt) > LLM_MAX_CONTEXT_CHARS:
        txt = txt[-LLM_MAX_CONTEXT_CHARS:]  # Truncate from the end (recency > history)
    return txt

# ========================
# Whisper Loading with Resilient, Multi-OS Cache
# ========================
def hf_cache_dir() -> str:
    """Returns a platform-agnostic cache directory inside the user's home."""
    # This is robust for Windows, macOS, and Linux
    cache_path = Path.home() / ".cache" / "hf_models"
    cache_path.mkdir(parents=True, exist_ok=True)
    return str(cache_path)

def load_whisper_model() -> WhisperModel:
    cache_root = hf_cache_dir()
    try:
        print(f"[model] Loading {MODEL_NAME} in {cache_root}")
        return WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE, download_root=cache_root)
    except Exception:
        # Attempt to purge a potentially corrupted cache for this model
        model_folder_name = f"models--Systran--faster-whisper-{MODEL_NAME}"
        corrupted_path = os.path.join(cache_root, model_folder_name)
        if os.path.isdir(corrupted_path):
            print(f"[model] Corrupted cache detected, purging: {corrupted_path}")
            shutil.rmtree(corrupted_path, ignore_errors=True)
        # Retry download
        return WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE, download_root=cache_root)

# ========================
# LLM Response Sanitization
# ========================
COACHING_PATTERNS = [
    r"\bHappy to elaborate if useful\b.*",
    r"\bLet me know if you'd like more details\b.*",
    r"\bYou (?:should|could|can)\b.*",
    r"\bYou demonstrate\b.*",
    r"\bOne (?:improvement|area to improve)\b.*",
    r"\bYour answer\b.*",
]

def sanitize_candidate_voice(text: str) -> str:
    """Removes coaching/feedback phrases and reformulates direct address."""
    # Remove feedback/coaching phrases
    for pat in COACHING_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    # Avoid direct address "you/your" -> neutral reformulation
    text = re.sub(r"\byou\b", "the interviewer", text, flags=re.IGNORECASE)
    text = re.sub(r"\byour\b", "the", text, flags=re.IGNORECASE)
    # Clean up whitespace
    return re.sub(r"\s{2,}", " ", text).strip(" .")

# ========================
# LLM Analyzer (more robust)
# ========================
@dataclass
class ConversationSegment:
    text: str
    timestamp: float

@dataclass
class QuestionCandidate:
    question: str
    context: str
    confidence: float
    urgency: str
    topic_area: str
    timestamp: float
    should_answer: bool = False

class ImprovedLLMAnalyzer:
    def __init__(self, api_key: str):
        self.enabled = LLM_ENABLED and bool(api_key.startswith("sk-"))
        self.client = None
        self.conversation_memory = deque(maxlen=MEMORY_SEGMENTS)
        self.previous_response_id = None
        self.last_llm_emit = 0.0
        self.seen_questions: Dict[str, float] = {}
        self.answers_timestamps = deque(maxlen=64)
        self.sem = asyncio.Semaphore(MAX_CONCURRENT_LLM)

        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                print(f"[llm] Ready (model={GPT_MODEL}, effort={GPT_EFFORT})")
            except Exception as e:
                print(f"[llm] Init error: {e}")
                self.enabled = False
        else:
            print("[llm] Disabled (no valid API key)")

    def _trim_seen(self):
        now = time.time()
        expired = [k for k, t in self.seen_questions.items() if now - t > SEEN_TTL_SEC]
        for k in expired:
            self.seen_questions.pop(k, None)

    async def analyze_segment(self, segment: ConversationSegment) -> List[QuestionCandidate]:
        if not self.enabled:
            return []
        self.conversation_memory.append(segment)
        self._trim_seen()
        return await self._detect_questions(segment)

    def _build_context_summary(self) -> str:
        text = build_llm_context_text()
        return f"Transcript (numbered lines):\n{text}"

    async def _detect_questions(self, segment: ConversationSegment) -> List[QuestionCandidate]:
        # Rate limiting
        if time.time() - self.last_llm_emit < LLM_MIN_GAP_SEC:
            return []

        qs = self._extract_questions_aggressive(segment.text)
        if not qs:
            return []
        if DEBUG:
            print(f"[llm] Found {len(qs)} potential questions: {[q[:40] + '...' for q in qs]}")

        candidates: List[QuestionCandidate] = []
        for q in qs:
            k = _qkey(q)
            if not k or k in self.seen_questions:
                continue
            
            decide = await asyncio.get_running_loop().run_in_executor(None, self._should_answer, q)
            if decide.get("should_answer"):
                self.seen_questions[k] = time.time()
                cand = QuestionCandidate(
                    question=q, context=segment.text, confidence=decide.get("confidence", 0.6),
                    urgency=decide.get("urgency", "relevant"), topic_area="interview",
                    timestamp=segment.timestamp, should_answer=True,
                )
                candidates.append(cand)
                if DEBUG:
                    print(f"[llm] Will answer: {q}")
        return candidates[:3]

    def _extract_questions_aggressive(self, text: str) -> List[str]:
        """Strict yet tolerant extraction for implicit and mid-sentence questions."""
        text = _norm(text)
        if not text: return []

        sentences = re.split(r'(?<=[.?!])\s+', text)
        questions: List[str] = []

        # Explicit questions (end with '?')
        for s in sentences:
            if s.endswith("?") and len(s.split()) >= 5:
                questions.append(s)

        # Implicit questions
        head_terms = [
            "what", "how", "why", "when", "where", "which", "who", "whom",
            "can you", "could you", "would you", "will you", "do you", "are you",
            "tell me", "explain", "describe", "walk me through",
        ]
        for s in sentences:
            sl = s.lower()
            if any(sl.startswith(term + " ") for term in head_terms) and len(s.split()) >= 6:
                questions.append(s if s.endswith("?") else s + "?")

        # Deduplicate and clean
        seen = set(); clean: List[str] = []
        for q in questions:
            k = _qkey(q)
            if k and k not in seen:
                seen.add(k)
                clean.append(q.strip(" ,;:-–—"))
        return clean[:3]

    def _should_answer(self, question: str) -> Dict[str, Any]:
        if not self.enabled: return {"should_answer": False}
        
        # RPM cap
        now = time.time()
        while self.answers_timestamps and now - self.answers_timestamps[0] > 60:
            self.answers_timestamps.popleft()
        if len(self.answers_timestamps) >= ANSWERS_PER_MIN:
            return {"should_answer": False}

        # Local heuristic as a fallback (permissive if it looks like a question)
        ql = question.lower().strip()
        wh = ("what", "how", "why", "when", "where", "which", "who", "whom", "can", "could", "would", "will", "do", "is", "are", "should")
        looks_like_q = ql.endswith("?") or any(ql.startswith(w + " ") for w in wh)

        try:
            resp = self.client.responses.create(
                model=GPT_MODEL, reasoning={"effort": GPT_EFFORT},
                input=[
                    {"role": "developer", "content": "Is the following an interview question that the candidate should answer? Reply with EXACTLY one token: YES or NO."},
                    {"role": "user", "content": f"Question: {question}\n\nReply ONLY 'YES' or 'NO'."},
                ],
            )
            out = (getattr(resp, "output_text", "") or "").strip().lower()
            decided_yes = "yes" in out
        except Exception as e:
            if DEBUG: print(f"[llm] Decision error: {e}")
            decided_yes = looks_like_q  # Permissive fallback

        if decided_yes:
            ts = time.time()
            self.last_llm_emit = ts
            self.answers_timestamps.append(ts)

        return {"should_answer": decided_yes, "confidence": 0.75 if decided_yes else 0.3}

    async def generate_answer(self, candidate: QuestionCandidate) -> str:
        if not self.enabled: return "[LLM disabled]"
        async with self.sem:
            return await asyncio.get_running_loop().run_in_executor(None, self._gen, candidate)

    def _gen(self, candidate: QuestionCandidate) -> str:
        try:
            q = (candidate.question or "").strip()
            
            if PERSONA == "candidate":
                system_prompt = (
                    "You are a tech professional answering questions in a job interview. "
                    "Speak in the first person singular ('I'). Provide 3-5 concise, complete sentences. "
                    "Start with a direct answer, then provide concrete points or a brief, relevant example from a tech domain "
                    "(e.g., web services, data pipelines, ML models). "
                    "Do not use bullet points, lists, or any meta-commentary/coaching phrases."
                )
            else: # Fallback "coach" persona
                system_prompt = "Provide a concise, helpful answer in 2-4 sentences."

            inputs = [{"role": "developer", "content": system_prompt}]
            
            if LLM_INCLUDE_FULL_TRANSCRIPT:
                full_tx = build_llm_context_text()
                if full_tx:
                    inputs.append({"role": "user", "content": f"Full Interview Transcript (for context):\n{full_tx}"})

            inputs.extend([
                {"role": "user", "content": f"Question: {q}"},
                {"role": "user", "content": (
                    "Answer now as a final spoken response. Do not address the interviewer as 'you'. "
                    "Avoid phrases like ‘Happy to elaborate’, ‘you should’, or other feedback."
                )},
            ])

            resp = self.client.responses.create(
                model=GPT_MODEL, reasoning={"effort": GPT_EFFORT}, input=inputs, max_output_tokens=MAX_OUTTOK
            )

            # Robust text extraction from response object
            raw_text = (getattr(resp, "output_text", "") or "").strip()
            
            # Sanitize and clean
            cleaned = re.sub(r"^\s*(yes|no)\s*[:\-\.,]*\s*", "", raw_text, flags=re.IGNORECASE)
            cleaned = sanitize_candidate_voice(cleaned)

            if not cleaned.strip():
                return "I would evaluate the model on a held-out test set to measure its generalization performance on unseen data."

            return cleaned

        except Exception as e:
            return f"[Answer error] {str(e)[:120]}"

# ========================
# Global Server State
# ========================
@dataclass
class ClientState:
    ws: websockets.WebSocketServerProtocol
    queue: asyncio.Queue
    sender_task: Optional[asyncio.Task] = None
    wants_broadcast: bool = True
    role: str = "ui"

clients: Dict[websockets.WebSocketServerProtocol, ClientState] = {}
latest_text = ""
detected = deque(maxlen=500)
qa_log = deque(maxlen=200)

transcript_lines: List[str] = []
MAX_TRANSCRIPT_LINES = int(os.getenv("STT_MAX_TRANSCRIPT_LINES", "5000"))
sentence_buf = ""
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.?!])\s+')

pcm_buf = bytearray()
bytes_since_last = 0

WIN_BYTES = int(SAMPLE_RATE * 2 * WINDOW_SECONDS)
HOP_BYTES = int(SAMPLE_RATE * 2 * HOP_SECONDS)
MAX_BUFFER_BYTES = WIN_BYTES * 2

audio_lock = asyncio.Lock()
server_shutdown = asyncio.Event()
transcriber_task: Optional[asyncio.Task] = None

# ========================
# Broadcast with Per-Client Queues (Backpressure-Friendly)
# ========================
async def _client_sender(client: ClientState):
    client_id = f"{client.ws.remote_address}"
    if DEBUG: print(f"[ws_sender] Starting sender for {client_id}")
    try:
        while True:
            msg = await client.queue.get()
            await client.ws.send(msg)
    except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError) as e:
        if DEBUG: print(f"[ws_sender] Connection closed for {client_id}: {getattr(e, 'code', 'closed')}")
    except Exception as e:
        if DEBUG: print(f"[ws_sender] Sender error for {client_id}: {e}")
    finally:
        try:
            await client.ws.close()
        except Exception: pass
        if DEBUG: print(f"[ws_sender] Sender stopped for {client_id}")

async def broadcast(data: dict):
    if not clients: return
    msg = json.dumps(data, ensure_ascii=False)
    dead = []
    for c in list(clients.values()):
        if not c.wants_broadcast: continue # Do not send to audio streamers
        try:
            if c.queue.full():
                try: _ = c.queue.get_nowait()
                except Exception: pass
            c.queue.put_nowait(msg)
        except Exception:
            dead.append(c)
    for d in dead: clients.pop(d.ws, None)

# ========================
# WebSocket Handler (Audio Ingestion Only)
# ========================
async def handler(ws: websockets.WebSocketServerProtocol):
    global transcriber_task, bytes_since_last
    print(f"[ws] Client connected from {ws.remote_address}")
    client = ClientState(ws=ws, queue=asyncio.Queue(maxsize=100))
    client.sender_task = asyncio.create_task(_client_sender(client))
    clients[ws] = client

    try:
        await ws.send(json.dumps({
            "snapshot": {"transcript": latest_text, "detected": list(detected), "lines": transcript_lines[-200:]}
        }))
    except Exception: pass

    if transcriber_task is None or transcriber_task.done():
        transcriber_task = asyncio.create_task(read_and_transcribe_loop())

    try:
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                async with audio_lock:
                    pcm_buf.extend(msg)
                    bytes_since_last += len(msg)
                    if len(pcm_buf) > MAX_BUFFER_BYTES:
                        excess = len(pcm_buf) - MAX_BUFFER_BYTES // 2
                        del pcm_buf[:excess]
            else:
                try:
                    data = json.loads(msg)
                    cmd = (data.get("cmd") or "").lower()
                    if cmd == "hello":
                        role = (data.get("client") or data.get("role") or "").lower()
                        if role in ("audio_streamer", "streamer", "ingest"):
                            client.role = "streamer"
                            client.wants_broadcast = False
                            if DEBUG: print(f"[ws] Marked {ws.remote_address} as streamer; broadcasts OFF")
                    elif cmd == "reset":
                        async with audio_lock:
                            pcm_buf.clear()
                            bytes_since_last = 0
                        reset_state()
                        print("[ws] Reset completed")
                except Exception as e:
                    if DEBUG: print(f"[ws] Control message parse error: {e}")
    except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError) as e:
        if DEBUG: print(f"[ws] Client closed: {getattr(e, 'code', 'closed')}")
    except Exception as e:
        print(f"[ws] Feed error: {e}")
    finally:
        clients.pop(ws, None)
        if client.sender_task: client.sender_task.cancel()
        print("[ws] Client disconnected")

# ========================
# Global Transcriber Loop
# ========================
async def read_and_transcribe_loop():
    global bytes_since_last, latest_text, sentence_buf, transcript_lines
    loop = asyncio.get_running_loop()
    last_hashes = deque(maxlen=6)
    process_count = 0
    print("[stt] Transcription loop started (global)")

    while not server_shutdown.is_set():
        await asyncio.sleep(0.05)
        async with audio_lock:
            ready = (bytes_since_last >= HOP_BYTES and len(pcm_buf) >= WIN_BYTES)
            if not ready: continue
            bytes_since_last = 0
            window_bytes = bytes(pcm_buf[-WIN_BYTES:])
        
        process_count += 1
        arr = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        energy = float(np.mean(arr * arr))
        if energy < ENERGY_GATE: continue
        
        if DEBUG and process_count % 10 == 0:
            print(f"[stt] Processing chunk #{process_count}: {len(window_bytes)} bytes, energy: {np.sqrt(energy):.4f}")

        def _transcribe():
            try:
                segments, _ = whisper.transcribe(
                    arr, language=FORCE_LANG, vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 200},
                    initial_prompt=INITIAL_PROMPT if not latest_text else None
                )
                return list(segments)
            except Exception as e:
                print(f"[stt] Transcription error: {e}")
                return []

        segments = await loop.run_in_executor(None, _transcribe)
        if not segments: continue

        for segment in segments:
            text = _norm(segment.text)
            if not text or len(text) < 2: continue

            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h in last_hashes: continue
            last_hashes.append(h)

            sentence_buf = (sentence_buf + " " + text).strip()
            parts = SENTENCE_SPLIT_RE.split(sentence_buf)
            
            complete_sentences = []
            if sentence_buf and sentence_buf[-1] in ".?!":
                complete_sentences = parts
                sentence_buf = ""
            elif len(parts) > 1:
                complete_sentences = parts[:-1]
                sentence_buf = parts[-1]
            
            latest_text = sentence_buf

            for s in complete_sentences:
                s = _norm(s)
                if not s: continue
                transcript_lines.append(s)
                if len(transcript_lines) > MAX_TRANSCRIPT_LINES:
                    del transcript_lines[:MAX_TRANSCRIPT_LINES // 10]
                if DEBUG: print(f"[stt] Line: {s}")
                await broadcast({"line": {"n": len(transcript_lines), "text": s}})
            
            await broadcast({"partial": text})

            if llm_analyzer and llm_analyzer.enabled:
                try:
                    seg = ConversationSegment(text=text, timestamp=time.time())
                    cands = await llm_analyzer.analyze_segment(seg)
                    for cand in cands:
                        item = {
                            "q": cand.question, "t": int(cand.timestamp * 1000), "a": None,
                            "context": cand.context[:150], "urgency": "relevant", "confidence": cand.confidence,
                        }
                        detected.append(item)
                        await broadcast({"question_detected": item})
                        ans = await llm_analyzer.generate_answer(cand)
                        item["a"] = ans
                        qa_log.append(item)
                        await broadcast({"qa": item})
                        if DEBUG: print(f"[llm] Response: {ans[:60]}...")
                except Exception as e:
                    print(f"[llm] Analysis error: {e}")

# ========================
# Utilities
# ========================
def reset_state():
    global latest_text, sentence_buf, transcript_lines
    latest_text, sentence_buf = "", ""
    transcript_lines.clear()
    detected.clear()
    qa_log.clear()
    if llm_analyzer:
        llm_analyzer.conversation_memory.clear()
        llm_analyzer.seen_questions.clear()
        llm_analyzer.answers_timestamps.clear()

# ========================
# Main
# ========================
async def main():
    global whisper, llm_analyzer
    print("🚀 OPTIMIZED STT SERVER V3")
    print("=" * 50)
    print("[startup] Loading Whisper model...")
    whisper = load_whisper_model()
    print("[startup] Whisper loaded.")
    print("[startup] Initializing LLM Analyzer...")
    llm_analyzer = ImprovedLLMAnalyzer(OPENAI_API_KEY)

    print(f"\n📋 CONFIGURATION:")
    print(f"   Audio: {WINDOW_SECONDS}s window, {HOP_SECONDS}s hop")
    print(f"   Buffer: max {MAX_BUFFER_BYTES} bytes")
    print(f"   LLM: {'Enabled' if llm_analyzer.enabled else 'Disabled'}")
    print(f"   Debug: {'ON' if DEBUG else 'OFF'}")

    try:
        async with websockets.serve(
            handler, HOST, PORT, max_size=2**20, max_queue=32,
            ping_interval=10, ping_timeout=30, close_timeout=5, compression=None
        ):
            print(f"\n🎤 Server ready on ws://{HOST}:{PORT}/")
            print("Waiting for audio streams...")
            await asyncio.Future()
    except Exception as e:
        print(f"[error] Server failed to start: {e}")

if __name__ == "__main__":
    try:
        whisper: WhisperModel
        llm_analyzer: Optional[ImprovedLLMAnalyzer]
        asyncio.run(main())
    except KeyboardInterrupt:
        server_shutdown.set()
        print("\n[shutdown] Server stopped.")
    except Exception as e:
        print(f"\n[fatal_error] {e}")
        import traceback

        traceback.print_exc()
