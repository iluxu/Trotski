import asyncio, json, os, time, re, hashlib, shutil
from typing import Set, Optional, List, Dict, Any
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path  # For multi-OS compatibility

import numpy as np
import websockets
from faster_whisper import WhisperModel
import requests

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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Advanced LLM Context Control (restored from French version)
LLM_INCLUDE_FULL_TRANSCRIPT = os.getenv("STT_LLM_INCLUDE_FULL_TRANSCRIPT", "1") not in ("0", "false", "False")
TECH_INTERVIEW_MODE = os.getenv("STT_TECH_INTERVIEW_MODE", "1") not in ("0", "false", "False")
LLM_CONTEXT_MODE = os.getenv("STT_LLM_CONTEXT_MODE", "full").lower()
LLM_WINDOW_LINES = int(os.getenv("STT_LLM_WINDOW_LINES", "160"))
LLM_HEAD_LINES = int(os.getenv("STT_LLM_HEAD_LINES", "60"))
LLM_TAIL_LINES = int(os.getenv("STT_LLM_TAIL_LINES", "300"))
LLM_MAX_CONTEXT_CHARS = int(os.getenv("STT_LLM_MAX_CONTEXT_CHARS", "300000"))
MAX_OUTTOK = int(os.getenv("STT_MAX_OUTTOK", "512"))
PERSONA = os.getenv("STT_LLM_PERSONA", "candidate").lower()

# LLM Rate Limiting
LLM_MIN_GAP_SEC = float(os.getenv("STT_LLM_MIN_GAP_SEC", "1.0"))
ANSWERS_PER_MIN = int(os.getenv("STT_LLM_ANSWERS_PER_MIN", "8"))
SEEN_TTL_SEC = float(os.getenv("STT_SEEN_TTL_SEC", "60"))
MAX_CONCURRENT_LLM = int(os.getenv("STT_MAX_CONCURRENT_LLM", "2"))

# Debug
DEBUG = os.getenv("STT_DEBUG", "1") not in ("0", "false", "False")
VERBOSE_BUFFER = os.getenv("STT_VERBOSE_BUFFER", "0") not in ("0", "false", "False")

LICENSE_FILE = Path.home() / ".interview_copilot_license"
GUMROAD_PRODUCT_PERMALINK = "himlkf"  

# ========================
# Small Helpers & Context Builders (restored from French version)
# ========================
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _qkey(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", _norm(s).lower())

def get_last_n_lines(n: int) -> List[str]:
    """Returns the last N numbered lines with their actual line numbers."""
    if n <= 0 or not transcript_lines: return []
    start_idx = max(0, len(transcript_lines) - n)
    return [f"{i+1}. {transcript_lines[i]}" for i in range(start_idx, len(transcript_lines))]

def get_head_tail_lines(head: int, tail: int) -> List[str]:
    """Returns a numbered 'head + ... + tail' to preserve global context without sending everything."""
    total = len(transcript_lines)
    if total == 0: return []
    head = max(0, min(head, total))
    tail = max(0, min(tail, total - head))
    parts = []
    for i in range(head):
        parts.append(f"{i+1}. {transcript_lines[i]}")
    if head + tail < total:
        parts.append("...")
    start_tail = max(head, total - tail)
    for i in range(start_tail, total):
        parts.append(f"{i+1}. {transcript_lines[i]}")
    return parts

def build_llm_context_text() -> str:
    """Constructs the context to send to the LLM based on LLM_CONTEXT_MODE."""
    mode = (LLM_CONTEXT_MODE or "window").lower()
    if mode == "full":
        lines = [f"{i+1}. {line}" for i, line in enumerate(transcript_lines)]
    elif mode == "headtail":
        lines = get_head_tail_lines(LLM_HEAD_LINES, LLM_TAIL_LINES)
    else:  # "window"
        lines = get_last_n_lines(LLM_WINDOW_LINES)
    
    txt = "\n".join(lines)
    if LLM_MAX_CONTEXT_CHARS and len(txt) > LLM_MAX_CONTEXT_CHARS:
        txt = txt[-LLM_MAX_CONTEXT_CHARS:]
    return txt

def verify_license(key):
    """Verify key with Gumroad"""
    try:
        resp = requests.post(
            "https://api.gumroad.com/v2/licenses/verify",
            data={
                "product_id": "ADySSUI1rySCo72YL8L4hA==",
                "license_key": key,
            },
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json().get("success", False)
    except:
        # If can't reach Gumroad, allow (offline mode)
        return True
    return False

def check_license():
    """Check if valid license exists"""
    if LICENSE_FILE.exists():
        key = LICENSE_FILE.read_text().strip()
        if verify_license(key):
            print("[License] Valid")
            return True
    
    # No valid license
    print("\n" + "="*60)
    print("LICENSE REQUIRED")
    print("Get license: https://mucciaccio.gumroad.com/l/himlkf")
    print("="*60)
    
    key = input("\nEnter license key (or 'trial' for 7-day trial): ").strip()
    
    if key.lower() == "trial":
        # Save trial start date
        trial_file = Path.home() / ".interview_copilot_trial"
        if not trial_file.exists():
            import datetime
            trial_file.write_text(str(datetime.date.today()))
            print("\n7-day trial started!")
            return True
        else:
            import datetime
            start = datetime.date.fromisoformat(trial_file.read_text())
            days = (datetime.date.today() - start).days
            if days <= 7:
                print(f"\nTrial active ({7-days} days left)")
                return True
            else:
                print("\nTrial expired. Purchase license to continue.")
                return False
    
    if verify_license(key):
        LICENSE_FILE.write_text(key)
        print("\nLicense activated!")
        return True
    else:
        print("\nInvalid license key")
        return False


# ========================
# Whisper Loading with Resilient, Multi-OS Cache
# ========================
def hf_cache_dir() -> str:
    """Returns a platform-agnostic cache directory inside the user's home."""
    cache_path = Path.home() / ".cache" / "hf_models"
    cache_path.mkdir(parents=True, exist_ok=True)
    return str(cache_path)

def load_whisper_model() -> WhisperModel:
    cache_root = hf_cache_dir()
    try:
        print(f"[model] Loading {MODEL_NAME} in {cache_root}")
        return WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE, download_root=cache_root)
    except Exception:
        model_folder_name = f"models--Systran--faster-whisper-{MODEL_NAME}"
        corrupted_path = os.path.join(cache_root, model_folder_name)
        if os.path.isdir(corrupted_path):
            print(f"[model] Corrupted cache detected, purging: {corrupted_path}")
            shutil.rmtree(corrupted_path, ignore_errors=True)
        return WhisperModel(MODEL_NAME, compute_type=COMPUTE_TYPE, download_root=cache_root)

# ========================
# LLM Response Sanitization (restored from French version)
# ========================
COACHING_PATTERNS = [
    r"\bHappy to elaborate if useful\b.*", r"\bLet me know if you'd like more details\b.*",
    r"\bYou (?:should|could|can)\b.*", r"\bYou demonstrate\b.*",
    r"\bOne (?:improvement|area to improve)\b.*", r"\bYour answer\b.*",
]

def sanitize_candidate_voice(text: str) -> str:
    for pat in COACHING_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou\b", "the interviewer", text, flags=re.IGNORECASE)
    text = re.sub(r"\byour\b", "the", text, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", text).strip(" .")

# ========================
# LLM Analyzer (more robust)
# ========================
@dataclass
class ConversationSegment:
    text: str; timestamp: float

@dataclass
class QuestionCandidate:
    question: str; context: str; confidence: float; urgency: str; topic_area: str; timestamp: float
    should_answer: bool = False

class ImprovedLLMAnalyzer:
    def __init__(self, api_key: str):
        self.enabled = LLM_ENABLED and bool(api_key.startswith("sk-"))
        self.client = None
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
                print(f"[llm] Init error: {e}"); self.enabled = False
        else:
            print("[llm] Disabled (no valid API key)")

    def _trim_seen(self):
        now = time.time()
        expired = [k for k, t in self.seen_questions.items() if now - t > SEEN_TTL_SEC]
        for k in expired: self.seen_questions.pop(k, None)

    async def analyze_segment(self, segment: ConversationSegment) -> List[QuestionCandidate]:
        if not self.enabled: return []
        self._trim_seen()
        return await self._detect_questions(segment)

    async def _detect_questions(self, segment: ConversationSegment) -> List[QuestionCandidate]:
        if time.time() - self.last_llm_emit < LLM_MIN_GAP_SEC: return []
        qs = self._extract_questions_aggressive(segment.text)
        if not qs: return []
        if DEBUG: print(f"[llm] Found {len(qs)} potential questions: {[q[:40] + '...' for q in qs]}")

        candidates: List[QuestionCandidate] = []
        for q in qs:
            k = _qkey(q)
            if not k or k in self.seen_questions: continue
            
            decide = await asyncio.get_running_loop().run_in_executor(None, self._should_answer, q)
            if decide.get("should_answer"):
                self.seen_questions[k] = time.time()
                cand = QuestionCandidate(
                    question=q, context=segment.text, confidence=decide.get("confidence", 0.7),
                    urgency="relevant", topic_area="interview", timestamp=segment.timestamp, should_answer=True,
                )
                candidates.append(cand)
                if DEBUG: print(f"[llm] Will answer: {q}")
        return candidates[:3]

    def _extract_questions_aggressive(self, text: str) -> List[str]:
        """Advanced 6-stage question extraction (restored from French version)."""
        text = _norm(text)
        if not text: return []
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]
        questions: List[str] = []
        for s in sentences:
            if s.endswith("?") and len(s.split()) >= 5: questions.append(s)
        
        head_terms = ["what","how","why","when","where","which","who","can you","could you","tell me","explain"]
        for s in sentences:
            sl = s.lower()
            if any(sl.startswith(term + " ") for term in head_terms) and len(s.split()) >= 6:
                questions.append(s if s.endswith("?") else s + "?")

        seen = set(); clean: List[str] = []
        for q in questions:
            k = _qkey(q)
            if k and k not in seen:
                seen.add(k)
                clean.append(q.strip(" ,;:-–—"))
        return clean[:3]

    def _should_answer(self, question: str) -> Dict[str, Any]:
        if not self.enabled: return {"should_answer": False}
        now = time.time()
        while self.answers_timestamps and now - self.answers_timestamps[0] > 60:
            self.answers_timestamps.popleft()
        if len(self.answers_timestamps) >= ANSWERS_PER_MIN:
            return {"should_answer": False}

        ql = question.lower().strip()
        wh = ("what", "how", "why", "when", "where", "which", "who", "can", "could", "do", "is", "are")
        looks_like_q = ql.endswith("?") or any(ql.startswith(w + " ") for w in wh)

        try:
            resp = self.client.responses.create(
                model=GPT_MODEL, reasoning={"effort": GPT_EFFORT},
                input=[
                    {"role": "developer", "content": "Is this an interview question needing an answer? Reply ONLY 'YES' or 'NO'."},
                    {"role": "user", "content": f"Context:\n{build_llm_context_text()[-1000:]}\n\nQuestion: {question}"},
                ],
            )
            out = (getattr(resp, "output_text", "") or "").strip().lower()
            decided_yes = "yes" in out and "no" not in out
        except Exception as e:
            if DEBUG: print(f"[llm] Decision error: {e}")
            decided_yes = looks_like_q

        if decided_yes:
            ts = time.time(); self.last_llm_emit = ts; self.answers_timestamps.append(ts)
        return {"should_answer": decided_yes, "confidence": 0.75 if decided_yes else 0.3}

    async def generate_answer(self, candidate: QuestionCandidate) -> str:
        if not self.enabled: return "[LLM disabled]"
        async with self.sem:
            return await asyncio.get_running_loop().run_in_executor(None, self._gen, candidate)

    def _gen(self, candidate: QuestionCandidate) -> str:
        """Advanced answer generation with persona, context, and retry (restored from French version)."""
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
                if full_tx: inputs.append({"role": "user", "content": f"Full Interview Transcript (for context):\n{full_tx}"})
            
            if TECH_INTERVIEW_MODE:
                inputs.append({"role": "user", "content": "Domain context: general software engineering, data science, cloud infrastructure."})

            inputs.extend([
                {"role": "user", "content": f"Question: {q}"},
                {"role": "user", "content": "Answer now as a final spoken response. Avoid addressing the interviewer as 'you'."},
            ])

            resp = self.client.responses.create(model=GPT_MODEL, reasoning={"effort": GPT_EFFORT}, input=inputs, max_output_tokens=MAX_OUTTOK)
            raw_text = (getattr(resp, "output_text", "") or "").strip()
            cleaned = sanitize_candidate_voice(raw_text)

            if not cleaned.strip(): # Retry logic
                if DEBUG: print("[llm] Response was empty, retrying with simpler prompt...")
                retry_inputs = [
                    {"role": "developer", "content": "Answer the following question as a tech job candidate in the first person."},
                    {"role": "user", "content": f"Question: {q}"}
                ]
                resp2 = self.client.responses.create(model=GPT_MODEL, input=retry_inputs, max_output_tokens=MAX_OUTTOK)
                raw2 = (getattr(resp2, "output_text", "") or "").strip()
                cleaned = sanitize_candidate_voice(raw2)

            return cleaned if cleaned.strip() else "I would evaluate the model on a held-out test set to measure its generalization performance."
        except Exception as e:
            return f"[Answer error] {str(e)[:120]}"

# ========================
# Global Server State
# ========================
@dataclass
class ClientState:
    ws: websockets.WebSocketServerProtocol; queue: asyncio.Queue
    sender_task: Optional[asyncio.Task] = None; wants_broadcast: bool = True; role: str = "ui"

clients: Dict[websockets.WebSocketServerProtocol, ClientState] = {}
latest_text, sentence_buf = "", ""
detected = deque(maxlen=500); qa_log = deque(maxlen=200)
transcript_lines: List[str] = []
MAX_TRANSCRIPT_LINES = int(os.getenv("STT_MAX_TRANSCRIPT_LINES", "5000"))
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.?!])\s+')
pcm_buf = bytearray(); bytes_since_last = 0
WIN_BYTES = int(SAMPLE_RATE * 2 * WINDOW_SECONDS); HOP_BYTES = int(SAMPLE_RATE * 2 * HOP_SECONDS)
MAX_BUFFER_BYTES = WIN_BYTES * 2
audio_lock = asyncio.Lock(); server_shutdown = asyncio.Event(); transcriber_task: Optional[asyncio.Task] = None

# ========================
# Broadcast & WebSocket Handling
# ========================
async def _client_sender(client: ClientState):
    try:
        while True: await client.ws.send(await client.queue.get())
    except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError): pass
    finally:
        try: await client.ws.close()
        except Exception: pass

async def broadcast(data: dict):
    if not clients: return
    msg = json.dumps(data, ensure_ascii=False)
    for c in list(clients.values()):
        if c.wants_broadcast:
            try:
                if c.queue.full(): _ = c.queue.get_nowait()
                c.queue.put_nowait(msg)
            except Exception: clients.pop(c.ws, None)

async def handler(ws: websockets.WebSocketServerProtocol):
    global transcriber_task, bytes_since_last
    print(f"[ws] Client connected from {ws.remote_address}")
    client = ClientState(ws=ws, queue=asyncio.Queue(maxsize=100))
    client.sender_task = asyncio.create_task(_client_sender(client))
    clients[ws] = client

    try:
        await ws.send(json.dumps({"snapshot": {"transcript": latest_text, "detected": list(detected), "lines": transcript_lines[-200:]}}))
    except Exception: pass

    if transcriber_task is None or transcriber_task.done():
        transcriber_task = asyncio.create_task(read_and_transcribe_loop())

    try:
        async for msg in ws:
            if isinstance(msg, (bytes, bytearray)):
                async with audio_lock:
                    pcm_buf.extend(msg); bytes_since_last += len(msg)
                    if len(pcm_buf) > MAX_BUFFER_BYTES: del pcm_buf[:len(pcm_buf) - MAX_BUFFER_BYTES // 2]
            else:
                try:
                    data = json.loads(msg)
                    cmd = (data.get("cmd") or "").lower()
                    if cmd == "hello" and (data.get("client") or "").lower() in ("audio_streamer", "ingest"):
                        client.wants_broadcast = False
                        if DEBUG: print(f"[ws] Marked {ws.remote_address} as streamer; broadcasts OFF")
                    elif cmd == "reset":
                        async with audio_lock: pcm_buf.clear(); bytes_since_last = 0
                        reset_state(); print("[ws] Reset completed")
                except Exception: pass
    except (websockets.ConnectionClosedOK, websockets.ConnectionClosedError): pass
    finally:
        clients.pop(ws, None)
        if client.sender_task: client.sender_task.cancel()
        print("[ws] Client disconnected")

# ========================
# Global Transcriber Loop
# ========================
async def read_and_transcribe_loop():
    global bytes_since_last, latest_text, sentence_buf, transcript_lines
    loop, last_hashes, process_count = asyncio.get_running_loop(), deque(maxlen=8), 0
    print("[stt] Transcription loop started (global)")
    PROMPT_WORDS = {w.strip(".,").lower() for w in INITIAL_PROMPT.split()} if INITIAL_PROMPT else set()

    while not server_shutdown.is_set():
        await asyncio.sleep(0.05)
        async with audio_lock:
            if not (bytes_since_last >= HOP_BYTES and len(pcm_buf) >= WIN_BYTES): continue
            bytes_since_last = 0; window_bytes = bytes(pcm_buf[-WIN_BYTES:])
        
        arr = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if np.mean(arr*arr) < ENERGY_GATE: continue
        
        def _transcribe():
            try:
                segments, _ = whisper.transcribe(arr, language=FORCE_LANG, vad_filter=True, initial_prompt=INITIAL_PROMPT if not latest_text else None)
                return list(segments)
            except Exception as e:
                print(f"[stt] Transcription error: {e}"); return []

        segments = await loop.run_in_executor(None, _transcribe)
        if not segments: continue

        for segment in segments:
            text = _norm(segment.text)
            if not text or len(text) < 2: continue
            
            # Advanced filtering (restored from French version)
            toks = [w.strip(".,").lower() for w in text.split()]
            if toks and PROMPT_WORDS and sum(1 for w in toks if w in PROMPT_WORDS) / max(1, len(toks)) > 0.6:
                if DEBUG: print("[stt] Dropped segment: initial_prompt echo")
                continue
            if re.search(r"\b(\w+(?:,\s*\w+){1,})\b(?:\s+\1\b){2,}", text, flags=re.I):
                if DEBUG: print("[stt] Dropped segment: repetitive phrase")
                continue

            h = hashlib.md5(text.encode("utf-8")).hexdigest()
            if h in last_hashes: continue
            last_hashes.append(h)

            sentence_buf = (sentence_buf + " " + text).strip()
            parts = SENTENCE_SPLIT_RE.split(sentence_buf)
            
            complete = parts[:-1] if len(parts) > 1 and sentence_buf[-1] not in ".?!" else (parts if sentence_buf and sentence_buf[-1] in ".?!" else [])
            sentence_buf = "" if complete == parts else parts[-1]
            latest_text = sentence_buf

            for s in complete:
                s = _norm(s)
                if not s: continue
                transcript_lines.append(s)
                if len(transcript_lines) > MAX_TRANSCRIPT_LINES: del transcript_lines[:MAX_TRANSCRIPT_LINES // 10]
                if DEBUG: print(f"[stt] Line: {s}")
                await broadcast({"line": {"n": len(transcript_lines), "text": s}})
            
            await broadcast({"partial": text})

            if llm_analyzer and llm_analyzer.enabled:
                try:
                    cands = await llm_analyzer.analyze_segment(ConversationSegment(text=text, timestamp=time.time()))
                    for cand in cands:
                        item = {"q": cand.question, "t": int(cand.timestamp * 1000), "a": None}
                        detected.append(item)
                        await broadcast({"question_detected": item})
                        ans = await llm_analyzer.generate_answer(cand)
                        item["a"] = ans; qa_log.append(item)
                        await broadcast({"qa": item})
                        if DEBUG: print(f"[llm] Response: {ans[:60]}...")
                except Exception as e:
                    print(f"[llm] Analysis error: {e}")

# ========================
# Utilities & Main
# ========================
def reset_state():
    global latest_text, sentence_buf, transcript_lines
    latest_text, sentence_buf = "", ""; transcript_lines.clear(); detected.clear(); qa_log.clear()
    if llm_analyzer:
        llm_analyzer.seen_questions.clear(); llm_analyzer.answers_timestamps.clear()

async def main():
    if not check_license():
        return
    global whisper, llm_analyzer

    print("🚀 OPTIMIZED STT SERVER V4 (Best of Both Worlds)")
    print("=" * 50)
    print("[startup] Loading Whisper model..."); whisper = load_whisper_model()
    print("[startup] Initializing LLM Analyzer..."); llm_analyzer = ImprovedLLMAnalyzer(OPENAI_API_KEY)
    print(f"\n📋 CONFIG: LLM {'Enabled' if llm_analyzer.enabled else 'Disabled'}, Debug {'ON' if DEBUG else 'OFF'}")

    try:
        async with websockets.serve(handler, HOST, PORT, max_size=2**20, ping_interval=10, ping_timeout=30):
            print(f"\n🎤 Server ready on ws://{HOST}:{PORT}/"); await asyncio.Future()
    except Exception as e:
        print(f"[error] Server failed to start: {e}")

if __name__ == "__main__":
    try:
        whisper: WhisperModel; llm_analyzer: Optional[ImprovedLLMAnalyzer]
        asyncio.run(main())
    except KeyboardInterrupt:
        server_shutdown.set(); print("\n[shutdown] Server stopped.")
    except Exception as e:
        print(f"\n[fatal_error] {e}"); import traceback; traceback.print_exc()
