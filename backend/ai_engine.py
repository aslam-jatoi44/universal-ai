import os
import uuid
import time
import base64
import asyncio
from typing import List, Dict, Any, Optional, Generator
from pathlib import Path
from dotenv import load_dotenv

# Try importing google.genai
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    types = None
    genai = None

# Try importing edge-tts
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    edge_tts = None

# Load .env from backend directory or project root
current_dir = Path(__file__).parent
env_paths = [current_dir / ".env", current_dir.parent / ".env"]
for env_file in env_paths:
    if env_file.exists():
        load_dotenv(dotenv_path=env_file)
        break

DEFAULT_SYSTEM_PROMPT = """
You are UNIVERSAL AI, an elite software engineering, multimodal intelligence, and precision execution core.
STRICT DIRECTIVES:
1. PERMANENT IDENTITY: Your name and identity is strictly UNIVERSAL AI. Never refer to yourself as Gemini, Google AI, or any other assistant. If asked who you are, state ONLY that you are UNIVERSAL AI.
2. ZERO FLUFF: Analyze the query and deliver ONLY the exact, actionable answer immediately. Never use greetings, pleasantries, or filler introductions unless requested.
3. MULTILINGUAL PRECISION: Default system interface language is English, but automatically detect and respond in the exact language the user writes or speaks (Sindhi, Urdu, English, etc.) while keeping technical code in English.
"""

# آڊيو فائلون محفوظ ڪرڻ لاءِ عارضي فولڊر
AUDIO_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

# ٻولي جي حساب سان بهترين پروفيشنل آواز چونڊڻ
VOICE_MAP = {
    "en": "en-US-AndrewNeural",      # پروفيشنل انگلش مرداڻو آواز
    "en-f": "en-US-AvaNeural",       # پروفيشنل انگلش زنانو آواز
    "ur": "ur-PK-AsadNeural",        # پاڪستاني اردو مرداڻو آواز
    "ur-f": "ur-PK-UzmaNeural",      # پاڪستاني اردو زنانو آواز
    "ar": "ar-SA-HamedNeural",       # عربي آواز (سنڌي اچار جي ويجهو)
}

def detect_voice(text: str) -> str:
    """جيڪڏهن ٽيڪسٽ ۾ عربي/سنڌي/اردو اکر آهن ته مقامي لهجي وارو آواز چونڊيو"""
    if any("\u0600" <= ch <= "\u06FF" for ch in text):
        # خاص سنڌي اکر چيڪ ڪرڻ (ڀ، ٺ، ڳ، ڄ، ڏ، ڙ، ڻ، ٻ)
        sindhi_chars = set("ڀٺڳڄڏڙڻڪڱڦڇڃٻ")
        if any(ch in sindhi_chars for ch in text):
            return VOICE_MAP["ur"]  # AsadNeural سنڌي ۽ اردو ٻئي صاف پڙهي ٿو
        return VOICE_MAP["ur"]
    return VOICE_MAP["en"]

def normalize_sindhi_for_tts(text: str) -> str:
    """سنڌي اکرن ۽ وقفن کي صوتي (Phonetic) طور صاف بڻائي ٿو ته جيئن TTS بلڪل درست اچار پڙهي"""
    phonetic_map = {
        "ڏ": "ڊ",
        "ڳ": "گ",
        "ڄ": "ج",
        "ٻ": "ب",
        "ڃ": "نج",
        "ڱ": "نگ",
        "ڻ": "ن",
        "ڪ": "ک",
        "ڦ": "پھ",
        "ٺ": "ٹھ",
        "ڀ": "بھ",
        "ڇ": "چھ",
        "۔": ". ",
        "،": ", ",
        "؟": "? "
    }
    for sindhi_char, phonetic_char in phonetic_map.items():
        text = text.replace(sindhi_char, phonetic_char)
    return text

async def text_to_speech_file(text: str, voice_override: str = None) -> Optional[str]:
    """ٽيڪسٽ کي MP3 آڊيو فائل ۾ تبديل ڪري ٿو"""
    # Markdown جا نشان (*, #, `) صاف ڪرڻ ته جيئن آواز قدرتي رهي
    clean_text = text.replace("*", "").replace("#", "").replace("`", "").strip()
    if not clean_text:
        return None

    primary_voice = voice_override if voice_override else detect_voice(clean_text)

    # Apply Sindhi phonetic normalization when Arabic/Sindhi/Urdu script or Urdu voice is used
    tts_text = (
        normalize_sindhi_for_tts(clean_text)
        if (any("\u0600" <= ch <= "\u06FF" for ch in clean_text) or primary_voice.startswith("ur-"))
        else clean_text
    )
    
    # Fallback voice candidates if primary drops connection
    candidate_voices = [primary_voice]
    if primary_voice.startswith("en-"):
        for alt in ["en-US-GuyNeural", "en-US-JennyNeural", "en-US-AriaNeural"]:
            if alt not in candidate_voices:
                candidate_voices.append(alt)
    elif primary_voice.startswith("ur-"):
        for alt in ["ur-PK-UzmaNeural", "ur-PK-AsadNeural"]:
            if alt not in candidate_voices:
                candidate_voices.append(alt)

    filename = f"{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    last_error = None
    for voice in candidate_voices:
        for attempt in range(2):
            try:
                communicate = edge_tts.Communicate(tts_text, voice)
                await communicate.save(filepath)
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    return filename
            except Exception as e:
                last_error = e
                await asyncio.sleep(0.5)

    if last_error:
        raise last_error
    return None

def transcribe_audio_gemini(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """Gemini Multimodal ذريعي سنڌي، اردو، ۽ انگريزي آواز کي بلڪل درست لکت ۾ بدلائي ٿو"""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Error: GEMINI_API_KEY missing."

    client = genai.Client(api_key=api_key)
    prompt = (
        "Transcribe this audio accurately into text. "
        "Detect the spoken language automatically (English, Sindhi, or Urdu). "
        "Output ONLY the exact spoken words without any commentary or translation."
    )
    
    # Ordered failover chain prioritizing Flash models that remain resilient during peak 503 demand
    candidate_models = [
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
    ]
    last_error = None
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                ],
            )
            return response.text.strip()
        except Exception as e:
            last_error = e
            err_upper = str(e).upper()
            if any(k in err_upper for k in ("API_KEY_INVALID", "PERMISSION_DENIED", "403")):
                break
            continue

    return f"Transcription error: {str(last_error)}"


class AIEngine:
    """
    UNIVERSAL AI — Precision Studio Core Engine
    High-performance orchestration engine using Google GenAI SDK.
    Supports Multi-modal analysis, Precision Prompt Optimization,
    Code Architecture, Reasoning with Thinking models, Live Streaming,
    Edge-TTS Multilingual Voice (Sindhi, Urdu, English), and Gemini Audio Transcription.
    """

    FALLBACK_MODELS = [
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-3.1-flash-lite",
    ]

    SUPPORTED_MODELS = [
        {
            "id": "gemini-3.5-flash",
            "name": "Gemini 3.5 Flash",
            "tag": "Primary / Ultra-Fast",
            "description": "High-throughput multimodal model with automatic failover to healthy Flash nodes.",
            "supports_vision": True,
            "supports_thinking": True,
            "max_tokens": 8192,
        },
        {
            "id": "gemini-3-flash-preview",
            "name": "Gemini 3 Flash Preview",
            "tag": "High Availability",
            "description": "Fast preview multimodal model ideal during peak demand spikes.",
            "supports_vision": True,
            "supports_thinking": True,
            "max_tokens": 8192,
        },
        {
            "id": "gemini-3.5-flash-lite",
            "name": "Gemini 3.5 Flash Lite",
            "tag": "Low Latency / Resilient",
            "description": "Ultra-low latency Flash Lite model optimized for multilingual speed and reliability.",
            "supports_vision": True,
            "supports_thinking": False,
            "max_tokens": 8192,
        },
        {
            "id": "gemini-flash-lite-latest",
            "name": "Gemini Flash Lite Latest",
            "tag": "Always Latest Lite",
            "description": "Auto-routed to Google's latest stable Flash Lite generation.",
            "supports_vision": True,
            "supports_thinking": False,
            "max_tokens": 8192,
        },
        {
            "id": "gemini-flash-latest",
            "name": "Gemini Flash Latest",
            "tag": "Always Latest Flash",
            "description": "Auto-routed to Google's latest stable Flash generation.",
            "supports_vision": True,
            "supports_thinking": True,
            "max_tokens": 8192,
        },
        {
            "id": "gemini-3.1-pro-preview",
            "name": "Gemini 3.1 Pro",
            "tag": "Deep Precision & Code",
            "description": "Top-tier flagship reasoning model for complex architectural blueprints (auto-routes if quota limited).",
            "supports_vision": True,
            "supports_thinking": True,
            "max_tokens": 8192,
        },
    ]

    POPULAR_VOICES = [
        {"id": "auto", "name": "Auto-Detect (English / Sindhi / Urdu)", "gender": "Smart", "locale": "auto"},
        {"id": "en-US-AndrewNeural", "name": "Andrew (Professional US Male)", "gender": "Male", "locale": "en-US"},
        {"id": "en-US-AvaNeural", "name": "Ava (Professional US Female)", "gender": "Female", "locale": "en-US"},
        {"id": "ur-PK-AsadNeural", "name": "Asad (Pakistani Urdu & Sindhi Male)", "gender": "Male", "locale": "ur-PK"},
        {"id": "ur-PK-UzmaNeural", "name": "Uzma (Pakistani Urdu Female)", "gender": "Female", "locale": "ur-PK"},
        {"id": "ar-SA-HamedNeural", "name": "Hamed (Arabic / Classical)", "gender": "Male", "locale": "ar-SA"},
    ]

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.client = None
        self._init_client()

    def _init_client(self):
        """Initializes the google.genai Client if an API key is available."""
        if GENAI_AVAILABLE and self.api_key and self.api_key.strip():
            try:
                http_opts = types.HttpOptions(timeout=60000) if types else None
                self.client = genai.Client(api_key=self.api_key.strip(), http_options=http_opts)
            except Exception as e:
                print(f"[AIEngine] Failed to initialize genai Client: {e}")
                self.client = None
        else:
            self.client = None

    def set_api_key(self, api_key: str, save_to_env: bool = True) -> Dict[str, Any]:
        """Sets a new API key dynamically and optionally persists it to .env."""
        cleaned_key = api_key.strip()
        self.api_key = cleaned_key
        os.environ["GEMINI_API_KEY"] = cleaned_key
        self._init_client()

        saved = False
        if save_to_env and cleaned_key:
            try:
                env_file = current_dir / ".env"
                existing_lines = []
                if env_file.exists():
                    existing_lines = env_file.read_text(encoding="utf-8").splitlines()
                
                new_lines = [line for line in existing_lines if not line.startswith("GEMINI_API_KEY=")]
                new_lines.append(f"GEMINI_API_KEY={cleaned_key}")
                env_file.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
                saved = True
            except Exception as e:
                print(f"[AIEngine] Could not save to .env: {e}")

        is_valid = self.client is not None
        return {
            "status": "success" if is_valid else "warning",
            "configured": bool(cleaned_key),
            "client_ready": is_valid,
            "saved_to_env": saved,
            "masked_key": f"{cleaned_key[:4]}...{cleaned_key[-4:]}" if len(cleaned_key) > 8 else "***"
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns engine operational telemetry, TTS readiness, and configuration state."""
        has_key = bool(self.api_key and self.api_key.strip())
        return {
            "system": "UNIVERSAL AI — Precision Studio Core",
            "phase": "Phase 1: Core Engine",
            "sdk_installed": GENAI_AVAILABLE,
            "tts_installed": EDGE_TTS_AVAILABLE,
            "audio_transcription_ready": has_key,
            "api_key_configured": has_key,
            "client_initialized": self.client is not None,
            "mode": "Live Production" if (self.client is not None) else "Precision Simulation & Evaluation Mode",
            "default_model": "gemini-3.5-flash",
            "default_voice": "Auto-Detect (English, Sindhi, Urdu)",
            "total_supported_models": len(self.SUPPORTED_MODELS),
            "masked_key": f"{self.api_key[:4]}...{self.api_key[-4:]}" if has_key and len(self.api_key) > 8 else None,
        }

    def get_models(self) -> List[Dict[str, Any]]:
        """Returns the list of supported precision studio models."""
        return self.SUPPORTED_MODELS

    def get_voices(self) -> List[Dict[str, Any]]:
        """Returns curated neural voices for TTS synthesis."""
        return self.POPULAR_VOICES

    def _build_config(
        self,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        max_output_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
        response_mime_type: Optional[str] = None,
    ) -> Any:
        """Constructs types.GenerateContentConfig for google.genai."""
        if not GENAI_AVAILABLE:
            return None

        kwargs = {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_output_tokens": int(max_output_tokens),
            "http_options": types.HttpOptions(timeout=60000)
        }
        if top_k is not None:
            kwargs["top_k"] = int(top_k)

        # Enforce UNIVERSAL AI permanent core identity and precision directives globally
        if system_instruction and system_instruction.strip():
            kwargs["system_instruction"] = (
                f"{DEFAULT_SYSTEM_PROMPT.strip()}\n\n"
                f"ADDITIONAL DIRECTIVES:\n{system_instruction.strip()}"
            )
        else:
            kwargs["system_instruction"] = DEFAULT_SYSTEM_PROMPT.strip()

        if response_mime_type:
            kwargs["response_mime_type"] = response_mime_type
        if thinking_budget is not None and thinking_budget > 0:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=int(thinking_budget),
                include_thoughts=True
            )

        return types.GenerateContentConfig(**kwargs)

    def _get_candidate_models(self, preferred_model: str) -> List[str]:
        """Builds an ordered, deduplicated list of models with healthy failover targets."""
        candidates = []
        if preferred_model:
            candidates.append(preferred_model)
        for fallback in self.FALLBACK_MODELS:
            if fallback not in candidates:
                candidates.append(fallback)
        return candidates

    @staticmethod
    def _is_retryable_error(error_str: str) -> bool:
        """Identifies transient availability, quota, or timeout errors safe for failover."""
        upper = error_str.upper()
        if any(auth in upper for auth in ("API_KEY_INVALID", "PERMISSION_DENIED", "UNREGISTERED", "403")):
            return False
        retry_markers = (
            "503", "UNAVAILABLE", "HIGH DEMAND", "OVERLOADED",
            "504", "DEADLINE_EXCEEDED", "DEADLINEEXCEEDED", "DEADLINE EXPIRED",
            "404", "NOT_FOUND",
            "429", "RESOURCE_EXHAUSTED", "QUOTA",
            "TIMED OUT", "TIMEOUT",
            "500", "INTERNAL"
        )
        return any(marker in upper for marker in retry_markers)

    @staticmethod
    def _strip_thinking_config(config: Any) -> Any:
        """Returns a copy of GenerateContentConfig without thinking_config for lite fallback models."""
        if config is None or not getattr(config, "thinking_config", None):
            return config
        return config.model_copy(update={"thinking_config": None})

    def _generate_with_failover(
        self,
        preferred_model: str,
        contents: Any,
        config: Any = None,
    ) -> tuple[Any, str]:
        """Executes generate_content with automatic failover across healthy Gemini models."""
        candidates = self._get_candidate_models(preferred_model)
        last_error = None

        for idx, m in enumerate(candidates):
            is_last = idx == len(candidates) - 1
            active_config = config
            if active_config is not None and types:
                attempt_timeout = 60000 if is_last else 20000
                active_config = active_config.model_copy(
                    update={"http_options": types.HttpOptions(timeout=attempt_timeout)}
                )
            for _attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=m,
                        contents=contents,
                        config=active_config,
                    )
                    return response, m
                except Exception as inner_e:
                    last_error = inner_e
                    err_str = str(inner_e)
                    # If a lite fallback rejects thinking_config, strip it and retry same model once
                    if ("400" in err_str or "INVALID_ARGUMENT" in err_str.upper()) and getattr(active_config, "thinking_config", None):
                        active_config = self._strip_thinking_config(active_config)
                        continue
                    break

            if last_error and not self._is_retryable_error(str(last_error)):
                raise last_error

        if last_error:
            raise last_error
        raise RuntimeError("No candidate models available for generation.")

    def generate(
        self,
        prompt: str,
        model: str = "gemini-3.5-flash",
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        max_output_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Executes precision content generation.
        Falls back to comprehensive simulation if API key is not configured.
        """
        start_time = time.time()

        if self.client is None:
            return self._simulate_chat_response(prompt, model, system_instruction, history)

        try:
            config = self._build_config(
                system_instruction=system_instruction,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=max_output_tokens,
                thinking_budget=thinking_budget,
            )

            contents = []
            if history:
                for item in history:
                    role = "user" if item.get("role") == "user" else "model"
                    contents.append(
                        types.Content(
                            role=role,
                            parts=[types.Part.from_text(text=item.get("content", ""))]
                        )
                    )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)]
                )
            )

            response, used_model = self._generate_with_failover(
                preferred_model=model,
                contents=contents,
                config=config,
            )

            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            response_text = response.text or ""
            estimated_tokens = len(response_text) // 4
            mode_label = (
                "Live Engine Execution"
                if used_model == model
                else f"Live Engine Execution (Auto-Routed → {used_model})"
            )

            return {
                "status": "success",
                "text": response_text,
                "model": used_model,
                "requested_model": model,
                "latency_ms": elapsed_ms,
                "tokens_estimated": estimated_tokens,
                "mode": mode_label,
            }
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            error_str = str(e)
            if "API_KEY_INVALID" in error_str or "403" in error_str or "unregistered" in error_str:
                return {
                    "status": "auth_error",
                    "text": f"⚠️ **API Authentication Error:** The configured Gemini API Key is invalid or expired.\n\n*Error details:* `{error_str}`\n\nPlease check or update your key in the **Engine Telemetry & Settings** panel.",
                    "model": model,
                    "latency_ms": elapsed_ms,
                    "error": error_str,
                    "mode": "Authentication Required"
                }
            return {
                "status": "error",
                "text": f"⚠️ **Execution Error:** `{error_str}`",
                "model": model,
                "latency_ms": elapsed_ms,
                "error": error_str,
                "mode": "Error Fallback"
            }

    def generate_stream(
        self,
        prompt: str,
        model: str = "gemini-3.5-flash",
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_output_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Streams generation chunks via Server-Sent Events or generators with model failover."""
        if self.client is None:
            simulated = self._simulate_chat_response(prompt, model, system_instruction)
            text = simulated["text"]
            words = text.split(" ")
            for i in range(0, len(words), 3):
                chunk = " ".join(words[i:i+3]) + " "
                time.sleep(0.04)
                yield chunk
            return

        config = self._build_config(
            system_instruction=system_instruction,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
        )

        last_error = None
        for m in self._get_candidate_models(model):
            active_config = config
            emitted_any = False
            try:
                stream = self.client.models.generate_content_stream(
                    model=m,
                    contents=prompt,
                    config=active_config,
                )
                for chunk in stream:
                    if chunk.text:
                        emitted_any = True
                        yield chunk.text
                return
            except Exception as e:
                last_error = e
                if emitted_any or not self._is_retryable_error(str(e)):
                    break
                continue

        if last_error:
            yield f"\n[Streaming Error: {str(last_error)}]"

    def analyze_vision(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        analysis_mode: str = "deep_inspection",
        model: str = "gemini-3.5-flash",
    ) -> Dict[str, Any]:
        """Performs precision multimodal vision analysis."""
        start_time = time.time()

        mode_directives = {
            "deep_inspection": "Perform an exhaustive, high-precision technical inspection of this image. Identify structural patterns, textures, anomalous elements, and semantic context.",
            "ui_to_code": "Analyze this user interface design. Deconstruct its layout, typography, color palette, components, spacing, and output responsive Tailwind CSS + HTML structure.",
            "ocr_extraction": "Extract all text, numbers, tabular data, and labels visible in this image with exact typographical accuracy and hierarchy.",
            "creative_critique": "Critique this visual asset with emphasis on composition, aesthetic harmony, color balance, visual weight, and lighting."
        }
        enhanced_prompt = f"Mode: {analysis_mode.upper()}\nDirectives: {mode_directives.get(analysis_mode, mode_directives['deep_inspection'])}\n\nTask: {prompt}"

        if self.client is None:
            return self._simulate_vision_response(analysis_mode, prompt, len(image_bytes))

        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
            config = self._build_config(temperature=0.4, max_output_tokens=4096)

            response, used_model = self._generate_with_failover(
                preferred_model=model,
                contents=[image_part, enhanced_prompt],
                config=config,
            )

            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "status": "success",
                "analysis": response.text or "",
                "mode": analysis_mode,
                "latency_ms": elapsed_ms,
                "model": used_model
            }
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "status": "error",
                "analysis": f"Vision analysis failed: {str(e)}",
                "latency_ms": elapsed_ms,
                "error": str(e)
            }

    def optimize_prompt(
        self,
        raw_prompt: str,
        objective: Optional[str] = None,
        target_model: str = "gemini-3.5-flash"
    ) -> Dict[str, Any]:
        """Transforms a raw or fuzzy prompt into a 5-pillar precision-engineered prompt."""
        start_time = time.time()
        
        system_instruction = (
            "You are the UNIVERSAL AI Precision Prompt Architect. Your objective is to transform raw, imprecise "
            "prompts into world-class enterprise system prompts. Structure your output with these distinct sections:\n"
            "1. Core Objective & Role Definition\n"
            "2. Tactical Execution Directives\n"
            "3. Negative Constraints & Boundary Conditions\n"
            "4. Structured Output Format Specification (JSON or Markdown)\n"
            "5. Few-Shot Demonstration / Calibration Anchor"
        )
        optimization_query = (
            f"Optimize the following user prompt for high-precision model execution:\n\n"
            f"RAW PROMPT: \"{raw_prompt}\"\n"
            f"PRIMARY OBJECTIVE / USE-CASE: \"{objective or 'General High-Precision Execution'}\"\n"
            f"TARGET ENGINE: \"{target_model}\"\n\n"
            f"Deliver the perfected prompt ready for immediate deployment."
        )

        if self.client is None:
            return self._simulate_prompt_optimization(raw_prompt, objective)

        try:
            config = self._build_config(system_instruction=system_instruction, temperature=0.5)
            response, used_model = self._generate_with_failover(
                preferred_model=target_model,
                contents=optimization_query,
                config=config,
            )
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "status": "success",
                "raw_prompt": raw_prompt,
                "optimized_prompt": response.text or "",
                "latency_ms": elapsed_ms,
                "model": used_model
            }
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "status": "error",
                "optimized_prompt": f"Prompt optimization encountered an error: {str(e)}",
                "latency_ms": elapsed_ms,
                "error": str(e)
            }

    def architect_code(
        self,
        specification: str,
        language: str = "python",
        framework: Optional[str] = None,
        model: str = "gemini-3.1-pro-preview"
    ) -> Dict[str, Any]:
        """Generates enterprise-grade architectural blueprints and production code."""
        start_time = time.time()

        system_instruction = (
            "You are the UNIVERSAL AI Principal Software Architect. You deliver battle-tested, modular, "
            "production-grade software solutions with comprehensive error handling, type annotations, "
            "algorithmic complexity analysis, and unit test suites."
        )
        task_query = (
            f"Generate a complete, production-ready code blueprint for the following specification:\n\n"
            f"SPECIFICATION: {specification}\n"
            f"TARGET LANGUAGE: {language}\n"
            f"FRAMEWORK / STACK: {framework or 'Standard Idiomatic / Clean Architecture'}\n\n"
            f"Structure your response with:\n"
            f"1. Architectural Blueprint & Design Patterns\n"
            f"2. Complete Implementation (fully written, no placeholders)\n"
            f"3. Algorithmic Complexity (Time & Space)\n"
            f"4. Production Edge Cases & Error Strategies\n"
            f"5. Unit Tests (e.g. pytest, jest, or standard runner)"
        )

        if self.client is None:
            return self._simulate_code_architect(specification, language, framework)

        try:
            config = self._build_config(system_instruction=system_instruction, temperature=0.3)
            response, used_model = self._generate_with_failover(
                preferred_model=model,
                contents=task_query,
                config=config,
            )

            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "status": "success",
                "code_blueprint": response.text if response else "",
                "language": language,
                "latency_ms": elapsed_ms,
                "model": used_model
            }
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "status": "error",
                "code_blueprint": f"Code generation encountered an issue: {str(e)}",
                "latency_ms": elapsed_ms,
                "error": str(e)
            }

    def generate_live_ui(
        self,
        prompt: str,
        model_name: str = "gemini-3.5-flash"
    ) -> Dict[str, Any]:
        """Generates a complete, single-file responsive HTML5 + Tailwind + Lucide application."""
        start_time = time.time()

        ui_system_prompt = """
    You are UNIVERSAL AI's Principal Frontend UI/UX Architect.
    Generate a COMPLETE, single-file, responsive, production-grade HTML5 web application using Tailwind CSS (<script src="https://cdn.tailwindcss.com"></script>) and Lucide Icons (<script src="https://unpkg.com/lucide@latest"></script>).
    CRITICAL OUTPUT RULES:
    - Output ONLY raw valid HTML starting with <!DOCTYPE html> and ending with </html>.
    - Do NOT wrap the output in markdown fences (no ```html or ```).
    - Do NOT include any conversational text before or after the HTML.
    - Include rich interactive JavaScript (lucide.createIcons(), working tabs/modals/cart/tickers), modern typography, and realistic content.
    """

        if self.client is None:
            return self._simulate_live_ui(prompt, model_name)

        try:
            config = self._build_config(
                system_instruction=ui_system_prompt,
                temperature=0.35,
                max_output_tokens=8192,
            )
            response, used_model = self._generate_with_failover(
                preferred_model=model_name,
                contents=prompt,
                config=config,
            )
            raw_html = response.text or ""
            cleaned_html = raw_html.replace("```html", "").replace("```HTML", "").replace("```", "").strip()
            # Ensure output starts cleanly at <!DOCTYPE html> if stray text precedes it
            doc_idx = cleaned_html.lower().find("<!doctype html")
            if doc_idx > 0:
                cleaned_html = cleaned_html[doc_idx:]
            end_idx = cleaned_html.lower().rfind("</html>")
            if end_idx != -1:
                cleaned_html = cleaned_html[: end_idx + len("</html>")]

            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "status": "success",
                "html": cleaned_html,
                "model": used_model,
                "requested_model": model_name,
                "latency_ms": elapsed_ms,
                "mode": "Live UI Execution" if used_model == model_name else f"Live UI Execution (Auto-Routed → {used_model})"
            }
        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            return {
                "status": "error",
                "html": "",
                "model": model_name,
                "latency_ms": elapsed_ms,
                "error": str(e)
            }

    # =========================================================================
    # SIMULATION & DEMO ENGINE (Active when GEMINI_API_KEY is not yet supplied)
    # =========================================================================

    def _simulate_chat_response(
        self,
        prompt: str,
        model: str,
        system_instruction: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Provides rich precision evaluation response when GEMINI_API_KEY is pending."""
        sample_response = f"""### 🚀 UNIVERSAL AI — Precision Studio Core
> **Status:** Running in **Precision Core Evaluation Mode**  
> **Model Target:** `{model}` | **Engine Phase:** `Phase 1: Core Engine`

---

#### 💡 Architectural Telemetry & Analysis
You dispatched the instruction:
> *"{prompt.strip()}"*

**Precision Evaluation Breakdown:**
1. **Semantic Intent:** High-confidence directive detected.
2. **Context Routing:** Evaluated under system profile `{system_instruction or 'Standard Precision Engine'}`.
3. **Execution Latency:** ~18.4 ms simulated internal pipeline latency.

```json
{{
  "studio_core": "UNIVERSAL_AI_V1",
  "pipeline": "FastAPI + Google GenAI SDK + Edge-TTS (English / Sindhi / Urdu)",
  "status": "ONLINE_STANDBY",
  "instructions": "Enter your GEMINI_API_KEY in the Engine Telemetry panel to activate live generative inference."
}}
```

#### 🔑 Ready to Activate Live Gemini Generation?
1. Click the **API Key** button in the top navigation or go to **Engine Telemetry & Settings**.
2. Paste your Google AI Studio API key (`AIzaSy...`).
3. Click **Connect Engine** to switch instantaneously from Simulation to Live Generation!"""

        return {
            "status": "simulation",
            "text": sample_response,
            "model": model,
            "latency_ms": 18.4,
            "tokens_estimated": len(sample_response) // 4,
            "mode": "Precision Simulation (API Key Pending)"
        }

    def _simulate_vision_response(self, analysis_mode: str, prompt: str, byte_count: int) -> Dict[str, Any]:
        """Provides simulated vision analysis when API key is pending."""
        text = f"""### 👁️ UNIVERSAL AI — Multimodal Vision Lab
> **Mode:** `{analysis_mode.upper()}` | **Payload Size:** `{byte_count} bytes`

#### 🔬 Optical & Structural Breakdown (Evaluation Mode)
- **Resolution & Aspect Ratio:** Verified high-density visual payload.
- **Analyzed Feature Maps:** Contrast gradients, structural hierarchy, and color vectors calculated.
- **User Query:** *"{prompt}"*

```yaml
vision_telemetry:
  mode: "{analysis_mode}"
  detected_channels: "RGB"
  contrast_score: 0.94
  readiness: "Ready for live Gemini 3.5 Flash Multimodal analysis"
```

*To run live deep inspection on this visual asset, enter your `GEMINI_API_KEY` in the top bar.*"""
        return {
            "status": "simulation",
            "analysis": text,
            "mode": analysis_mode,
            "latency_ms": 24.1,
            "model": "gemini-3.5-flash (Simulated)"
        }

    def _simulate_prompt_optimization(self, raw_prompt: str, objective: Optional[str]) -> Dict[str, Any]:
        """Provides simulated prompt engineering template."""
        optimized = f"""### 🎯 Optimized Precision System Prompt Blueprint

```markdown
# ROLE & CORE OBJECTIVE
You are an expert AI execution agent specializing in high-precision reasoning and algorithmic clarity.
Your objective is to: {objective or raw_prompt}

# DIRECTIVES & EXECUTION RULES
1. Provide mathematically rigorous and deterministic logic where applicable.
2. Structure all explanations with explicit headings, concise rationale, and concrete examples.
3. Validate all inputs against edge conditions before presenting conclusions.

# NEGATIVE CONSTRAINTS
- Avoid hand-wavy generalizations or ambiguous filler text.
- Do not make unverified assumptions without stating hypotheses.

# STRUCTURED OUTPUT SCHEMA
Return your output using Markdown with:
- Executive Summary
- Step-by-Step Technical Blueprint
- Verification Matrix

# CALIBRATION PROMPT
"{raw_prompt}"
```

*(Generated by UNIVERSAL AI Precision Prompt Optimizer. Connect your GEMINI_API_KEY to customize live dynamically.)*"""
        return {
            "status": "simulation",
            "raw_prompt": raw_prompt,
            "optimized_prompt": optimized,
            "latency_ms": 14.2,
            "model": "gemini-3.5-flash"
        }

    def _simulate_code_architect(self, specification: str, language: str, framework: Optional[str]) -> Dict[str, Any]:
        """Provides simulated architecture blueprint."""
        blueprint = f"""### 🏗️ UNIVERSAL AI Architectural Blueprint

#### 1. System Topology & Pattern
- **Target Language:** `{language.upper()}`
- **Pattern:** Clean Architecture / Hexagonal Ports & Adapters
- **Framework:** `{framework or 'Idiomatic Standard'}`

#### 2. Core Implementation
```{language.lower()}
# UNIVERSAL AI - Precision Core Blueprint
# Spec: {specification}

import asyncio
from typing import Dict, Any, Optional

class CoreService:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._initialized: bool = True

    async def execute_precision_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        \"\"\"Executes deterministic workload with boundary checks.\"\"\"
        if not payload:
            raise ValueError("Payload cannot be empty")
        
        # Core execution logic
        result = {{
            "status": "COMPLETED",
            "metric_score": 0.998,
            "data": payload
        }}
        return result

if __name__ == "__main__":
    service = CoreService(config={{"env": "production"}})
    print("Service initialized and verified.")
```

#### 3. Complexity & Edge Cases
- **Time Complexity:** $\\mathcal{{O}}(1)$ dispatch, scalable linear processing.
- **Space Complexity:** $\\mathcal{{O}}(N)$ bounded memory buffers.
- **Edge Conditions:** Null payload checks, timeout boundaries, asynchronous non-blocking event loop.
"""
        return {
            "status": "simulation",
            "code_blueprint": blueprint,
            "language": language,
            "latency_ms": 28.6,
            "model": "gemini-3.1-pro-preview"
        }


    def _simulate_live_ui(self, prompt: str, model_name: str) -> Dict[str, Any]:
        """Provides a standalone Tailwind + Lucide HTML template when API key is pending."""
        safe_prompt = prompt.replace("<", "&lt;").replace(">", "&gt;")
        html = f"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>UNIVERSAL AI — Live UI Preview</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex items-center justify-center p-6">
  <div class="max-w-xl w-full bg-slate-900 border border-slate-800 rounded-2xl p-6 space-y-4 shadow-2xl">
    <div class="flex items-center gap-3">
      <i data-lucide="layout-dashboard" class="w-6 h-6 text-emerald-400"></i>
      <h1 class="text-lg font-bold">UNIVERSAL AI — Live UI Studio (Standby)</h1>
    </div>
    <p class="text-sm text-slate-300">Simulation preview for prompt:</p>
    <div class="bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono text-emerald-300">{safe_prompt}</div>
    <p class="text-xs text-slate-400">Connect a valid GEMINI_API_KEY in Telemetry & Audio Config to synthesize complete interactive applications.</p>
  </div>
  <script>lucide.createIcons();</script>
</body>
</html>"""
        return {
            "status": "simulation",
            "html": html,
            "model": model_name,
            "latency_ms": 12.0,
            "mode": "Precision Simulation (API Key Pending)"
        }


# Global Singleton Instance for easy import
ai_engine = AIEngine()


def generate_live_ui(prompt: str, model_name: str = "gemini-3.5-flash") -> str:
    """Generates raw HTML5 + Tailwind + Lucide output using UNIVERSAL AI with automatic failover."""
    result = ai_engine.generate_live_ui(prompt=prompt, model_name=model_name)
    if result.get("status") in ("success", "simulation"):
        return result.get("html", "")
    raise RuntimeError(result.get("error") or "Live UI generation failed")
