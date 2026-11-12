import io
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure backend directory is in path
current_dir = Path(__file__).parent.resolve()
frontend_dir = (current_dir.parent / "frontend").resolve()
audio_dir = (current_dir / "temp_audio").resolve()
audio_dir.mkdir(parents=True, exist_ok=True)

if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from ai_engine import (
    ai_engine,
    generate_live_ui,
    text_to_speech_file,
    transcribe_audio_gemini,
    detect_voice,
    AUDIO_DIR
)

app = FastAPI(
    title="UNIVERSAL AI — Precision Studio Core",
    description="High-Precision Multi-modal GenAI, Edge-TTS (Sindhi/Urdu/English) & Audio Transcription",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================================
# Request / Response Schemas
# =========================================================================

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'model'")
    content: str = Field(..., description="Message text content")

class ChatRequest(BaseModel):
    prompt: str = Field(..., description="The user query or instruction")
    model: str = Field(default="gemini-3.5-flash", description="Target Gemini Model ID")
    system_instruction: Optional[str] = Field(default=None, description="System persona and instructions")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    max_output_tokens: int = Field(default=4096, ge=64, le=8192)
    thinking_budget: Optional[int] = Field(default=None, ge=0)
    history: Optional[List[ChatMessage]] = Field(default=None, description="Conversation history")

class ConfigRequest(BaseModel):
    api_key: str = Field(..., description="Google Gemini API Key")

class OptimizePromptRequest(BaseModel):
    prompt: str = Field(..., description="Raw user prompt to engineer")
    objective: Optional[str] = Field(default=None, description="Target domain or objective")
    model: str = Field(default="gemini-3.5-flash")

class CodeArchitectRequest(BaseModel):
    specification: str = Field(..., description="Software specification or requirements")
    language: str = Field(default="python", description="Target programming language")
    framework: Optional[str] = Field(default=None, description="Preferred architecture or framework")
    model: str = Field(default="gemini-3.1-pro-preview")

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize into neural speech")
    voice: Optional[str] = Field(default=None, description="Voice identifier or None for auto-detect")

class LiveUIRequest(BaseModel):
    prompt: str = Field(..., description="Frontend UI/UX web application specification")
    model: str = Field(default="gemini-3.5-flash")

class UIBuildRequest(BaseModel):
    prompt: str
    model: str = "gemini-3.5-flash"

class UIExportRequest(BaseModel):
    html: str
    project_name: str = "universal_ai_site"


# =========================================================================
# API Routes
# =========================================================================

@app.get("/api/status")
async def get_system_status():
    """Returns engine operational telemetry, TTS readiness, and connection status."""
    return ai_engine.get_status()

@app.get("/api/models")
async def get_models():
    """Returns list of all available models with capability tags."""
    return {"models": ai_engine.get_models()}

@app.get("/api/tts/voices")
async def get_tts_voices():
    """Returns curated Edge-TTS multilingual voice profiles (Sindhi, Urdu, English, Arabic)."""
    return {"voices": ai_engine.get_voices()}

@app.post("/api/tts/file")
async def tts_file_endpoint(req: TTSRequest):
    """
    Synthesizes text into an MP3 file using Edge-TTS.
    Automatically detects Sindhi/Urdu/English script if no voice override is given.
    """
    try:
        voice_override = None if (not req.voice or req.voice == "auto") else req.voice
        filename = await text_to_speech_file(req.text, voice_override=voice_override)
        if not filename:
            raise HTTPException(status_code=400, detail="Empty text payload")
        
        voice_used = voice_override if voice_override else detect_voice(req.text)
        return {
            "status": "success",
            "filename": filename,
            "audio_url": f"/audio/{filename}",
            "voice": voice_used
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/audio/{filename}")
async def serve_audio_file(filename: str):
    """Serves generated audio MP3 files."""
    filepath = Path(AUDIO_DIR) / filename
    if filepath.exists():
        return FileResponse(str(filepath), media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio file not found")

@app.post("/api/audio/transcribe")
async def transcribe_audio_endpoint(
    audio: UploadFile = File(...)
):
    """
    Transcribes spoken audio in English, Sindhi, or Urdu using Gemini Multimodal Audio.
    """
    try:
        content = await audio.read()
        mime_type = audio.content_type or "audio/webm"
        transcription = await run_in_threadpool(transcribe_audio_gemini, content, mime_type=mime_type)
        return {
            "status": "success",
            "transcription": transcription
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/config")
def update_config(config: ConfigRequest):
    """Dynamically configures or updates the Gemini API Key."""
    result = ai_engine.set_api_key(config.api_key)
    return result

@app.post("/api/chat")
def chat_generate(req: ChatRequest):
    """Executes precision multi-turn chat and reasoning generation."""
    history_dicts = None
    if req.history:
        history_dicts = [{"role": msg.role, "content": msg.content} for msg in req.history]

    response = ai_engine.generate(
        prompt=req.prompt,
        model=req.model,
        system_instruction=req.system_instruction,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_output_tokens=req.max_output_tokens,
        thinking_budget=req.thinking_budget,
        history=history_dicts,
    )
    return response

@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """Streams generative response chunks via Server-Sent Events."""
    def event_generator():
        for chunk in ai_engine.generate_stream(
            prompt=req.prompt,
            model=req.model,
            system_instruction=req.system_instruction,
            temperature=req.temperature,
            top_p=req.top_p,
            max_output_tokens=req.max_output_tokens,
            thinking_budget=req.thinking_budget,
        ):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/vision")
async def analyze_vision_endpoint(
    image: UploadFile = File(...),
    prompt: str = Form(default="Analyze this image in detail"),
    mode: str = Form(default="deep_inspection"),
    model: str = Form(default="gemini-3.5-flash")
):
    """Processes uploaded visual media and runs precision visual inspection."""
    try:
        content = await image.read()
        mime_type = image.content_type or "image/png"
        result = await run_in_threadpool(
            ai_engine.analyze_vision,
            image_bytes=content,
            mime_type=mime_type,
            prompt=prompt,
            analysis_mode=mode,
            model=model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prompt/optimize")
def optimize_prompt_endpoint(req: OptimizePromptRequest):
    """Transforms a raw prompt into a 5-pillar precision system prompt."""
    result = ai_engine.optimize_prompt(
        raw_prompt=req.prompt,
        objective=req.objective,
        target_model=req.model
    )
    return result

@app.post("/api/code/architect")
def code_architect_endpoint(req: CodeArchitectRequest):
    """Generates an architectural blueprint and production code implementation."""
    result = ai_engine.architect_code(
        specification=req.specification,
        language=req.language,
        framework=req.framework,
        model=req.model
    )
    return result

@app.post("/api/ui/generate")
def live_ui_generate_endpoint(req: LiveUIRequest):
    """Generates a complete single-file responsive HTML5 + Tailwind + Lucide application."""
    result = ai_engine.generate_live_ui(
        prompt=req.prompt,
        model_name=req.model
    )
    return result

@app.post("/api/ui-builder")
def build_live_ui(request: UIBuildRequest):
    if not request.prompt.strip():
        return {"error": "Prompt required."}
    html_code = generate_live_ui(request.prompt, request.model)
    return {"html": html_code}

@app.post("/api/ui-builder/zip")
def export_ui_zip(request: UIExportRequest):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("index.html", request.html)
        zf.writestr("README.txt", "Built with UNIVERSAL AI — Ready for Hostinger / Vercel deployment.")
    zip_buffer.seek(0)
    filename = f"{request.project_name.strip() or 'universal_ai_site'}.zip"
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/health")
async def health():
    """Health probe endpoint."""
    return {"status": "healthy", "service": "UNIVERSAL AI — Precision Studio Core"}


# =========================================================================
# Serve Frontend Static Assets
# =========================================================================

index_path = frontend_dir / "index.html"

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serves the Precision Studio UI."""
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>UNIVERSAL AI — Precision Studio Core</h1><p>Frontend loading...</p>")

if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
