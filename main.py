from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import os
import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from pyannote.audio import Pipeline
import librosa
import numpy as np
from typing import Optional
import json
from datetime import datetime
import time
import math 
from dotenv import load_dotenv
load_dotenv()
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

app = FastAPI(title="Audio Transcription API", version="1.0.0")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
transcription_model = None
transcription_processor = None
diarization_pipeline = None

# Supported languages for Whisper
SUPPORTED_LANGUAGES = {
    "auto": "Auto-detect",
    "en": "English",
    "es": "Spanish", 
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "hi": "Hindi",
    "nl": "Dutch",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "tr": "Turkish",
    "cs": "Czech",
    "hu": "Hungarian",
    "th": "Thai",
    "vi": "Vietnamese"
}

@app.on_event("startup")
async def load_models():
    """Load models on startup"""
    global transcription_model, transcription_processor, diarization_pipeline
    
    try:
        # Load Whisper model for transcription
        model_id = "openai/whisper-base"
        transcription_processor = AutoProcessor.from_pretrained(model_id)
        transcription_model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        
        # Load pyannote speaker diarization pipeline
        # Note: You need to accept user conditions at https://huggingface.co/pyannote/speaker-diarization-3.1
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=HUGGINGFACE_TOKEN
        )
        
        print("✅ Models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("Note: Make sure you have accepted the user conditions for pyannote models on Hugging Face")

def preprocess_audio(file_path: str, target_sr: int = 16000):
    """Preprocess audio file for transcription"""
    try:
        # Load audio with librosa
        audio, sr = librosa.load(file_path, sr=target_sr)
        return audio, sr
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing audio: {str(e)}")

def transcribe_audio(audio_path: str, language: str = "auto") -> str:
    """Transcribe audio using Whisper model with language specification"""
    try:
        # Load and preprocess audio
        audio, sr = preprocess_audio(audio_path)
        
        # Process with Whisper
        inputs = transcription_processor(audio, sampling_rate=sr, return_tensors="pt")
        
        # Set up generation kwargs
        generation_kwargs = {"input_features": inputs["input_features"]}
        
        # Add language forcing if specified
        if language != "auto" and language in SUPPORTED_LANGUAGES:
            # Force the model to use the specified language
            forced_decoder_ids = transcription_processor.get_decoder_prompt_ids(
                language=language, 
                task="transcribe"
            )
            generation_kwargs["forced_decoder_ids"] = forced_decoder_ids
        
        # Generate transcription
        with torch.no_grad():
            predicted_ids = transcription_model.generate(**generation_kwargs)
        
        # Decode the transcription
        transcription = transcription_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription.strip()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription error: {str(e)}")

def perform_speaker_diarization(audio_path: str):
    """Perform speaker diarization using pyannote"""
    try:
        # Apply diarization pipeline
        diarization = diarization_pipeline(audio_path)
        
        # Convert diarization results to a more usable format
        speakers_timeline = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers_timeline.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": speaker,
                "duration": float(turn.end - turn.start)
            })
        
        return speakers_timeline
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Speaker diarization error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Audio Transcription API is running!", "timestamp": datetime.now().isoformat()}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    models_loaded = all([
        transcription_model is not None,
        transcription_processor is not None,
        diarization_pipeline is not None
    ])
    
    return {
        "status": "healthy" if models_loaded else "loading",
        "models_loaded": models_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "supported_languages": SUPPORTED_LANGUAGES,
        "default": "auto",
        "note": "Use 'auto' for automatic language detection"
    }

@app.post("/transcribe")
async def transcribe_only(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """Transcribe audio without speaker diarization"""
    
    if not transcription_model or not transcription_processor:
        raise HTTPException(status_code=503, detail="Transcription model not loaded yet")
    
    # Validate language
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language: {language}. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}"
        )
    
    # Validate file type
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Transcribe the audio with specified language
        transcription = transcribe_audio(temp_file_path, language)
        
        return JSONResponse(content={
            "transcription": transcription,
            "language": language,
            "language_name": SUPPORTED_LANGUAGES.get(language, "Unknown"),
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    return time.strftime('%H:%M:%S', time.gmtime(math.floor(seconds)))

def segment_transcription_by_speakers(transcription: str, speakers_timeline: list):
    words = transcription.split()
    total_words = len(words)
    total_segments = len(speakers_timeline)

    if total_segments == 0:
        return []

    avg_words_per_segment = max(total_words // total_segments, 1)
    lines = []
    word_idx = 0

    for i, segment in enumerate(speakers_timeline):
        segment_words = words[word_idx:word_idx + avg_words_per_segment]
        word_idx += avg_words_per_segment

        lines.append({
            "id": i,
            "speaker": segment["speaker"],
            "timestamp": segment["start"],
            "timestampFormatted": format_timestamp(segment["start"]),
            "text": " ".join(segment_words)
        })

    return lines


@app.post("/transcribe-with-speakers")
async def transcribe_with_speakers(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """Transcribe audio with speaker diarization and return speaker-labeled segments"""
    
    if not all([transcription_model, transcription_processor, diarization_pipeline]):
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported language: {language}. Supported languages: {list(SUPPORTED_LANGUAGES.keys())}"
        )

    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name

    try:
        transcription = transcribe_audio(temp_file_path, language)
        speakers_timeline = perform_speaker_diarization(temp_file_path)

        unique_speakers = list(set([segment["speaker"] for segment in speakers_timeline]))
        lines = segment_transcription_by_speakers(transcription, speakers_timeline)

        return JSONResponse(content={
            "transcription": transcription,
            "language": language,
            "language_name": SUPPORTED_LANGUAGES.get(language, "Unknown"),
            "speakers_timeline": speakers_timeline,
            "unique_speakers": unique_speakers,
            "total_speakers": len(unique_speakers),
            "lines": lines,  # ✅ new addition for frontend
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@app.post("/diarize-only")
async def diarize_only(file: UploadFile = File(...)):
    """Perform only speaker diarization without transcription"""
    
    if not diarization_pipeline:
        raise HTTPException(status_code=503, detail="Diarization model not loaded yet")
    
    # Validate file type
    if not file.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Perform speaker diarization
        speakers_timeline = perform_speaker_diarization(temp_file_path)
        
        # Get unique speakers
        unique_speakers = list(set([segment["speaker"] for segment in speakers_timeline]))
        
        return JSONResponse(content={
            "speakers_timeline": speakers_timeline,
            "unique_speakers": unique_speakers,
            "total_speakers": len(unique_speakers),
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 
