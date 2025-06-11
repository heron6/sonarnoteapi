#!/usr/bin/env python3
"""
Startup script for the Audio Transcription API
"""
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Main function to start the server"""
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    
    print("🎙️  Starting Audio Transcription API...")
    print(f"📡 Server will run on: http://{host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔄 Auto-reload: {reload}")
    print("\n⚠️  Note: Models will load on first startup - this may take a few minutes!")
    print("\n🤗 Make sure you have:")
    print("   1. Accepted user conditions for pyannote models on Hugging Face")
    print("   2. Set your HF_TOKEN environment variable if required")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 
