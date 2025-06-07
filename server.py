import os
import tempfile
from flask import Flask, request, jsonify
import whisper
from pyannote.audio import Pipeline

app = Flask(__name__)

# Load environment variables
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HUGGINGFACE_TOKEN environment variable")

print("Loading OpenAI Whisper medium model...")
whisper_model = whisper.load_model("medium").to("cuda")

print("Loading pyannote diarization pipeline...")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HF_TOKEN)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    f = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        f.save(tmp.name)
        audio_path = tmp.name

    try:
        result = whisper_model.transcribe(audio_path)
        segments = result.get("segments", [])
        results = []
        for segment in segments:
            results.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
        return jsonify({"segments": results, "language": result.get("language", "unknown")})
    finally:
        os.remove(audio_path)


@app.route("/process", methods=["POST"])
def process():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    f = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        f.save(tmp.name)
        audio_path = tmp.name

    try:
        # Speaker diarization
        diarization = pipeline(audio_path)

        # Transcription
        result = whisper_model.transcribe(audio_path)
        segments = result.get("segments", [])

        # Combine diarization and transcription
        results = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Find transcription segments overlapping diarization turn
            texts = []
            for seg in segments:
                if seg["start"] >= turn.start and seg["end"] <= turn.end:
                    texts.append(seg["text"].strip())

            combined_text = " ".join(texts) if texts else ""

            results.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker,
                "text": combined_text
            })

        return jsonify(results)
    finally:
        os.remove(audio_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
