import os
import tempfile
import subprocess
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
from pyannote.audio import Pipeline
import torch
from collections import defaultdict

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load Whisper model
print("Loading Whisper...")
whisper_model = whisper.load_model("medium", device=device)
print("Whisper loaded.")

# === Check for HuggingFace token
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    raise EnvironmentError("Missing HUGGINGFACE_TOKEN environment variable.")

# === Load pyannote SpeakerDiarization pipeline
print("Loading pyannote SpeakerDiarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token
)
pipeline.to(device)
print(f"pyannote pipeline loaded on {device}")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    f = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        f.save(tmp.name)
        audio_path = tmp.name

    # Safe normalized path
    base, ext = os.path.splitext(audio_path)
    normalized_path = f"{base}_normalized{ext}"

    try:
        # === Normalize with ffmpeg
        print("Normalizing audio...")
        start = time.time()
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", "16000",
            normalized_path
        ], check=True)
        print(f"Normalization took {time.time() - start:.2f}s")

        # === Speaker diarization
        print("Running speaker diarization...")
        start = time.time()
        diarization = pipeline({'uri': 'file', 'audio': normalized_path})
        diarization_turns = list(diarization.itertracks(yield_label=True))
        print(f"Diarization took {time.time() - start:.2f}s")

        # === Whisper transcription
        print("Running Whisper transcription...")
        start = time.time()
        result = whisper_model.transcribe(
            normalized_path,
            language="en",
            fp16=(device.type == "cuda")
        )
        print(f"Whisper took {time.time() - start:.2f}s")

        # === Collect Whisper segments
        segments = [{
            "start": s["start"],
            "end": s["end"],
            "text": s["text"]
        } for s in result["segments"]]

        # === Align segments to diarization speaker turns with improved logic
        print("Aligning speaker labels to segments...")
        turn_segments = defaultdict(list)

        for s in segments:
            seg_start = s["start"]
            seg_end = s["end"]
            best_match = None
            best_overlap = 0

            for i, (turn, _, speaker) in enumerate(diarization_turns):
                overlap = max(0, min(seg_end, turn.end) - max(seg_start, turn.start))
                duration = seg_end - seg_start
                overlap_ratio = overlap / duration if duration > 0 else 0

                # Only assign if overlap is at least 25% of the segment duration
                if overlap_ratio > 0.25 and overlap > best_overlap:
                    best_overlap = overlap
                    best_match = i

            if best_match is not None:
                turn, _, speaker = diarization_turns[best_match]
                turn_segments[best_match].append({
                    "start": s["start"],
                    "end": s["end"],
                    "speaker": speaker,
                    "text": s["text"]
                })

        # Flatten results
        results = []
        for i, seg_list in turn_segments.items():
            results.extend(seg_list)

        print("Detected speakers:", set(speaker for _, _, speaker in diarization_turns))

        # Optional deduplication to avoid repeated lines
        seen = set()
        unique_results = []
        for r in results:
            key = (r["start"], r["end"], r["speaker"], r["text"])
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        results = unique_results

        # === Full transcription (no speaker tags)
        full_text = " ".join([s["text"] for s in segments])

        return jsonify({
            "transcription": full_text,
            "speaker_segments": results
        })
    

    except subprocess.CalledProcessError as e:
        return jsonify({"error": "Audio normalization failed", "details": str(e)}), 500
    except Exception as e:
        return jsonify({"error": "Processing failed", "details": str(e)}), 500
    finally:
        os.remove(audio_path)
        if os.path.exists(normalized_path):
            os.remove(normalized_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
