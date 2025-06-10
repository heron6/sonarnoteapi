import os
import tempfile
import subprocess
from flask import Flask, request, jsonify
from pyannote.audio import Pipeline
import torch
import whisper
from flask_cors import CORS
from collections import defaultdict

# === GPU Diagnostics ===
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

try:
    test_tensor = torch.tensor([1.0]).to("cuda" if torch.cuda.is_available() else "cpu")
    print("PyTorch tensor test device:", test_tensor.device)
except Exception as e:
    print("Tensor allocation failed:", str(e))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# === Error Handling ===
@app.errorhandler(Exception)
def handle_exception(e):
    import traceback
    traceback.print_exc()
    return jsonify({"error": str(e)}), 500

# Load HuggingFace token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HUGGINGFACE_TOKEN environment variable")

# Load Whisper (OpenAI local model)
print("Loading OpenAI Whisper model...")
whisper_model = whisper.load_model("medium", device=device)
print("Whisper model loaded.")

# === Whisper GPU diagnostics ===
print("Whisper model device:", next(whisper_model.parameters()).device)
for name, param in whisper_model.named_parameters():
    print(f"Param '{name}' loaded on: {param.device}")
    break  # one example is enough

print("GPU Memory Allocated:", torch.cuda.memory_allocated() / 1e6, "MB")
print("GPU Memory Reserved:", torch.cuda.memory_reserved() / 1e6, "MB")

# Load pyannote pipeline
print("Loading pyannote speaker diarization pipeline...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

# Move all submodels to GPU (if available)
for name, model in pipeline._models.items():
    model.to(device)
    print(f"{name} moved to: {next(model.parameters()).device}")

print("pyannote pipeline loaded.")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    f = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        f.save(tmp.name)
        audio_path = tmp.name

    normalized_path = audio_path.replace(".wav", "_normalized.wav")

    try:
        # === Normalize audio with ffmpeg ===
        print("Normalizing audio with ffmpeg...")
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", "16000",
            normalized_path
        ], check=True)
        print("Audio normalization complete.")

        # Run speaker diarization
        diarization = pipeline(normalized_path)
        diarization_turns = list(diarization.itertracks(yield_label=True))

        # Run Whisper transcription (local model)
        result = whisper_model.transcribe(normalized_path, verbose=False)
        segments = result.get("segments", [])

        # Assign Whisper segments to diarization turns
        turn_segments = defaultdict(list)
        for seg in segments:
            seg_start, seg_end = seg['start'], seg['end']
            max_overlap = 0
            assigned_turn_idx = None

            for i, (turn, _, _) in enumerate(diarization_turns):
                overlap = max(0, min(seg_end, turn.end) - max(seg_start, turn.start))
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_turn_idx = i

            if assigned_turn_idx is not None:
                turn_segments[assigned_turn_idx].append(seg['text'].strip())

        # Identify and remap speakers
        speaker_first_appearance = {}
        for turn, _, speaker in diarization_turns:
            if speaker not in speaker_first_appearance:
                speaker_first_appearance[speaker] = turn.start
        sorted_speakers = sorted(speaker_first_appearance.items(), key=lambda x: x[1])
        speaker_mapping = {old: f"SPEAKER_{i:02d}" for i, (old, _) in enumerate(sorted_speakers)}

        # Build initial result
        results = []
        for i, (turn, _, speaker) in enumerate(diarization_turns):
            combined_text = " ".join(turn_segments[i]) if i in turn_segments else ""
            if combined_text:  # Only include if there's actual speech
                results.append({
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": speaker_mapping.get(speaker, speaker),
                    "text": combined_text,
                })

        # Merge consecutive same-speaker blocks
        merged_results = []
        prev = None
        for r in results:
            if prev and r["speaker"] == prev["speaker"]:
                prev["end"] = r["end"]
                if r["text"]:
                    prev["text"] = f"{prev['text']} {r['text']}".strip()
            else:
                if prev:
                    merged_results.append(prev)
                prev = r
        if prev:
            merged_results.append(prev)

        # Full transcript
        full_text = result["text"]

        return jsonify({
            "transcription": full_text,
            "lines": merged_results,
            "file": None
        })

    finally:
        os.remove(audio_path)
        if os.path.exists(normalized_path):
            os.remove(normalized_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
