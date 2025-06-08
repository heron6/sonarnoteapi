import os
import tempfile
from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torch
from flask_cors import CORS
from collections import defaultdict
import ctranslate2

# === GPU Diagnostics ===
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("PyTorch tensor test device:", torch.tensor([1.0]).device)
print("CTranslate2 device:", ctranslate2.Device.from_string("cuda" if torch.cuda.is_available() else "cpu"))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Load HuggingFace token
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HUGGINGFACE_TOKEN environment variable")

# Load Whisper with GPU support
print("Loading faster-whisper model...")
# Use float32 for broader compatibility
whisper_model = WhisperModel("medium", device=device, compute_type="float32")
print("faster-whisper model loaded.")
print("Whisper model backend device:", whisper_model.model.device)

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

    try:
        # Run speaker diarization
        diarization = pipeline(audio_path)
        diarization_turns = list(diarization.itertracks(yield_label=True))

        # Run faster-whisper transcription
        segments_gen, _ = whisper_model.transcribe(audio_path)
        segments = list(segments_gen)

        # Assign Whisper segments to diarization turns
        turn_segments = defaultdict(list)
        for seg in segments:
            seg_start, seg_end = seg.start, seg.end
            max_overlap = 0
            assigned_turn_idx = None

            for i, (turn, _, _) in enumerate(diarization_turns):
                overlap = max(0, min(seg_end, turn.end) - max(seg_start, turn.start))
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_turn_idx = i

            if assigned_turn_idx is not None:
                turn_segments[assigned_turn_idx].append(seg.text.strip())

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
        full_text = " ".join([s.text for s in segments])

        return jsonify({
            "transcription": full_text,
            "lines": merged_results,
            "file": None
        })
    finally:
        os.remove(audio_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
