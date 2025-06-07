import os
import tempfile
from flask import Flask, request, jsonify
import whisper
from pyannote.audio import Pipeline
import torch
from flask_cors import CORS
from collections import defaultdict

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")


app = Flask(__name__)
CORS(app)

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Please set HUGGINGFACE_TOKEN environment variable")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper model on {device}...")
whisper_model = whisper.load_model("medium").to(device)
result = whisper_model.transcribe(audio_path, fp16=(device == "cuda"))


print("Loading pyannote speaker diarization pipeline (3.1)...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    f = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        f.save(tmp.name)
        audio_path = tmp.name

    try:
        diarization = pipeline(audio_path)
        segments = result.get("segments", [])

        # Map from diarization turn index to assigned segment texts
        turn_segments = defaultdict(list)

        # Convert diarization turns to list for indexed access
        diarization_turns = list(diarization.itertracks(yield_label=True))

        # Assign whisper segments to diarization turns by max overlap
        for seg in segments:
            max_overlap = 0
            assigned_turn_idx = None
            seg_start = seg["start"]
            seg_end = seg["end"]

            for i, (turn, _, speaker) in enumerate(diarization_turns):
                overlap = max(0, min(seg_end, turn.end) - max(seg_start, turn.start))
                if overlap > max_overlap:
                    max_overlap = overlap
                    assigned_turn_idx = i

            if assigned_turn_idx is not None:
                turn_segments[assigned_turn_idx].append(seg["text"].strip())

        # 1. Identify first appearance time of each speaker
        speaker_first_appearance = {}
        for turn, _, speaker in diarization_turns:
            if speaker not in speaker_first_appearance:
                speaker_first_appearance[speaker] = turn.start

        # 2. Sort speakers by first appearance and remap to SPEAKER_00, SPEAKER_01, ...
        sorted_speakers = sorted(speaker_first_appearance.items(), key=lambda x: x[1])
        speaker_mapping = {old: f"SPEAKER_{i:02d}" for i, (old, _) in enumerate(sorted_speakers)}

        # 3. Build initial results list with remapped speakers
        results = []
        for i, (turn, _, speaker) in enumerate(diarization_turns):
            combined_text = " ".join(turn_segments[i]) if i in turn_segments else ""
            results.append({
                "start": round(turn.start, 2),
                "end": round(turn.end, 2),
                "speaker": speaker_mapping.get(speaker, speaker),
                "text": combined_text,
            })

        # 4. Merge consecutive segments by same speaker to avoid fragmentation
        merged_results = []
        prev = None
        for r in results:
            if prev and r["speaker"] == prev["speaker"]:
                # Extend previous segment's end time and append text
                prev["end"] = r["end"]
                if r["text"]:
                    if prev["text"]:
                        prev["text"] += " " + r["text"]
                    else:
                        prev["text"] = r["text"]
            else:
                if prev:
                    merged_results.append(prev)
                prev = r
        if prev:
            merged_results.append(prev)

        return jsonify({
            "transcription": result.get("text", ""),
            "lines": merged_results,
            "file": None,
        })
    finally:
        os.remove(audio_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
