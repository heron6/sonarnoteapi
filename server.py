import os
import tempfile
import subprocess
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import defaultdict
from faster_whisper import WhisperModel
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.audio import Model, Inference
import torch

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === Load Faster-Whisper model
print("Loading Faster-Whisper...")
whisper_model = WhisperModel("medium", device=device, compute_type="float16" if device == "cuda" else "int8")
print("Whisper loaded.")

# === Load pyannote v2.1 pipeline
print("Loading pyannote 2.1 SpeakerDiarization pipeline...")
pipeline = SpeakerDiarization(segmentation="pyannote/segmentation")
pipeline.instantiate({
    "segmentation": {
        "device": device
    }
})
print("pyannote 2.1 pipeline loaded.")

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
        # === Normalize with ffmpeg
        start = time.time()
        print("Normalizing audio...")
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-ac", "1", "-ar", "16000",
            normalized_path
        ], check=True)
        print(f"Normalization took {time.time() - start:.2f}s")

        # === Diarization
        start = time.time()
        diarization = pipeline({'uri': 'file', 'audio': normalized_path})
        diarization_turns = list(diarization.itertracks(yield_label=True))
        print(f"Diarization took {time.time() - start:.2f}s")

        # === Whisper transcription
        start = time.time()
        segments, _ = whisper_model.transcribe(normalized_path, language="en")
        segments = list(segments)
        print(f"Whisper took {time.time() - start:.2f}s")

        # === Map transcription segments to speaker turns
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

        # === Remap speaker labels
        speaker_first_appearance = {}
        for turn, _, speaker in diarization_turns:
            if speaker not in speaker_first_appearance:
                speaker_first_appearance[speaker] = turn.start
        sorted_speakers = sorted(speaker_first_appearance.items(), key=lambda x: x[1])
        speaker_mapping = {old: f"SPEAKER_{i:02d}" for i, (old, _) in enumerate(sorted_speakers)}

        # === Build final results
        results = []
        for i, (turn, _, speaker) in enumerate(diarization_turns):
            combined_text = " ".join(turn_segments[i]) if i in turn_segments else ""
            if combined_text:
                results.append({
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": speaker_mapping.get(speaker, speaker),
                    "text": combined_text,
                })

        # === Merge consecutive same-speaker blocks
        merged_results = []
        prev = None
        for r in results:
            if prev and r["speaker"] == prev["speaker"]:
                prev["end"] = r["end"]
                prev["text"] = f"{prev['text']} {r['text']}".strip()
            else:
                if prev:
                    merged_results.append(prev)
                prev = r
        if prev:
            merged_results.append(prev)

        full_text = " ".join([s.text for s in segments])

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
