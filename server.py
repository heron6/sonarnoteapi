from flask import Flask, request
from flask_cors import CORS
import os
import whisper
import torchaudio
import subprocess

app = Flask(__name__)
CORS(app)

# Load Whisper model on GPU
whisper_model = whisper.load_model("medium", device="cuda")

@app.route("/")
def index():
    return "Whisper transcription server running (GPU-only, no document output)."

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    timestamps_enabled = request.form.get('timestamps') == 'true'

    os.makedirs('temp', exist_ok=True)
    original_path = os.path.join('temp', file.filename)
    file.save(original_path)

    try:
        filepath = convert_to_wav(original_path)
    except RuntimeError as e:
        return str(e), 500

    # Optional: limit to 60 seconds for testing
    waveform, sample_rate = torchaudio.load(filepath)
    max_samples = 60 * sample_rate
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
        torchaudio.save(filepath, waveform, sample_rate)

    print("Starting transcription.")
    result = whisper_model.transcribe(filepath, language="en")
    segments = result.get("segments", [])
    print("Transcription done.")

    # Group segments (no speaker info)
    grouped_segments = []
    if segments:
        current_group = {
            "start": segments[0]["start"],
            "end": segments[0]["end"],
            "text": segments[0]["text"]
        }

        for segment in segments[1:]:
            current_group["end"] = segment["end"]
            current_group["text"] += " " + segment["text"]
        grouped_segments.append(current_group)

    transcription_text = " ".join([seg["text"] for seg in grouped_segments])

    return {
        'transcription': transcription_text,
        'segments': grouped_segments,
    }

def convert_to_wav(input_path):
    output_path = input_path.rsplit('.', 1)[0] + '.wav'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', input_path, output_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return output_path
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to convert audio to WAV format.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
