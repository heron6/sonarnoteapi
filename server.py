from flask import Flask, request
from flask_cors import CORS
import os
import whisper
import torchaudio
import subprocess
from pyannote.audio import Pipeline
import torch

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Whisper model on GPU
whisper_model = whisper.load_model("medium", device=device)

# Load pyannote speaker diarization pipeline on GPU
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN")  # Set this in your environment
).to(device)

@app.route("/")
def index():
    return "Whisper + Pyannote transcription server running (GPU-only, with speaker diarization)."

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

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

    print("Running speaker diarization.")
    diarization = diarization_pipeline(filepath)
    print("Diarization complete.")

    # Match Whisper segments to speaker turns
    speaker_segments = []
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]
        speaker_label = "Unknown"

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= start <= turn.end or turn.start <= end <= turn.end:
                speaker_label = speaker
                break

        speaker_segments.append({
            "start": start,
            "end": end,
            "speaker": speaker_label,
            "text": text
        })

    full_text = " ".join([f"{seg['speaker']}: {seg['text']}" for seg in speaker_segments])

    return {
        'transcription': full_text,
        'segments': speaker_segments
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
