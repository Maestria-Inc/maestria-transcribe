import os
import tempfile
import urllib.request
from flask import Flask, request, jsonify
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi
import music21

app = Flask(__name__)

def midi_to_abc(midi_path: str, title: str = "Untitled") -> str:
    """Convert MIDI file to ABC notation using music21."""
    score = music21.converter.parse(midi_path)

    # Keep only the first part (melody) if multiple tracks
    parts = score.parts
    if len(parts) > 1:
        # Find the part with the most notes (likely the melody)
        melody = max(parts, key=lambda p: len(p.flat.notes))
        score = music21.stream.Score([melody])

    # Set title
    score.metadata = music21.metadata.Metadata()
    score.metadata.title = title

    # Export to ABC
    exporter = music21.converter.subConverters.ConverterABC()
    abc_str = exporter.write(score, fmt='abc', fp=None)
    return abc_str

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    audio_url = data.get('audioUrl')
    title     = data.get('title', 'Untitled')

    if not audio_url:
        return jsonify({'error': 'audioUrl required'}), 400

    try:
        # Download audio to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_audio:
            urllib.request.urlretrieve(audio_url, tmp_audio.name)
            audio_path = tmp_audio.name

        # Transcribe with Basic Pitch
        model_output, midi_data, note_events = predict(
            audio_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,   # ms — filters out noise
            minimum_frequency=27.5,   # A0 — lowest piano key
            maximum_frequency=4186.0, # C8 — highest piano key
            melodia_trick=True,       # improves melody extraction
        )

        # Save MIDI to temp file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp_midi:
            midi_data.write(tmp_midi.name)
            midi_path = tmp_midi.name

        # Convert MIDI to ABC
        abc = midi_to_abc(midi_path, title)

        # Extract note events for piano animation
        # Format: [{ pitch, startTime, endTime, velocity }]
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch':     note.pitch,       # MIDI note number (0-127)
                    'startTime': note.start,        # seconds
                    'endTime':   note.end,           # seconds
                    'velocity':  note.velocity,      # 0-127
                })
        # Sort by start time
        notes.sort(key=lambda n: n['startTime'])

        # Cleanup temp files
        os.unlink(audio_path)
        os.unlink(midi_path)

        return jsonify({
            'abc':   abc,
            'notes': notes,       # used for piano virtual animation
            'count': len(notes),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
