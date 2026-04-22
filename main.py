import os
import tempfile
import urllib.request
import json
from flask import Flask, request, jsonify
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi
import music21

app = Flask(__name__)

def midi_to_abc(midi_path: str, title: str = "Untitled") -> str:
    """Convert MIDI file to ABC notation using music21."""
    score = music21.converter.parse(midi_path)

    # Keep only the part with the most notes (melody)
    parts = score.parts
    if len(parts) > 1:
        melody = max(parts, key=lambda p: len(p.flat.notes))
        score = music21.stream.Score([melody])

    # Set metadata
    score.metadata = music21.metadata.Metadata()
    score.metadata.title = title

    # Write to a temp ABC file then read it back
    with tempfile.NamedTemporaryFile(suffix='.abc', delete=False) as tmp:
        tmp_path = tmp.name

    score.write('abc', fp=tmp_path)

    with open(tmp_path, 'r', encoding='utf-8') as f:
        abc_str = f.read()

    os.unlink(tmp_path)
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
            req = urllib.request.Request(
                audio_url,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            with urllib.request.urlopen(req) as response:
                tmp_audio.write(response.read())
            audio_path = tmp_audio.name

        # Transcribe with Basic Pitch
        model_output, midi_data, note_events = predict(
            audio_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=0.5,
            frame_threshold=0.3,
            minimum_note_length=58,
            minimum_frequency=27.5,   # A0 — lowest piano key
            maximum_frequency=4186.0, # C8 — highest piano key
            melodia_trick=True,
        )

        # Save MIDI to temp file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp_midi:
            midi_data.write(tmp_midi.name)
            midi_path = tmp_midi.name

        # Convert MIDI to ABC
        abc = midi_to_abc(midi_path, title)

        # Extract note events for piano animation
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch':     note.pitch,
                    'startTime': round(note.start, 3),
                    'endTime':   round(note.end, 3),
                    'velocity':  note.velocity,
                })
        notes.sort(key=lambda n: n['startTime'])

        # Cleanup
        os.unlink(audio_path)
        os.unlink(midi_path)

        return jsonify({
            'abc':   abc,
            'notes': notes,
            'count': len(notes),
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
