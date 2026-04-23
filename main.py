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
    score = music21.converter.parse(midi_path)

    # Keep only the part with the most notes (melody)
    parts = score.parts
    if len(parts) > 1:
        melody = max(parts, key=lambda p: len(p.flat.notes))
        score = music21.stream.Score([melody])

    score.metadata = music21.metadata.Metadata()
    score.metadata.title = title

    # music21 always appends .abc — use a fixed known path
    out_base = '/tmp/maestria_out'
    out_path = out_base + '.abc'

    # Remove stale file if exists
    if os.path.exists(out_path):
        os.unlink(out_path)

    score.write('abc', fp=out_base)

    # Read the file music21 actually wrote
    if not os.path.exists(out_path):
        # Fallback: maybe it wrote without extension
        if os.path.exists(out_base):
            out_path = out_base
        else:
            return ''

    with open(out_path, 'r', encoding='utf-8') as f:
        abc_str = f.read()

    os.unlink(out_path)
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
        # Download audio
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
            minimum_frequency=27.5,
            maximum_frequency=4186.0,
            melodia_trick=True,
        )

        # Save MIDI
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp_midi:
            midi_data.write(tmp_midi.name)
            midi_path = tmp_midi.name

        # Convert MIDI → ABC
        abc = midi_to_abc(midi_path, title)

        # Extract notes for piano animation
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append({
                    'pitch':     int(note.pitch),
                    'startTime': round(float(note.start), 3),
                    'endTime':   round(float(note.end), 3),
                    'velocity':  int(note.velocity),
                })
        notes.sort(key=lambda n: n['startTime'])

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
