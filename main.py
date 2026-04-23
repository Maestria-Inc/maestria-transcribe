import os
import tempfile
import urllib.request
import math
from flask import Flask, request, jsonify
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH
import pretty_midi

app = Flask(__name__)


# ── Pure Python MIDI-to-ABC converter (no music21) ─────────────────────────

NOTE_NAMES = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B']

def midi_to_abc_note(midi_pitch):
    """Convert MIDI pitch to ABC note name."""
    octave = midi_pitch // 12 - 1  # MIDI 60 = C4
    note_idx = midi_pitch % 12
    name = NOTE_NAMES[note_idx]

    # ABC: C4=C  C3=C,  C2=C,,  C5=c  C6=c'
    if octave <= 4:
        result = name
        if octave < 4:
            result += ',' * (4 - octave)
    else:
        if name.startswith('^'):
            result = '^' + name[1:].lower()
        else:
            result = name.lower()
        if octave > 5:
            result += "'" * (octave - 5)
    return result


def dur_to_abc(sixteenths):
    """Duration in 16th-note units to ABC length string (base L:1/16)."""
    s = max(1, sixteenths)
    return '' if s == 1 else str(s)


def notes_to_abc(notes, title="Untitled", tempo=72):
    """Convert note dicts to renderable ABC notation."""
    if not notes:
        return ''

    sorted_notes = sorted(notes, key=lambda n: n['startTime'])

    beat_dur = 60.0 / tempo
    sixteenth = beat_dur / 4.0
    measure_slots = 16  # 16 sixteenths in 4/4

    # Quantize to 16th-note grid
    events = {}
    for n in sorted_notes:
        slot = round(n['startTime'] / sixteenth)
        dur = max(1, min(16, round((n['endTime'] - n['startTime']) / sixteenth)))
        vel = n.get('velocity', 64)
        if slot not in events:
            events[slot] = []
        events[slot].append((n['pitch'], dur, vel))

    if not events:
        return ''

    # Filter: max 4 notes per chord, keep highest velocity
    for slot in events:
        if len(events[slot]) > 4:
            events[slot].sort(key=lambda x: x[2], reverse=True)
            events[slot] = events[slot][:4]

    max_slot = max(events.keys())

    # Detect key from most common pitch class
    pc_counts = {}
    for n in sorted_notes:
        pc = n['pitch'] % 12
        pc_counts[pc] = pc_counts.get(pc, 0) + 1
    root_pc = max(pc_counts, key=pc_counts.get)
    key_map = ['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B']
    key = key_map[root_pc]

    header = f'X:1\nT:{title}\nM:4/4\nL:1/16\nQ:1/4={tempo}\nK:{key}\n'

    measures = []
    slot = 0
    max_measures = 120

    while slot <= max_slot and len(measures) < max_measures:
        bar = ''
        pos = 0

        while pos < measure_slots:
            if slot > max_slot:
                # Fill rest of measure
                remaining = measure_slots - pos
                if remaining > 0:
                    bar += 'z' + dur_to_abc(remaining)
                break

            if slot in events:
                pitches = events[slot]
                # Deduplicate by pitch
                seen = set()
                unique = []
                for p, d, v in pitches:
                    if p not in seen:
                        seen.add(p)
                        unique.append((p, d))

                max_d = max(d for _, d in unique)
                max_d = min(max_d, measure_slots - pos)

                if len(unique) == 1:
                    p, d = unique[0]
                    d = min(d, measure_slots - pos)
                    bar += midi_to_abc_note(p) + dur_to_abc(d)
                else:
                    # Sort chord low to high
                    unique.sort(key=lambda x: x[0])
                    bar += '[' + ''.join(midi_to_abc_note(p) for p, _ in unique) + ']'
                    bar += dur_to_abc(max_d)

                pos += max_d
                slot += max_d
            else:
                # Find consecutive rests
                rest = 0
                while (slot + rest <= max_slot and
                       (slot + rest) not in events and
                       pos + rest < measure_slots):
                    rest += 1
                rest = max(1, min(rest, measure_slots - pos))
                bar += 'z' + dur_to_abc(rest)
                pos += rest
                slot += rest

        measures.append(bar)

    # Format: 4 measures per line
    lines = []
    for i in range(0, len(measures), 4):
        chunk = measures[i:i+4]
        lines.append('|'.join(chunk) + '|')

    body = '\n'.join(lines)
    body += ']'  # Final barline

    return header + body


# ── Routes ─────────────────────────────────────────────────────────────────

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

        # Basic Pitch transcription (higher thresholds = fewer ghost notes)
        model_output, midi_data, note_events = predict(
            audio_path,
            ICASSP_2022_MODEL_PATH,
            onset_threshold=0.6,
            frame_threshold=0.45,
            minimum_note_length=80,
            minimum_frequency=27.5,
            maximum_frequency=4186.0,
            melodia_trick=True,
        )

        # Extract notes
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

        # Generate ABC (pure Python)
        # Filter weak notes for cleaner score (keep top 60% by velocity)
        if notes:
            velocities = sorted([n['velocity'] for n in notes])
            vel_threshold = velocities[int(len(velocities) * 0.4)]
            score_notes = [n for n in notes if n['velocity'] >= vel_threshold]
        else:
            score_notes = notes
        abc = notes_to_abc(score_notes, title)

        os.unlink(audio_path)

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
    
