"""
Maestria Piano Transcription Service v3
========================================
ByteDance piano_transcription_inference + music21 for notation.

Endpoints:
  POST /transcribe  { audioUrl, title? }  ->  { taskId }
  GET  /status?taskId=X                   ->  { status, notes?, abc?, noteCount? }
  GET  /health                            ->  { status, model, device }
"""

import os
import json
import uuid
import tempfile
import traceback
import threading
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests as http_requests
import librosa
import numpy as np
import pretty_midi
import gc

app = Flask(__name__)
CORS(app)

# ── Task store ───────────────────────────────────────────────────────────────
tasks = {}

# ── Load ByteDance model at startup ──────────────────────────────────────────
from piano_transcription_inference import PianoTranscription, sample_rate

DEVICE = 'cuda' if os.environ.get('USE_CUDA') else 'cpu'
print(f"[Maestria] Loading ByteDance piano transcription model on {DEVICE}...")
transcriptor = PianoTranscription(device=DEVICE, checkpoint_path=None)
print("[Maestria] Model loaded successfully.")

transcribe_lock = threading.Lock()

# ── Import music21 ───────────────────────────────────────────────────────────
import music21
from music21 import converter, instrument, stream, tempo, key, meter, note, chord

# Configure music21 to use LilyPond if available (optional, for PDF)
# For ABC output, no external renderer needed
print("[Maestria] music21 loaded successfully.")


# ── music21 MIDI → ABC conversion ───────────────────────────────────────────

def midi_to_notation(midi_path, title='Untitled'):
    """
    Convert MIDI file to properly quantized ABC notation using music21.
    Returns ABC string.
    """
    try:
        # Parse MIDI with music21
        score = converter.parse(midi_path)

        # Set metadata
        score.metadata = music21.metadata.Metadata()
        score.metadata.title = title
        score.metadata.composer = 'Maestria'

        # Quantize — music21 handles this properly
        # This fixes note alignment, measure boundaries, proper rhythmic values
        score.quantize(inPlace=True)

        # Split into treble and bass if not already
        parts = score.parts
        if len(parts) == 1:
            # Single track — split by pitch at middle C (MIDI 60)
            original = parts[0]
            treble = stream.Part()
            bass = stream.Part()
            treble.insert(0, instrument.Piano())
            bass.insert(0, instrument.Piano())

            for element in original.recurse():
                if isinstance(element, note.Note):
                    if element.pitch.midi >= 60:
                        treble.append(element)
                    else:
                        bass.append(element)
                elif isinstance(element, chord.Chord):
                    upper = chord.Chord([p for p in element.pitches if p.midi >= 60])
                    lower = chord.Chord([p for p in element.pitches if p.midi < 60])
                    if upper.pitches:
                        upper.quarterLength = element.quarterLength
                        treble.append(upper)
                    if lower.pitches:
                        lower.quarterLength = element.quarterLength
                        bass.append(lower)
                elif isinstance(element, (meter.TimeSignature, key.KeySignature, tempo.MetronomeMark)):
                    treble.append(element)
                    bass.append(element)

            # Make measures
            treble.makeMeasures(inPlace=True)
            bass.makeMeasures(inPlace=True)

            # Set clefs
            from music21 import clef
            treble.insert(0, clef.TrebleClef())
            bass.insert(0, clef.BassClef())

            score = stream.Score()
            score.metadata = music21.metadata.Metadata()
            score.metadata.title = title
            score.metadata.composer = 'Maestria'
            score.insert(0, treble)
            score.insert(0, bass)

        # Generate ABC notation
        abc_text = ''
        try:
            # music21 can write ABC directly
            with tempfile.NamedTemporaryFile(suffix='.abc', delete=False, mode='w') as f:
                abc_path = f.name

            score.write('abc', fp=abc_path)

            with open(abc_path, 'r') as f:
                abc_text = f.read()

            os.unlink(abc_path)
        except Exception as abc_err:
            print(f"[Maestria] ABC generation via music21 failed: {abc_err}")
            # Fallback: generate basic ABC from the parsed score
            abc_text = fallback_abc(score, title)

        return abc_text

    except Exception as e:
        print(f"[Maestria] music21 conversion failed: {e}")
        traceback.print_exc()
        return ''


def fallback_abc(score, title='Untitled'):
    """Simple fallback ABC if music21's ABC writer fails."""
    try:
        # Try MusicXML as intermediate
        with tempfile.NamedTemporaryFile(suffix='.musicxml', delete=False) as f:
            xml_path = f.name
        score.write('musicxml', fp=xml_path)

        # Re-parse and try ABC again
        score2 = converter.parse(xml_path)
        with tempfile.NamedTemporaryFile(suffix='.abc', delete=False, mode='w') as f:
            abc_path = f.name
        score2.write('abc', fp=abc_path)
        with open(abc_path, 'r') as f:
            abc_text = f.read()

        os.unlink(xml_path)
        os.unlink(abc_path)
        return abc_text
    except:
        return f"X:1\nT:{title}\nM:4/4\nL:1/8\nK:C\n|z8|"


# ── Background transcription worker ─────────────────────────────────────────

def transcribe_worker(task_id, audio_url, title):
    try:
        print(f"[Maestria] [{task_id[:8]}] Downloading: {audio_url[:80]}...")
        resp = http_requests.get(audio_url, timeout=60)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(resp.content)
            audio_path = f.name

        print(f"[Maestria] [{task_id[:8]}] Downloaded {len(resp.content)} bytes")

        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        duration = len(audio) / sample_rate
        print(f"[Maestria] [{task_id[:8]}] Audio: {duration:.1f}s at {sample_rate}Hz")

        midi_path = audio_path.replace('.mp3', '.mid')
        print(f"[Maestria] [{task_id[:8]}] Transcribing...")

        with transcribe_lock:
            transcriptor.transcribe(audio, midi_path)

        # Parse notes for the falling notes display
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        all_notes = []
        for instr in midi_data.instruments:
            for n in instr.notes:
                all_notes.append({
                    'pitch': n.pitch,
                    'startTime': round(n.start, 4),
                    'endTime': round(n.end, 4),
                    'velocity': n.velocity,
                })
        all_notes.sort(key=lambda n: (n['startTime'], n['pitch']))

        # Generate notation with music21 (uses the MIDI file directly)
        print(f"[Maestria] [{task_id[:8]}] Generating notation with music21...")
        abc = midi_to_notation(midi_path, title)
        print(f"[Maestria] [{task_id[:8]}] Done: {len(all_notes)} notes, {len(abc)} chars ABC")

        # Cleanup
        try: os.unlink(midi_path)
        except: pass
        try: os.unlink(audio_path)
        except: pass
        gc.collect()

        tasks[task_id] = {
            'status': 'complete',
            'result': {
                'notes': all_notes,
                'abc': abc,
                'noteCount': len(all_notes),
            }
        }

    except Exception as e:
        traceback.print_exc()
        print(f"[Maestria] [{task_id[:8]}] Failed: {e}")
        tasks[task_id] = {
            'status': 'failed',
            'error': str(e),
        }


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json or {}
    audio_url = data.get('audioUrl')
    title = data.get('title', 'Untitled')

    if not audio_url:
        return jsonify({'error': 'audioUrl required'}), 400

    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'processing'}

    thread = threading.Thread(target=transcribe_worker, args=(task_id, audio_url, title))
    thread.daemon = True
    thread.start()

    return jsonify({'taskId': task_id, 'status': 'processing'})


@app.route('/status', methods=['GET'])
def status():
    task_id = request.args.get('taskId')
    if not task_id or task_id not in tasks:
        return jsonify({'status': 'not_found'}), 404

    task = tasks[task_id]

    if task['status'] == 'processing':
        return jsonify({'status': 'processing'})
    elif task['status'] == 'complete':
        return jsonify({
            'status': 'complete',
            **task['result'],
        })
    else:
        return jsonify({
            'status': 'failed',
            'error': task.get('error', 'Unknown error'),
        })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'bytedance/piano_transcription',
        'notation': 'music21',
        'device': DEVICE,
        'active_tasks': sum(1 for t in tasks.values() if t['status'] == 'processing'),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
