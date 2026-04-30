"""
MaestrIA Piano Transcription Service v2.1
==========================================
ByteDance piano_transcription_inference — trained on MAESTRO dataset.
Async architecture: POST /transcribe returns taskId, GET /status polls result.

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
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests as http_requests
import librosa
import numpy as np
import pretty_midi
import gc

app = Flask(__name__)
CORS(app)

# ── Task store (in-memory) ───────────────────────────────────────────────────
tasks = {}

# ── Load ByteDance model at startup ──────────────────────────────────────────
from piano_transcription_inference import PianoTranscription, sample_rate

DEVICE = 'cuda' if os.environ.get('USE_CUDA') else 'cpu'
print(f"[MaestrIA] Loading ByteDance piano transcription model on {DEVICE}...")
transcriptor = PianoTranscription(device=DEVICE, checkpoint_path=None)
print("[MaestrIA] Model loaded successfully.")

# Lock to prevent concurrent transcriptions (model is not thread-safe)
transcribe_lock = threading.Lock()


# ── ABC generation ───────────────────────────────────────────────────────────

NOTE_NAMES = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B']

def midi_to_abc_note(midi_pitch):
    octave = midi_pitch // 12 - 1
    note_idx = midi_pitch % 12
    name = NOTE_NAMES[note_idx]
    if octave >= 5:
        base = name[0].lower() if name[0] != '^' else '^' + name[1].lower()
        return base + "'" * (octave - 5)
    elif octave == 4:
        return name
    else:
        return name + "," * (4 - octave)


def quantize_duration(dur_in_sixteenths):
    if dur_in_sixteenths <= 0: return '1'
    if dur_in_sixteenths < 1.5: return ''
    elif dur_in_sixteenths < 3: return '2'
    elif dur_in_sixteenths < 5: return '4'
    elif dur_in_sixteenths < 7: return '6'
    elif dur_in_sixteenths < 10: return '8'
    elif dur_in_sixteenths < 14: return '12'
    else: return '16'


def detect_key(notes):
    if not notes: return 'C'
    histogram = [0] * 12
    for n in notes:
        pc = n['pitch'] % 12
        weight = (n['endTime'] - n['startTime']) * n['velocity']
        histogram[pc] += weight
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    key_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    best_key, best_corr = 'C', -1
    for shift in range(12):
        shifted = histogram[shift:] + histogram[:shift]
        corr = np.corrcoef(shifted, major_profile)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_key = key_names[shift]
    return best_key


def detect_tempo(notes):
    if len(notes) < 10: return 72
    onsets = sorted(set(round(n['startTime'], 2) for n in notes))
    if len(onsets) < 5: return 72
    intervals = [onsets[i+1] - onsets[i] for i in range(min(len(onsets)-1, 100)) if onsets[i+1] - onsets[i] > 0.05]
    if not intervals: return 72
    median_interval = sorted(intervals)[len(intervals)//2]
    bpm = 60.0 / (median_interval * 2)
    while bpm < 40: bpm *= 2
    while bpm > 180: bpm /= 2
    return int(round(bpm))


def notes_to_abc(notes, title='Untitled', split_threshold=55):
    if not notes: return ''
    key = detect_key(notes)
    tempo = detect_tempo(notes)
    rh_notes = [n for n in notes if n['pitch'] >= split_threshold]
    lh_notes = [n for n in notes if n['pitch'] < split_threshold]
    sec_per_16th = 60.0 / tempo / 4

    def render_voice(voice_notes):
        if not voice_notes: return 'z16|'
        voice_notes.sort(key=lambda n: n['startTime'])
        measures, measure_content, current_measure_ticks = [], [], 0
        ticks_per_measure = 16
        for note in voice_notes:
            tick = round(note['startTime'] / sec_per_16th)
            dur_ticks = max(1, round((note['endTime'] - note['startTime']) / sec_per_16th))
            gap = tick - current_measure_ticks
            while gap >= ticks_per_measure:
                remaining = ticks_per_measure - (current_measure_ticks % ticks_per_measure)
                if 0 < remaining < ticks_per_measure:
                    measure_content.append(f'z{remaining}')
                measures.append(' '.join(measure_content) if measure_content else f'z{ticks_per_measure}')
                measure_content = []
                current_measure_ticks += remaining
                gap = tick - current_measure_ticks
            if gap > 0:
                measure_content.append(f'z{gap}')
                current_measure_ticks += gap
            measure_content.append(f'{midi_to_abc_note(note["pitch"])}{quantize_duration(dur_ticks)}')
            current_measure_ticks += dur_ticks
            if current_measure_ticks % ticks_per_measure == 0 and measure_content:
                measures.append(' '.join(measure_content))
                measure_content = []
        if measure_content:
            remaining = ticks_per_measure - (current_measure_ticks % ticks_per_measure)
            if 0 < remaining < ticks_per_measure:
                measure_content.append(f'z{remaining}')
            measures.append(' '.join(measure_content))
        lines = []
        for i in range(0, len(measures), 4):
            lines.append('|'.join(measures[i:i+4]) + '|')
        return '\n'.join(lines) if lines else 'z16|'

    return f"""X:1
T:{title}
M:4/4
L:1/16
Q:1/4={tempo}
K:{key}
V:RH clef=treble
{render_voice(rh_notes)}
V:LH clef=bass
{render_voice(lh_notes)}"""


# ── Background transcription worker ─────────────────────────────────────────

def transcribe_worker(task_id, audio_url, title):
    try:
        print(f"[MaestrIA] [{task_id[:8]}] Downloading: {audio_url[:80]}...")
        resp = http_requests.get(audio_url, timeout=60)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(resp.content)
            audio_path = f.name

        print(f"[MaestrIA] [{task_id[:8]}] Downloaded {len(resp.content)} bytes")

        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        duration = len(audio) / sample_rate
        print(f"[MaestrIA] [{task_id[:8]}] Audio: {duration:.1f}s at {sample_rate}Hz")

        midi_path = audio_path.replace('.mp3', '.mid')
        print(f"[MaestrIA] [{task_id[:8]}] Transcribing...")

        with transcribe_lock:
            transcriptor.transcribe(audio, midi_path)

        midi_data = pretty_midi.PrettyMIDI(midi_path)
        all_notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                all_notes.append({
                    'pitch': note.pitch,
                    'startTime': round(note.start, 4),
                    'endTime': round(note.end, 4),
                    'velocity': note.velocity,
                })

        all_notes.sort(key=lambda n: (n['startTime'], n['pitch']))

        try: os.unlink(midi_path)
        except: pass
        try: os.unlink(audio_path)
        except: pass
        gc.collect()

        abc = notes_to_abc(all_notes, title)
        print(f"[MaestrIA] [{task_id[:8]}] Done: {len(all_notes)} notes, {len(abc)} chars ABC")

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
        print(f"[MaestrIA] [{task_id[:8]}] Failed: {e}")
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
        'device': DEVICE,
        'active_tasks': sum(1 for t in tasks.values() if t['status'] == 'processing'),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
