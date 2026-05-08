"""
Maestria Piano Transcription Service v4.2
==========================================
ByteDance transcription + post-processing + MuseScore PDF.

Pipeline:
  ByteDance AMT → MIDI brut
       ↓
  Post-processing (clamp durations, filter ghosts, reduce clusters)
       ↓
  pretty_midi → JSON notes (for falling notes in app)
       ↓
  MuseScore (MIDI → PDF directly — handles quantization, hand split, layout)
       ↓
  Partitura → MusicXML (optional, for in-app OSMD — non-blocking if it fails)

Outputs:
  - notes (JSON) — for falling notes display (post-processed)
  - musicxml (string) — for in-app OSMD rendering (best-effort)
  - pdf_url — served from this service for download

Endpoints:
  POST /transcribe  { audioUrl, title? }  ->  { taskId }
  GET  /status?taskId=X                   ->  { status, notes?, musicxml?, noteCount? }
  GET  /score/:taskId.pdf                 ->  PDF file
  GET  /health
"""

import os
import uuid
import tempfile
import traceback
import threading
import subprocess
from flask import Flask, request, jsonify, send_file, abort
from flask_cors import CORS
import requests as http_requests
import librosa
import numpy as np
import pretty_midi
import gc

app = Flask(__name__)
CORS(app)

tasks = {}
SCORES_DIR = tempfile.mkdtemp(prefix='maestria_scores_')

# ── Load ByteDance model ─────────────────────────────────────────────────────
from piano_transcription_inference import PianoTranscription, sample_rate

DEVICE = 'cuda' if os.environ.get('USE_CUDA') else 'cpu'
print(f"[Maestria] Loading ByteDance model on {DEVICE}...")
transcriptor = PianoTranscription(device=DEVICE, checkpoint_path=None)
print("[Maestria] Model loaded.")

transcribe_lock = threading.Lock()

# ── Load Partitura (optional — for MusicXML only) ───────────────────────────
try:
    import partitura as pt
    print("[Maestria] Partitura loaded.")
    HAS_PARTITURA = True
except ImportError:
    print("[Maestria] Partitura not available — MusicXML will be empty")
    HAS_PARTITURA = False

# ── Check MuseScore ──────────────────────────────────────────────────────────
mscore_cmd = None
for cmd in ['musescore', 'mscore', 'musescore4', 'musescore3']:
    result = subprocess.run(['which', cmd], capture_output=True, text=True)
    if result.stdout.strip():
        mscore_cmd = cmd
        print(f"[Maestria] MuseScore found: {result.stdout.strip()}")
        break
if not mscore_cmd:
    print("[Maestria] WARNING: MuseScore not found — PDF generation will fail")


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def postprocess_midi(midi_path, audio_path=None):
    """
    Clean up ByteDance MIDI output and split into RH/LH tracks.

    Constraints applied:
    - Ghost notes filtered (velocity < 35)
    - Micro-notes filtered (< 40ms)
    - Duration clamped (tempo-aware)
    - Max 4 simultaneous onset notes per hand
    - Hand span max 15 semitones (a 10th) per chord
    - Max 3 sustained notes per hand at any time
    - Split into RH/LH tracks for MuseScore

    Returns (clean_midi_path, detected_tempo).
    """
    midi = pretty_midi.PrettyMIDI(midi_path)

    # ── Tempo detection ──
    if audio_path:
        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if hasattr(tempo, '__len__'):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                tempo = float(tempo)

            if tempo > 140:
                tempo = tempo / 2.0
                print(f"[Maestria] Tempo halved (likely double-detected): {tempo:.0f} BPM")

            beat_duration = 60.0 / tempo
            print(f"[Maestria] Detected tempo: {tempo:.0f} BPM (beat = {beat_duration:.3f}s)")
        except Exception as e:
            print(f"[Maestria] Tempo detection failed, using default: {e}")
            tempo = 120.0
            beat_duration = 0.5
    else:
        tempo = 120.0
        beat_duration = 0.5

    # ── Max note duration ──
    if tempo < 72:
        max_duration = beat_duration * 4.0
    elif tempo > 120:
        max_duration = beat_duration * 2.0
    else:
        max_duration = beat_duration * 3.0

    max_duration = max(max_duration, 1.5)
    max_duration = min(max_duration, 6.0)

    print(f"[Maestria] Max note duration: {max_duration:.2f}s")

    # ── Collect all notes ──
    all_notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        all_notes.extend(instrument.notes)

    all_notes.sort(key=lambda n: (n.start, n.pitch))

    # ── Step 1: Filter ghost notes ──
    all_notes = [n for n in all_notes if n.velocity >= 35]

    # ── Step 2: Remove micro-notes ──
    all_notes = [n for n in all_notes if (n.end - n.start) >= 0.04]

    # ── Step 3: Clamp durations ──
    notes_by_pitch = {}
    for n in all_notes:
        notes_by_pitch.setdefault(n.pitch, []).append(n)

    for pitch, pnotes in notes_by_pitch.items():
        pnotes.sort(key=lambda n: n.start)
        for i, n in enumerate(pnotes):
            n.end = min(n.end, n.start + max_duration)
            if i + 1 < len(pnotes):
                n.end = min(n.end, pnotes[i + 1].start - 0.02)
            if n.end <= n.start:
                n.end = n.start + 0.05

    # ── Step 4: Reduce clusters + enforce hand span ──
    MAX_PER_HAND = 4
    MAX_SPAN = 15  # semitones — a 10th, generous but realistic
    SPLIT_POINT = 60

    onset_groups = []
    current_group = []
    for n in sorted(all_notes, key=lambda n: n.start):
        if not current_group or abs(n.start - current_group[0].start) <= 0.05:
            current_group.append(n)
        else:
            onset_groups.append(current_group)
            current_group = [n]
    if current_group:
        onset_groups.append(current_group)

    notes_to_remove = set()
    for group in onset_groups:
        rh = sorted([n for n in group if n.pitch >= SPLIT_POINT], key=lambda n: -n.velocity)
        lh = sorted([n for n in group if n.pitch < SPLIT_POINT], key=lambda n: -n.velocity)

        # Limit count per hand
        for excess in rh[MAX_PER_HAND:]:
            notes_to_remove.add(id(excess))
        for excess in lh[MAX_PER_HAND:]:
            notes_to_remove.add(id(excess))

        # Enforce hand span — keep highest-velocity notes within span
        for hand_notes in [rh[:MAX_PER_HAND], lh[:MAX_PER_HAND]]:
            if len(hand_notes) < 2:
                continue
            # Sort by pitch to check span
            by_pitch = sorted(hand_notes, key=lambda n: n.pitch)
            while len(by_pitch) > 1 and (by_pitch[-1].pitch - by_pitch[0].pitch) > MAX_SPAN:
                # Remove the note with lowest velocity from extremes
                if by_pitch[0].velocity <= by_pitch[-1].velocity:
                    notes_to_remove.add(id(by_pitch.pop(0)))
                else:
                    notes_to_remove.add(id(by_pitch.pop(-1)))

    all_notes = [n for n in all_notes if id(n) not in notes_to_remove]

    # ── Step 5: Deduplicate ──
    deduped = []
    seen = set()
    for n in sorted(all_notes, key=lambda n: (n.start, n.pitch, -n.velocity)):
        key = (n.pitch, round(n.start * 25))
        if key not in seen:
            seen.add(key)
            deduped.append(n)
    all_notes = deduped

    # ── Step 6: Limit sustained polyphony (max 3 per hand) ──
    # This is the key constraint for readability: no more than 3 notes
    # sounding at the same time per hand. This prevents MuseScore from
    # creating 4 independent voices with competing stems/beams.
    MAX_SUSTAINED = 3
    all_notes.sort(key=lambda n: n.start)
    final_notes = []
    for n in all_notes:
        is_rh = n.pitch >= SPLIT_POINT
        active = sum(1 for fn in final_notes
                     if fn.start <= n.start < fn.end
                     and (fn.pitch >= SPLIT_POINT) == is_rh)
        if active < MAX_SUSTAINED:
            final_notes.append(n)
    all_notes = final_notes

    print(f"[Maestria] After filtering: {len(all_notes)} notes")

    # ── Step 7: Split into two tracks (RH / LH) ──
    clean_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

    rh_inst = pretty_midi.Instrument(program=0, name='Piano RH')
    lh_inst = pretty_midi.Instrument(program=0, name='Piano LH')

    for n in all_notes:
        note_copy = pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch,
            start=n.start, end=n.end
        )
        if n.pitch >= SPLIT_POINT:
            rh_inst.notes.append(note_copy)
        else:
            lh_inst.notes.append(note_copy)

    clean_midi.instruments.append(rh_inst)
    clean_midi.instruments.append(lh_inst)

    clean_path = midi_path.replace('.mid', '_clean.mid')
    clean_midi.write(clean_path)

    total = len(rh_inst.notes) + len(lh_inst.notes)
    print(f"[Maestria] Post-processed: {total} notes (RH={len(rh_inst.notes)}, LH={len(lh_inst.notes)}), tempo={tempo:.0f}")

    return clean_path, tempo


# ══════════════════════════════════════════════════════════════════════════════
# PDF — MuseScore imports MIDI directly
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_pdf(midi_path, task_id, title='Untitled'):
    """
    Let MuseScore handle everything: MIDI import, quantization, hand split, layout, PDF.
    MuseScore has 15+ years of MIDI import logic — no need to reinvent it.
    """
    pdf_path = os.path.join(SCORES_DIR, f'{task_id}.pdf')

    if not mscore_cmd:
        print("[Maestria] MuseScore not available — skipping PDF")
        return None

    try:
        # Primary: xvfb-run for headless rendering
        result = subprocess.run(
            ['xvfb-run', '-a', mscore_cmd, '-o', pdf_path, midi_path],
            capture_output=True, text=True, timeout=120
        )

        if os.path.exists(pdf_path):
            print(f"[Maestria] PDF generated: {pdf_path}")
            return pdf_path

        # Fallback: QT offscreen
        env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
        result2 = subprocess.run(
            [mscore_cmd, '-o', pdf_path, midi_path],
            capture_output=True, text=True, timeout=120, env=env
        )

        if os.path.exists(pdf_path):
            print(f"[Maestria] PDF generated (offscreen): {pdf_path}")
            return pdf_path

        print(f"[Maestria] MuseScore failed — stderr: {result.stderr[:500]}")
        print(f"[Maestria] MuseScore fallback stderr: {result2.stderr[:500]}")
        return None

    except Exception as e:
        print(f"[Maestria] PDF generation failed: {e}")
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MusicXML — Partitura (best-effort, non-blocking)
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_musicxml(midi_path, title='Untitled'):
    """
    Try to produce MusicXML via Partitura. If it fails, return empty string.
    This is used for in-app OSMD display — the PDF is the critical output.
    """
    if not HAS_PARTITURA:
        return ''

    try:
        score = pt.load_score_midi(
            midi_path,
            part_voice_assign_mode=0,
            quantization_unit=None,
            estimate_voice_info=False,  # Disabled — crashes on AMT output
            estimate_key=False,
            assign_note_ids=True,
        )

        with tempfile.NamedTemporaryFile(suffix='.musicxml', delete=False) as f:
            xml_path = f.name

        pt.save_musicxml(score, xml_path)

        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()

        os.unlink(xml_path)
        return xml_content

    except Exception as e:
        print(f"[Maestria] MusicXML generation failed (non-critical): {e}")
        return ''


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND WORKER
# ══════════════════════════════════════════════════════════════════════════════

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
        print(f"[Maestria] [{task_id[:8]}] Audio: {duration:.1f}s")

        midi_path = audio_path.replace('.mp3', '.mid')
        print(f"[Maestria] [{task_id[:8]}] Transcribing audio...")

        with transcribe_lock:
            transcriptor.transcribe(audio, midi_path)

        # ── Post-process ──
        print(f"[Maestria] [{task_id[:8]}] Post-processing MIDI...")
        clean_midi_path, detected_tempo = postprocess_midi(midi_path, audio_path)

        # ── JSON notes for falling display ──
        midi_data = pretty_midi.PrettyMIDI(clean_midi_path)
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
        print(f"[Maestria] [{task_id[:8]}] {len(all_notes)} notes extracted")

        # ── PDF via MuseScore (critical path) ──
        print(f"[Maestria] [{task_id[:8]}] Generating PDF via MuseScore...")
        pdf_path = midi_to_pdf(clean_midi_path, task_id, title)
        has_pdf = pdf_path is not None and os.path.exists(pdf_path)
        print(f"[Maestria] [{task_id[:8]}] PDF: {'yes' if has_pdf else 'no'}")

        # ── MusicXML via Partitura (best-effort) ──
        print(f"[Maestria] [{task_id[:8]}] Generating MusicXML (best-effort)...")
        musicxml = midi_to_musicxml(clean_midi_path, title)
        print(f"[Maestria] [{task_id[:8]}] MusicXML: {len(musicxml)} chars")

        # ── Cleanup ──
        for p in [midi_path, clean_midi_path, audio_path]:
            try: os.unlink(p)
            except: pass
        gc.collect()

        print(f"[Maestria] [{task_id[:8]}] Complete: {len(all_notes)} notes")

        tasks[task_id] = {
            'status': 'complete',
            'result': {
                'notes': all_notes,
                'musicxml': musicxml,
                'abc': '',  # Deprecated
                'noteCount': len(all_notes),
                'hasPdf': has_pdf,
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


@app.route('/score/<task_id>.pdf', methods=['GET'])
def serve_pdf(task_id):
    pdf_path = os.path.join(SCORES_DIR, f'{task_id}.pdf')
    if os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype='application/pdf', download_name=f'score_{task_id[:8]}.pdf')
    abort(404)


@app.route('/health', methods=['GET'])
def health():
    mscore_ok = False
    try:
        result = subprocess.run(
            [mscore_cmd or 'musescore', '--version'],
            capture_output=True, text=True, timeout=5
        )
        mscore_ok = result.returncode == 0
    except:
        pass

    return jsonify({
        'status': 'ok',
        'model': 'bytedance/piano_transcription',
        'pdf_renderer': 'musescore',
        'musescore': mscore_ok,
        'partitura': HAS_PARTITURA,
        'device': DEVICE,
        'active_tasks': sum(1 for t in tasks.values() if t['status'] == 'processing'),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
