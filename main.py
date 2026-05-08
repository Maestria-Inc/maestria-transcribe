"""
Maestria Piano Transcription Service v4.3
==========================================
ByteDance transcription + post-processing + MuseScore PDF + Supabase Storage.

Pipeline:
  ByteDance AMT → MIDI brut
       ↓
  Post-processing (clamp durations, filter ghosts, reduce clusters)
       ↓
  pretty_midi → JSON notes (for falling notes in app)
       ↓
  MuseScore (MIDI → PDF directly)
       ↓
  Upload PDF to Supabase Storage → permanent public URL
       ↓
  Partitura → MusicXML (optional, best-effort)

Endpoints:
  POST /transcribe  { audioUrl, title?, compositionId? }  ->  { taskId }
  GET  /status?taskId=X  ->  { status, notes?, noteCount?, scorePdfUrl?, hasPdf? }
  GET  /health
"""

import os
import uuid
import tempfile
import traceback
import threading
import subprocess
from flask import Flask, request, jsonify, abort
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

# ── Supabase client ──────────────────────────────────────────────────────────
from supabase import create_client

SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_SERVICE_KEY = os.environ.get('SUPABASE_SERVICE_KEY', '')
SCORE_BUCKET = 'scores'

if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    print(f"[Maestria] Supabase connected: {SUPABASE_URL[:40]}...")
else:
    supabase = None
    print("[Maestria] WARNING: Supabase not configured — PDF upload disabled")

# ── Load ByteDance model ─────────────────────────────────────────────────────
from piano_transcription_inference import PianoTranscription, sample_rate

DEVICE = 'cuda' if os.environ.get('USE_CUDA') else 'cpu'
print(f"[Maestria] Loading ByteDance model on {DEVICE}...")
transcriptor = PianoTranscription(device=DEVICE, checkpoint_path=None)
print("[Maestria] Model loaded.")

transcribe_lock = threading.Lock()

# ── Load Partitura (optional) ────────────────────────────────────────────────
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
    Clean up ByteDance MIDI output.

    Constraints:
    - Ghost notes filtered (velocity < 35)
    - Micro-notes filtered (< 40ms)
    - Duration clamped (tempo-aware)
    - Max 3 sounding notes per hand at any time
    - Hand span max 15 semitones per sounding chord
    - Single-track output for MuseScore grand staff

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

    # ── Step 4: Unified playability filter ──
    MAX_SOUNDING = 3
    MAX_SPAN = 15
    SPLIT_POINT = 60

    all_notes.sort(key=lambda n: (n.start, -n.velocity))

    accepted = []
    seen_dedup = set()

    for n in all_notes:
        dedup_key = (n.pitch, round(n.start * 25))
        if dedup_key in seen_dedup:
            continue

        is_rh = n.pitch >= SPLIT_POINT
        active = [a for a in accepted
                  if a.end > n.start and (a.pitch >= SPLIT_POINT) == is_rh]

        if len(active) >= MAX_SOUNDING:
            continue

        if active:
            pitches = [a.pitch for a in active] + [n.pitch]
            span = max(pitches) - min(pitches)
            if span > MAX_SPAN:
                continue

        accepted.append(n)
        seen_dedup.add(dedup_key)

    all_notes = accepted

    print(f"[Maestria] After filtering: {len(all_notes)} notes")

    # ── Step 5: Write single-track MIDI ──
    clean_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0, name='Piano')

    for n in all_notes:
        piano.notes.append(pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch,
            start=n.start, end=n.end
        ))

    clean_midi.instruments.append(piano)

    clean_path = midi_path.replace('.mid', '_clean.mid')
    clean_midi.write(clean_path)

    rh_count = sum(1 for n in all_notes if n.pitch >= SPLIT_POINT)
    lh_count = len(all_notes) - rh_count
    print(f"[Maestria] Post-processed: {len(all_notes)} notes (RH≈{rh_count}, LH≈{lh_count}), tempo={tempo:.0f}")

    return clean_path, tempo


# ══════════════════════════════════════════════════════════════════════════════
# PDF — MuseScore + Supabase Storage
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_pdf(midi_path, task_id, title='Untitled'):
    """Generate PDF via MuseScore. Returns local path or None."""
    pdf_path = os.path.join(SCORES_DIR, f'{task_id}.pdf')

    if not mscore_cmd:
        print("[Maestria] MuseScore not available — skipping PDF")
        return None

    try:
        result = subprocess.run(
            ['xvfb-run', '-a', mscore_cmd, '-o', pdf_path, midi_path],
            capture_output=True, text=True, timeout=120
        )

        if os.path.exists(pdf_path):
            print(f"[Maestria] PDF generated: {pdf_path}")
            return pdf_path

        env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
        result2 = subprocess.run(
            [mscore_cmd, '-o', pdf_path, midi_path],
            capture_output=True, text=True, timeout=120, env=env
        )

        if os.path.exists(pdf_path):
            print(f"[Maestria] PDF generated (offscreen): {pdf_path}")
            return pdf_path

        print(f"[Maestria] MuseScore failed — stderr: {result.stderr[:500]}")
        return None

    except Exception as e:
        print(f"[Maestria] PDF generation failed: {e}")
        traceback.print_exc()
        return None


def upload_pdf_to_supabase(pdf_path, composition_id=None):
    """
    Upload PDF to Supabase Storage bucket 'scores'.
    Returns the public URL or None.
    Also updates the composition record if compositionId is provided.
    """
    if not supabase:
        print("[Maestria] Supabase not configured — skipping upload")
        return None

    try:
        filename = os.path.basename(pdf_path)
        storage_path = f"{filename}"

        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()

        # Upload to Supabase Storage
        res = supabase.storage.from_(SCORE_BUCKET).upload(
            storage_path,
            pdf_bytes,
            file_options={"content-type": "application/pdf", "upsert": "true"}
        )

        # Build public URL
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SCORE_BUCKET}/{storage_path}"
        print(f"[Maestria] PDF uploaded: {public_url}")

        # Update composition record if ID provided
        if composition_id:
            supabase.table('compositions').update({
                'score_pdf_url': public_url
            }).eq('id', composition_id).execute()
            print(f"[Maestria] Composition {composition_id[:8]} updated with PDF URL")

        return public_url

    except Exception as e:
        print(f"[Maestria] PDF upload failed: {e}")
        traceback.print_exc()
        return None


# ══════════════════════════════════════════════════════════════════════════════
# MusicXML — Partitura (best-effort, non-blocking)
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_musicxml(midi_path, title='Untitled'):
    if not HAS_PARTITURA:
        return ''
    try:
        score = pt.load_score_midi(
            midi_path,
            part_voice_assign_mode=0,
            quantization_unit=None,
            estimate_voice_info=False,
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

def transcribe_worker(task_id, audio_url, title, composition_id=None):
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

        # ── PDF via MuseScore ──
        print(f"[Maestria] [{task_id[:8]}] Generating PDF via MuseScore...")
        pdf_path = midi_to_pdf(clean_midi_path, task_id, title)
        has_pdf = pdf_path is not None and os.path.exists(pdf_path)
        print(f"[Maestria] [{task_id[:8]}] PDF: {'yes' if has_pdf else 'no'}")

        # ── Upload PDF to Supabase Storage ──
        score_pdf_url = None
        if has_pdf:
            print(f"[Maestria] [{task_id[:8]}] Uploading PDF to Supabase...")
            score_pdf_url = upload_pdf_to_supabase(pdf_path, composition_id)
            # Clean up local PDF after upload
            try: os.unlink(pdf_path)
            except: pass

        # ── MusicXML via Partitura (best-effort) ──
        print(f"[Maestria] [{task_id[:8]}] Generating MusicXML (best-effort)...")
        musicxml = midi_to_musicxml(clean_midi_path, title)
        print(f"[Maestria] [{task_id[:8]}] MusicXML: {len(musicxml)} chars")

        # ── Cleanup ──
        for p in [midi_path, clean_midi_path, audio_path]:
            try: os.unlink(p)
            except: pass
        gc.collect()

        print(f"[Maestria] [{task_id[:8]}] Complete: {len(all_notes)} notes, PDF URL: {score_pdf_url or 'none'}")

        tasks[task_id] = {
            'status': 'complete',
            'result': {
                'notes': all_notes,
                'musicxml': musicxml,
                'noteCount': len(all_notes),
                'hasPdf': has_pdf and score_pdf_url is not None,
                'scorePdfUrl': score_pdf_url,
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
    composition_id = data.get('compositionId')  # Optional — for direct DB update

    if not audio_url:
        return jsonify({'error': 'audioUrl required'}), 400

    task_id = str(uuid.uuid4())
    tasks[task_id] = {'status': 'processing'}

    thread = threading.Thread(
        target=transcribe_worker,
        args=(task_id, audio_url, title, composition_id)
    )
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
        'storage': 'supabase' if supabase else 'none',
        'musescore': mscore_ok,
        'partitura': HAS_PARTITURA,
        'device': DEVICE,
        'active_tasks': sum(1 for t in tasks.values() if t['status'] == 'processing'),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
