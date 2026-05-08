"""
Maestria Piano Transcription Service v4.0
==========================================
ByteDance transcription + post-processing + Partitura notation + LilyPond PDF.

Pipeline:
  ByteDance AMT → MIDI brut
       ↓
  Post-processing (clamp durations, filter ghosts, reduce clusters, tempo-aware)
       ↓
  pretty_midi → JSON notes (for falling notes in app)
       ↓
  Partitura (quantization + voice separation + pitch spelling + key estimation)
       ↓
  MusicXML export → LilyPond PDF

Outputs:
  - notes (JSON) — for falling notes display (post-processed)
  - musicxml (string) — for in-app OSMD rendering
  - pdf_url — served from this service for download

Endpoints:
  POST /transcribe  { audioUrl, title? }  ->  { taskId}
  GET  /status?taskId=X                   ->  { status, notes?, musicxml?, noteCount? }
  GET  /score/:taskId.pdf                 ->  PDF file
  GET  /health
"""

import os
import json
import uuid
import tempfile
import traceback
import threading
import subprocess
import re
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

# ── Load Partitura ───────────────────────────────────────────────────────────
import partitura as pt

print("[Maestria] Partitura loaded.")

# Check LilyPond
lily_path = subprocess.run(['which', 'lilypond'], capture_output=True, text=True).stdout.strip()
if lily_path:
    print(f"[Maestria] LilyPond found: {lily_path}")
else:
    print("[Maestria] WARNING: LilyPond not found — PDF generation will fail")


# ══════════════════════════════════════════════════════════════════════════════
# POST-PROCESSING — The critical layer between AMT and notation
# ══════════════════════════════════════════════════════════════════════════════

def postprocess_midi(midi_path, audio_path=None):
    """
    Clean up ByteDance MIDI output for both falling notes display and notation.
    
    Problems solved:
    1. Notes too long (sustain/reverb captured as duration)
    2. Ghost notes (low velocity harmonics/artifacts)
    3. Impossible clusters (>6 simultaneous notes per hand)
    4. Overlapping notes on same pitch
    
    Returns the path to the cleaned MIDI file.
    """
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    # Estimate tempo from audio if available, otherwise from MIDI
    if audio_path:
        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            # librosa can return an array; extract scalar
            if hasattr(tempo, '__len__'):
                tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
            else:
                tempo = float(tempo)
            beat_duration = 60.0 / tempo
            print(f"[Maestria] Detected tempo: {tempo:.0f} BPM (beat = {beat_duration:.3f}s)")
        except Exception as e:
            print(f"[Maestria] Tempo detection failed, using default: {e}")
            tempo = 120.0
            beat_duration = 0.5
    else:
        tempo = 120.0
        beat_duration = 0.5
    
    # Max note duration: 2 beats (context-aware)
    # For slow pieces (tempo < 72), allow up to 3 beats
    # For fast pieces (tempo > 100), cap at 1.5 beats
    if tempo < 72:
        max_duration = beat_duration * 3.0
    elif tempo > 100:
        max_duration = beat_duration * 1.5
    else:
        max_duration = beat_duration * 2.0
    
    # Absolute ceiling: 4 seconds regardless of tempo
    max_duration = min(max_duration, 4.0)
    
    print(f"[Maestria] Max note duration: {max_duration:.2f}s")
    
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        
        # Sort by start time then pitch
        instrument.notes.sort(key=lambda n: (n.start, n.pitch))
        
        # ── Step 1: Filter ghost notes (low velocity artifacts) ──
        instrument.notes = [n for n in instrument.notes if n.velocity >= 25]
        
        # ── Step 2: Remove very short notes (< 30ms — likely onset artifacts) ──
        instrument.notes = [n for n in instrument.notes if (n.end - n.start) >= 0.03]
        
        # ── Step 3: Clamp durations ──
        # For each note, the end time is the minimum of:
        #   a) its current end time
        #   b) the start of the next note on the same pitch (minus tiny gap)
        #   c) max_duration after its start
        notes_by_pitch = {}
        for n in instrument.notes:
            notes_by_pitch.setdefault(n.pitch, []).append(n)
        
        for pitch, pnotes in notes_by_pitch.items():
            pnotes.sort(key=lambda n: n.start)
            for i, n in enumerate(pnotes):
                # Cap at max duration
                n.end = min(n.end, n.start + max_duration)
                # Don't overlap with next note on same pitch
                if i + 1 < len(pnotes):
                    next_start = pnotes[i + 1].start
                    # Leave a tiny gap (20ms) for note separation
                    n.end = min(n.end, next_start - 0.02)
                # Ensure minimum duration after clamping
                if n.end <= n.start:
                    n.end = n.start + 0.05
        
        # ── Step 4: Reduce impossible clusters ──
        # No more than 5 simultaneous notes per hand (split at MIDI 60 = C4)
        # If more, keep the ones with highest velocity
        MAX_PER_HAND = 5
        SPLIT_POINT = 60
        
        # Group notes by approximate onset (within 50ms)
        onset_groups = []
        current_group = []
        for n in sorted(instrument.notes, key=lambda n: n.start):
            if not current_group or abs(n.start - current_group[0].start) <= 0.05:
                current_group.append(n)
            else:
                onset_groups.append(current_group)
                current_group = [n]
        if current_group:
            onset_groups.append(current_group)
        
        notes_to_remove = set()
        for group in onset_groups:
            # Split into hands
            rh = sorted([n for n in group if n.pitch >= SPLIT_POINT], key=lambda n: -n.velocity)
            lh = sorted([n for n in group if n.pitch < SPLIT_POINT], key=lambda n: -n.velocity)
            
            # Mark excess notes for removal
            for excess in rh[MAX_PER_HAND:]:
                notes_to_remove.add(id(excess))
            for excess in lh[MAX_PER_HAND:]:
                notes_to_remove.add(id(excess))
        
        instrument.notes = [n for n in instrument.notes if id(n) not in notes_to_remove]
        
        # ── Step 5: Deduplicate near-simultaneous same-pitch notes ──
        # If two notes on the same pitch start within 40ms, keep the louder one
        deduped = []
        seen = set()
        for n in sorted(instrument.notes, key=lambda n: (n.start, n.pitch, -n.velocity)):
            key = (n.pitch, round(n.start * 25))  # ~40ms buckets
            if key not in seen:
                seen.add(key)
                deduped.append(n)
        instrument.notes = deduped
    
    # Write cleaned MIDI
    clean_path = midi_path.replace('.mid', '_clean.mid')
    
    # Set tempo in the MIDI file (important for Partitura quantization)
    # Remove existing tempo changes and set a single tempo
    midi.tempo_changes = []  # Clear — we'll set via initial tempo
    # pretty_midi uses initial_tempo in the constructor, but we can adjust
    # by writing a tempo change at time 0
    # The simplest approach: write to a new PrettyMIDI with correct tempo
    clean_midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    for inst in midi.instruments:
        clean_inst = pretty_midi.Instrument(program=inst.program, is_drum=inst.is_drum, name=inst.name)
        clean_inst.notes = inst.notes
        clean_inst.control_changes = inst.control_changes
        clean_midi.instruments.append(clean_inst)
    
    clean_midi.write(clean_path)
    
    # Stats
    total_notes = sum(len(inst.notes) for inst in clean_midi.instruments)
    print(f"[Maestria] Post-processed: {total_notes} notes, tempo={tempo:.0f}")
    
    return clean_path, tempo


# ══════════════════════════════════════════════════════════════════════════════
# NOTATION — Partitura-based pipeline
# ══════════════════════════════════════════════════════════════════════════════

def midi_to_score_partitura(midi_path, title='Untitled'):
    """
    Load MIDI as a score via Partitura with:
    - Automatic quantization
    - Voice separation (Chew & Wu algorithm)
    - Pitch spelling (PS13 algorithm)
    - Key estimation (Krumhansl profiles)
    
    Returns a Partitura Score object.
    """
    score = pt.load_score_midi(
        midi_path,
        part_voice_assign_mode=0,    # All notes in one Part, voices assigned by algorithm
        quantization_unit=None,       # Auto-detect best quantization grid
        estimate_voice_info=True,     # Chew & Wu voice separation
        estimate_key=True,            # Krumhansl key profiles
        assign_note_ids=True,
    )
    
    return score


def score_to_musicxml_partitura(score, title='Untitled'):
    """Export Partitura score to MusicXML string."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.musicxml', delete=False, mode='w') as f:
            xml_path = f.name
        
        pt.save_musicxml(score, xml_path)
        
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        
        # Inject title and composer into MusicXML if not present
        if '<work-title>' not in xml_content and '<movement-title>' not in xml_content:
            # Add identification block after the first <score-partwise> tag
            ident_block = f"""
  <work>
    <work-title>{title}</work-title>
  </work>
  <identification>
    <creator type="composer">Maestria</creator>
  </identification>"""
            xml_content = xml_content.replace(
                '<score-partwise',
                f'<score-partwise',
                1
            )
            # Insert after the opening tag
            insert_pos = xml_content.find('>', xml_content.find('<score-partwise')) + 1
            xml_content = xml_content[:insert_pos] + ident_block + xml_content[insert_pos:]
        
        os.unlink(xml_path)
        return xml_content
    except Exception as e:
        print(f"[Maestria] MusicXML export failed: {e}")
        traceback.print_exc()
        return ''


def score_to_pdf_lilypond(score, task_id, title='Untitled'):
    """
    Export Partitura score to PDF via LilyPond.
    
    Pipeline: Partitura → MusicXML → music21 (only for LilyPond export) → .ly → LilyPond → PDF
    
    Alternative: Partitura → MusicXML → MuseScore CLI → PDF
    But LilyPond gives better typesetting quality.
    """
    pdf_path = os.path.join(SCORES_DIR, f'{task_id}.pdf')
    output_base = os.path.join(SCORES_DIR, task_id)
    ly_path = os.path.join(SCORES_DIR, f'{task_id}.ly')
    xml_path = os.path.join(SCORES_DIR, f'{task_id}_temp.musicxml')
    
    try:
        # Step 1: Save Partitura score as MusicXML
        pt.save_musicxml(score, xml_path)
        
        # Step 2: Use music21 ONLY for the LilyPond conversion
        # (Partitura doesn't have a native LilyPond exporter)
        import music21
        from music21 import converter as m21_converter
        
        m21_score = m21_converter.parse(xml_path)
        
        # Set metadata
        m21_score.metadata = music21.metadata.Metadata()
        m21_score.metadata.title = title
        m21_score.metadata.composer = 'Maestria'
        
        # Write LilyPond source
        m21_score.write('lily', fp=ly_path)
        
        # Clean up temp XML
        try: os.unlink(xml_path)
        except: pass
        
        # Step 3: Post-process the .ly file
        with open(ly_path, 'r', encoding='utf-8', errors='replace') as f:
            ly_content = f.read()
        
        # Remove lilypond-book-preamble (forces compact layout)
        ly_content = ly_content.replace('\\include "lilypond-book-preamble.ly"', '')
        
        # Remove music21's color function definition
        ly_content = re.sub(
            r'color = #\(define-music-function.*?#\}\)', '', ly_content, flags=re.DOTALL
        )
        
        # Wrap staves in PianoStaff for brace + connected barlines
        ly_content = ly_content.replace(
            '\\score  { \n      <<',
            '\\score  { \n      \\new PianoStaff <<'
        )
        ly_content = ly_content.replace(
            '\\score  {\n      <<',
            '\\score  {\n      \\new PianoStaff <<'
        )
        ly_content = ly_content.replace(
            '\\score { \n<<',
            '\\score { \n\\new PianoStaff <<'
        )
        
        # Inject layout block
        layout_block = r"""
  \layout {
    \context {
      \Staff
      \override VerticalAxisGroup.remove-empty = ##f
      \override VerticalAxisGroup.remove-first = ##f
    }
  }
"""
        last_brace = ly_content.rfind('}')
        if last_brace > 0:
            ly_content = ly_content[:last_brace] + layout_block + ly_content[last_brace:]
        
        # Paper block
        paper_block = r"""
\paper {
  #(set-paper-size "a4")
  indent = 12\mm
  short-indent = 5\mm
  top-margin = 15\mm
  bottom-margin = 15\mm
  left-margin = 15\mm
  right-margin = 15\mm
  system-system-spacing = #'((basic-distance . 16) (padding . 3))
  ragged-last-bottom = ##t
}
"""
        # Replace header
        ly_content = re.sub(
            r'\\header\s*\{[^}]*\}',
            r'\\header {\n  title = "' + title.replace('"', '\\"') +
            r'"\n  composer = "Maestria"\n  tagline = \\markup { \\small \\italic "© 2026 Maestria — Original Composition" }\n}',
            ly_content
        )
        
        # Insert paper after \version line
        version_end = ly_content.find('\n', ly_content.find('\\version'))
        if version_end > 0:
            ly_content = ly_content[:version_end + 1] + paper_block + ly_content[version_end + 1:]
        
        with open(ly_path, 'w', encoding='utf-8') as f:
            f.write(ly_content)
        
        # Step 4: Run LilyPond
        result = subprocess.run(
            ['lilypond', '-dno-point-and-click', '--pdf', '-o', output_base, ly_path],
            capture_output=True, text=True, timeout=120
        )
        
        try: os.unlink(ly_path)
        except: pass
        
        if os.path.exists(pdf_path):
            print(f"[Maestria] PDF generated: {pdf_path}")
            return pdf_path
        
        print(f"[Maestria] LilyPond failed — stderr: {result.stderr[:500]}")
        return None
        
    except Exception as e:
        print(f"[Maestria] PDF generation failed: {e}")
        traceback.print_exc()
        # Cleanup
        for p in [xml_path, ly_path]:
            try: os.unlink(p)
            except: pass
        return None


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

        # ── Post-process the MIDI ──
        print(f"[Maestria] [{task_id[:8]}] Post-processing MIDI...")
        clean_midi_path, detected_tempo = postprocess_midi(midi_path, audio_path)

        # ── Notes for falling display (from cleaned MIDI) ──
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

        # ── Notation via Partitura ──
        print(f"[Maestria] [{task_id[:8]}] Generating notation via Partitura...")
        score = midi_to_score_partitura(clean_midi_path, title)

        musicxml = score_to_musicxml_partitura(score, title)
        print(f"[Maestria] [{task_id[:8]}] MusicXML: {len(musicxml)} chars")

        # PDF generation
        pdf_path = score_to_pdf_lilypond(score, task_id, title)
        has_pdf = pdf_path is not None and os.path.exists(pdf_path)
        print(f"[Maestria] [{task_id[:8]}] PDF: {'yes' if has_pdf else 'no'}")

        # Cleanup source files
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
                'abc': '',  # Deprecated — kept for backward compat
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
    """Serve the generated PDF score."""
    pdf_path = os.path.join(SCORES_DIR, f'{task_id}.pdf')
    if os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype='application/pdf', download_name=f'score_{task_id[:8]}.pdf')
    abort(404)


@app.route('/health', methods=['GET'])
def health():
    lily_ok = False
    try:
        result = subprocess.run(['lilypond', '--version'], capture_output=True, text=True, timeout=5)
        lily_ok = result.returncode == 0
    except:
        pass

    return jsonify({
        'status': 'ok',
        'model': 'bytedance/piano_transcription',
        'notation': 'partitura',
        'lilypond': lily_ok,
        'device': DEVICE,
        'active_tasks': sum(1 for t in tasks.values() if t['status'] == 'processing'),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
