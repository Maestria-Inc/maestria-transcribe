"""
Maestria Piano Transcription Service v3.1
==========================================
ByteDance transcription + music21 notation + LilyPond PDF rendering.

Outputs:
  - notes (JSON) — for falling notes display
  - musicxml (string) — for in-app OSMD rendering
  - pdf_url — served from this service for download

Endpoints:
  POST /transcribe  { audioUrl, title? }  ->  { taskId }
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
import base64
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

# ── Load music21 ─────────────────────────────────────────────────────────────
import music21
from music21 import converter, instrument, stream, tempo, key, meter, note, chord, clef

# Tell music21 where LilyPond is
lily_path = subprocess.run(['which', 'lilypond'], capture_output=True, text=True).stdout.strip()
if lily_path:
    music21.environment.set('lilypondPath', lily_path)
    print(f"[Maestria] LilyPond found: {lily_path}")
else:
    print("[Maestria] WARNING: LilyPond not found — PDF generation will fail")

print("[Maestria] music21 loaded.")


# ── MIDI → music21 Score ─────────────────────────────────────────────────────

def midi_to_score(midi_path, title='Untitled'):
    """Let music21 parse the MIDI directly and clean up the result."""
    
    # Let music21 handle the MIDI parsing natively
    score = converter.parse(midi_path)
    
    # Set metadata
    score.metadata = music21.metadata.Metadata()
    score.metadata.title = title
    score.metadata.composer = 'Maestria'
    
    # Quantize
    score.quantize(inPlace=True)
    
    # Fix all micro-durations recursively
    MIN_QL = 0.25
    for el in score.recurse():
        if hasattr(el, 'quarterLength'):
            if 0 < el.quarterLength < MIN_QL:
                el.quarterLength = MIN_QL
    
    # If single part, split into treble/bass
    parts = score.parts
    if len(parts) == 1:
        original = parts[0]
        
        treble = stream.Part()
        treble.id = 'RH'
        treble.insert(0, clef.TrebleClef())
        treble.insert(0, instrument.Piano())
        
        bass = stream.Part()
        bass.id = 'LH'  
        bass.insert(0, clef.BassClef())
        bass.insert(0, instrument.Piano())
        
        for m in original.getElementsByClass(stream.Measure):
            tm = stream.Measure(number=m.number)
            bm = stream.Measure(number=m.number)
            
            for el in m.notesAndRests:
                if isinstance(el, note.Rest):
                    continue
                elif isinstance(el, note.Note):
                    n_copy = el.transpose(0)  # deep copy
                    if el.pitch.midi >= 60:
                        tm.insert(el.offset, n_copy)
                    else:
                        bm.insert(el.offset, n_copy)
                elif isinstance(el, chord.Chord):
                    upper = [p for p in el.pitches if p.midi >= 60]
                    lower = [p for p in el.pitches if p.midi < 60]
                    if upper:
                        c = chord.Chord(upper)
                        c.quarterLength = el.quarterLength
                        tm.insert(el.offset, c)
                    if lower:
                        c = chord.Chord(lower)
                        c.quarterLength = el.quarterLength
                        bm.insert(el.offset, c)
            
            # Copy time signature, key signature, tempo from first measure
            for sig in m.getElementsByClass((meter.TimeSignature, key.KeySignature, tempo.MetronomeMark)):
                tm.insert(sig.offset, sig)
                bm.insert(sig.offset, sig)
            
            treble.append(tm)
            bass.append(bm)
        
        # Build new score
        from music21 import layout
        score = stream.Score()
        score.metadata = music21.metadata.Metadata()
        score.metadata.title = title
        score.metadata.composer = 'Maestria'
        
        sg = layout.StaffGroup([treble, bass], symbol='brace', barTogether=True)
        score.insert(0, treble)
        score.insert(0, bass)
        score.insert(0, sg)
    
    # Final cleanup pass
    for el in score.recurse():
        if hasattr(el, 'quarterLength') and 0 < el.quarterLength < MIN_QL:
            el.quarterLength = MIN_QL
    
    return score


def _quantize_ql(ql):
    """Quantize quarter length to nearest standard musical duration."""
    standard = [0.25, 0.375, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    if ql < 0.125:
        return 0.25
    closest = min(standard, key=lambda s: abs(s - ql))
    return closest


def score_to_musicxml(score):
    """Export score to MusicXML string."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.musicxml', delete=False) as f:
            xml_path = f.name
        score.write('musicxml', fp=xml_path)
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_content = f.read()
        os.unlink(xml_path)
        return xml_content
    except Exception as e:
        print(f"[Maestria] MusicXML export failed: {e}")
        return ''


def score_to_pdf(score, task_id):
    """Export score to PDF via LilyPond with proper page layout."""
    pdf_path = os.path.join(SCORES_DIR, f'{task_id}.pdf')
    output_base = os.path.join(SCORES_DIR, task_id)
    ly_path = os.path.join(SCORES_DIR, f'{task_id}.ly')

    try:
        # Write LilyPond source
        score.write('lily', fp=ly_path)

        # Read the generated file
        with open(ly_path, 'r', encoding='utf-8', errors='replace') as f:
            ly_content = f.read()

        # CRITICAL: Remove lilypond-book-preamble — it overrides page layout
        # and forces compact single-system formatting
        ly_content = ly_content.replace('\\include "lilypond-book-preamble.ly"', '')

        # Also remove the color function definition that music21 adds (unnecessary)
        import re
        ly_content = re.sub(r'color = #\(define-music-function.*?\n.*?\n.*?\n', '', ly_content, flags=re.DOTALL)

        # Inject paper and layout right after \version
        paper_and_layout = r"""
\paper {
  #(set-paper-size "a4")
  indent = 12\mm
  short-indent = 5\mm
  top-margin = 15\mm
  bottom-margin = 15\mm
  left-margin = 15\mm
  right-margin = 15\mm
  system-system-spacing = #'((basic-distance . 16) (padding . 3))
  score-markup-spacing = #'((basic-distance . 12))
  ragged-last-bottom = ##t
}

\header {
  tagline = \markup { \small \italic "© 2026 Maestria — Original Composition" }
}
"""
        # Insert after \version line
        version_end = ly_content.find('\n', ly_content.find('\\version'))
        if version_end > 0:
            ly_content = ly_content[:version_end+1] + paper_and_layout + ly_content[version_end+1:]
        else:
            ly_content = paper_and_layout + ly_content

        with open(ly_path, 'w', encoding='utf-8') as f:
            f.write(ly_content)

        # Run LilyPond
        result = subprocess.run(
            ['lilypond', '-dno-point-and-click', '--pdf', '-o', output_base, ly_path],
            capture_output=True, text=True, timeout=120
        )

        # Clean up .ly file
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
        return None


# ── Also keep simple ABC for backward compatibility ──────────────────────────

def score_to_abc(score):
    """Try to export ABC from music21. Non-critical — MusicXML is primary."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.abc', delete=False) as f:
            abc_path = f.name
        converter.freeze(score, fmt='abc', fp=abc_path)
        with open(abc_path, 'r') as f:
            abc_text = f.read()
        os.unlink(abc_path)
        # Validate it's actual ABC, not a Python repr
        if abc_text.startswith('<') or 'music21' in abc_text:
            return ''
        return abc_text
    except Exception as e:
        print(f"[Maestria] ABC export failed (non-critical): {e}")
        return ''


# ── Background worker ────────────────────────────────────────────────────────

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

        # ── Notes for falling display ──
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
        print(f"[Maestria] [{task_id[:8]}] {len(all_notes)} notes extracted")

        # ── Notation via music21 ──
        print(f"[Maestria] [{task_id[:8]}] Generating notation...")
        score = midi_to_score(midi_path, title)

        musicxml = score_to_musicxml(score)
        print(f"[Maestria] [{task_id[:8]}] MusicXML: {len(musicxml)} chars")

        abc = score_to_abc(score)

        # PDF generation
        pdf_path = score_to_pdf(score, task_id)
        has_pdf = pdf_path is not None and os.path.exists(pdf_path)
        print(f"[Maestria] [{task_id[:8]}] PDF: {'yes' if has_pdf else 'no'}")

        # Cleanup source files
        try: os.unlink(midi_path)
        except: pass
        try: os.unlink(audio_path)
        except: pass
        gc.collect()

        print(f"[Maestria] [{task_id[:8]}] Complete: {len(all_notes)} notes")

        tasks[task_id] = {
            'status': 'complete',
            'result': {
                'notes': all_notes,
                'musicxml': musicxml,
                'abc': abc,
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
    # Check LilyPond
    lily_ok = False
    try:
        result = subprocess.run(['lilypond', '--version'], capture_output=True, text=True, timeout=5)
        lily_ok = result.returncode == 0
    except:
        pass

    return jsonify({
        'status': 'ok',
        'model': 'bytedance/piano_transcription',
        'notation': 'music21',
        'lilypond': lily_ok,
        'device': DEVICE,
        'active_tasks': sum(1 for t in tasks.values() if t['status'] == 'processing'),
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
