"""
MaestrIA Piano Transcription Service v2
========================================
ByteDance piano_transcription_inference — trained on MAESTRO dataset.
Replaces Basic Pitch for significantly better piano-specific accuracy.

Endpoint: POST /transcribe { audioUrl, title? }
Returns:  { notes: [...], abc: "..." }

Each note: { pitch, startTime, endTime, velocity }
Same format as v1 (Basic Pitch) — drop-in replacement.
"""

import os
import json
import tempfile
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import librosa
import numpy as np
import pretty_midi

import gc

app = Flask(__name__)
CORS(app)

# ── Load ByteDance model at startup ──────────────────────────────────────────
from piano_transcription_inference import PianoTranscription, sample_rate

# Use CPU on Railway (no GPU tier needed for inference on ~2min audio)
DEVICE = 'cuda' if os.environ.get('USE_CUDA') else 'cpu'
print(f"[MaestrIA] Loading ByteDance piano transcription model on {DEVICE}...")
transcriptor = PianoTranscription(device=DEVICE, checkpoint_path=None)
print("[MaestrIA] Model loaded successfully.")


# ── ABC generation (pure Python, no music21) ─────────────────────────────────

NOTE_NAMES = ['C', '^C', 'D', '^D', 'E', 'F', '^F', 'G', '^G', 'A', '^A', 'B']

def midi_to_abc_note(midi_pitch):
    """Convert MIDI pitch to ABC notation."""
    octave = midi_pitch // 12 - 1
    note_idx = midi_pitch % 12
    name = NOTE_NAMES[note_idx]
    
    if octave >= 5:
        # Lowercase for octave 5+
        base = name[0].lower() if name[0] != '^' else '^' + name[1].lower()
        extra = octave - 5
        return base + "'" * extra
    elif octave == 4:
        return name
    else:
        extra = 4 - octave
        return name + "," * extra


def quantize_duration(dur_in_sixteenths):
    """Quantize to nearest musical duration in ABC."""
    if dur_in_sixteenths <= 0:
        return '1'  # sixteenth
    
    # Map to standard durations
    if dur_in_sixteenths < 1.5:
        return ''       # sixteenth (L:1/16 default)
    elif dur_in_sixteenths < 3:
        return '2'      # eighth
    elif dur_in_sixteenths < 5:
        return '4'      # quarter
    elif dur_in_sixteenths < 7:
        return '6'      # dotted quarter
    elif dur_in_sixteenths < 10:
        return '8'      # half
    elif dur_in_sixteenths < 14:
        return '12'     # dotted half
    else:
        return '16'     # whole


def detect_key(notes):
    """Simple key detection based on pitch class histogram."""
    if not notes:
        return 'C'
    
    histogram = [0] * 12
    for n in notes:
        pc = n['pitch'] % 12
        weight = (n['endTime'] - n['startTime']) * n['velocity']
        histogram[pc] += weight
    
    # Major key profiles (Krumhansl)
    major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    key_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    
    best_key = 'C'
    best_corr = -1
    
    for shift in range(12):
        shifted = histogram[shift:] + histogram[:shift]
        corr = np.corrcoef(shifted, major_profile)[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_key = key_names[shift]
    
    return best_key


def detect_tempo(notes):
    """Estimate tempo from onset intervals."""
    if len(notes) < 10:
        return 72
    
    onsets = sorted(set(round(n['startTime'], 2) for n in notes))
    if len(onsets) < 5:
        return 72
    
    intervals = [onsets[i+1] - onsets[i] for i in range(min(len(onsets)-1, 100)) if onsets[i+1] - onsets[i] > 0.05]
    if not intervals:
        return 72
    
    median_interval = sorted(intervals)[len(intervals)//2]
    
    # Assume median interval ≈ eighth note
    bpm = 60.0 / (median_interval * 2)
    
    # Clamp to reasonable range
    while bpm < 40: bpm *= 2
    while bpm > 180: bpm /= 2
    
    return int(round(bpm))


def notes_to_abc(notes, title='Untitled', split_threshold=55):
    """
    Convert note list to ABC with two staves (treble + bass).
    Split at MIDI 55 (G3) — standard piano split point.
    """
    if not notes:
        return ''
    
    key = detect_key(notes)
    tempo = detect_tempo(notes)
    
    rh_notes = [n for n in notes if n['pitch'] >= split_threshold]
    lh_notes = [n for n in notes if n['pitch'] < split_threshold]
    
    sec_per_16th = 60.0 / tempo / 4
    
    def render_voice(voice_notes):
        if not voice_notes:
            return 'z16|'
        
        voice_notes.sort(key=lambda n: n['startTime'])
        measures = []
        current_measure_ticks = 0
        measure_content = []
        ticks_per_measure = 16  # 4/4 time, L:1/16
        
        for note in voice_notes:
            tick = round(note['startTime'] / sec_per_16th)
            dur_ticks = max(1, round((note['endTime'] - note['startTime']) / sec_per_16th))
            
            # Fill gaps with rests
            gap = tick - current_measure_ticks
            while gap >= ticks_per_measure:
                # Fill remaining measure with rest
                remaining = ticks_per_measure - (current_measure_ticks % ticks_per_measure)
                if remaining > 0 and remaining < ticks_per_measure:
                    measure_content.append(f'z{remaining}')
                measures.append(' '.join(measure_content) if measure_content else f'z{ticks_per_measure}')
                measure_content = []
                current_measure_ticks += remaining
                gap = tick - current_measure_ticks
            
            if gap > 0:
                measure_content.append(f'z{gap}')
                current_measure_ticks += gap
            
            # Add note
            abc_note = midi_to_abc_note(note['pitch'])
            dur_str = quantize_duration(dur_ticks)
            measure_content.append(f'{abc_note}{dur_str}')
            current_measure_ticks += dur_ticks
            
            # Check measure boundary
            measure_pos = current_measure_ticks % ticks_per_measure
            if measure_pos == 0 and measure_content:
                measures.append(' '.join(measure_content))
                measure_content = []
        
        # Flush remaining
        if measure_content:
            remaining = ticks_per_measure - (current_measure_ticks % ticks_per_measure)
            if remaining > 0 and remaining < ticks_per_measure:
                measure_content.append(f'z{remaining}')
            measures.append(' '.join(measure_content))
        
        # Format: 4 measures per line
        lines = []
        for i in range(0, len(measures), 4):
            chunk = measures[i:i+4]
            lines.append('|'.join(chunk) + '|')
        
        return '\n'.join(lines) if lines else 'z16|'
    
    rh_abc = render_voice(rh_notes)
    lh_abc = render_voice(lh_notes)
    
    abc = f"""X:1
T:{title}
M:4/4
L:1/16
Q:1/4={tempo}
K:{key}
V:RH clef=treble
{rh_abc}
V:LH clef=bass
{lh_abc}"""
    
    return abc


# ── Main transcription endpoint ──────────────────────────────────────────────

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json or {}
    audio_url = data.get('audioUrl')
    title = data.get('title', 'Untitled')
    
    if not audio_url:
        return jsonify({'error': 'audioUrl required'}), 400
    
    try:
        # Download audio
        print(f"[MaestrIA] Downloading: {audio_url[:80]}...")
        resp = requests.get(audio_url, timeout=60)
        resp.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            f.write(resp.content)
            audio_path = f.name
        
        print(f"[MaestrIA] Downloaded {len(resp.content)} bytes")
        
        # Load audio at model's expected sample rate
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        duration = len(audio) / sample_rate
        print(f"[MaestrIA] Audio loaded: {duration:.1f}s at {sample_rate}Hz")
        
        # Process in chunks to avoid OOM on CPU (max 60s per chunk)
        CHUNK_SEC = 60
        OVERLAP_SEC = 2  # overlap to catch notes at boundaries
        all_notes = []
        
        if duration <= CHUNK_SEC + 10:
            # Short enough to process in one shot
            midi_path = audio_path.replace('.mp3', '.mid')
            transcriptor.transcribe(audio, midi_path)
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    all_notes.append({
                        'pitch': note.pitch,
                        'startTime': round(note.start, 4),
                        'endTime': round(note.end, 4),
                        'velocity': note.velocity,
                    })
            try: os.unlink(midi_path)
            except: pass
        else:
            # Chunk processing
            chunk_samples = CHUNK_SEC * sample_rate
            overlap_samples = OVERLAP_SEC * sample_rate
            step = chunk_samples - overlap_samples
            n_chunks = max(1, int(np.ceil((len(audio) - overlap_samples) / step)))
            
            print(f"[MaestrIA] Processing in {n_chunks} chunks of {CHUNK_SEC}s...")
            
            for i in range(n_chunks):
                start_sample = i * step
                end_sample = min(start_sample + chunk_samples, len(audio))
                chunk = audio[start_sample:end_sample]
                offset_sec = start_sample / sample_rate
                
                chunk_midi = audio_path.replace('.mp3', f'_chunk{i}.mid')
                
                print(f"[MaestrIA] Chunk {i+1}/{n_chunks}: {offset_sec:.1f}s - {end_sample/sample_rate:.1f}s")
                
                try:
                    transcriptor.transcribe(chunk, chunk_midi)
                    midi_data = pretty_midi.PrettyMIDI(chunk_midi)
                    
                    for instrument in midi_data.instruments:
                        for note in instrument.notes:
                            all_notes.append({
                                'pitch': note.pitch,
                                'startTime': round(note.start + offset_sec, 4),
                                'endTime': round(note.end + offset_sec, 4),
                                'velocity': note.velocity,
                            })
                except Exception as chunk_err:
                    print(f"[MaestrIA] Chunk {i+1} failed: {chunk_err}")
                finally:
                    try: os.unlink(chunk_midi)
                    except: pass
                    gc.collect()
            
            # Deduplicate notes from overlap regions
            all_notes.sort(key=lambda n: (n['startTime'], n['pitch']))
            deduped = []
            for n in all_notes:
                if not deduped or not (
                    abs(n['startTime'] - deduped[-1]['startTime']) < 0.05
                    and n['pitch'] == deduped[-1]['pitch']
                ):
                    deduped.append(n)
            all_notes = deduped
        
        print(f"[MaestrIA] Transcribed: {len(all_notes)} notes")
        
        # Generate ABC notation
        abc = notes_to_abc(all_notes, title)
        print(f"[MaestrIA] ABC generated: {len(abc)} chars")
        
        # Cleanup
        try: os.unlink(audio_path)
        except: pass
        gc.collect()
        
        return jsonify({
            'notes': all_notes,
            'abc': abc,
            'noteCount': len(all_notes),
        })
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'model': 'bytedance/piano_transcription', 'device': DEVICE})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
