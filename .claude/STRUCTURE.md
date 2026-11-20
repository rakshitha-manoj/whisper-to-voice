# Repo structure

Whisper to Voice: whispered speech in, voiced speech out, via LPC source filter resynthesis.
16 kHz mono float32, frame 320, hop 160. Contract in README.md. Full spec in docs/whisper_to_voice_team_guide.pdf.

## Pipeline (offline)
load_audio -> frame_signal -> frame_energy -> classify_frames -> lpc_all (-> warp_poles) -> build_f0 -> make_excitation -> synthesize -> output_stage -> wav

## Modules
- src/framing.py (A): load_audio, frame_signal, frame_energy, frame_zcr
- src/detect.py (A): voicing_strength, calibrate_thresholds, classify_frames (labels only), plot_labels, plot_spectrogram
- src/lpc.py (B): preemphasis, autocorr, levinson, lpc_frame, lpc_all, residual, lpc_envelope, log_spectral_distortion, warp_poles
- src/pitch.py (C): rosenberg_pulse, impulse_train, build_f0, make_excitation
- src/synth.py (C): synthesize, output_stage, convert, ablation variants
- src/live.py (C+A): causal per-hop pipeline, state dict carried between hops
- src/stream.py (A): sounddevice duplex stream, calls live processor, latency log
- src/evaluate.py (C): WER (Whisper + jiwer), LSD per variant, f0 histogram, listening sheet
- src/mocks.py: shared mocks for testing a module alone

## Entry points
- run_offline.py: one wav in, one wav out (optional variants)
- run_eval.py: ablation table over data/raw/spk*_s*_whisper.wav -> results/
- run_live.py: real time demo
- app.py: Gradio file mode demo with sliders (fallback for the live demo)
- record.py: record a clip

## Tests and validation
- tests/: pytest suite, run `python -m pytest tests -q`
- tests/fixtures.py: synthetic pseudo whisper (formant filtered noise) for tests without recordings
- notebooks/{a,b,c}_*_validation.ipynb: rubric validation against librosa/scipy references

## Data
- data/raw/: recordings (git ignored except shared_test.wav, transcripts.txt). Naming spk01_s03_whisper.wav / _normal.wav; sentence id maps to transcripts.txt line
- results/: generated wavs, figures, tables (not committed)
