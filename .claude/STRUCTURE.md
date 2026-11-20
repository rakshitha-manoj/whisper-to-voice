# Repo structure

Whisper to Voice: whispered speech in, voiced speech out, via LPC source filter resynthesis.
16 kHz mono float32, frame 320, hop 160. Contract in README.md. Full spec in docs/whisper_to_voice_team_guide.pdf.

## Pipeline (offline)
load_audio (mono, 16 kHz, dc_block) -> frame_signal -> frame_energy -> classify_frames -> lpc_all (-> warp_poles) -> build_f0 -> make_excitation -> synthesize -> output_stage (radiation + de-emphasis + DC blocker + RMS match) -> peak_protect -> wav
Offline loops the same per-hop step functions the live path calls (f0_step for live only; build_f0 offline knows whole phrases).

## Modules
- src/framing.py (A): load_audio, frame_signal, frame_energy, frame_zcr
- src/detect.py (A): voicing_strength, calibrate_thresholds, label_frame, classify_frames (labels only), plot_labels, plot_spectrogram, save_handoff
- src/lpc.py (B): preemphasis, autocorr, levinson, lpc_frame, lpc_all, residual, prediction_gain_db, lpc_envelope, log_spectral_distortion, lsd_frames, envelope_lsd, warp_poles
- src/pitch.py (C): impulse_train, rosenberg_pulse (flow, not differentiated), build_f0 (offline, per phrase), init_f0_state/f0_step/build_f0_causal (live), init_excitation_state/excitation_step/make_excitation
- src/synth.py (C): synth_step/synthesize, init_output_state/output_step/output_stage, peak_protect, convert, VARIANTS/render_variant/run_ablation
- src/live.py (C+A): default_params, init_live_state, process_hop_live, process_file_live, latency_report, plot_latency
- src/stream.py (A): sounddevice duplex stream, calls live processor, latency log
- src/evaluate.py (C): transcripts, discover_clips, Whisper ASR (CPU, numpy input), WER, LSD per variant, ablation csv/md, listening sheet + aggregation, report figures
- src/mocks.py: shared mocks for testing a module alone

## Entry points
- run_offline.py: one wav in, one wav out (optional variants)
- run_eval.py: ablation table over data/raw/spk*_s*_whisper.wav -> results/
- run_live.py: real time demo
- app.py: Gradio file mode demo with sliders (fallback for the live demo)
- record.py: record a clip
- tools/make_dev_clip.py: SAPI TTS sentences whisperised by LPC + noise -> results/dev (synthetic, pipeline check only)
- app_a.py: Module A diagnostic Gradio view

## Tests and validation
- tests/: pytest suite, run `python -m pytest tests -q`
- tests/fixtures.py: synthetic pseudo whisper (formant filtered noise) for tests without recordings
- notebooks/{a,b,c}_*_validation.ipynb: rubric validation against librosa/scipy references

## Data
- data/raw/: recordings (git ignored except shared_test.wav, transcripts.txt). Naming spk01_s03_whisper.wav / _normal.wav; sentence id maps to transcripts.txt line
- results/: generated wavs, figures, tables (not committed)
