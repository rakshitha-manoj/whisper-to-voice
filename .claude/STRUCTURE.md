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
- src/evaluate.py (C): transcripts, discover_clips, filter_speakers, Whisper ASR (CPU, numpy input), WER, LSD per variant, ablation csv/md, listening sheet + aggregation, report figures
- src/mocks.py: shared mocks for testing a module alone

## Entry points
- run_offline.py: one wav in, one wav out (optional variants)
- run_eval.py: ablation table over data/raw/spk*_s*_whisper.wav -> results/ (--speakers 02 restricts to speaker ids via evaluate.filter_speakers; add "shared" for shared_test.wav)
- run_live.py: real time demo
- app.py: tabbed Gradio app (fallback for the live demo): Convert (sliders, A/B, download, compare variants, Transcribe input vs output with optional WER; get_asr caches Whisper models per name, whisper imported lazily), Analysis (Module A features/labels, Module B per-frame LPC envelope, pole plot and F1-F3 formant tracks), Live (clip through process_file_live: per-hop timing, live vs offline envelope, latency summary), Convert accordions for Module C graphs (excitation zoom, f0 breakdown, loudness) and variant comparison (spectrogram grid, LSD bars, LTAS), Results (reads results/**/ablation.md, figures, variant wavs). Pure helpers, gradio imported in build_app only
- record.py: record a clip (simple, one file)
- tools/record_session.py: guided cross-platform session, all transcript sentences x whisper/normal for --speaker NN, keep/redo/skip, quality warnings, resume, --shared, --zip; I/O injectable (tests/test_record_session.py)
- tools/make_dev_clip.py: SAPI TTS sentences whisperised by LPC + noise -> results/dev (synthetic, pipeline check only)
- app_a.py: Module A diagnostic Gradio view (older, standalone; app.py Analysis tab covers it)

## Tests and validation
- tests/: pytest suite, run `python -m pytest tests -q`
- tests/test_app.py: app.py helpers on the synthetic fixture
- tests/test_record_session.py: record_session with fake rec/play/ask in tmp dirs
- tests/fixtures.py: synthetic pseudo whisper (formant filtered noise) for tests without recordings
- notebooks/walkthrough.ipynb: end to end explainer (all stages, plots, audio, live vs offline, evaluation), committed with outputs
- notebooks/{a,b,c}_*_validation.ipynb: rubric validation against librosa/scipy references

## Docs
- docs/PROJECT.md: project description (theory, architecture, design decisions, evaluation, dev results, novel contribution, limitations)
- docs/TESTING.md: how to test, every test case, notebook/app/live/evaluation checklists
- docs/RECORDING.md: instructions for the three recorders (setup, whisper vs normal, commands, sentence table, sending the zip, Raks's eval steps)
- docs/viva/module_{a,b,c}.md: per member viva guides (theory, function walkthrough, weak spots, Q&A, cheat sheet)
- docs/whisper_to_voice_team_guide.pdf: original spec

## Data
- data/raw/: recordings (git ignored except shared_test.wav, transcripts.txt; 20 sentences, 01-06 Harvard, 07-20 conversational; spk01 Hafsa, spk02 Akhila, spk03 Raks). Naming spk01_s03_whisper.wav / _normal.wav; sentence id maps to transcripts.txt line
- results/: generated wavs, figures, tables (not committed)
