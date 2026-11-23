# Testing guide

How to check that Whisper to Voice works: the automated suite, the validation notebooks, the manual checklists for the app and the live mode, and the end-to-end evaluation. For what the system is and why, see `docs/PROJECT.md`. For setup and daily commands, see `README.md`.

All commands run from the repository root (`whisper_to_voice/`). One test reads `data/raw/transcripts.txt` by a relative path and skips itself if you start elsewhere.

## 1. Setup

Python 3.10 or newer.

    pip install -r requirements.txt

`requirements.txt` installs numpy, scipy, matplotlib, soundfile, sounddevice, librosa, gradio, jiwer, pystoi, pytest, nbformat, nbconvert and ipykernel. (`pystoi` is installed because the team guide lists it, but no code in the repository imports it.)

`openai-whisper` is optional and is needed only for the WER part of `run_eval.py` and the Transcribe button in `app.py`:

    pip install openai-whisper

It pulls in PyTorch, which is a large download. The code runs Whisper on the CPU (`device="cpu"`, `fp16=False`), so no GPU is required. The first run downloads the `base` model (about 140 MB). Everything else in this guide works without Whisper.

The automated suite, the notebooks and the app need no recordings. `data/raw/shared_test.wav` is listed in `.gitignore` as an exception, but it is not in the repository at the moment. When it is missing, the notebooks and the app use the synthetic pseudo-whisper from `tests/fixtures.py` and say so in their output.

## 2. Automated test suite

    python -m pytest tests -q                      # whole suite
    python -m pytest tests/test_lpc.py -q          # one file
    python -m pytest tests/test_lpc.py::test_levinson_matches_toeplitz -q    # one test
    python -m pytest "tests/test_module_c.py::test_hopwise_equals_single_pass[125.0]" -q   # one parametrisation
    python -m pytest tests -v                      # one line per test
    python -m pytest tests -k "live" -v            # by keyword
    python -m pytest tests -s -k "speed or timing or fast_enough"   # the three timing tests, with their printed timings
    python -m pytest tests --collect-only -q       # list the tests without running them

Expected result, verified on 2026-11-22 (Windows, Python 3.12):

    112 passed in about 35 to 65 s

Allow roughly 20 to 45 seconds depending on the machine. If the count is not 112 (111 passed and 1 skipped if openai-whisper or its cached `base` model is absent), either a test file changed or something failed to import.

Notes:

* The suite needs no microphone. `test_list_devices_subprocess` runs `run_live.py --list-devices` and only needs PortAudio to import.
* `app.py` imports Gradio only inside `build_app()`, so only `test_build_app_constructs` needs `gradio`; the other app tests and `test_imports_without_device` run without it. `test_imports_without_device` imports `src.stream`, which imports `sounddevice`, so PortAudio must load (no device is opened).
* Three tests assert timings (`test_speed` under 1 ms per LPC frame, `test_step_functions_fast_enough_for_live` under 2 ms per hop, `test_timing` mean under 2 ms and p95 under 5 ms per live hop). On a heavily loaded machine they can fail even though the code is correct. Re-run them alone before debugging.
* `tests/test_module_c.py::test_convert_end_to_end` and `test_run_ablation_writes_four_wavs` replace Modules A and B with mocks (see below); the live, app and evaluation tests use the real modules.

### Helpers used by the tests

* `tests/fixtures.py`: a synthetic pseudo-whisper, 2.57 s at 16 kHz. Vowels are white noise through three formant resonances, "s" is high-pass filtered noise, silence is low-level noise (RMS 0.002). The plan is silence 0.4 s, vowel "a" 0.25 s, "s" fricative 0.12 s, vowel "i" 0.25 s, silence 0.6 s, vowel "u" 0.25 s, "s" 0.10 s, "a" 0.30 s, silence 0.30 s. `pseudo_whisper(voiced=True)` excites the same filters with a 120 Hz impulse train instead, which gives a voiced twin for comparison. Formant frequencies and bandwidths come from the `VOWELS` table. `write_fixture(path)` writes it as a wav. It is deterministic (seeded).
* `src/mocks.py`: `mock_frames`, `mock_energy`, `mock_labels` (all 2) and `mock_lpc` (a = [1, 0, ...], G = 1). They let one module be tested before a teammate's module is ready. `test_module_c.py` uses `mock_frames` and `mock_energy`; `test_lpc.py` uses `mock_frames` to cut the fixture into frames.
* `tests/validate_module_a.py` is a script, not a test (its name does not match `test_*.py`, so pytest does not collect it, and it has no assertions). Run it with `python tests/validate_module_a.py`. It processes `data/raw/spk01_s01_whisper.wav` and `_normal.wav` if both exist (label plot, spectrogram and a whisper versus normal comparison figure into `results/`), otherwise it prints "Skipping recordings", and then it always runs a small synthetic noise-burst check and prints the label counts. Expected output without recordings: the skip line and a label distribution line such as `Synthetic test label distribution: {np.int64(0): np.int64(93), np.int64(1): np.int64(12), np.int64(2): np.int64(44)}` (NumPy 2 prints the `np.int64` wrappers; the noise is not seeded, so counts can vary slightly).

### 2.1 tests/test_module_a.py (Module A: framing, features, labelling) - 21 tests

| Test | What it checks | Why it matters |
|---|---|---|
| `test_load_audio_stereo_44k` | Writes a 1.5 s stereo 44.1 kHz wav and loads it. Result is 16 kHz, float32, 1-D, length within 2 samples of 24000, peak at most 1. | Users upload arbitrary files in the app; the contract says mono 16 kHz float32. |
| `test_frame_signal_shape_and_librosa` | Frame count is `1 + (n - 320) // 160`, dtype float32, and frames equal `librosa.util.frame` exactly. | Every later stage depends on identical framing. |
| `test_frame_signal_is_copy` | Zeroing the input after framing does not change the frames; no shared memory. | The sliding window view is copied, so later in-place edits cannot corrupt frames. |
| `test_frame_signal_short_input` | 100 samples give shape (0, 320); exactly 320 samples give (1, 320). | Short clips must not crash; they produce zero frames and the app shows a friendly error. |
| `test_energy_and_zcr_vs_librosa` | RMS equals `librosa.feature.rms` (atol 1e-6). ZCR times 319/320 equals `librosa.feature.zero_crossing_rate`. | Reference check for A3. The factor documents that we divide by 319 sample pairs, librosa by 320. |
| `test_voicing_strength_sine_vs_noise` | A 120 Hz sine scores above 0.95 in every frame; white noise below 0.45 (median below 0.35). | The autocorrelation peak must separate periodic from noisy signals. |
| `test_voicing_strength_whisper_vs_voiced` | Averaged over the vowel segments, the voiced twin of the fixture scores more than 0.1 higher than the whisper. | Confirms the measure sees the missing source. It is a diagnostic, not used by the classifier. |
| `test_classify_contract_labels_only` | `classify_frames` returns an integer ndarray of shape (n_frames,) with values in {0,1,2}; passing `zcr` explicitly gives the same labels. | Function contract in the README. |
| `test_classify_return_thresholds` | With `return_thresholds=True` the result is (labels, zcr_thr, silence_thr) with 0 < silence_thr < 0.05 and 0 < zcr_thr < 1. | The app and the live path read these thresholds. |
| `test_classify_fixture_segments` | In each fixture segment (margin 3 frames) more than 75 percent of frames have the expected label: silence 0, "s" 1, vowels 2. | End-to-end sanity of the classifier on known content. |
| `test_classify_explicit_zcr_threshold` | `zcr_thr=0.0` leaves no vowel-like frames; `zcr_thr=1.0` leaves no consonant frames. | A given ZCR threshold must override the percentile rule (used by the Analysis tab slider). |
| `test_clip_starting_with_speech_not_all_silence` | Cutting the first 0.4 s of silence off the fixture still labels under 50 percent of frames as silence and over 80 percent of frames 2 to 19 as non-silent. | Calibration must not take speech as the noise floor. |
| `test_calibrate_guard_on_speech_start` | On the same speech-first clip the silence threshold is below 0.03 and below the naive 3 x median of the first 0.3 s. | Tests the fallback to the 10th percentile of energies. |
| `test_calibrate_empty_and_leading_silence` | Empty input returns the defaults (1e-4, 0.5). On the fixture, silence threshold lies between 0.004 and 0.01. | Edge case and normal case of `calibrate_thresholds`. |
| `test_label_frame_matches_pre_smoothing` | Per-hop `label_frame` equals the vectorised rule; thresholds returned by `classify_frames` match `calibrate_thresholds`; smoothed labels agree with raw labels on over 90 percent of frames; three direct calls give 0, 1, 2. | The live path uses `label_frame`, so it must match the offline rule before smoothing. |
| `test_hangover_is_bounded` | A 10-frame burst of energy: frames before it and frames from index 36 on are silence. | The 5-frame hangover must extend speech by a bounded amount only. |
| `test_save_handoff` | `save_handoff` writes `labels.npy` and `energy.npy` that load back unchanged. | Team handoff files. |
| `test_plot_spectrogram_shape` | `plot_spectrogram` returns an array of shape (len(f), len(t)) and writes the png. | Plot helper contract. |
| `test_load_audio_removes_dc_and_classifier_survives_offset` | A wav with +0.3 DC loads with mean below 0.01 after the first 4000 samples, and over 80 percent of labels still match those of the clean signal. | A DC offset from a sound card must not break calibration. |
| `test_app_a_process_audio_runs` | `app_a.process_audio` on the fixture completes, writes all five plots (into a temporary folder) and its summary shows a finite, positive noise floor. | Regression test for the undefined `noise_floor` that broke every Process click in `app_a.py`. |
| `test_dc_block_starts_in_steady_state_and_carries_state` | A constant 0.3 input (plus small noise) gives output below 0.05 over the first 400 samples, so there is no decaying start-up step (steady-state start). Splitting the signal into chunks, including empty ones, and carrying `zi` reproduces the one-shot output (atol 1e-6). | The live path calls `dc_block` per hop and must give the same result as the offline call. |

### 2.2 tests/test_lpc.py (Module B: LPC) - 16 tests

| Test | What it checks | Why it matters |
|---|---|---|
| `test_preemphasis` | `y[0] == x[0]` and `y[n] = x[n] - 0.9 x[n-1]` otherwise. | Definition of the pre-emphasis filter. |
| `test_autocorr_matches_numpy` | For 10 random frames, `autocorr(x, 16)` equals `np.correlate` and `r[0]` is the energy. | Reference check B3. |
| `test_levinson_matches_toeplitz` | For fixture frames and random frames, orders 4 to 24 in steps of 4: coefficients equal `scipy.linalg.solve_toeplitz` to within 1e-8, `a[0] == 1`, and the returned error equals `r[0] + a.r`. | Reference check B4 for the core algorithm. |
| `test_poles_inside_unit_circle` | For every fixture frame, `a` and the gain are finite and all roots of A(z) have magnitude below 1. | A stable synthesis filter is required. |
| `test_silent_frames` | An all-zero frame gives gain 0, `a = [1, 0, ..., 0]`, length 13. | Avoids NaN or huge gains on silence. |
| `test_lpc_all_shapes` | Zero frames give shapes (0, 13) and (0,). On fixture frames with order 10 the shapes are (n, 11) and (n,), `A[:,0] == 1`, all finite. | Contract of `lpc_all`. |
| `test_speed` | Mean time of `lpc_frame` is under 1 ms. | Real-time budget. |
| `test_prediction_gain` | A synthetic vowel frame has prediction gain above 0 dB and residual energy below frame energy. | The all-pole fit must remove structure. |
| `test_gain_matches_rms` | Unit-RMS noise through G/A(z) has RMS within a factor of 2 of the pre-emphasised frame. | Validates the gain formula. |
| `test_envelope_peaks_near_formants` | For the vowel "a" (formants 700 and 1220 Hz), the two largest envelope peaks of the de-emphasised envelope are within 15 percent of the true formants on average over 5 seeds. | Checks that LPC finds formants. |
| `test_envelope_vs_librosa` | For vowels a, i, u the envelope of `lpc_frame` correlates above 0.9 with the envelope from `librosa.lpc`. | Reference check B8. |
| `test_warp_identity_and_stability` | Factor 1.0 returns the input; for 7 factors from 0.85 to 1.0 the output has the same shape, float dtype, `a[0] == 1` and all poles inside the unit circle. | Warping must never produce an unstable filter. |
| `test_warp_moves_formants` | Warping by 0.9 moves both envelope peaks down, and the ratio of the new to the old peak frequency is 0.9 within 0.06. | Confirms the warp does what the factor says. |
| `test_warp_degenerate` | A filter with no poles (`a = [1, 0, ...]`) is returned unchanged. | Silent frames pass through the live path. |
| `test_lsd` | LSD of a filter with itself is exactly 0; between vowels "a" and "i" it exceeds 0.5 dB; `lsd_frames` is 0 for identical sets, positive for different sets, and 0 when the mask selects only identical frames. | The LSD used in the evaluation. |
| `test_envelope_lsd_reasonable` | Envelope-versus-FFT LSD at order 16 is positive and lower than at order 4; a zero frame gives 0. | Behaviour the order sweep relies on. |

### 2.3 tests/test_module_c.py (Module C: pitch, excitation, synthesis, output) - 29 tests

| Test | What it checks | Why it matters |
|---|---|---|
| `test_impulse_train_harmonics` | A 1 s train at 120 Hz has spectral peaks at harmonics 1 to 5, each more than 5 times the level between harmonics; the sum is within 1 of 120. | Reference for the basic pulse train (C1). |
| `test_rosenberg_pulse_shape` | For T0 in {80, 133.3, 200}: length `round(T0)`, starts at 0, peaks near 40 percent of the period at value 1, closed (exactly 0) after 40 plus 16 percent, never negative. | Defines the glottal flow shape (C2). |
| `test_excitation_f0_autocorrelation[120.0\|200.0]` (2) | The autocorrelation estimate of f0 of `make_excitation` is within 3 percent of the target. | The team guide's acceptance check for C3. |
| `test_excitation_glide_tracked` | A glide from 100 to 150 Hz: start and end estimates are within 5 percent of the local mean f0, and the end is more than 25 percent above the start. | Running phase must follow a moving f0. |
| `test_hopwise_equals_single_pass[125.0\|120.0]` (2) | Excitation built hop by hop equals one 60-hop call to `excitation_step` (atol 1e-6), with aspiration 0. | Hop-by-hop processing (live) and whole-clip processing give the same signal. |
| `test_hopwise_matches_independent_train` | At 125 Hz (period 128 samples) the excitation equals an independently built train of unit-RMS Rosenberg pulses (atol 1e-5). | Checks the phase and pulse placement logic from scratch. |
| `test_no_hop_rate_modulation[0.0\|0.15]` (2) | Per-hop RMS over a constant-f0 run stays between 0.75 and 1.25, with and without aspiration. | A pulse cut by a hop boundary must not make loudness pulse at the hop rate (100 Hz buzz). |
| `test_build_f0_basic_properties` | f0 has the label shape, is 0 wherever label is not 2, stays in 70 to 400 Hz, and consecutive voiced frames never jump more than 15 percent. | Contour sanity (C4). |
| `test_build_f0_declination_and_reset` | With stress and jitter off: each of two phrases falls (start more than 8 percent above end), both start within 2 percent of each other (reset after a 0.6 s gap), and the first phrase never rises. | Declination per phrase. |
| `test_build_f0_short_gap_does_not_reset` | A 0.3 s gap inside a phrase does not reset the declination. | The phrase gap is 0.5 s. |
| `test_build_f0_louder_is_higher` | With declination off, f0 correlates above 0.9 with log energy. | Stress: louder syllables rise. |
| `test_build_f0_flat_when_neutral` | With decl, stress and jitter all 0, f0 equals base_f0 (150) on voiced frames and 0 elsewhere. | Parameters do what they say; used by `v1_robot`. |
| `test_build_f0_causal_properties` | Causal tracker: 0 off voiced frames, within 70 to 400; declines within a phrase; resets on a new phrase; is flat when neutral; correlates above 0.5 with log energy. | The live path uses `f0_step`. |
| `test_excitation_unit_rms_per_label` | Label 0 hops are exactly zero, label 1 hops have RMS 1 within 1e-4, label 2 hops have RMS near 1 (mean within 0.1). | Spec requirement: unit-RMS excitation per label. |
| `test_label2_without_f0_is_noise` | A vowel hop with f0 = 0 falls back to unit-RMS noise. | No silent holes if the f0 tracker gives 0. |
| `test_aspiration_changes_excitation_and_is_reproducible` | aspiration 0.15 differs from 0.0; two runs with 0.15 are identical; RMS stays within 0.25 of 1. | Seeded noise gives repeatable output. |
| `test_synthesize_hopwise_equals_lfilter` | `synthesize` with a constant filter equals one `scipy.signal.lfilter` call over the whole signal (rtol 1e-5). | Filter state is carried across hops without clicks. |
| `test_output_stage_follows_input_loudness` | Output hop RMS is within 0.5x to 2x of the input energy and the log correlation is above 0.9. | Loudness following (C7). |
| `test_output_step_radiation_deemphasis_dc_block_applied_once` | `output_step` over two hops equals radiation, then de-emphasis, then DC blocker, then the gain (and its linear ramp), computed with `lfilter` in one pass. | Guards against applying a filter twice or resetting its state. |
| `test_output_step_digital_silence_then_onset_is_bounded` | A zero hop returns zeros, and a loud onset after it stays below 1.0. | The gain ramp and cap must not blow up after silence. |
| `test_convert_end_to_end` | With mocked Modules A and B, `convert` writes a wav of `(n // 160) * 160` samples, finite, peak between 0.01 and 2, file length matches, and log RMS correlates above 0.5 with input energy on loud frames; a warped, aspiration-free call is also finite. | Offline pipeline integration for Module C alone. |
| `test_run_ablation_writes_four_wavs` | `run_ablation` writes exactly four `{prefix}_{variant}.wav` files; a custom prefix is honoured; v1 differs from v2 and v3 from v4. | C9 deliverable. |
| `test_step_functions_fast_enough_for_live` | 1000 iterations of f0_step, excitation_step, synth_step and output_step take under 2 ms per hop. | Real-time budget for Module C. |
| `test_output_has_little_energy_below_60hz` | Less than 5 percent of output power lies below 60 Hz. | The DC blocker does its job (the Rosenberg flow has a large mean). |
| `test_render_shorter_than_one_frame[None\|0.92]` (2) | Features with zero frames render to silence of the input length, with or without warp. | Short clips must not crash. |

### 2.4 tests/test_live.py (live path and entry points) - 15 tests

These use the real Modules A, B and C on the pseudo-whisper fixture.

| Test | What it checks | Why it matters |
|---|---|---|
| `test_output_basic` | `process_file_live` output has the input length, is finite, peak at most 1 and above 0.01, and the first 0.3 s are exactly zero. | The calibration period is silent. |
| `test_loudness_follows_input` | Log-RMS correlation of output and input is above 0.85 on loud frames after calibration. | Live loudness following. |
| `test_output_is_voiced` | Voicing strength in vowel regions is at least 0.1 higher in the output than in the whisper. | The output actually has a periodic source. |
| `test_timing` | Mean per-hop time under 2 ms, p95 under 5 ms; `latency_report` gives algorithmic delay 20 ms and total 50 ms when told device latency is 0.01 plus 0.02 s. | The per-hop time must stay under the 10 ms hop. |
| `test_plot_latency` | `plot_latency` writes a non-empty png. | The latency figure for the demo. |
| `test_params_change_midstream` | Changing `base_f0` from 100 to 200 in `state["params"]` halfway shows up as an f0 more than 1.5 times higher afterwards. | The app and sliders change parameters while streaming. |
| `test_live_vs_offline_envelope` | The log-RMS envelope of the live output correlates above 0.8 with the offline output (live shifted by one hop). | Live and offline must sound alike in loudness. |
| `test_imports_without_device` | `app`, `run_live` and `src.stream` import without an audio device. | Allows running tests and the app on any machine. |
| `test_run_live_file_subprocess` | `run_live.py --file in.wav out.wav` exits with 0 and writes a non-trivial wav. | The no-microphone demo mode. |
| `test_list_devices_subprocess` | `run_live.py --list-devices` exits with 0. | Basic device enumeration works. |
| `test_device_from_args_keeps_a_lone_in_or_out` | `run_live.device_from_args` returns None with no indices, the pair with both, and `(1, None)` / `(None, 3)` with only one, so the other side uses its default device. | A lone `--in` or `--out` used to be silently ignored. |
| `test_stream_callback_survives_processing_error` | With `process_hop_live` patched to raise, the stream callback outputs silence for that hop and records `error_count` and `first_error`; with the real function restored the next hop processes normally. | One exception in the audio callback used to stop the whole stream. |
| `test_live_zcr_threshold_adapts_to_offline_value_and_finds_consonants` | After streaming the fixture, the live ZCR threshold is within 30 percent of the offline threshold, and at least 20 hops were labelled consonant. | The adaptive live ZCR rule works without the whole clip. |
| `test_live_fixed_zcr_threshold_param_overrides` | `zcr_thr=0.9` fixes the threshold: `zcr_live` stays at its initial 0.12 and no hop is labelled consonant. | The parameter overrides adaptation. |
| `test_live_zcr_threshold_keeps_adapting_when_window_not_multiple_of_ten` | With `zcr_window_s=2.55`, a switch from low-ZCR to high-ZCR input raises `zcr_live` by more than 0.2. | Regression test: an earlier update counter stopped updating once the window was full. |

### 2.5 tests/test_evaluate.py (evaluation tooling) - 9 tests

None of these needs Whisper; the ASR is replaced by a stub where needed.

| Test | What it checks | Why it matters |
|---|---|---|
| `test_load_transcripts_with_odd_line` | Lines "01 ..." and "02 ..." parse to ids "01", "02"; a line starting with "s01" is ignored with a `UserWarning`. | Tolerant transcript parsing. |
| `test_load_real_transcripts` | `data/raw/transcripts.txt` parses to ids 01 to 06 (with the same warning for the "s01" line). Skipped if run from another directory. | Guards the real transcript file. |
| `test_normalise_text` | Lower-casing, hyphen to space, punctuation removal, whitespace collapse. | WER should not penalise punctuation or case. |
| `test_wer_counts` | Exact match gives (0, 3), empty hypothesis gives (3, 3), one substitution gives (1, 3). | Correct error and reference counts. |
| `test_discover_clips` | Finds `shared_test.wav` (sentence ids 01 and 02) and `spkXX_sNN_whisper.wav`, pairs each with its `_normal.wav` when present, ignores other files, and returns [] for a missing folder. | Defines the recording naming convention. |
| `test_lsd_variant` | LSD of the fixture against itself is 0; against white noise (RMS 0.05) it exceeds 0.5 dB. | Sanity of the evaluation LSD. |
| `test_run_ablation_eval_no_asr` | With `asr=False`: five systems (raw_whisper plus four variants), raw LSD 0 and variants above 0, writes `ablation_per_clip.csv`, `ablation.csv`, `ablation.md` and four variant wavs; the table header matches. | The whole evaluation flow minus Whisper. |
| `test_run_ablation_eval_monkeypatched_asr` | With a stub ASR that always returns "the birch canoe" for the reference "The birch canoe slid" (4 words): WER is 0.25, per clip and corpus. | Checks the WER bookkeeping. |
| `test_listening_sheet_and_aggregate` | The sheet has 5 shuffled items (raw plus 4 variants) with a key, 5 listeners times 5 items = 25 rows, a README; aggregation averages clarity and naturalness per system, skips rows with a missing score, never reports the unscored system, and the scores appear in the next `ablation.md`. | The blinded listening test pipeline. |

### 2.6 tests/test_app.py (Gradio app helpers) - 22 tests

These test the pure helper functions on the fixture. The browser layer is covered by the manual checklist in section 4.

| Test | What it checks | Why it matters |
|---|---|---|
| `test_convert_clip_audio` | `convert_clip` returns (16000, y) with finite y and two figures. | The Convert tab's main function. |
| `test_convert_full_download` | `convert_full` with `v4_warp` returns the input path and a downloadable wav whose length equals the output. | Download button. |
| `test_convert_clip_requires_audio` | No clip raises `ValueError`. | Becomes the friendly "Upload or record a clip first" message. |
| `test_compare_variants` | Five results: custom plus `v1_robot`, `v2_contour`, `v3_aspiration`, `v4_warp`, all finite. | "Compare all variants" button. |
| `test_module_a_figure_and_thresholds` | The Module A figure has 4 axes; a given ZCR threshold is honoured; a larger silence multiplier raises the silence threshold. | Analysis tab sliders. |
| `test_lpc_frame_figure[first-1.0\|last-0.85\|mid-0.92]` (3) | Figure and "prediction gain" text for the first, middle and last frames; envelopes are finite; warp 1.0 gives identical original and warped envelopes. | Frame slider edge values and warp slider extremes. |
| `test_analyse` | `analyse` returns two figures, text and a frame count above 5. | Analysis tab button. |
| `test_results_discovery_empty` | Empty or missing results folder gives no folders and no figures; the empty message mentions `run_eval.py`. | Results tab with nothing to show. |
| `test_results_discovery_populated` | A folder with `ablation.md`, a figure and variant wavs is discovered; clip stems and variant paths are read from file names. | Results tab with results. |
| `test_build_app_constructs` | `build_app()` builds without error. | Catches Gradio wiring errors early. |
| `test_output_players_are_not_inputs_for_the_user` | The "Input (A)" and "Voiced output (B)" players are built with `interactive=False`. | The output player also feeds Transcribe; without the pin Gradio would turn it into an upload/record box, and a user's own recording would be transcribed as "voiced output". |
| `test_too_short_clip_raises_friendly` | An 80-sample clip makes `convert_clip`, `compare_variants` and `analyse` raise `ValueError` mentioning "shorter than one" frame. | Friendly error instead of a crash. |
| `test_f0_params_follow_variant` | The label/f0 plot uses the variant's own settings (`v1_robot` is flat, `v3_aspiration` default) and the sliders only for "custom". | The plot must match the audio. |
| `test_transcribe_both_signals` | With a fake ASR model, `transcribe_clip` returns the transcript of the input and of the output; no reference gives `wer` None and no WER row in the table. | Transcribe button, without loading Whisper. |
| `test_transcribe_wer_against_reference` | Reference "The cat sat down." against "the cat sat" gives 1 error in 4 words for both; an exact reference gives 0 of 3; the table shows "1/4 = 25.0%". | WER uses `evaluate.wer_counts` (case and punctuation normalised). |
| `test_transcribe_uses_output_player_value` | When the output player value belongs to the loaded clip, that exact signal (int16 scaled) is transcribed; for a stale clip the output is re-rendered. | The output transcript must match what the user hears. |
| `test_transcribe_requires_clip` | No clip raises `ValueError` "Upload or record a clip first". | Friendly error. |
| `test_transcribe_whisper_missing` | An `ImportError` from the loader becomes a `ValueError` containing "pip install openai-whisper". | The app and other tabs must work without whisper. |
| `test_asr_model_cache_reuses_object` | `get_asr` loads each model name once and returns the same object; an unknown name raises `ValueError`. | Repeat clicks must not reload the model. |
| `test_transcribe_real_base_does_not_crash` | Real Whisper `base` on the fixture returns strings and a WER denominator of 2. Skipped unless `openai-whisper` is importable and `~/.cache/whisper/base.pt` exists. | End to end check; the fixture is noise so the text itself is not asserted. |

Totals: 21 + 16 + 29 + 15 + 9 + 22 = 112, which equals the collected count.

## 3. Validation notebooks

These are the rubric validations against reference implementations. Executing a notebook rewrites its outputs, so run them on a copy if you do not want to change the committed file (the `--output-dir` form below does that). The in-place form is the one in the README.

    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/a_framing_validation.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/b_lpc_validation.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/c_synthesis_validation.ipynb
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/walkthrough.ipynb

    # without touching the repository copy
    python -m jupyter nbconvert --to notebook --execute --output-dir /tmp/nb notebooks/b_lpc_validation.ipynb

Each notebook finds the repository root itself and runs top to bottom in about 10 to 25 seconds with `nbconvert` (including kernel start-up). Without `data/raw/shared_test.wav` they use the pseudo-whisper fixture and print which source they used. "Pass" means it runs without an error cell and the printed numbers match the table below (values from a run on 2026-11-22 on the fixture; on a real recording the numbers differ but the criteria are the same).

| Notebook | What it validates, against what | What a pass looks like |
|---|---|---|
| `a_framing_validation.ipynb` | A1 load and waveform. A2 `frame_signal` against `librosa.util.frame`. A3 `frame_energy` and `frame_zcr` against `librosa.feature.rms` and `zero_crossing_rate`. A4 voicing strength on a 120 Hz sine, white noise and the whisper and voiced fixtures. A5 thresholds and label plot. A6 spectrograms of whisper (no stripes) and voiced (harmonic stripes). | Shape (256, 320) equals the expected count; "identical to librosa.util.frame: True"; RMS max abs diff 0.0; ZCR diff 0.0023 raw and about 1e-16 after scaling by 319/320; sine strength about 0.9999, noise about 0.22; silence threshold about 0.0058, ZCR threshold about 0.138; label plot switches sensibly. |
| `b_lpc_validation.ipynb` | B1 pre-emphasis spectrum tilt. B2 window. B3 `autocorr` against `np.correlate`. B4 `levinson` against `scipy.linalg.solve_toeplitz` for orders 4 to 24. B5 and B6 `lpc_all` shapes, finiteness, pole radius, time per frame. B7 residual and prediction gain. B8 envelope against `librosa.lpc`. B9 order sweep 4 to 24 by `envelope_lsd`. B10 `warp_poles` identity, stability and pole plot. | Autocorrelation diff 0.0; Levinson error below 1e-8 (8.9e-9); all finite, max pole radius 0.994 (below 1), about 0.1 ms per frame (machine dependent); prediction gain 22 dB on the loudest frame, 17.8 dB mean; envelope correlation with librosa 0.990; order sweep falls from 7.52 dB (order 4) to 5.05 dB (12), 4.83 dB (14), 4.35 dB (24), and the notebook picks 12; warp identity error 0.0, `a[0] = 1`, max radius 0.976. |
| `c_synthesis_validation.ipynb` | C1 impulse train harmonics. C2 Rosenberg pulse shape and spectrum. C3 excitation f0 by autocorrelation (120 and 200 Hz, and a glide). C4 `build_f0` against `build_f0_causal` on two phrases. C5 excitation by label. C6 synthesis across hop boundaries. C7 output stage loudness following and filters applied once. Uses the real `lpc_all`, `classify_frames`, `frame_signal`. | 120 impulses in 1 s; pulse peak at 0.398 of the period; f0 error 0.25 percent at 120 Hz and 0.00 percent at 200 Hz; glide estimates 105 to 144 Hz for a 100 to 150 Hz target; per-hop RMS of a constant run between 0.91 and 1.16; label RMS {0: 0, 1: 1.0, 2: 1.0}; output over input ratio 0.60 to 1.80 and log-RMS correlation 0.92; "hop 0 matches lfilter: True". |
| `walkthrough.ipynb` | Explains every stage with plots and audio on one clip, then: section 7 the four variants; section 8 live versus offline including the proof cell; section 9 LSD per variant and, if present, the stored ablation table. Uses only `src` functions. | Runs without errors. Section 8 proof cell prints `max |difference| = 6.0e-08` (offline chain fed with the live decisions against the live output), `live output here equals process_file_live output: True`, label agreement 91 percent, f0 methods 5.7 Hz mean apart (max 13.2 Hz). The loudness-envelope correlation of live and offline is 0.952, and the sample correlation is only 0.080, which is expected and explained in the notebook. LSD about 12 to 13.5 dB per variant on the fixture. Section 9 shows `results/dev_eval/ablation.md` if it exists; the committed outputs show the 2026-11-22 dev table, the same as `docs/PROJECT.md` section 6. |

Remarks on the numbers: the order sweep falls steadily with order (more poles always fit the envelope better), so "order 12" is an elbow choice, not a minimum. The notebook's rule is the first order where the next step of 2 gains less than 0.25 dB. The 12 to 14 step gains 0.21 dB. The walkthrough notebook is committed with outputs; its numbers come from the fixture because no real recording is in the repository.

## 4. Manual checklist for the app

Start it with

    python app.py

and open the printed local address (normally http://127.0.0.1:7860). Allow microphone access in the browser if you want to record. Gradio 6.x is installed here.

**Convert tab**

1. Press "Load demo clip". The input player is filled (the shared recording if present, otherwise a generated pseudo-whisper).
2. Press "Convert". (Loading a different clip clears the output players, so an old output is never shown or transcribed next to a new input.) Expected: the "Input (A)" and "Voiced output (B)" players, a download file, a spectrogram figure (input noise, output with horizontal harmonic lines) and a labels and f0 figure (f0 only on vowel frames, falling over a phrase). Compare A and B by ear.
3. Sliders: base pitch 70 to 250 Hz, intonation (declination) 0 to 0.3, stress 0 to 0.5, aspiration 0 to 0.3, pole warp 0.85 to 1.0, volume 0.2 to 2.0. Move base pitch and convert again: the pitch of B and the f0 plot change. Move the volume slider: B gets louder or quieter.
4. Warp 1.0 means "off" and must give the same audio as no warp; 0.85 gives a visibly lower formant position and a darker voice.
5. Variant dropdown: choose `v1_robot` and convert. The f0 plot is flat and the voice monotone. The named variants ignore the intonation, stress, aspiration and warp sliders (they still use base pitch and volume); only "custom" uses all the sliders.
6. "Compare all variants": five players appear (custom plus the four variants). Listen to the order robot, contour, aspiration, warp.
7. Download: the wav downloaded plays and has the same length as the output.
8. Transcribe: with a clip converted, choose a model (base is the default) and press "Transcribe". Expected: a table with the transcript of the input (whisper) and of the voiced output, after a few seconds (the first use of a model downloads it). Type the sentence in "Reference sentence" and press again: a WER row (errors/reference words) appears for both. Change a slider without pressing Convert and transcribe: the output transcript still matches the audio in the output player. Press Transcribe with no clip: error "Upload or record a clip first". On the synthetic dev set Whisper recognises the raw whisper better than the voiced output, so the input column may be the more accurate one. The "Whisper" in the note is the ASR model, not whispered speech.

**Edge cases on the Convert tab**

* No clip loaded, press Convert: a red error box says "Upload or record a clip first, or press Load demo clip." (no stack trace).
* A clip shorter than 20 ms (for example 100 samples): the error "Clip is shorter than one 20 ms frame (320 samples at 16 kHz); record or upload a longer clip." Same message from Compare and Analyse.
* A stereo 44.1 kHz upload works (it is converted to mono 16 kHz). Make a test file with
  `python -c "import numpy as np, soundfile as sf; t=np.arange(44100*2)/44100; sf.write('stereo44.wav', np.stack([0.2*np.sin(2*np.pi*200*t), 0.2*np.sin(2*np.pi*300*t)], 1), 44100)"`
  (a tone is not whisper, so only check that it loads and converts without error).
* Record from the microphone in the browser: stay silent for the first 0.3 s, then whisper a sentence.

**Analysis tab** (uses the clip from the Convert tab)

1. Press "Analyse clip". Expected: the Module A figure (RMS with the silence threshold line, ZCR with its threshold line, voicing strength, label track) and the Module B figure (FFT magnitude of one frame, LPC envelope of order 12, warped envelope if the warp factor is below 1) plus a line "Frame N: prediction gain X dB, LPC gain Y".
2. Move "Silence multiplier" and "ZCR threshold" (0 means automatic) and release: the Module A figure updates, the threshold lines move, labels change.
3. Frame index: after "Analyse clip" the slider maximum becomes the last frame. Check frame 0 and the last frame (both must plot, no error). Move to a vowel frame and a silent frame and compare the envelopes.
4. Warp factor 1.0: the warped curve is not drawn. 0.85: formant peaks move down.
5. Press "Analyse clip" with no clip loaded: error "Upload or record a clip on the Convert tab first, or press Load demo clip."

**Results tab**

* If `results/` (or one of its subfolders, such as `results/dev_eval`) contains `ablation.md`, the tab shows a folder dropdown, the ablation table, the figures gallery, a clip dropdown and four variant players. Switch folder and clip and check that the table, figures and players follow.
* With no results the tab shows "No results found. Run `python run_eval.py` (or `python run_eval.py --dev`) first." To see it, temporarily move `results/` away and restart the app. The helper behaviour is also covered by `test_results_discovery_empty`.
* The folders are discovered once, when the app starts. After running `run_eval.py` you must restart `python app.py` to see new results.

The Convert tab also shows a note that live mode runs from the command line (see the next section).

## 5. Live mode checks

Use headphones for every live test, otherwise the output feeds back into the microphone. Close other programs that use the microphone. On Windows, check Settings, Privacy, Microphone and switch off audio enhancements for the device.

1. List devices:

       python run_live.py --list-devices

   Each line has the index, name, host API and channel counts. Note an input index and an output index. On Windows the same hardware can appear under several host APIs (for example MME and WASAPI); WASAPI usually has the lower latency.
2. Mic check without processing:

       python run_live.py --passthrough

   You should hear your own voice with a small delay and no crackle. Ctrl-C stops it. (Passthrough prints no latency report.)
3. Choose devices:

       python run_live.py --in 1 --out 3

   You can give `--in`, `--out` or both; a side that is not given uses its default device, and the script prints which input and output it uses ("default" for an unspecified side).
4. On Windows add `--wasapi` if the latency is high or the stream fails to open:

       python run_live.py --wasapi
       python run_live.py --wasapi --in 5 --out 6

   Without explicit devices it uses the WASAPI default input and output, in shared mode with automatic sample-rate conversion. If WASAPI settings are rejected it retries without them.
5. Run the conversion. Stay silent for the first 0.3 s after "Stream started": the silence threshold is calibrated from it, and the output is exactly zero during this time. Anything said in these 0.3 s is lost, and if you speak almost immediately (within a few tens of milliseconds) the threshold is set on your speech and quiet speech is then treated as silence; stop with Ctrl-C and restart. Then whisper a sentence. Expected: a voiced version of your whisper, delayed by the 20 ms algorithmic delay (a model: two hops) plus the device buffers. Options: `--base-f0 150`, `--aspiration 0.2`, `--warp 0.92` (1.0 means off), `--volume 1.5`.
6. Stop with Ctrl-C. In conversion mode the program prints the latency report and saves the latency plot:

       Hops: N  mean X ms  p95 X ms  max X ms  over 10 ms: Y%
       Algorithmic delay 20.0 ms, device latency D ms, total T ms
       Latency plot saved to results/latency.png

   The plot `results/latency.png` shows per-hop processing time against the 10 ms budget. `latency_report` returns these fields: `hops`, `mean_ms`, `p95_ms`, `max_ms`, `frac_over_budget`, `algorithmic_ms` (two hops, 20 ms), `device_ms` (input plus output latency reported by the sound device, `None` when unknown) and `total_ms`. A line "Stream status events: N" means the sound card reported underruns or overruns during the run; fix the device settings if N is large.
   Pass criteria: mean per-hop time well under 10 ms and `over 10 ms` at or near 0 percent. On the development machine the fixture usually ran at about 0.3 ms mean, 0.5 to 0.7 ms p95 and 0.9 to 2.5 ms max per hop; in one test run on a busy machine the mean was 0.75 ms and a single hop took 28 ms, so check the maximum and `over 10 ms`, not only the mean. The total latency on a real device has not been measured yet; record it from step 6 on the demo machine. The team guide's target is a delay a listener can speak against (the guide mentions 30 to 50 ms as the expected range).
7. No microphone available (or as the fallback demo):

       python run_live.py --file data/raw/shared_test.wav results/live.wav --play

   This sends the file through the same per-hop live code, writes `results/live.wav`, prints the per-hop timing and algorithmic delay (no device latency and no latency plot in this mode) and plays the result if `--play` is given. Use any 16 kHz mono wav; `results/dev/dev_s01_whisper.wav` works after `python tools/make_dev_clip.py`. The first 0.3 s of the output are silent by design.
8. Things to listen for: no clicks at hop boundaries, loudness following the whisper, silence between words silent (after a short decaying tail of a few hops), a pitch that falls slowly over a phrase, no buzz at 100 Hz. Whisper at a constant level for a few seconds, then louder: the ZCR threshold adapts over the first seconds, so consonant labelling is rougher at the start of a stream.

## 6. End-to-end evaluation

### Recordings

Record 16 kHz mono clips with headphones on, quiet room, mic 2 to 5 cm from the mouth, and stay silent for the first 0.3 s.

    python record.py data/raw/shared_test.wav 12                 # sentences 01 and 02 back to back, whispered
    python record.py data/raw/spk01_s03_whisper.wav 5            # speaker 01, sentence 03, whispered
    python record.py data/raw/spk01_s03_normal.wav 5             # same sentence, spoken normally

Naming: `spkXX_sNN_whisper.wav` and `spkXX_sNN_normal.wav`, where XX is the speaker and NN is the line number in `data/raw/transcripts.txt`. Each line of `transcripts.txt` is "NN sentence text". Currently ids 01 to 06 are the Harvard sentences; the last line (`s01 Hey, how are you?`) does not start with a number and is ignored with a warning, which is expected. The `_normal.wav` is optional; when it exists it gives a `normal_recording` row in the table (the ASR ceiling). `shared_test.wav` is read as sentences 01 and 02 together. Recordings in `data/raw/` stay local and are git-ignored, except `shared_test.wav`, which is meant to be committed. The team guide asks for 20 sentences per member; the transcripts file has 6 so far. Split by speaker for tuning and testing as the guide says.

### Run

    python run_eval.py                          # WER (Whisper base, CPU) and LSD for every clip and variant
    python run_eval.py --no-asr                 # LSD only, no Whisper needed
    python run_eval.py --model small            # another Whisper model
    python run_eval.py --listening              # write the blinded listening sheet
    python run_eval.py --aggregate-listening results/listening/sheet.csv    # after the sheet is filled
    python run_eval.py --dev                    # same, on synthetic dev clips (see below)

Outputs, all under `results/` (not committed):

* `ablation.md`: the table (raw_whisper, v1_robot, v2_contour, v3_aspiration, v4_warp, normal_recording) with WER (mean per clip and corpus), LSD and the listening scores if available.
* `ablation.csv` (aggregates) and `ablation_per_clip.csv` (per clip and system: WER, error count, reference word count, LSD, the Whisper transcript).
* `<clip>_<variant>.wav`: the four rendered variants for each clip.
* `figures/fig_ablation_bar.png`, `fig_waveform_spectrogram.png`, `fig_envelope_overlay.png`, `fig_f0.png` (the last three use the first clip and variant `v4_warp`).
* `listening/` with `item_NN.wav`, `sheet.csv`, `key.csv` and `README.txt` (written by `--listening`).
* `listening_scores.csv` in the results folder itself (not in `listening/`), written by `--aggregate-listening`; `ablation.md` reads it from there.

Without recordings, `run_eval.py` prints how to record them and exits.

Check the result: `ablation_per_clip.csv` should list five systems per clip (six with a normal recording); the transcripts for `normal_recording` should be close to the reference sentence (WER near 0); open a few variant wavs and listen. WER can exceed 1.0 for a clip when the ASR inserts words. Report the numbers honestly, including when the voiced variants do not beat the raw whisper.

### Listening test

1. `python run_eval.py --listening` writes shuffled files `item_01.wav` and so on (raw whisper and the four variants for every clip), `sheet.csv` with rows for listeners L1 to L5, and `key.csv` mapping items to systems.
2. Give listeners `item_XX.wav` and `sheet.csv` only. Do not show `key.csv`. Each listener scores clarity and naturalness from 1 to 5 per item. Follow `results/listening/README.txt` (voluntary participation, no personal data, only ids L1 to L5).
3. Collect the filled sheet and run `python run_eval.py --aggregate-listening results/listening/sheet.csv`. It prints mean clarity and naturalness per system and writes `results/listening_scores.csv`. Rows with a missing score are skipped.
4. Run `python run_eval.py` again (with Whisper; `--no-asr` would leave the WER column empty) so that the scores appear in `ablation.md`. For the dev data add `--dev` to both commands; the scores then go to `results/dev_eval/listening_scores.csv`.

### Synthetic dev run (pipeline check only)

When no recordings exist, check the whole chain on synthetic data. This needs Windows (it uses the built-in SAPI speech synthesiser through PowerShell):

    python tools/make_dev_clip.py --ids 01 02 03 04 05 06
    python run_eval.py --dev

`make_dev_clip.py` speaks each transcript sentence with SAPI, turns it into a pseudo-whisper by filtering white noise through its own per-frame LPC filters, adds a small noise floor and 0.5 s of silence at each end, and writes `results/dev/dev_sNN_{normal,whisper}.wav` (default ids are 01 to 03 only). `run_eval.py --dev` reads those and writes into `results/dev_eval/`, and its table carries a "SYNTHETIC DEV DATA" banner. These numbers say that the pipeline runs; they say nothing about real whispered speech. The run is deterministic (three runs on 2026-11-22 gave identical tables) and takes under a minute with Whisper base on the CPU.

## 7. Pre-viva checklist

Before the day:

- [ ] `git status` clean on `main`; everyone has pulled.
- [ ] `python -m pytest tests -q` shows 112 passed (111 passed, 1 skipped without the cached Whisper base model).
- [ ] The four notebooks run without errors (section 3).
- [ ] `python run_offline.py data/raw/shared_test.wav results/out.wav --variants` produces `out.wav` and four variant wavs; listen to all.
- [ ] Real recordings exist: `data/raw/shared_test.wav` and at least a few `spkXX_sNN_whisper.wav` with transcripts. The repository currently has none, so the notebooks show the synthetic clip.
- [ ] `python run_eval.py` has been run on real recordings (with Whisper), `results/ablation.md` exists, and the team can say what it shows, including numbers that are unfavourable.
- [ ] Listening test done and aggregated (5 listeners), or the team can say it was not done.
- [ ] `python run_live.py --list-devices` works on the demo machine; headphones tested with `--passthrough`; `--in N --out M` (and `--wasapi` on Windows) tested; the real latency line from step 6 of section 5 written down.
- [ ] Fallback ready: `python run_live.py --file ... --play` and `python app.py` both work offline, plus pre-rendered variant wavs.
- [ ] `python app.py` tested tab by tab (section 4), restarted after the last `run_eval.py`.
- [ ] Each member can run the full demo, explain the pipeline in two sentences, and explain their own module's functions, tests and numbers (see `docs/viva/` and `docs/PROJECT.md`).
