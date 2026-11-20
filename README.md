# Whisper to Voice

MAI522 Speech Processing and Recognition, CIA III group mini project.
Converts whispered speech into voiced speech using source filter synthesis: LPC extracts the vocal tract filter, a synthetic glottal pulse train replaces the missing vocal cord buzz.

Full details, tasks and prompts are in docs/whisper_to_voice_team_guide.pdf. Read INSTRUCTIONS.md for how to use this repo.

## Team and modules

* Hafsa (Mac): Module A, src/framing.py, src/detect.py, src/stream.py, run_live.py
* Akhila: Module B, src/lpc.py
* Raks (Victus, GPU): Module C, src/pitch.py, src/synth.py, run_offline.py, Whisper evaluation

## Setup

    pip install -r requirements.txt
    pip install openai-whisper        # only for the WER evaluation

Use headphones for every live test. When recording or starting the live stream, stay silent for the first 0.3 s: the silence threshold is calibrated from it.

## How to run

Record (16 kHz mono):

    python record.py data/raw/shared_test.wav 12                # whisper sentences 01 and 02
    python record.py data/raw/spk01_s03_whisper.wav 5           # spkXX_sNN_{whisper,normal}.wav, NN = line in transcripts.txt

Offline, one file in, one file out (C8), optionally all ablation variants into results/ (C9):

    python run_offline.py data/raw/shared_test.wav results/out.wav
    python run_offline.py data/raw/shared_test.wav results/out.wav --base-f0 150 --aspiration 0.2 --warp 0.92 --variants

Live demo (A7, A8). List devices, then pick input and output; on Windows add --wasapi if latency is high:

    python run_live.py --list-devices
    python run_live.py --in 1 --out 3
    python run_live.py --passthrough                             # mic check only
    python run_live.py --file data/raw/shared_test.wav results/live.wav --play   # no mic: file through the live code path

Gradio file mode with sliders (A10, and the demo fallback):

    python app.py

Evaluation (C10): WER with Whisper base, LSD, ablation table, figures, listening test:

    python run_eval.py                                           # writes results/ablation.md, results/figures/
    python run_eval.py --listening                               # blinded sheet in results/listening/
    python run_eval.py --aggregate-listening results/listening/sheet.csv
    python run_eval.py --dev                                     # synthetic TTS dev clips (python tools/make_dev_clip.py first)

Tests and validation notebooks:

    python -m pytest tests -q
    python -m jupyter nbconvert --to notebook --execute --inplace notebooks/b_lpc_validation.ipynb

## Pipeline

    x, fs = load_audio(path)
    frames = frame_signal(x); energy = frame_energy(frames)
    labels = classify_frames(frames, energy)
    A, G = lpc_all(frames, order=12)                 # optional warp_poles per frame
    f0 = build_f0(labels, energy, base_f0=120)
    e = make_excitation(labels, f0)                  # Rosenberg flow pulses + aspiration, noise, zeros
    y = synthesize(A, G, e, labels, energy)          # lfilter with zi carried across hops
    y = output_stage(y, energy)                      # lip radiation, de-emphasis, DC blocker, per hop RMS match

The live path (src/live.py) runs the same per hop step functions (f0_step, excitation_step, synth_step, output_step) one 160 sample block at a time, so offline and live produce the same sound.

## Function contract

Audio is 16 kHz mono float32, frame 320 samples, hop 160 samples.

* load_audio(path) returns x, fs
* frame_signal(x, frame_len=320, hop=160) returns frames (n_frames, 320)
* frame_energy(frames) returns RMS (n_frames,)
* classify_frames(frames, energy) returns int labels (n_frames,), 0 silence, 1 consonant, 2 vowel like
* lpc_frame(frame, order=12) returns a (order+1,) with a[0]=1, and gain
* lpc_all(frames, order=12) returns A (n_frames, order+1), G (n_frames,)
* warp_poles(a, factor) returns a new a
* build_f0(labels, energy, base_f0=120, decl=0.12, stress=0.25) returns f0 per frame in Hz, 0 where not voiced
* make_excitation(labels, f0, hop=160, fs=16000) returns e of length n_frames*hop
* synthesize(A, G, e, labels, energy, hop=160) returns y

Change a signature only after telling the whole group. Optional keyword arguments added since (defaults keep the contract): classify_frames(..., zcr=None, silence_mult=3.0, zcr_thr=None, return_thresholds=False), build_f0(..., jitter, phrase_gap_s, seed), make_excitation(..., aspiration=0.15, seed=0).

## Working rules

* Edit only your own files. Pull before you start, commit and push at least once a day.
* Test each function alone on data/raw/shared_test.wav before announcing it is ready.
* Until a teammate's module is ready, use src/mocks.py.
* Handoff files go in data/handoff/: labels.npy and energy.npy (Hafsa), A.npy and G.npy (Akhila). Due end of Sat 14 Nov.
* Raw recordings stay local except shared_test.wav and transcripts.txt. Figures and wavs go in results/ and are not committed.

## Timeline

* Fri 13 Nov: setup, first tasks
* Sat 14 Nov: modules validated, handoff files pushed
* Sun 15 Nov: offline integration session
* Mon 16 Nov: live stream, causal versions of build_f0 and synthesize
* Tue 17 to Wed 18 Nov: evaluation, slides
* Thu 19 Nov: rehearsal
* Fri 20 Nov: buffer and submission
