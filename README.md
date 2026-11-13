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
    python run_offline.py data/raw/shared_test.wav results/out.wav

Use headphones for every live test.

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

Change a signature only after telling the whole group.

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
