# Whisper to Voice: project description

MAI522 Speech Processing and Recognition, CIA III group mini project. Team of three: Hafsa (Module A), Akhila (Module B), Raks (Module C).

This document describes the problem, the theory, the design, the evaluation and the limits of the system in one place. Every parameter value below is taken from the code in `src/`. For testing and verification procedures see `docs/TESTING.md`; for the commands to run the project see `README.md`.

## 1. Problem and motivation

A whisper is speech produced without vibration of the vocal folds. It keeps the words, because the mouth and throat still shape the sound, but it has no pitch. It is also quiet and breathy, and it is hard to understand in noise. People who cannot use their vocal folds, for example after throat surgery or a vocal fold injury, can often still whisper. Listeners find their speech tiring to follow, and the speaker cannot signal emphasis or a question by pitch.

The project takes whispered speech from a microphone or a file and produces voiced-sounding speech from it, in real time, using classical signal processing only: no cloud text-to-speech and no trained model. Speech is not recognised and re-synthesised from text. The words are kept because the output is built from the whisper's own spectral envelope.

Who may benefit: people with temporary or permanent voice loss, people who whisper in places where speaking aloud is not possible and want to be heard on a call, and anyone who wants an interpretable baseline for whisper-to-speech conversion.

## 2. Background theory

### 2.1 Source-filter model

Speech production is modelled as an excitation signal (the source) passed through a linear filter (the vocal tract):

$$S(z) = E(z)\, H(z)\, R(z), \qquad H(z) = \frac{G}{A(z)}, \qquad R(z) = 1 - 0.98\, z^{-1}$$

| Part | Voiced speech | Whisper |
|---|---|---|
| Source $e[n]$ | Periodic glottal flow pulses at the fundamental frequency $f_0$ | Turbulent air noise, no periodicity |
| Filter $H(z)$ | Vocal tract resonances (formants) | Same resonances, somewhat different bandwidths and positions |
| Radiation $R(z)$ | Lips act roughly as a differentiator, about +6 dB per octave | Same |

The filter carries the words (which vowel, which consonant). The source carries pitch and voicing. The idea of the project is to keep the filter of the whisper and replace the noise source by a synthetic voiced source with a pitch we choose.

### 2.2 Linear prediction (LPC) as an all-pole vocal tract

Over a short frame (20 ms) the signal is treated as stationary and each sample is predicted from the previous $p$ samples:

$$\hat{x}[n] = -\sum_{k=1}^{p} a_k\, x[n-k], \qquad e[n] = x[n] - \hat{x}[n]$$

so that $e = A(z)\,x$ with $A(z) = 1 + \sum_{k=1}^{p} a_k z^{-k}$. Inverting this, the signal is the prediction error (the excitation) passed through the all-pole filter $1/A(z)$. The poles of $1/A(z)$ lie near the unit circle at the vocal tract resonances. A pole at radius $r$ and angle $\theta$ gives a resonance at $F = \theta f_s / 2\pi$ whose bandwidth is narrower the closer $r$ is to 1. The magnitude $|G/A(e^{j\omega})|$ is a smooth spectral envelope that follows the formants and ignores harmonic fine structure.

The coefficients minimise the energy of $e$ and solve the normal equations built from the autocorrelation $r[k] = \sum_n x[n]\,x[n+k]$:

$$\sum_{k=1}^{p} a_k\, r[\,|i-k|\,] = -\,r[i], \quad i = 1,\dots,p$$

The Levinson-Durbin recursion solves this in $O(p^2)$ operations. At step $i$ it computes the reflection coefficient and updates the filter and the error:

$$k_i = -\frac{r[i] + \sum_{j=1}^{i-1} a_j^{(i-1)}\, r[i-j]}{E_{i-1}}, \qquad a_j^{(i)} = a_j^{(i-1)} + k_i\, a_{i-j}^{(i-1)},\quad a_i^{(i)} = k_i, \qquad E_i = (1-k_i^2)\,E_{i-1}$$

The synthesis filter is stable when every $|k_i| < 1$, which the autocorrelation method guarantees.

### 2.3 Glottal excitation

In voiced speech the vocal folds open and close $f_0$ times per second. The airflow through the glottis is a smooth pulse per cycle, not an impulse. A train of unit impulses has a flat line spectrum (all harmonics equally strong) and sounds buzzy. A smooth glottal flow pulse has a spectrum that falls with frequency, like a real voice. The Rosenberg model describes one pulse with a rising phase, a falling phase and a closed phase. Because $f_0$ changes over time, the pulse positions are generated from a running phase $\phi[n] = \phi[n-1] + f_0[n]/f_s$: a new pulse starts every time the phase crosses an integer.

Natural speech also contains a little breath noise in the voiced signal (aspiration noise) and small cycle-to-cycle pitch variation (jitter). Over a phrase, pitch usually drifts down (declination) and rises on stressed syllables.

## 3. Architecture

### 3.1 Pipeline

```
 mic or wav, 16 kHz mono float32
        |
        v
 load_audio  (mono, resample to 16 kHz, DC blocker)                  Module A
        |
        v
 frame_signal (320 samples = 20 ms, hop 160 = 10 ms) -> frame_energy (RMS), frame_zcr
        |
        v
 classify_frames -> label per hop: 0 silence, 1 consonant, 2 vowel-like   Module A
        |                      |
        |                      +--------------------------+
        v                                                 v
 lpc_all (order 12) -> A (n,13), G (n,)  Module B     build_f0 -> f0 per hop (Hz, 0 if not label 2)   Module C
        |  optional warp_poles(a, 0.92)                    |
        |                                                  v
        |                                      make_excitation: label 2 -> Rosenberg pulses + aspiration,
        |                                      label 1 -> white noise, label 0 -> zeros (all unit RMS)   Module C
        |                                                  |
        +------------------------> synthesize: lfilter G/A(z) per hop, filter state carried          Module C
                                                           |
                                                           v
                  output_stage: radiation 1 - 0.98 z^-1 -> de-emphasis 1/(1 - 0.97 z^-1)
                                -> DC blocker -> per-hop RMS match to input energy x volume      Module C
                                                           |
                                                           v
                                      peak_protect (offline) or soft limiter (live) -> wav / headphones
```

### 3.2 Modules, owners and files

| Module | Owner | Files | Responsibility |
|---|---|---|---|
| A | Hafsa | `src/framing.py`, `src/detect.py`, `src/stream.py`, `run_live.py` | Loading, DC removal, framing, RMS and ZCR, voicing strength, frame classification and thresholds, plots, the sound-device stream and the live entry point |
| B | Akhila | `src/lpc.py` | Pre-emphasis, autocorrelation, Levinson-Durbin, gain, envelope, residual, prediction gain, LSD, pole warping |
| C | Raks | `src/pitch.py`, `src/synth.py`, `run_offline.py`, `src/evaluate.py`, `run_eval.py` | f0 contour, glottal pulses and excitation, synthesis, output stage, ablation variants, offline conversion, WER, LSD and listening-test tooling |
| A and C | shared | `src/live.py` | Causal per-hop pipeline, latency report and plot |
| all | shared | `app.py`, `src/mocks.py`, `tests/`, `notebooks/` | Gradio app, mocks, tests, validation and walkthrough notebooks |

### 3.3 Function and data contract

Audio is 16 kHz mono float32, frame 320 samples, hop 160 samples. These signatures were fixed at the start so that three people could build separate parts:

| Function | Returns |
|---|---|
| `load_audio(path)` | `x, fs` |
| `frame_signal(x, frame_len=320, hop=160)` | frames `(n_frames, 320)`, no window |
| `frame_energy(frames)` | RMS `(n_frames,)` |
| `classify_frames(frames, energy)` | int labels `(n_frames,)`: 0 silence, 1 consonant, 2 vowel-like |
| `lpc_frame(frame, order=12)` | `a` of length `order+1` with `a[0]=1`, and gain |
| `lpc_all(frames, order=12)` | `A (n_frames, order+1)`, `G (n_frames,)` |
| `warp_poles(a, factor)` | new `a` |
| `build_f0(labels, energy, base_f0=120, decl=0.12, stress=0.25)` | f0 per frame in Hz, 0 where not voiced |
| `make_excitation(labels, f0, hop=160, fs=16000)` | `e` of length `n_frames * hop` |
| `synthesize(A, G, e, labels, energy, hop=160)` | `y` |

Optional keyword arguments were added later with defaults that keep the contract (for example `aspiration`, `seed`, `jitter`, `phrase_gap_s`, `zcr_thr`, `return_thresholds`).

### 3.4 Offline and live paths

The offline path (`src/synth.py`, `run_offline.py`) processes a whole clip: thresholds come from the clip, labels are smoothed with a median filter that looks one frame ahead, and `build_f0` sees whole phrases.

The live path (`src/live.py`, `src/stream.py`) processes one 160-sample hop per call and may use only the past. `process_hop_live(block, state)` takes one block and returns one block, carrying a `state` dictionary (previous block, filter memories, f0 tracker state, excitation phase and pulse tail, output gain, timings). Per hop it:

1. removes DC (`dc_block` with carried state);
2. forms the 320-sample frame from the previous block and the current block (the frame ends at the current hop);
3. during the first 0.3 s only measures the noise floor and outputs zeros;
4. then labels the hop, applies a causal hangover, runs `lpc_frame`, optional `warp_poles`, `f0_step`, `excitation_step`, `synth_step` and `output_step`, and applies a soft limiter.

The three per-hop step functions `excitation_step`, `synth_step` and `output_step` are the same functions the offline path loops over (`make_excitation`, `synthesize` and `output_stage` are loops around them). What differs between the paths is only what feeds them:

| Aspect | Offline | Live |
|---|---|---|
| Silence threshold | 3 x median RMS of first 0.3 s of the clip (with the 10th-percentile fallback, see 4.3) | Same function, given only the calibration hops of the first 0.3 s of the stream, so it always takes the fallback: 3 x 10th percentile of those RMS values; then frozen |
| ZCR threshold | 70th percentile of ZCR of all non-silent frames | Starts at 0.12, then 70th percentile of the last 3 s of non-silent frames, once at least 0.5 s of speech was heard, updated every 10 non-silent hops |
| Label smoothing | 5-frame hangover, then 3-frame median | 5-frame causal hangover, no median |
| Frame for hop $i$ | starts at hop $i$ | ends at hop $i$ (one hop earlier in time) |
| f0 | `build_f0` per phrase | `f0_step`: running phrase time, running mean energy |
| Peak handling | `peak_protect` at 0.99 | `tanh` soft limiter above 0.9 |

### 3.5 Gradio app and walkthrough notebook

`app.py` is a tabbed Gradio app and the fallback demo when the live stream cannot be used. Convert: upload or record a clip, choose a variant or the sliders (base pitch 70 to 250 Hz, declination, stress, aspiration, pole warp 0.85 to 1.0, volume), listen to input and output, see spectrograms and the label and f0 plot, download the wav, compare all variants, and press Transcribe to run the Whisper ASR model (tiny, base or small, on CPU) on the input and on the voiced output shown in the player, side by side, with word error rate for each if a reference sentence is typed (reusing `evaluate.transcribe` and `wer_counts`). On the synthetic dev set Whisper recognised the raw whispered clips better than the voiced output, so the Transcribe button may well show the input transcript as the better one; that is a real result of the current synthesis, not a bug in the button. Analysis: Module A features and labels with adjustable thresholds, and Module B's LPC envelope for any frame. Results: tables, figures and variant players produced by `run_eval.py`. The app's helpers are pure functions that are covered by tests; Gradio is imported only when the app is built.

`notebooks/walkthrough.ipynb` runs every stage on one clip with plots, audio and explanations, including the live versus offline proof and the LSD per variant. It runs without recordings, devices or Whisper.

## 4. Design decisions and parameters

### 4.1 Framing and pre-processing

| Choice | Value | Reason |
|---|---|---|
| Sample rate | 16 kHz | Covers the speech band up to 8 kHz and matches Whisper's input rate. |
| Frame and hop | 320 samples (20 ms), 160 samples (10 ms) | Speech is roughly stationary over about 20 ms. A 10 ms hop gives smooth filter updates and a 10 ms real-time budget. |
| Input DC removal | One-pole high-pass $(1 - z^{-1})/(1 - 0.995 z^{-1})$, cutoff about 13 Hz, started in steady state | Sound cards often add an offset that would disturb the energy-based silence detection. The same function is used per hop in the live path with carried state. |
| Framing window | None in `frame_signal` | The Hamming window is applied inside the LPC step. RMS and ZCR use the raw frame. |

### 4.2 LPC analysis (Module B)

* Pre-emphasis $y[n] = x[n] - 0.97\,x[n-1]$ flattens the spectral tilt, so low frequencies do not dominate the fit.
* Hamming window (`np.hamming(320)`) reduces edge effects.
* Autocorrelation to lag $p$, with $r[0] \leftarrow r[0](1 + 10^{-9}) + 10^{-12}$ as a tiny white-noise correction that keeps the recursion well conditioned.
* Levinson-Durbin as above. A silent frame (energy below $10^{-12}$) gives $a = [1, 0, \dots]$ and gain 0.
* Gain $G = \sqrt{E_p / \sum_n w[n]^2}$, where $E_p$ is the final prediction error of the windowed frame. Because the all-pole model reproduces the autocorrelation $r[0..p]$, unit-variance white noise through $G/A(z)$ has exactly the window-weighted RMS of the pre-emphasised frame, $\sqrt{\sum_n (w[n]\,x_\text{pre}[n])^2 / \sum_n w[n]^2}$ (checked numerically: ratio 1.000 on every non-silent fixture frame). Against the plain RMS of the pre-emphasised frame the ratio is close to 1 but not exact (median 0.99, 5 to 95 percent range 0.75 to 1.13 on the fixture), because a frame is not stationary. It does not match the RMS of the raw (not pre-emphasised) frame. Since the output stage re-scales every hop to the input energy anyway, $G$ has little effect on the final loudness.
* **Order 12.** The common rule of thumb is $f_s/1000 + 2 = 18$ at 16 kHz (two poles per formant, about one formant per kHz, plus two for spectral tilt). The project uses 12 because (a) the order sweep in the Module B notebook shows diminishing returns: mean envelope-versus-spectrum LSD over the non-silent frames of the synthetic fixture is 5.05 dB at order 12, 4.83 dB at 14 and 4.35 dB at 24, so the step from 12 to 14 gains only 0.21 dB, against 0.32 dB from 10 to 12; (b) the team guide warns that a high order fits harmonics and noise and makes the synthesis sound unstable; (c) it is cheaper per hop. The sweep always decreases with order, so this is a judgement at the elbow, not a minimum. The notebook's rule (the first order whose next step of 2 gains less than 0.25 dB) uses a threshold we chose; a 0.2 dB threshold would have picked 14. The sweep ran on the synthetic fixture, whose vowels are made with only three formant resonances (a 6th-order all-pole filter), so it cannot show what real speech needs. The choice must be re-checked on real recordings.

### 4.3 Frame classes and calibration (Module A)

Each hop gets one label: 0 silence, 1 consonant (noisy, high ZCR), 2 vowel-like (loud, low ZCR).

* **Silence threshold** = 3 x median RMS of the first 0.3 s (30 hops), never below $10^{-4}$. This is why the speaker must stay silent for the first 0.3 s. If the median of those frames is above the 25th percentile of all frame energies (for example because the clip starts with speech), the noise floor falls back to the 10th percentile of all energies. On a clip that is mostly silence the fallback can trigger even when the start is quiet, because the 25th percentile then lies inside the noise floor; this happens on the synthetic fixture, where the result (0.0058) is within 4 percent of the plain rule (0.0060). The live path calls the same function with only the 29 calibration hops (hops 1 to 29; hop 0's frame is half zeros), whose median is always above their own 25th percentile, so live always uses 3 x the 10th percentile of the calibration RMS values (0.0057 on the fixture).
* **ZCR threshold** = 70th percentile of the ZCR of the non-silent frames. ZCR is the fraction of the 319 adjacent sample pairs that change sign. Fricatives such as "s" cross zero often; vowels do not.
* **Hangover**: after a non-silent frame, a silent frame inherits the previous label for up to 5 frames (50 ms), so quiet word endings are not cut.
* **3-frame median filter** (offline only) removes single-frame flips.

A voicing-strength measure (peak of the normalised autocorrelation for lags 40 to 228 samples, i.e. 400 to 70 Hz) is implemented and shown in the app and the notebooks, but the classifier does not use it. In a whisper there is no periodicity to find; the measure is a diagnostic that shows how much more periodic the output is than the input.

### 4.4 Pitch contour (Module C)

For label-2 frames only, per phrase (a phrase ends at a silence longer than 0.5 s):

$$f_0[i] = f_\text{base}\,\bigl(1 - d\,\tfrac{i - i_0}{i_1 - i_0}\bigr)\,\bigl(1 + s\,\Delta_i\bigr)\,\bigl(1 + j\,u_i\bigr)$$

where $i_0$ and $i_1$ are the first and last non-silent frames of the phrase, with $f_\text{base} = 120$ Hz, declination $d = 0.12$, stress $s = 0.25$, jitter $j = 0.005$, $u_i$ uniform in $[-1, 1]$ (seeded), and $\Delta_i = \operatorname{clip}(\ln e_i - \overline{\ln e}, -1, 1)$ the log RMS of the frame relative to its mean over the voiced frames of the phrase. Each voiced run is smoothed by a 5-point median followed by a 3-point moving average, and f0 is clipped to 70 to 400 Hz. `f0_step` (live) uses $\min(t/2\,\text{s}, 1)$ as the declination position, a running mean energy for $\Delta$, and a one-pole smoother with factor 0.5. A phrase resets after 0.5 s of silence in both versions.

Reasons: 120 Hz is a typical male speaking pitch and is a slider (70 to 250 Hz in the app); a 12 percent fall per phrase and a 25 percent stress effect are the values the team guide suggests as a starting point; loud frames are taken as stressed, since whisper carries stress in loudness.

### 4.5 Excitation

* **Rosenberg pulse**: for a period of $L = \text{round}(f_s/f_0)$ samples, opening phase $T_p = 0.40\,L$ samples with $g[n] = \tfrac{1}{2}\bigl(1 - \cos(\pi n / T_p)\bigr)$, closing phase $T_n = 0.16\,L$ samples with $g[m] = \cos\bigl(\pi m / (2 T_n)\bigr)$, then closed (zero) for the rest. This is the glottal flow, not its derivative; the lip radiation filter in the output stage supplies the derivative-like effect. (The team guide suggests also differencing the pulse; we apply the differencing once, as the radiation filter, so it is not applied twice.) Each pulse is normalised to unit RMS.
* **Running phase**: $f_0$ is interpolated linearly across each hop, the phase accumulates, and a pulse is started whenever the phase crosses an integer. The tail of a pulse that crosses a hop boundary is carried over (401 samples), so there is no click and no 100 Hz modulation at the hop rate. $f_0$ is limited to 40 to 800 Hz at this stage.
* **Aspiration**: with weight $a = 0.15$, $e = \bigl((1-a)\,p + a\,\nu\bigr)/\sqrt{(1-a)^2 + a^2}$, where $p$ is the unit-RMS pulse signal and $\nu$ unit-RMS white noise. The result has about unit RMS (exactly so only on average, since pulse and noise are uncorrelated and the pulse signal's RMS per hop varies between about 0.9 and 1.16), and the noise carries $a^2/((1-a)^2+a^2) \approx 3$ percent of the power. The team guide asks for 10 to 20 percent noise; the weight 0.15 is in that range.
* **Label 1**: unit-RMS white noise (the same kind of source a whisper has, so fricatives keep their character). **Label 0**: exactly zero. A label-2 hop with $f_0 = 0$ falls back to noise.
* Everything has unit RMS per hop so loudness is controlled in one place, the output stage.

### 4.6 Pole warping

`warp_poles(a, factor)` finds the roots of $A(z)$ and multiplies the angle of every complex pole by `factor`, keeping the radius (capped at 0.9999) and leaving real poles unchanged; it rebuilds a real polynomial from conjugate pairs:

$$z_k = r_k e^{j\theta_k} \;\longrightarrow\; r_k e^{j\alpha\theta_k}, \qquad F_k = \tfrac{\theta_k f_s}{2\pi} \;\longrightarrow\; \alpha F_k$$

With $\alpha = 0.92$ every formant moves down by 8 percent. The team guide gives the reason: whispered formants sit higher than in voiced speech of the same speaker, and warping by 0.90 to 0.95 compensates. It can also be read as a slightly longer vocal tract. Radius and conjugate symmetry are preserved, so the filter stays stable (tested for factors 0.85 to 1.0). The factor is fixed, not estimated per speaker.

### 4.7 Synthesis and output stage

* **Synthesis**: `scipy.signal.lfilter([G], a, e_hop, zi=zi)` per hop with the filter state `zi` carried across hops. Resetting it every hop would click. With constant coefficients the hop-wise result equals one `lfilter` call over the whole signal (tested).
* **Lip radiation** $1 - 0.98\,z^{-1}$, as in the team guide.
* **De-emphasis** $1/(1 - 0.97\,z^{-1})$. The LPC was fitted on pre-emphasised speech, so its envelope still contains the +6 dB per octave of the pre-emphasis; without de-emphasis the output is too bright and the high band hides the harmonics. The team guide lists radiation only, so this is an addition. Radiation and de-emphasis together are nearly flat above a few hundred Hz (about $-1$ dB at 100 Hz, $-0.3$ dB at 200 Hz, $0$ dB above 500 Hz) with a DC gain of about 0.67.
* **DC blocker** $(1 - z^{-1})/(1 - 0.99\,z^{-1})$, cutoff about 25 Hz. The Rosenberg flow pulse has a large mean; without removal the RMS match would spend loudness on inaudible DC. A test confirms that less than 5 percent of output power lies below 60 Hz.
* **RMS match**: the gain for each hop is target RMS (input frame energy x volume) divided by the hop's RMS, capped at 50. The gain is ramped linearly across the hop from the previous gain, which is itself limited to within a factor 2 of the new gain, so loudness follows the whisper without steps. Hops with RMS below $10^{-12}$ are passed through unscaled (they are effectively zero) and do not update the gain.
* **Peak protection**: offline, the whole signal is scaled down only if a sample would exceed 0.99. Live, samples above 0.9 pass through $0.9 + 0.1\tanh((|y|-0.9)/0.1)$, because the whole signal is not known.

## 5. Evaluation method

### 5.1 Ablation variants

Four cumulative variants (`VARIANTS` in `src/synth.py`) share the same labels, energy and LPC; only the source or filter changes.

| Variant | Settings | What it adds |
|---|---|---|
| `v1_robot` | declination 0, stress 0, jitter 0, aspiration 0, no warp | Baseline: a monotone pulse source at a fixed pitch (120 Hz), Rosenberg pulses, noise for consonants |
| `v2_contour` | default declination, stress and jitter, aspiration 0 | Adds the f0 contour |
| `v3_aspiration` | adds aspiration 0.15 | Adds breath noise to the voiced source (the default pipeline) |
| `v4_warp` | adds pole warp 0.92 | Adds formant warping |

The raw whisper is also scored as `raw_whisper`, and the normal recording of the same sentence, if present, as `normal_recording` (an upper bound for the ASR).

### 5.2 Word error rate with Whisper

The intelligibility judge is Whisper ASR (`openai-whisper`, model `base`, on the CPU, `language="en"`, temperature 0, no conditioning on previous text). The system output is transcribed and compared with the known sentence text (`data/raw/transcripts.txt`). Text is normalised (lower case, hyphens to spaces, punctuation removed). With `jiwer`:

$$\text{WER} = \frac{S + D + I}{N}$$

where $S$, $D$, $I$ are substitutions, deletions and insertions and $N$ the number of reference words. WER can exceed 1. The table reports the mean over clips and the corpus WER (total errors over total reference words). Whisper is trained mostly on normal speech and the sentences are short, so the number is a proxy for intelligibility and carries large uncertainty on a handful of sentences.

### 5.3 Log spectral distortion

Per frame, the LPC envelopes (order 12, 512-point grid, 257 frequencies from 0 to 8 kHz, gain included) of the input whisper and the output are compared:

$$\text{LSD}_i = \sqrt{\frac{1}{K}\sum_{k=1}^{K}\bigl(20\log_{10}|H_\text{in}(\omega_k)| - 20\log_{10}|H_\text{out}(\omega_k)|\bigr)^2}$$

and the reported value is the mean of $\text{LSD}_i$ over the non-silent frames (labels from the input). The raw whisper row is 0 by definition. Lower means the output keeps the input's spectral shape. Because the gain and the output's overall tilt enter the comparison, LSD is also affected by level and tilt differences, not only by formant positions, and a deliberate warp can change it.

### 5.4 Blinded listening test

`run_eval.py --listening` writes, for every clip, the raw whisper and the four variants as shuffled files `item_NN.wav` (fixed seed), a score sheet for five listeners (L1 to L5), and a key file that maps items to systems and is not shown to listeners. Listeners score clarity and naturalness from 1 (worst) to 5 (best). `--aggregate-listening` computes the mean per system and writes `listening_scores.csv`; the scores appear in the ablation table the next time `run_eval.py` is run. The listening test has not been carried out yet. Participation is voluntary, only scores and listener ids are stored.

### 5.5 Other checks

Per-hop processing time against the 10 ms budget (latency report), f0 histograms of the output against the paired normal recording (`fig_f0.png`, with `librosa.pyin`), spectrogram and envelope figures, and the unit tests and notebooks in `docs/TESTING.md`.

## 6. Results

**There are no results on real whispered recordings yet.** `data/raw/shared_test.wav` and the speaker recordings are not in the repository, and `results/` is not committed. Real-recording numbers are pending; this section will be filled when `python run_eval.py` has been run on real recordings, and the listening test has been done.

What exists is a pipeline check on **synthetic dev data**: six Harvard sentences were spoken by the Windows SAPI text-to-speech voice and turned into pseudo-whispers by filtering white noise through their own per-frame LPC filters (`tools/make_dev_clip.py`). These are not whispered speech from a person. They show that the evaluation code runs end to end and nothing else.

Table produced on 2026-11-22 with the committed code (commit b0e1301) by `python run_eval.py --dev` (written to `results/dev_eval/`), on the six dev clips in `results/dev/` (Whisper base on CPU, 49 reference words). Three separate runs gave identical numbers, so the evaluation is deterministic:

| System | WER, mean per clip (corpus) | LSD (dB) |
|---|---|---|
| raw_whisper (synthetic) | 0.328 (0.327) | 0 (reference) |
| v1_robot | 0.782 (0.796) | 10.49 |
| v2_contour | 0.736 (0.755) | 11.42 |
| v3_aspiration | 0.492 (0.510) | 6.77 |
| v4_warp | 0.750 (0.776) | 6.38 |
| normal_recording (SAPI speech) | 0.042 (0.041) | n/a |

An earlier run (2026-11-20 21:20, kept locally in `results/dev_eval_2026-11-20/`, before the later commits that added DC removal at input and output and fixed the pole warp) gave WER 0.707, 0.784, 0.545 and 0.594 and LSD 11.23, 12.03, 6.63 and 6.57 for the four variants (raw whisper and normal speech unchanged). Since the evaluation is deterministic, the differences between the two runs come from the code changes, not from chance. That small processing changes move the WER of single variants by up to 0.16 shows how sensitive a six-sentence evaluation is.

What can be said from this, and what cannot:

* On this synthetic data every voiced variant has a higher WER than the raw synthetic whisper. The conversion did not make Whisper's transcription better here. The team guide warned that an ASR trained on normal speech may not improve on rule-based pitch.
* The aspiration variant has the lowest WER of the four in both runs and a much lower LSD than the first two (6.6 to 6.8 dB against 10.5 to 12 dB). A plausible reason is that noise between the harmonics fills the spectral valleys that a pure pulse train leaves empty. This is a hypothesis, not something the experiment tests.
* The ordering of `v1_robot`, `v2_contour` and `v4_warp` changes from run to run. No claim about the contour or the warp can be made from this data.
* The raw "whisper" here is itself an LPC resynthesis of clean TTS speech, so it is easier for Whisper than a real whisper in a real room would be.

## 7. Live versus offline

The live and offline outputs are not sample-identical and are not meant to be: the live path cannot see the future. A direct comparison of the waveforms on the walkthrough clip gives a sample correlation of only 0.08, which says little because two voices with slightly different pitch-pulse positions decorrelate quickly. The loudness envelopes correlate at 0.95.

The informative check is the proof cell in section 8 of `notebooks/walkthrough.ipynb`. It runs the clip through `process_hop_live` one hop at a time, records every decision the live path made (label, f0 and the 320-sample frame), then runs the offline chain (`lpc_all`, `make_excitation`, `synthesize`, `output_stage`) on those recorded decisions. The result differs from the live output by at most $6.0 \times 10^{-8}$ (rounding error, float32). The live soft limiter did not engage on this clip (live peak 0.77, knee 0.9); the offline chain has no limiter, so on a louder clip the two would differ above 0.9. Since both paths call the same step functions, this is a check that the offline loops, the carried states and `lpc_all` on the recorded frames behave exactly like the per-hop live calls, not an independent re-implementation. So the synthesis and output code is the same in both paths, and the audible difference comes only from the causal inputs: on the same clip the live labels agree with the offline labels on 91 percent of hops (aligned by one hop, after calibration), and the f0 from `f0_step` differs from `build_f0` on the same labels by 5.7 Hz on average (maximum 13.2 Hz). These numbers were re-run for this document with the committed code and reproduced exactly. They come from the synthetic fixture because no real recording is available.

Processing time on the development machine for the same clip (257 hops), over repeated runs: mean about 0.3 ms, p95 about 0.5 to 0.6 ms, max 0.9 to 1.3 ms per hop, none over the 10 ms budget in those runs. Single slow hops do occur: in one test run on the same machine the mean was 0.75 ms but one hop took 28 ms, which in a real stream could cause a dropout unless the device buffer absorbs it. The algorithmic delay is defined in `latency_report` as two hops (20 ms): one hop to collect the 160-sample input block before the callback can run, and one hop for the returned block to be played out. It is a model, not a measurement. The device latency on a real sound card, and so the total delay the speaker experiences, has not been measured yet.

## 8. Novel contribution

The project is a classical LPC source-filter converter. Compared with a basic version that excites an LPC filter with an impulse train, these are the contributions of this project. Several of the ideas (Rosenberg pulse, aspiration, declination and stress rules, pole warping, WER evaluation) are suggested in the team guide and are known techniques; what is claimed here is their implementation from first principles, their integration into one causal and offline system, and their measurement, not that the ideas are new in the literature.

1. **Rosenberg glottal flow pulse plus aspiration noise instead of a plain impulse train.** The voiced source in `src/pitch.py` is a unit-RMS Rosenberg flow pulse (opening 40 percent, closing 16 percent of the period) with a carried phase and pulse tail, mixed with 0.15 weight of white noise. The impulse train (`impulse_train`) is kept only as a reference and for tests. The `v3_aspiration` variant isolates the effect of the noise; the pulse shape itself is not isolated by any variant (all four use Rosenberg pulses), so its benefit over an impulse train is argued from the spectra, not measured.
2. **A synthetic, stress- and declination-driven f0 contour, built without any pitch to copy.** A whisper has no f0, so the contour is constructed from the label track and the frame energies: linear declination per phrase, a stress term from the log energy, jitter, smoothing and phrase reset after 0.5 s of silence (`build_f0` offline, `f0_step` live). The `v2_contour` variant isolates the effect.
3. **Formant (pole) warping of the LPC filter.** `warp_poles` scales the angle of every complex pole of $A(z)$ by a factor (0.92 in `v4_warp`) while keeping the radius, so the filter stays stable and formants move down proportionally. The `v4_warp` variant isolates the effect.
4. **A causal live path that shares its per-hop step functions with the offline path, with a demonstration that the offline chain fed with the live decisions reproduces the live output.** `excitation_step`, `synth_step` and `output_step` are called by both paths, and `notebooks/walkthrough.ipynb` shows the maximum difference of $6.0 \times 10^{-8}$ when the offline chain is given the live labels, f0 and frames.
5. **Evaluation using Whisper ASR as the intelligibility judge (WER), alongside LSD and a blinded listening test, as an ablation over the four variants.** `src/evaluate.py` and `run_eval.py` produce per-clip and aggregate tables, figures and the blinded sheet for the same four variants. So far the tooling has only been run on synthetic dev data, and the listening test has not been done.

## 9. Limitations

* **Prosody is a rule, not the speaker's intonation.** The contour depends only on the label track and the loudness. It cannot know a question from a statement, and it does not follow the words; it sounds monotone-ish with a gentle fall and small bumps at loud frames. Stress in a whisper may not be carried by loudness alone.
* **Voiced/unvoiced labels are a two-feature rule** (RMS and ZCR with percentile thresholds). Weak vowels, breathy consonants and voiced consonants with high ZCR can be mislabelled: a fricative labelled 2 buzzes, a vowel labelled 1 stays whispered. Stop closures shorter than the 50 ms hangover are filled in. Live, the labels are less accurate than offline (no median filter, adaptive ZCR threshold, 91 percent agreement on the test clip).
* **LPC order 12 is a trade-off.** It is below the usual rule of thumb (18). It captures the main formants, but upper formants and spectral detail above about 4 kHz are modelled coarsely, and a whisper's envelope is only an approximation of the voiced tract (broader bandwidths, no voice bar at low frequencies). The order was chosen from a sweep on a synthetic clip.
* **Fixed warp factor and fixed base pitch.** The 0.92 warp and the 120 Hz base pitch are not estimated from the speaker. Gender, age and personal pitch range have to be set by the slider.
* **Noise and aspiration.** Background noise is part of the whisper and so shapes the LPC filter in active frames. In silent frames the excitation is exactly zero, so once the filter ringing has died away (a few hops, starting about 40 dB below speech) the output is digital silence, and the room's noise floor appears and disappears with the label. The aspiration weight is a constant and is not shaped per pitch cycle. Loudness is matched to the input RMS, which includes noise.
* **Calibration needs silence at the start.** The live output is exactly zero during the first 0.3 s, so anything said then is lost. Because live uses the 10th percentile of the calibration hops, a few silent hops are enough: on the fixture, 50 ms of silence before speech still gave the normal threshold, but with 20 ms or less the threshold was set on speech (about 18 times too high) and most quiet speech was labelled silence. Offline, the 25th-percentile guard catches a clip that starts with speech. The live ZCR threshold starts from a default (0.12) and needs at least 0.5 s of speech to adapt.
* **Offline output length.** Frames are 320 samples with a 160 hop, so the last 160 to 319 samples of a clip are not synthesised and are padded with zeros.
* **Latency.** The algorithmic delay is 20 ms by the model in `latency_report`. Device buffers add to it and have not been measured on the demo machine. Processing runs inside the sound-device callback in Python; a slow or busy machine can underrun (single hops of tens of milliseconds have been seen in a test run). If processing raises an exception in the callback, that hop is output as silence and the error is counted and printed after the stream stops, so the stream keeps running.
* **The ablation has no impulse-train baseline.** All four variants use Rosenberg pulses, so the benefit of the pulse shape is not measured.
* **Evaluation is small.** There are six sentences, one or a few speakers, an ASR (Whisper base) trained on voiced speech, and a small listening panel. The only table that exists is on synthetic data. LSD compares envelopes (including level and tilt) and is not a perceptual measure; WER depends on the ASR model.
* **No learned model.** The output is audibly synthetic. There is no neural vocoder, no speaker identity, no automatic formant correction, and no handling of languages other than the English test sentences (Whisper is forced to English in the evaluation).
* **Recordings are missing from the repository.** The committed notebooks therefore show the synthetic fixture, and real-recording results are pending.

## 10. Future work

* Record the real data set, run `run_eval.py` and the listening test, and report all numbers, including unfavourable ones. Split by speaker for tuning and testing.
* Re-run the order sweep and the warp factor on real whispers, and choose them by WER and listening scores instead of only by LSD. Estimate the warp per speaker.
* Better labelling: add voicing-strength and spectral-tilt features or a small supervised classifier trained on labelled whispers, and use hysteresis in the live thresholds.
* Better prosody: predict pitch from the word or syllable structure, or from text if a recogniser is available, and let the user draw or set intonation. Match base pitch to a chosen speaker.
* Smoother live behaviour: store the calibration for later sessions, adapt the silence threshold during the stream, and measure the round-trip latency with a loopback test on the demo machine.
* Evaluate with more ASR models and with additional objective measures such as STOI (the `pystoi` package is installed but not used yet).

## 11. How to run

See `README.md` for all commands and `docs/TESTING.md` for the test and verification procedures. The short version:

    pip install -r requirements.txt
    python -m pytest tests -q                                  # 112 tests
    python run_offline.py in.wav out.wav --variants             # one file in, one file out, plus the four variants
    python run_live.py --list-devices                           # then: python run_live.py --in N --out M
    python run_live.py --file in.wav out.wav --play             # live code path without a microphone
    python app.py                                               # Gradio app
    python run_eval.py                                          # WER, LSD, ablation table (pip install openai-whisper first)

Stay silent for the first 0.3 s of every clip or stream, and use headphones for live tests.

## 12. Repository map

| Path | Contents |
|---|---|
| `src/framing.py` | `dc_block`, `load_audio`, `frame_signal`, `frame_energy`, `frame_zcr` |
| `src/detect.py` | `voicing_strength`, `calibrate_thresholds`, `label_frame`, `classify_frames`, plots, `save_handoff` |
| `src/lpc.py` | `preemphasis`, `autocorr`, `levinson`, `lpc_frame`, `lpc_all`, `residual`, `prediction_gain_db`, `lpc_envelope`, LSD functions, `warp_poles` |
| `src/pitch.py` | `impulse_train`, `rosenberg_pulse`, `build_f0`, `f0_step`, `build_f0_causal`, `excitation_step`, `make_excitation` |
| `src/synth.py` | `synth_step`, `synthesize`, `output_step`, `output_stage`, `peak_protect`, `convert`, `VARIANTS`, `render_variant`, `run_ablation` |
| `src/live.py` | `default_params`, `init_live_state`, `process_hop_live`, `process_file_live`, `latency_report`, `plot_latency` |
| `src/stream.py` | sounddevice duplex stream, device listing, WASAPI helpers, latency print and plot |
| `src/evaluate.py` | Transcripts, clip discovery, Whisper ASR, WER, LSD per variant, ablation tables, listening sheet and aggregation, report figures |
| `src/mocks.py` | Mock inputs for testing one module alone |
| `run_offline.py`, `run_live.py`, `run_eval.py`, `record.py` | Command-line entry points |
| `app.py` | Gradio app (Convert incl. Transcribe, Analysis, Results); `app_a.py` is an older stand-alone Module A view |
| `tools/make_dev_clip.py` | Synthetic dev clips from Windows SAPI TTS |
| `tests/` | 112 pytest tests, `fixtures.py` (synthetic pseudo-whisper), `validate_module_a.py` (script) |
| `notebooks/` | `a_framing_validation`, `b_lpc_validation`, `c_synthesis_validation`, `walkthrough` |
| `data/raw/` | Recordings (git-ignored except `shared_test.wav` and `transcripts.txt`), `data/handoff/` for the handoff files |
| `results/` | Generated wavs, figures and tables (not committed) |
| `docs/` | `whisper_to_voice_team_guide.pdf` (specification), `TESTING.md`, `PROJECT.md`, `viva/` |
