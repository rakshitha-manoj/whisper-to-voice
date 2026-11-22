# Viva guide: Module A (Hafsa)

Audio, detection, live streaming and display. This guide is written so that you can learn the topic from this document alone, then be questioned on it. Every number below was checked against the code in this repository (and, where noted, by running small snippets). Where a reason for a choice is not written down anywhere in the code, comments or the team guide, the text says so and gives the standard engineering reason instead. Do not present those as "the team decided X because Y"; present them as "the usual reason is Y".

Files you own: `src/framing.py`, `src/detect.py`, `src/stream.py`, `run_live.py`, `record.py`, `app_a.py`, and the Module A parts of `src/live.py` (causal labelling, calibration, adaptive ZCR threshold, hangover, latency report). Tests: `tests/test_module_a.py`, your parts of `tests/test_live.py`. Notebooks: `notebooks/a_framing_validation.ipynb`, sections 1 to 3 and 8 of `notebooks/walkthrough.ipynb`.

Contents

1. Your module in one paragraph, and where it sits
2. Theory from scratch
3. Function by function walkthrough
4. Frame classification in depth (offline and live)
5. Real time streaming
6. Tests that cover your module
7. Weak spots and honest answers
8. Likely viva questions with model answers
9. Cheat sheet

---

## 1. Your module in one paragraph, and where it sits

Module A is everything that touches the raw audio and decides what each tiny slice of it is. It loads a recording (or takes microphone blocks), makes it mono, 16 kHz and free of DC offset, cuts it into overlapping 20 ms frames every 10 ms, measures how loud (RMS) and how noisy (zero-crossing rate) each frame is, and gives every frame a label: 0 silence, 1 noisy consonant, 2 vowel-like. It also provides a periodicity measure (`voicing_strength`) used only as a diagnostic, the spectrogram and label plots, the real-time sound-card stream, and the latency measurement. The labels and the per-frame energy are the contract with the rest of the team.

Where it sits (offline pipeline, from `.claude/STRUCTURE.md`):

```
load_audio -> frame_signal -> frame_energy -> classify_frames  ->  lpc_all -> build_f0 -> make_excitation -> synthesize -> output_stage -> wav
 [A]           [A]            [A]             [A]                  [B]        [C]         [C]               [C]           [C]
```

What your outputs feed:

| Your output | Consumer | Used for |
|---|---|---|
| `frames` (n_frames, 320) | Module B (`lpc_all`) | LPC analysis of each 20 ms frame |
| `labels` (0/1/2) | Module C (`build_f0`, `make_excitation`, `synthesize`) | 0: excitation exactly zero. 1: white noise, no pitch. 2: glottal pulse train with pitch |
| `energy` (RMS per frame) | Module C (`build_f0` stress, `output_stage` loudness match) | Pitch accent follows loudness; output RMS is matched to input RMS |
| `data/handoff/labels.npy`, `energy.npy` (`save_handoff`) | teammates working alone | Handoff files from the README |

In the live path, `process_hop_live` (in `src/live.py`) calls your pieces once per 160-sample hop: `dc_block`, then RMS and ZCR of the 320-sample frame ending now, then `label_frame` with causal thresholds and a causal hangover, then hands the label to Modules B and C's step functions. `src/stream.py` is the sound-card shell around that function.

Contract (README): audio is 16 kHz mono float32, frame 320, hop 160. `load_audio(path)` returns `x, fs`; `frame_signal(x, 320, 160)` returns `(n_frames, 320)`; `frame_energy(frames)` returns RMS; `classify_frames(frames, energy)` returns int labels `(n_frames,)`.

---

## 2. Theory from scratch

### 2.1 Sampling and why 16 kHz

A microphone gives a continuous voltage. A computer stores it as numbers measured at regular instants: samples. The sampling rate $f_s$ is how many per second. Sampling at $f_s$ can only represent frequencies below the Nyquist limit $f_s/2$. At 16 kHz that is 8 kHz.

Why that is enough: most of the information that makes speech intelligible (the formants, i.e. the resonances of the vocal tract, and the noise of fricatives) lies below about 4 to 8 kHz. The standard "wideband speech" rate is 16 kHz; telephone speech is 8 kHz. 16 kHz is also the rate Whisper expects for the WER evaluation in Module C (a general fact about Whisper). The team guide fixes 16 kHz as the contract; the reason is not argued in code, the above is the standard reason.

If a file is recorded at another rate (say 44.1 kHz), `load_audio` resamples it. When resampling down you must low-pass first, otherwise frequencies above 8 kHz fold back (alias) into the band. `scipy.signal.resample_poly` does the low-pass filtering as part of the polyphase resampling (its default is a Kaiser-window FIR anti-alias filter).

Each sample is a float32 between -1 and 1. The team guide warns against mixing int16 and float32; everything is converted once, at loading.

### 2.2 Why we cut the signal into frames: quasi-stationarity

Everything we want to measure (spectrum, pitch, loudness) is only well defined for a signal whose properties do not change while you measure it. Speech is not like that: the mouth moves, and sounds change every 50 to 100 ms or faster. But the vocal tract cannot move instantly, so over a very short window (roughly 10 to 30 ms) the signal behaves like a stationary one. This is called quasi-stationarity. The team guide says it plainly: "Speech changes slowly, so we study it in 20 ms pieces (320 samples), moving forward 10 ms (160 samples)". So we cut the waveform into short frames, analyse each as if it were stationary, and track how the results change from frame to frame.

### 2.3 Why 20 ms frames, 10 ms hop, 50% overlap

- Frame length: 320 samples / 16000 = 20 ms. Short enough to be quasi-stationary; long enough to contain at least one pitch period of a typical voice (a 120 Hz voice has an 8.3 ms period) and so give a meaningful spectrum. Note the limit: a 70 Hz voice has a 14.3 ms period, and you need about two periods to measure periodicity reliably, which is 28.6 ms, longer than the frame. So very low voices are measured poorly (see weak spots).
- Hop: 160 samples = 10 ms. A new label and a new set of LPC coefficients every 10 ms is fine enough to follow speech. It is also the real-time unit: the stream's `blocksize` is 160, so the hop is also the per-block processing budget (10 ms).
- Overlap: frame 320, hop 160 means each sample is in two frames, 50% overlap. Without overlap, a sound event sitting on a frame boundary would be cut in half and the features would jump. Overlap gives smooth feature tracks.
- Number of frames: $n_\text{frames} = 1 + \lfloor (N - 320)/160 \rfloor$ for $N \ge 320$ samples. The final partial frame is dropped (up to 159 trailing samples). For a clip shorter than 320 samples, `frame_signal` returns shape `(0, 320)`.
- Frame $i$ covers samples $[160 i,\ 160 i + 320)$ and its centre is at $(160 i + 160)/f_s$ seconds (this is what `plot_labels` uses for the marker time).

Whether these specific numbers are "best" is not argued in the code; they are the contract in the team guide ("320 samples (20 ms), hop 160 (10 ms)"), and they are the textbook values.

### 2.4 Windowing

When you cut a frame out of a signal you multiply by a rectangular window (1 inside, 0 outside). The abrupt edges leak energy across the spectrum: a pure tone smears into a broad skirt of fake frequencies (spectral leakage). A smooth taper such as a Hamming or Hann window brings the edges to (near) zero and reduces leakage, at the price of a slightly wider main lobe (a little less frequency resolution).

Important detail for your viva: `frame_signal` does NOT window (the spec says "without windowing"). Windowing is applied where it matters: Module B applies a Hamming window inside `lpc_frame` (`src/lpc.py`, `np.hamming`), and your `plot_spectrogram` applies a Hann window. RMS and ZCR are computed on the raw rectangular frame, because they are time-domain measures.

Frame length versus frequency resolution: the frequency spacing of the DFT bins is $f_s / N$. For the spectrogram window of 25 ms (400 samples) it is 16000/400 = 40 Hz. A longer window gives finer frequency resolution but blurs events in time (and breaks quasi-stationarity); a shorter window gives better time resolution but smears frequencies. With a Hann window the main lobe is about 4 bins wide (a standard property of the Hann window), so with 40 Hz bins the harmonics of a 120 Hz voice are just resolved, which is why normal speech shows horizontal harmonic stripes in a 25 ms spectrogram and a whisper does not (it has no harmonics).

### 2.5 RMS energy

Short-time energy tells you how loud a frame is. We use the root mean square:

$$\mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{n=0}^{N-1} x[n]^2}$$

Squaring makes everything positive; averaging gives power; the square root brings it back to the units of amplitude (so a sine of amplitude $A$ has RMS $A/\sqrt2$). Silence has tiny RMS; speech has larger. RMS is less sensitive to one stray click than peak amplitude. Module C also uses it to set the output loudness and the pitch accent.

### 2.6 Zero-crossing rate

ZCR is the fraction of adjacent sample pairs whose sign differs. In `frame_zcr`:

$$\mathrm{ZCR} = \frac{1}{N-1}\sum_{n=1}^{N-1} \mathbb{1}\big[\operatorname{sgn} x[n] \ne \operatorname{sgn} x[n-1]\big],\quad N=320$$

A sample equal to exactly 0 counts as positive (in the code `signs[signs == 0] = 1`). Why it separates sounds:

- A sinusoid of frequency $f$ crosses zero twice per cycle, so ZCR $\approx 2f/f_s$. For a 120 Hz tone, $2\cdot120/16000 = 0.015$; measured 0.01497. Voiced speech is dominated by energy at low frequencies (the fundamental and first formants, a few hundred Hz), so the ZCR of a vowel is low.
- White noise has a random sign each sample, so it crosses about every second sample: ZCR about 0.5 (measured 0.503). Fricatives like "s" and "sh" put their energy at high frequencies (several kHz) so their ZCR is high, between the vowel value and the noise limit.

So a high ZCR means "the energy is mostly high frequency / noise like", i.e. a fricative, and a low ZCR means "low frequency dominated", i.e. a vowel. This is the standard reasoning (and a team guide viva question). Caveat: ZCR is a crude proxy for dominant frequency; a whispered vowel is itself noise (breath through the vocal tract), so its ZCR is higher than a voiced vowel's, and the separation between whispered vowels and whispered consonants relies on the vowels still having more low-frequency energy (formants below 1 kHz) than fricatives.

Note on librosa: librosa divides crossings by `frame_length` (320), you divide by the number of pairs (319), so your values are larger by 320/319. The test checks `zcr * 319/320 == librosa` (difference 1e-16 in the validation notebook).

### 2.7 Autocorrelation voicing strength

Autocorrelation compares a frame with a shifted copy of itself. If the sound repeats with period $T$ samples (voiced speech: the vocal folds open and close periodically), the frame lines up with itself when shifted by $k = T$, so the correlation has a peak there. Noise has no repeating structure, so it never lines up and the correlation stays small at every lag above 0.

`voicing_strength` computes, for each lag $k$ from 40 to 228, a normalised cross correlation of the two overlapping parts of the frame, and keeps the maximum:

$$\rho[k] = \frac{\sum_{n=0}^{N-1-k} x[n]\,x[n+k]}{\sqrt{\left(\sum_{n=0}^{N-1-k} x[n]^2\right)\left(\sum_{n=k}^{N-1} x[n]^2\right)}},\qquad \text{strength} = \max_{40 \le k \le 228} \rho[k]$$

Normalising by the energies of the two overlapping segments (rather than by the total energy) matters: at lag $k$ only $N-k$ samples overlap, so a plain autocorrelation shrinks with $k$ even for a perfectly periodic frame. The normalised version scores near 1 for a periodic frame at any lag (by the Cauchy-Schwarz inequality $\rho \le 1$).

Lag range in Hz: lag $k$ samples corresponds to a pitch of $f_s / k$:

- $k = 40$: $16000/40 = 400$ Hz
- $k = 228$: $16000/228 \approx 70.2$ Hz

So 40..228 covers pitch from 400 Hz down to about 70 Hz, the range of human speaking voices (low male to high child/female); this range is given in the team guide ("lags 40 to 228 (400 Hz down to 70 Hz)"). Excluding small lags avoids the trivial high correlation between neighbouring samples of a low-passed signal.

Measured values (320-sample frames, my snippets): 120 Hz sine 0.9999; white noise mean 0.22 (range 0.14 to 0.38 across frames); the validation notebook prints 0.99988 and 0.217. White noise is not 0 because the maximum of 189 random correlation values is always positive and noticeably above 0, and the short overlap at large lags makes those values noisy.

Why whisper has no periodicity: a whisper is produced with the vocal folds held apart, so there is no periodic puff train. The sound source is turbulent airflow (breath noise), which is random. The vocal tract still filters it (so the formants are there), but the filtered noise is not periodic. Voiced speech has a periodic source (pulses at f0) and so a clear autocorrelation peak at the pitch period. The pitch is what the whisper lacks and what Module C adds back.

An honest nuance in the notebook (walkthrough section 2): a whisper scores clearly lower than voiced speech but not near 0 (synthetic clip about 0.57 over loud frames versus about 1.0 voiced), because narrow formant resonances make filtered noise partly predictable. `voicing_strength` is a diagnostic and for plots only: `classify_frames` uses only RMS and ZCR. (That is why label 2 is called "vowel-like" and not "voiced": in a whisper no frame is voiced.)

### 2.8 DC removal (`dc_block`)

DC offset is a constant (zero-frequency) bias added to the signal: cheap sound cards and microphones often have one. It matters here because:

- It adds to RMS: an offset of 0.3 on a silent signal gives RMS 0.3, so "silence" looks loud and the silence threshold and calibration break.
- It destroys ZCR: the signal no longer crosses zero, so ZCR collapses towards 0 and consonants look like vowels.
- Module C's output stage also has its own DC blocker.

The filter is a one-pole high-pass:

$$H(z) = \frac{1 - z^{-1}}{1 - r z^{-1}},\qquad r = 0.995$$

Difference equation $y[n] = x[n] - x[n-1] + r\,y[n-1]$. The numerator zero at $z=1$ removes exactly DC. The pole at $r$ close to 1 makes the notch narrow so that low speech frequencies pass. The cutoff is about $(1-r) f_s/(2\pi) = 0.005\cdot16000/(2\pi) \approx 12.7$ Hz (about 13 Hz); I computed the gain at 13 Hz as 0.715 (the -3 dB point is 0.707), at 100 Hz 0.994 and at 1 kHz 1.002. The value 0.995 is a conventional choice; the code docstring states the cutoff but does not argue for 0.995 specifically. The time constant is $1/(1-r) = 200$ samples = 12.5 ms.

`dc_block` returns `(y, zi)` so a stream can carry the filter memory between hops, and a test shows that processing in pieces equals processing the whole file. If no `zi` is given the state starts in steady state for the first sample (`lfilter_zi * x[0]`), otherwise a file with a constant offset would start with a decaying step. In the live path the state starts at zero (`dc_zi = np.zeros(1)`), so a mic stream with an offset begins with a decaying step: for an offset of 0.3 the output is 0.3 at the first sample, 0.11 after 12.5 ms and 0.015 after 37.5 ms. This happens inside the 0.3 s calibration window (output is muted then anyway) and the live floor uses a low percentile of the calibration frames (section 4.4), so the effect on the threshold is small.

### 2.9 Resampling and mono in `load_audio`

- Mono: if the file has several channels, `np.mean(x, axis=1)` averages them (a stereo file gives (L+R)/2).
- Resample: if `fs != 16000`, `resample_poly(x, up, down)` with `up = 16000/gcd`, `down = fs/gcd`. For 44100 Hz, gcd(44100, 16000) = 100, so up = 160, down = 441. Polyphase resampling does the interpolation and the anti-alias low-pass in one efficient filter. In the code the variable names are swapped relative to convention (`g` is the up factor, `r` the down factor) but the arithmetic is right; the stereo 44.1 kHz test confirms the output length.

---

## 3. Function by function walkthrough

### `src/framing.py`

Module constant: `DC_POLE = 0.995`.

**`dc_block(x, zi=None, r=DC_POLE)`** returns `(y float32, zi)`.
Applies $H(z) = (1 - z^{-1})/(1 - r z^{-1})$ with `scipy.signal.lfilter` in float64 and returns float32.
- `r = 0.995`: cutoff about 13 Hz at 16 kHz (documented in the docstring). Reason for 0.995 beyond that: not documented; standard DC blockers use a pole between 0.99 and 0.999 so the notch is narrow and does not touch speech (pitch is above 70 Hz).
- `zi=None`: start in steady state, `lfilter_zi([1,-1],[1,-r]) * x[0]`, so a file with an offset does not begin with a decaying step. The stream passes the returned `zi` back in.
- Empty input returns an empty float32 array and the state unchanged (or a zero state of length 1 if none was given). This edge case was a verifier finding fixed in a later commit.

**`load_audio(path, target_fs=16000)`** returns `(x float32 mono, fs)`.
Reads with `soundfile` as float32 (falls back to `scipy.io.wavfile` if soundfile is missing, dividing signed integer types by the dtype maximum), averages channels, resamples to `target_fs` with `resample_poly`, applies `dc_block`, and returns `fs == target_fs` always. Default 16000 is the contract.

**`frame_signal(x, frame_len=320, hop=160)`** returns `(n_frames, frame_len)` float32.
Uses `sliding_window_view(x, frame_len)[::hop].copy()`. `.copy()` matters: a sliding window view shares memory with the input, so without the copy a later change to `x` would silently change the frames (`test_frame_signal_is_copy`). Defaults 320 and 160 are the contract: 20 ms and 10 ms at 16 kHz. Returns `(0, frame_len)` if there are fewer than `frame_len` samples. No windowing and no padding (the last partial frame is dropped).

**`frame_energy(frames)`** returns `(n_frames,)`.
$\sqrt{\operatorname{mean}(x^2)}$ over the frame axis, computed in float32. Matches `librosa.feature.rms(center=False)` exactly (max difference 0.0 in the notebook).

**`frame_zcr(frames)`** returns `(n_frames,)`.
$\frac{1}{N-1}\sum \mathbb 1[\text{sign change}]$, zeros counted as positive (so an all-zero frame has ZCR 0, and `[1,-1,1,-1...]` has ZCR 1). Differs from librosa by the factor 320/319 (documented in the docstring).

### `src/detect.py`

**`voicing_strength(frames, fs=16000, lag_min=40, lag_max=228)`** returns `(n_frames,)` float32.
Formula in 2.7. Implementation: `np.correlate(frame, frame, 'full')[n-1:]` gives the raw autocorrelation $\sum x[n]x[n+k]$; a cumulative sum of squares `csum` gives the two overlap energies in O(1) per lag (`e_head = csum[n-k]`, `e_tail = csum[n] - csum[k]`); the denominator is $\sqrt{e_\text{head} e_\text{tail}} + 10^{-12}$ (the $10^{-12}$ prevents division by zero, so a digital-silence frame scores 0). The upper lag is clipped to `n - 2`. Defaults: 40 and 228 samples = 400 Hz to 70 Hz (spec). The `fs` argument is accepted but not used inside the function (lags are in samples), a harmless leftover. It is a per-frame Python loop, fine offline, and not used in the live path.

**`calibrate_thresholds(energy, zcr, fs=16000, hop=160, calib_s=0.3, silence_mult=3.0, zcr_pct=70)`** returns `(silence_thr, zcr_thr)`.
- `n_early = min(max(int(calib_s*fs//hop), 1), len(energy))`: with defaults 30 frames.
- `floor = median(energy[:n_early])`: the noise floor, the median of the first 0.3 s.
- Guard: if `floor > percentile(energy, 25)` (the first 0.3 s are louder than the quietest quarter of the whole clip, i.e. the clip starts with speech), use `percentile(energy, 10)` instead.
- `silence_thr = max(silence_mult * floor, 1e-4)`.
- `zcr_thr = percentile(zcr[energy >= silence_thr], zcr_pct)`, or 0.5 if there are no non-silent frames.
- Empty input returns `(1e-4, 0.5)`.

Why each value (what is documented and what is not):
- 0.3 s: from the team guide (noise floor from the first 0.3 s of the clip; "calibrate for 0.3 s at stream start"). Short enough that a speaker can stay quiet, long enough (30 frames) for a stable median.
- Median rather than mean: not documented; standard reason is robustness, since a cough or a click in the calibration window would drag a mean up but barely moves a median.
- Multiplier 3.0: not documented. Standard engineering reason: you need a margin above the noise floor so noise ripple does not trigger speech. $20\log_{10}3 = 9.5$ dB above the floor. It is a parameter (`silence_mult`; the Gradio app has a slider 1 to 10).
- Percentile 70: not documented; it means "the most noise-like 30% of the non-silent frames are called consonants". See the weak spots (it forces about 30% consonants by construction).
- Guard (25th and 10th percentiles): documented in the docstring and tested (`test_calibrate_guard_on_speech_start`): if the clip starts with speech, the first 0.3 s is not noise and 3x its median would set the threshold above most of the speech. The percentile values themselves are not argued.
- `1e-4` minimum: not documented; it stops a perfectly digital-silent calibration window (floor 0) from giving a threshold of 0, which would label everything as speech. 0.5 fallback ZCR: not documented; 0.5 is the ZCR of white noise, so a threshold there labels almost nothing as consonant (my reading, not a code comment).

**`label_frame(rms, zcr, silence_thr, zcr_thr) -> int`**: one hop. `rms < silence_thr` gives 0; else `zcr > zcr_thr` gives 1; else 2. The test shows this equals the vectorised rule used in `classify_frames` before smoothing. Equality counts as non-silent (`<`) and as not noisy (`>`).

**`classify_frames(frames, energy, zcr=None, silence_mult=3.0, zcr_thr=None, return_thresholds=False, hangover=5)`**.
Computes `zcr` if not given, calls `calibrate_thresholds` (with default `fs`, `hop`, `calib_s`, `zcr_pct`: only `silence_mult` is passed through), uses the given `zcr_thr` if any, applies the rule, then the hangover, then the median filter (section 4). Returns `labels` (int array), or `(labels, zcr_thr, silence_thr)` when `return_thresholds=True` (note the order: ZCR first, silence second). A given `zcr_thr` overrides the percentile rule. Default `hangover=5` frames = 50 ms (the notebook says the label is held 50 ms "so quiet word endings are not cut off"; the number 5 itself is not argued in code).

**`save_handoff(labels, energy, out_dir="data/handoff")`**: writes `labels.npy` and `energy.npy` for teammates (README "Handoff files"). Creates the directory.

**`plot_labels(x, fs, frames, labels, hop=160, output_path="results/labels.png")`**: waveform with a coloured marker per frame at the frame centre time $(i\cdot hop + N/2)/f_s$: red 0, orange 1, green 2. Saves and closes the figure, returns the path.

**`plot_spectrogram(x, fs=16000, win_ms=25.0, hop_ms=10.0, output_path="results/spectrogram.png")`**: Hann window of 25 ms (400 samples; `get_window('hann', 400, fftbins=True)`, a periodic Hann), hop 10 ms (160), `scipy.signal.stft` with `nfft = 400`, no padding or boundary extension, and dB scale $20\log_{10}(|Z| + 10^{-10})$. Returns `(S_db, f, t)` and saves the figure. 25 ms and 10 ms are from the spec (task A6). Bin spacing 40 Hz. The $10^{-10}$ avoids $\log(0)$.

### `src/live.py` (the Module A parts)

Constants: `BUDGET_MS = 10.0`, `LIMIT_KNEE = 0.9` (the knee belongs to Module C's soft limiter).

**`default_params()`** (Module A related keys): `calib_s=0.3`, `silence_mult=3.0`, `zcr_pct=70`, `hangover=5`, `zcr_thr=None` (None means adaptive; a number fixes it), `zcr_init=0.12`, `zcr_window_s=3.0`, `zcr_min_s=0.5`. They are read every hop, so a GUI can change them while streaming.
- `zcr_init = 0.12`, `zcr_window_s = 3.0`, `zcr_min_s = 0.5`: no documented reason for these values. `zcr_init` is the starting ZCR threshold, used until enough speech has been heard; on the synthetic fixture the offline percentile rule gives 0.138, so 0.12 is in the same region (my observation, not a stated reason). The 3 s window is long enough to hold several words and short enough to follow a changing speaker or room (generic reason). 0.5 s of speech (50 non-silent hops) is a minimum sample for a percentile to mean anything (generic reason).

**`init_live_state(params=None, fs=16000, hop=160)`**: builds the state dict: `prev` (previous 160-sample block, zeros), calibration lists, `thresholds=None`, `dc_zi`, `speech_zcr` list, `zcr_live` (starts at `zcr_init`), `zcr_updates`, hangover memory (`last_nonsilent`, `since_nonsilent = 10**6`), and the states of Modules B/C.

**`_frame_features(prev, block)`**: frame = previous block + current block (320 samples, the frame that ENDS at the current hop); returns RMS and ZCR of it.

**`_live_zcr_threshold(rms, zcr, silence_thr, state)`**: the adaptive ZCR threshold (section 4.4).

**`_causal_hangover(label, state, hangover)`**: the live hangover (section 4.3).

**`process_hop_live(block, state)`** (Module A steps): (1) DC-block the incoming block with carried state; (2) features of the frame ending now; (3) while `thresholds is None`: calibrate, output zeros; (4) afterwards: adaptive ZCR threshold, `label_frame`, `_causal_hangover`, then B and C's step functions. Also records the time per hop in `state["timings"]`.

**`process_file_live(x, params=None, fs=16000, hop=160, state=None)`**: feeds a whole signal hop by hop through `process_hop_live` and returns an output of the same length (the last partial block is zero padded and the extra output discarded). Stores `process_file_live.last_state`. This is the "file through live mode" used by `run_live.py --file` and by tests with no sound card.

**`latency_report(state, stream_latency_s=None)`**: returns a dict: `hops`, `mean_ms`, `p95_ms`, `max_ms`, `frac_over_budget` (fraction of hops above `BUDGET_MS`), `algorithmic_ms = 2 * hop_ms = 20 ms`, `device_ms` (sum of the stream's reported input and output latencies, or None), `total_ms = algorithmic_ms + device_ms`.

**`plot_latency(state, path="results/latency.png")`**: line plot of per-hop time with a dashed line at 10 ms (spec task A8 "saved as a plot"). Uses `matplotlib.figure.Figure` directly (not pyplot), so it does not touch global pyplot state.

### `src/stream.py`

**`FS = 16000`, `HOP = 160`.**
**`list_devices()`**: prints index, name, host API, in/out channels.
**`find_wasapi_devices()`**: returns the default input and output device indices of the WASAPI host API, or None.
**`_wasapi_settings()`**: `sd.WasapiSettings(exclusive=False, auto_convert=True)` (shared mode, let WASAPI convert the sample rate to the stream's), or None if unavailable.
**`make_callback(state, passthrough=False)`**: returns the audio callback `callback(indata, outdata, frames, time_info, status)`. It logs any `status` flags (under/overflow) to `state["status_log"]`, takes channel 0 of the input, and either copies it to the output (passthrough) or calls `process_hop_live` and writes the result. No plotting, no printing.
**`report_and_plot(state, stream_latency, path)`**: prints the latency summary and saves the plot after the stream stops.
**`run_stream(sample_rate=16000, blocksize=160, device=None, params=None, passthrough=False, wasapi=False, state=None)`**: opens `sd.Stream` (duplex, float32, 1 channel), starts it, waits (`sd.sleep(500)` loop) until Ctrl-C, stops, reads `stream.latency` and reports. If WASAPI settings are rejected it retries without them.

### `run_live.py`

`main(argv)`: `--list-devices`; `--in/--out` device indices (`device_from_args`: either or both; a side not given uses its default device); `--passthrough`; `--wasapi`; `--base-f0` (120), `--aspiration` (0.15), `--warp`, `--volume`; `--file IN OUT` runs a file through the live code path; `--play` plays that result. `params_from_args` only sets `warp` if it is below 1.0. `run_file` loads with `load_audio`, runs `process_file_live`, writes the wav, prints the per-hop mean, p95, max and algorithmic delay. (Note `load_audio` already DC-blocks, and the live path DC-blocks again per block; harmless.)

### `record.py`

Nine lines: `python record.py path seconds` records `seconds * 16000` samples mono at 16 kHz with `sounddevice.rec` and writes with `soundfile.write` (a float32 array to a .wav gives 16-bit PCM by default in soundfile). The README stresses staying silent for the first 0.3 s, because the calibration uses it. There is no countdown or level check.

### `app_a.py`

A standalone Gradio view for Module A only: loads a file (or mic), shows the waveform, a 100 ms zoom with frame boxes, the label strip, RMS, ZCR and voicing strength curves with thresholds, the spectrogram, a table of label percentages, and a "Whisper vs Normal" tab. Note the ZCR slider default 0.33 is a FIXED threshold passed as `zcr_thr`, so it overrides the percentile rule. It is the older view; `app.py`'s Analysis tab covers it (`.claude/STRUCTURE.md`). It had a bug (an undefined `noise_floor`), fixed on 2026-11-22; see section 7.

---

## 4. Frame classification in depth

### 4.1 The three classes and what they mean downstream

| Label | Meaning | Excitation made by Module C |
|---|---|---|
| 0 | silence (RMS below the silence threshold) | exactly zero |
| 1 | noisy consonant (loud enough, ZCR above the ZCR threshold) | white noise, no pitch |
| 2 | vowel-like (loud enough, ZCR at or below the threshold) | glottal pulse train with pitch (plus a little aspiration) |

Because a whisper contains no voiced sounds, "vowel-like" means "the sort of frame that in normal speech would be voiced": loud, low-frequency dominated. Vowels get the artificial voice, consonants stay noisy, silence stays silent.

### 4.2 Offline, exactly as coded (`classify_frames`)

1. `zcr = frame_zcr(frames)` unless given.
2. `silence_thr, auto_zcr_thr = calibrate_thresholds(energy, zcr, silence_mult=...)`.
   - `floor` = median of the first 30 frame energies, replaced by the 10th percentile of ALL energies if it exceeds the 25th percentile of all energies.
   - `silence_thr = max(3 * floor, 1e-4)`.
   - `zcr_thr` = 70th percentile of ZCR over frames with `energy >= silence_thr` (0.5 if none).
3. If the caller supplied `zcr_thr`, it replaces the automatic one.
4. Raw labels: `0` if `energy < silence_thr`; else `1` if `zcr > zcr_thr`; else `2`.
5. Hangover (`hangover=5`): scan frames in order, remember the last non-zero raw label; for the following raw-silent frames, while `since <= 5`, overwrite the 0 with the remembered label. So up to 5 frames (50 ms) after speech stops keep the speech label. The test `test_hangover_is_bounded` (loud frames 20..29) asserts silence for frames before 20 and from frame 36 on; in fact frames 30..34 are held and silence already starts at frame 35 (I ran it), the test bound is just a little loose.
6. Median filter of length 3 on the interior (`labels[1:-1]` replaced by the median of three neighbours; the first and last label are not changed). Removes isolated flips. Only if there are at least 3 frames.

On the fixture clip (2.57 s, 256 frames, the float array from `pseudo_whisper()` as used in `a_framing_validation.ipynb`) the raw rule gives counts [127, 38, 91] for labels 0, 1, 2 and the final labels [117, 47, 92]: the hangover turned silence frames into speech labels and the median filter tidied isolated frames. The walkthrough notebook prints slightly different numbers (ZCR threshold 0.137, labels 117 / 49 / 90) because it writes the fixture to a 16-bit wav and reads it back with `load_audio` (quantisation plus DC blocker). Both are correct for their input; if asked, say which notebook the number came from.

Median-filter quirk: the median is taken on the label numbers, so the order 0 < 1 < 2 matters: an isolated 0 between a 1 and a 2 becomes 1. It is a sensible result here, but it is a numeric accident of the label coding, not a designed rule.

### 4.3 Hangover, and why

Real speech has quiet tails: word endings, soft final consonants, trailing breath. Their energy can dip under the silence threshold for a few frames, making the output chop off or flicker. The hangover holds the previous speech label for up to `hangover` frames after the last non-silent frame, so quiet word endings are not cut off. The cost: a delayed release, up to 50 ms of speech-type excitation after the input fell below threshold (hardly audible, because the output RMS is matched to the tiny input RMS).

The hangover only looks at the past, so the live and offline versions are the same logic. Live (`_causal_hangover`):

```
if label != 0:  remember label, since = 0, return label
else: since += 1; if since <= hangover and a label was remembered: return the remembered label; else 0
```

Offline it runs over the whole array with the same rule; the difference is what comes after it: offline then applies the median filter, live does not.

### 4.4 Live (causal) labelling, differences from offline

Everything must be decided hop by hop using only past samples. What changes:

1. Calibration only at the start. For the first 30 hops (0.3 s, `n_calib = int(0.3 * 16000 // 160) = 30`) the output is zeros. Hop 0 is not used for calibration, because its frame is half made of the all-zero initial `prev` buffer. The thresholds are computed when `hop_count + 1 >= 30`, from 29 collected frame RMS values. After that the silence threshold is frozen for the whole stream. Practical consequence: stay quiet for the first 0.3 s and do not change the room or the microphone gain afterwards.
   - Subtle point: `calibrate_thresholds` is called with only these 29 calibration values. Its guard compares their median with their own 25th percentile; the median is almost always above the 25th percentile of the same numbers, so the guard fires and the floor becomes the 10th percentile of the calibration RMS values, not the median. On the fixture: median 0.00201, 10th percentile 0.00190, so the live silence threshold is 0.00571 (3 x p10), against 0.00583 offline. The effect is small (a slightly lower floor), but "3 x median" is exactly true offline and only approximately true live.
   - Side effect worth knowing: because the live floor is a low percentile, speech that starts part way through the calibration window does little harm. I streamed the fixture so that speech began 0.35, 0.25, 0.15 and 0.05 s after stream start: the live silence threshold was 0.00571, 0.00578, 0.00586, 0.00604. Only when speech fills essentially the whole 0.3 s (speech from the first sample) does it break: threshold 0.106, about 18 times too high, so almost everything is called silence. This robustness is an accident of the guard, not a designed feature.
   - The ZCR threshold that comes out of the same call (0.5, because no calibration frame is above the silence threshold) is ignored in live; the live code never reads `state["thresholds"][1]`.
2. Adaptive ZCR threshold (`_live_zcr_threshold`). Offline the 70th-percentile rule needs the ZCRs of all the non-silent frames, which do not exist yet at stream start (calibration hears only silence). So live does:
   - If `params["zcr_thr"]` is a number, use it, fixed (test: forcing 0.9 gives no consonants).
   - Otherwise start at `zcr_init = 0.12`.
   - For every hop whose RMS is at or above the silence threshold, append its ZCR to a list `speech_zcr` and count it in `zcr_updates`. The list keeps the last `zcr_window_s * 100 = 300` entries (3 s of non-silent frames).
   - Once the list holds at least `zcr_min_s * 100 = 50` entries, every 10th non-silent hop (`zcr_updates % 10 == 0`) the threshold is replaced by the 70th percentile of the list. The counter `zcr_updates` is independent of the window length, which fixed an earlier bug where, for a window length that was not a multiple of 10, `len(list) % 10` stopped updating (`test_live_zcr_threshold_keeps_adapting_when_window_not_multiple_of_ten`).
   - Updating only every 10 hops (100 ms) saves computation and stops the threshold jittering (the reason is my reading; the code does not state it).
3. No median filter (it would need the next frame, which does not exist yet). The labels can therefore flicker more than offline.
4. Frame alignment. Offline, label $i$ describes the frame starting at sample $160 i$. Live, the frame analysed at hop $j$ is the previous block plus the current block, so it ENDS now: it is offline frame $j-1$. The notebook's agreement figure shifts the comparison by one hop for this reason.
5. Hangover: same rule, applied to the raw label in `_causal_hangover`, and no median afterwards.
6. DC removal block by block with carried state; offline DC removal is done once at load. Numerically identical (test).

Agreement figure. In the committed walkthrough (section 8, proof cell) the label agreement between live and offline, aligned by one hop and after the calibration period, is 91%. The notebook used the synthetic pseudo-whisper (no real recording was available when it was run; its first output line says "SYNTHETIC pseudo-whisper"). When I fed the same fixture array directly (no wav round trip) I measured 94%; without the one-hop shift 89%. So the right way to say it: "about 91 to 94% on the synthetic test clip, with the disagreement coming from the missing median filter, the adaptive versus percentile ZCR threshold, the slightly different noise floor and the boundaries". Do not claim this number for real recordings; it has not been measured on any.

The same notebook cell shows the two paths share their synthesis code: running Module C's offline chain on the labels, f0 and frames the live path decided reproduces the live output to $6\times10^{-8}$ (rounding error). So the rest of the difference between live and offline output is caused only by the causal decisions (labels, f0, frame alignment), which is the Module A and C boundary.

---

## 5. Real time streaming

### 5.1 Duplex stream and blocksize = hop

`sd.Stream(samplerate=16000, blocksize=160, dtype="float32", channels=1, device=..., callback=...)` is a full-duplex stream: the sound card hands us a block of 160 input samples and expects 160 output samples back, in the same callback, every 10 ms. Choosing `blocksize = hop = 160` means that one callback is exactly one hop of the pipeline, so `process_hop_live` is called once per callback with no extra buffering logic, and the 10 ms hop is also the real-time deadline. (The team guide task A7 says "blocksize 160".)

### 5.2 The callback

`callback(indata, outdata, frames, time_info, status)`: `indata` is a (160, 1) float32 array; `outdata` is filled in place. The callback runs on the audio thread of the sound library: if it takes longer than one block the output underruns (clicks). Therefore:

- It does only `process_hop_live` (and, on error flags, appends the status string to a list).
- No plotting, no printing, no file I/O. The team guide lists "Doing plotting inside the audio callback" as the first common mistake: it causes crackling and dropped audio. All plotting happens after the stream stops (`report_and_plot`).
- Timing is recorded inside `process_hop_live` with `time.perf_counter()` and appended to a list, which is cheap.

### 5.3 WASAPI (Windows)

On Windows, the default host API often adds large buffers. The team guide: "if live latency is high, select the WASAPI host API in sounddevice and switch off audio enhancements in sound settings". `--wasapi` makes `run_stream` pick the WASAPI default input and output devices and pass `WasapiSettings(exclusive=False, auto_convert=True)`: shared mode (works while other apps use the device, no exclusive lock) with automatic format conversion so a device whose native rate is 44.1 or 48 kHz still accepts a 16 kHz stream. If creating the stream with those settings raises, the code retries without them. README: use headphones for every live test (otherwise the microphone hears the loudspeaker: feedback). The team guide's original plan names CoreAudio on a Mac; the code does not depend on the platform.

### 5.4 What "latency" means: three different things

1. Algorithmic delay: delay inherent in the algorithm even on an infinitely fast computer. `latency_report` sets it to 2 x hop = 20 ms. This is the code's model (`algo_ms = 2.0 * hop_ms`); the walkthrough explains it as "A hop is 10 ms and the frame is 20 ms, so the algorithmic delay is about two hops". Interpretation to use in the viva: the system cannot output a block until the whole input block has been captured (up to one hop, 10 ms), and the block it outputs is then played during the next block period (another 10 ms). Look-ahead is zero: the frame ends at the current hop, so there is no waiting for future samples. Note this is the model the code reports, not a separately measured value, and the code does not say which reading it intends (`docs/PROJECT.md` calls it "two hops, the frame length"); all three guides use the "one hop to capture, one hop to play out" reading, which is the standard one for block-based duplex audio.
2. Processing time per hop: how long `process_hop_live` takes to run on the CPU. This must be below the 10 ms hop or the stream falls behind. Measured by `time.perf_counter()` around each call and stored in `state["timings"]`. On the fixture file (257 hops): mean about 0.27 to 0.36 ms (two runs; the committed walkthrough shows 0.273), p95 0.5 to 0.8 ms, max 1.3 to 1.5 ms, 0% over budget. Numbers differ run to run and with machine load: a run while another heavy job was going gave mean 0.48 ms, max 1.8 ms, still far below 10 ms. So processing uses about 3 to 4% of the budget. `tests/test_live.py::test_timing` asserts mean < 2 ms and p95 < 5 ms.
3. Device (buffer) latency: time the audio spends in the operating system and driver input and output buffers. Reported by `stream.latency` (a tuple, input and output, in seconds); `latency_report` adds them: `device_ms = (input + output) * 1000`, `total_ms = algorithmic_ms + device_ms`. The team guide's viva question "Where does the 30 to 50 ms latency of the live system come from?" has this answer: roughly 20 ms algorithmic (two hops) plus tens of ms of device buffers (the test assumes 10 ms + 20 ms device and gets a 50 ms total). Targets: under 50 ms total delay and under 10 ms processing per hop (spec A7, A8).

Honest limit: the repo has no loopback measurement (play a click and record it back), so the reported total is "model plus what the driver says", not a physical measurement.

### 5.5 Passthrough check (A7)

`python run_live.py --passthrough` copies the input block to the output without processing. Purpose: prove the audio chain works (device selection, 16 kHz, blocksize 160, no crackle, no feedback with headphones, delay low) BEFORE the pipeline is plugged in, so a later problem can be blamed on the processing and not on the audio setup. This follows the spec ("First mic to headphone passthrough, then plug in process_hop"). In passthrough mode no latency report is printed.

### 5.6 File-through-live mode

`python run_live.py --file IN.wav OUT.wav [--play]` loads the file, runs `process_file_live` (exactly the same `process_hop_live` code as the real stream, one hop at a time with the same state), writes a wav and prints the per-hop timing summary. It needs no sound device. Uses: demo fallback if the microphone fails (README: "no mic: file through the live code path"), regression tests (`test_run_live_file_subprocess`), and measuring the per-hop processing time without hardware. What it does NOT test: device buffers, driver latency and underruns, which only exist in the real stream.

---

## 6. Tests that cover your module

Run: `python -m pytest tests/test_module_a.py tests/test_live.py -q` (33 passed when I ran them).

### `tests/test_module_a.py`

| Test | What it proves |
|---|---|
| `test_load_audio_stereo_44k` | Stereo 44.1 kHz file becomes mono float32, fs 16000, length within 2 samples of 1.5 s x 16000, peak at most 1 |
| `test_frame_signal_shape_and_librosa` | Shape `(1 + (N-320)//160, 320)`, float32, identical to `librosa.util.frame` (spec A2) |
| `test_frame_signal_is_copy` | Changing the input afterwards does not change the frames, no shared memory |
| `test_frame_signal_short_input` | 100 samples gives `(0, 320)`; exactly 320 gives one frame |
| `test_energy_and_zcr_vs_librosa` | RMS equals librosa's to 1e-6; your ZCR times 319/320 equals librosa's (spec A3) |
| `test_voicing_strength_sine_vs_noise` | 120 Hz sine above 0.95 in every frame; white noise below 0.45, median below 0.35 (spec A4) |
| `test_voicing_strength_whisper_vs_voiced` | In vowel segments the voiced fixture scores at least 0.1 higher than the whispered one |
| `test_classify_contract_labels_only` | Returns an int array, shape (n_frames,), values in {0,1,2}; same with or without precomputed ZCR |
| `test_classify_return_thresholds` | With the flag, returns `(labels, zcr_thr, silence_thr)` with sensible ranges |
| `test_classify_fixture_segments` | In each known segment of the fixture (silence, "s", vowels), more than 75% of frames have the expected label (spec A5) |
| `test_classify_explicit_zcr_threshold` | `zcr_thr=0` gives no vowels; `zcr_thr=1` gives no consonants (override works) |
| `test_clip_starting_with_speech_not_all_silence` | A clip that starts with speech is not wrongly labelled silent (the percentile guard) |
| `test_calibrate_guard_on_speech_start` | The guard gives a lower threshold than the naive 3 x median would |
| `test_calibrate_empty_and_leading_silence` | Empty input returns `(1e-4, 0.5)`; a normal clip gives a threshold between 0.004 and 0.01 |
| `test_label_frame_matches_pre_smoothing` | `label_frame` per hop equals the vectorised rule; more than 90% of final labels equal the pre-smoothing ones; three spot cases |
| `test_hangover_is_bounded` | Hangover holds at most 5 frames and never leaks further |
| `test_save_handoff` | The `.npy` files round trip |
| `test_plot_spectrogram_shape` | `S.shape == (len(f), len(t))` and the image file exists |
| `test_load_audio_removes_dc_and_classifier_survives_offset` | A +0.3 offset is removed (mean below 0.01 after the start) and more than 80% of the labels are unchanged |
| `test_dc_block_starts_in_steady_state_and_carries_state` | No start transient for a constant offset; processing in pieces (including empty pieces) equals one pass |

### Your parts of `tests/test_live.py`

| Test | What it proves |
|---|---|
| `test_output_basic` | Output has the input length, is finite, within [-1, 1], and exactly zero during the 0.3 s calibration |
| `test_timing` | Mean per-hop time below 2 ms, p95 below 5 ms; `algorithmic_ms == 20` and total 50 for device latencies (0.01, 0.02) s |
| `test_plot_latency` | The latency plot is written and non-empty |
| `test_imports_without_device` | `run_live`, `src.stream`, `app` import without a sound device |
| `test_run_live_file_subprocess` | `run_live.py --file` runs end to end as a subprocess and writes a wav |
| `test_list_devices_subprocess` | `--list-devices` exits cleanly |
| `test_live_zcr_threshold_adapts_to_offline_value_and_finds_consonants` | After the clip the live threshold is within 30% of the offline percentile threshold, and at least 20 consonant frames are found |
| `test_live_fixed_zcr_threshold_param_overrides` | With `zcr_thr=0.9` the fixed 0.9 is used, the adaptive value `zcr_live` is never updated (stays at its initial 0.12), and no consonants are labelled |
| `test_live_zcr_threshold_keeps_adapting_when_window_not_multiple_of_ten` | Regression test for the update counter: with `zcr_window_s=2.55` the threshold still tracks a change from low-ZCR to high-ZCR signal |

Other tests in the same file (`test_loudness_follows_input`, `test_output_is_voiced`, `test_params_change_midstream`, `test_live_vs_offline_envelope`) test the whole live chain and are mostly Module C's.

Validation notebook `notebooks/a_framing_validation.ipynb` shows A1 to A6: waveform; frames identical to librosa; RMS difference 0.0; ZCR difference 1e-16 after the 319/320 scaling; sine 0.9999 vs noise 0.217; thresholds on the fixture (silence 0.00583, ZCR 0.138; labels 117/47/92); spectrograms of whisper versus voiced fixture. The notebook ran on the synthetic fixture because no `shared_test.wav` was present.

---

## 7. Weak spots and honest answers

1. All evidence is on a synthetic clip. `data/raw/` contains only `transcripts.txt`; the committed notebook outputs (including the 91% label agreement and the label counts) come from `tests/fixtures.py`'s synthetic pseudo-whisper. Honest answer: the classifier is validated on a synthetic clip against known segment boundaries, plus unit tests; behaviour on real recordings (breath noise, room noise, real fricatives) is only as good as the two simple features allow, and the thresholds are tunable parameters. Say what you would do next: record real clips, plot the labels, adjust `silence_mult` and the ZCR percentile.
2. The ZCR percentile rule forces about 30% of the non-silent frames to be "consonant", whatever the content. A sentence with no consonants would still get about 30% label 1 before smoothing, and one with many fricatives would lose some. I measured 29.5% of non-silent frames above the threshold on the fixture, and a synthetic vowel-only input still gave 44 consonant frames out of 200 after smoothing. Mitigation: the `zcr_thr` override, and in live the 3 s window. A better design: a fixed threshold from data, or a two-cluster split.
3. Noisy rooms. The silence threshold is 3 x the first-0.3 s floor and then frozen. If the room is loud while calibrating, the threshold is high and quiet speech is called silence; if the noise rises later (a fan switches on) noise is labelled as speech; if it falls, nothing breaks but the margin is wasted. If the speaker talks during the whole of the first 0.3 s of a live stream, calibration is wrong and the threshold ends up far too high (0.106 instead of about 0.0057 on the fixture). Talking during only part of it is tolerated, because the live floor is the 10th percentile of the calibration frames (section 4.4); that is luck, not design.
4. ZCR is a crude feature. A whispered vowel is noise, so its ZCR is moderate; some whispered fricatives (such as /f/, /th/) are weak in high frequencies and can look vowel-like; voiced fricatives are mixed. There is no use of the spectral shape. The labels are "good enough to choose between pulse train and noise", not a phoneme classifier.
5. Hangover side effect: up to 50 ms of speech label after speech ends.
6. Median filter vs causality. Offline labels are smoother than live labels. Live labels can flicker at class boundaries; live labels are also aligned one hop later.
7. `voicing_strength` limits. A frame of 320 samples contains fewer than two periods at pitches below about 100 Hz, and at lag 228 only 92 samples overlap. A 60 Hz sine scores only 0.66, while a periodic signal with a period (or a multiple) inside 40..228 scores near 1, even above 400 Hz (a 500 Hz sine scores 1.0 via a multiple of its period). So it is not a strict pitch detector. It is O(N^2) per frame with `np.correlate` full mode and is not used in the live path. It is a diagnostic only; label decisions never use it.
8. Latency figures are partly a model. `algorithmic_ms` is defined as 2 hops; the repo has no loopback measurement. Device latency is whatever the driver reports, and no device figure has been recorded on the demo machine yet. PortAudio's reported latencies may already include part of the block buffering, so "20 ms + device" is an estimate that can double count a little; only a loopback test (play a click, record it) gives the true round trip.
9. Python is not a hard real-time system. The garbage collector or the OS scheduler can cause a rare spike beyond 10 ms; the measured max was about 1.5 ms on the test clip, but that is a file loop without a sound card in the way. Underruns are logged in `status_log` but not repaired.
10. Exceptions in the callback. Originally, if `process_hop_live` raised, the exception would stop the stream. Since 2026-11-22 the callback catches it, outputs silence for that hop, counts the errors and prints the count and the first error after the stream stops (`test_stream_callback_survives_processing_error`). The processing itself still aims to be total (`np.nan_to_num` on the output), so the guard is a safety net, not the normal path.
11. `load_audio` fallback path (no soundfile) converts signed integers by `/iinfo.max` (so int16 -32768 maps slightly below -1) and does not normalise unsigned 8-bit; with soundfile installed, the normal case, this path is not used.
12. DC state differences: offline starts the DC filter in steady state, live starts at zero, so a live stream with a large DC offset has a short transient in the first few ms (it decays with a 12.5 ms time constant).
13. `app_a.py` had a bug, fixed on 2026-11-22: `process_audio` used a variable `noise_floor` that was never defined, so each "Process" click ended in a NameError. It now computes the noise floor as silence threshold / multiplier (the floor the threshold was built from), and `test_app_a_process_audio_runs` guards it. `app.py`'s Analysis tab is still the maintained view.
14. Spec versus code on thresholds: the guide says "calibrate ... then freeze the threshold". The code freezes the silence threshold but lets the ZCR threshold adapt, because the calibration period contains no speech.
15. Interface point: you own the label and the RMS; Module C owns what happens after. A wrong label 1 on a vowel gives noise instead of a pulse train (whisper-like output); a wrong label 0 on speech gives silence.

---

## 8. Likely viva questions with model answers

### Concept

1. **Why does a whisper have no clear autocorrelation peak while a vowel does?**
   A voiced vowel is produced by vocal-fold vibration, a periodic train of airflow pulses at f0, so the waveform repeats every pitch period and the frame matches itself when shifted by that period: a peak near 1 at lag $f_s/f_0$. In a whisper the vocal folds do not vibrate; the source is turbulent airflow, random noise, which has no repeating structure, so the normalised autocorrelation stays low at all lags (about 0.2 for white noise in my measurements, somewhat higher for formant-filtered noise).

2. **Why do we window a frame, and what happens in the spectrum if we do not?**
   Cutting a frame is a multiplication by a rectangular window; its sharp edges spread each frequency over a broad skirt (spectral leakage), hiding weak components next to strong ones. A Hamming or Hann taper makes the edges smooth and cuts leakage at the price of a slightly wider main lobe. In this project `frame_signal` does not window; Module B applies a Hamming window in `lpc_frame`, and `plot_spectrogram` applies a Hann window. RMS and ZCR are time-domain features and are computed on the raw frame.

3. **What is the trade-off between frame length and frequency resolution in a spectrogram?**
   Bin spacing is $f_s/N$. A longer window resolves close frequencies (harmonics) but blurs fast changes in time and violates quasi-stationarity; a shorter window follows changes but smears frequency. With 25 ms and 16 kHz the spacing is 40 Hz, enough to show harmonics of a 120 Hz voice as stripes while keeping 10 ms time steps.

4. **Why is the lag range 40 to 228 samples for pitch at 16 kHz?**
   Pitch $= f_s/\text{lag}$. 16000/40 = 400 Hz, 16000/228 = 70.2 Hz, so the range covers speech pitch from a low male voice (about 70 Hz) to a high child or female voice (about 400 Hz). It follows the spec; smaller lags would be dominated by neighbouring-sample similarity; larger lags leave too few overlapping samples in a 320-sample frame.

5. **Why can zero crossing rate separate fricatives from vowels?**
   ZCR is about twice the dominant frequency over $f_s$. Vowels have most of their energy at low frequencies (the fundamental and first formants), so they cross zero rarely (a 120 Hz tone gives 0.015). Fricatives are high-frequency noise, so they cross often (white noise gives about 0.5). Caveat: it is a proxy, and in a whisper vowels are noisy too, so the separation is weaker.

6. **What is quasi-stationarity and why does it justify 20 ms frames?**
   The vocal tract moves slowly compared with the sound waveform, so over about 10 to 30 ms its shape, and the signal's statistics, are roughly constant. Within such a frame we can use stationary tools (spectrum, LPC, autocorrelation). Much longer and the sound has changed; much shorter and there are too few samples (less than a pitch period) to measure anything.

7. **Why 50% overlap?**
   So that a sound event on a frame boundary is fully inside one of the frames and the feature tracks change smoothly; it also gives a new analysis every 10 ms (which is also the real-time block). The cost is double the computation compared with no overlap.

8. **Why remove DC, and what filter did you use?**
   DC bias adds to RMS (silence looks loud) and removes zero crossings (ZCR collapses), so both features and the thresholds break. A one-pole high-pass $(1-z^{-1})/(1-0.995z^{-1})$ with cutoff about 13 Hz removes it without touching speech (pitch is above 70 Hz). The state is carried between blocks for streaming.

### Why this value

9. **How did you set the silence threshold?**
   It is 3 times the median RMS of the first 0.3 s (assumed to be background noise only), with an absolute floor of 1e-4. 3x is about 9.5 dB above the noise floor, a margin so that noise ripple does not trigger speech. The 3 is a tunable parameter (not argued from data in the code; standard margin choice). If the clip starts with speech (the first 0.3 s louder than the 25th percentile of all frames), the floor falls back to the 10th percentile of the frame energies. Live, the same function only gets the 29 calibration frames, so the fallback nearly always fires and the floor is their 10th percentile (0.00571 versus 0.00583 offline on the fixture).

10. **What fails in a noisy room?**
    Calibration measures the noise, so the threshold rises and quiet speech may fall below it and become silence; noise-like consonants may not stand out; and the threshold is fixed after 0.3 s, so a change in noise level later is not tracked. Mitigations: a multiplier slider, headphones and a quiet start; the future fix is a running noise-floor estimate.

11. **Why the 70th percentile for the ZCR threshold?**
    It splits the non-silent frames into a low-ZCR 70% (vowel-like) and a high-ZCR 30% (consonant-like). The number is a heuristic; no value is argued in the code. A consequence is that about 30% of speech frames are called consonants regardless of content, which is acceptable for deciding between pulse and noise excitation but is not phonetically exact.

12. **Why a 5 frame hangover?**
    It holds the last speech label for 50 ms after the signal drops below the silence threshold so quiet word endings are not cut off or flickering. 5 is a design choice (not derived in code) and is a parameter; larger values smooth more but extend speech into silence.

13. **Why a median filter of 3, and why not live?**
    It removes one-frame flips inside a run. A centred 3-frame median needs the next frame, which does not exist in a live stream, so live omits it (and lives with some flicker).

14. **Why is the frame 320 samples and the hop 160?**
    20 ms and 10 ms at 16 kHz: textbook values from the team guide. The frame is short enough to be quasi-stationary and long enough to contain a pitch period; the hop gives 50% overlap and a 10 ms processing unit equal to the stream's blocksize.

15. **Why `blocksize = 160`?**
    So one audio callback is exactly one hop, which keeps the live code simple and makes the hop time the real-time budget. A bigger block lowers overhead but adds delay; a smaller one would force the frame logic to buffer across callbacks.

### What if

16. **What if the speaker talks during the first 0.3 s?**
    Offline: the guard detects that the first 0.3 s is louder than the 25th percentile of the clip and uses the 10th percentile of all energies. Live: the guard only sees the 29 calibration frames, so it nearly always falls back to their 10th percentile. If some of those frames are quiet (speech started part way through the 0.3 s) the floor is still the noise level and nothing breaks (I tested speech starting 0.05 s after stream start: threshold 0.0060 versus 0.0057). If speech fills the whole window, the floor is speech level, the threshold is far too high (0.106 on the fixture) and most speech is treated as silence. Then you restart the stream and stay quiet.

17. **What if the room gets louder after calibration?**
    The silence threshold is frozen, so noise above it is labelled as speech (consonants or vowels depending on ZCR) and Module C synthesises voice on it. Restart to recalibrate; a running noise-floor tracker would be the next step.

18. **What if the microphone has a DC offset?**
    `dc_block` removes it per block with carried state. Without it RMS would be raised and ZCR would drop; a test adds 0.3 and shows more than 80% of the labels stay the same.

19. **What if processing takes longer than 10 ms?**
    The callback misses its deadline, the sound card's output buffer runs dry and you hear clicks or dropouts (an underrun flag arrives in the callback's `status` and is logged). `latency_report` gives `frac_over_budget` for this. On our test clip the mean is about 0.3 ms and the maximum about 1.5 ms.

20. **What if the clip is shorter than 320 samples?**
    `frame_signal` returns an empty `(0, 320)` array, `classify_frames` returns an empty array, `calibrate_thresholds` returns the fallbacks `(1e-4, 0.5)`.

21. **What if you set `zcr_thr` explicitly?**
    The percentile rule is bypassed in both paths; `zcr_thr=0` makes every loud frame a consonant, `zcr_thr=1` makes every loud frame a vowel (tests). In live, a numeric `params["zcr_thr"]` fixes it and the adaptation is skipped.

### Code specifics

22. **Why `.copy()` in `frame_signal`?**
    `sliding_window_view` returns a view with overlapping strides onto the same memory (read-only); without a copy, later edits of the signal would silently change what you think are the frames. The copy gives an independent, writable, contiguous array (tested).

23. **How is the voicing strength computed efficiently, and why normalise?**
    `np.correlate(frame, frame, 'full')` gives the raw autocorrelation for all lags at once; a cumulative sum of squares gives the energies of the head and tail segments for each lag in O(1). Dividing by $\sqrt{e_\text{head}e_\text{tail}}$ compensates the shrinking overlap, so a perfectly periodic frame scores near 1 at the true period regardless of lag. The `1e-12` stops division by zero.

24. **Explain `process_hop_live` as far as it is yours.**
    DC-block the block with carried state; concatenate with the previous block into a 320-sample frame that ends now; compute RMS and ZCR; for the first 30 hops collect calibration values and output zeros; then use the frozen silence threshold, update the ZCR threshold from the last 3 s of speech frames, label, apply the causal hangover, and pass the label to Module B and C's step functions; store `prev`; time the hop.

25. **Why is the live frame different from the offline frame?**
    Live can only use the past: the frame is the previous block plus the current one, ending now (offline frame $j-1$ in offline indexing). Offline, frame $i$ starts at hop $i$ and extends into the future. So live labels are aligned one hop later, and the comparison in the notebook shifts by one.

26. **What does the callback do and why must it be fast?**
    It is called by the sound system every 10 ms with a block of input and must fill the output block immediately. If it takes longer than the block duration the output is starved (clicks) and later blocks are late. So it only runs the per-hop processing; all plotting and printing are done after the stream stops.

27. **What does `latency_report` report and what is the 10 ms budget?**
    Per-hop processing mean, p95, max and the fraction of hops over 10 ms; the algorithmic delay (2 hops = 20 ms); the device latency from `stream.latency` and the total. 10 ms is the hop duration: to keep up, each hop must be processed in less time than the audio it represents.

### Live versus offline

28. **How do live and offline labels differ, and how much?**
    Live: calibrate once on 0.3 s (floor effectively the 10th percentile of the calibration frames), ZCR threshold adaptive (starting at 0.12, then the 70th percentile of the last 3 s of non-silent frames, updated every 10 speech hops), causal hangover, no median filter, one-hop frame shift. Offline: whole-clip percentile ZCR threshold, hangover then median filter. Agreement was 91% on the walkthrough's synthetic clip (94% on the raw fixture array), so live and offline outputs are not sample identical; the proof cell shows the offline synthesis chain, fed with the live decisions, reproduces the live output to 6e-8.

29. **Why not simply run the offline algorithm on a sliding buffer in live?**
    You could, but the percentile and median filter need future samples to be exact; waiting for them adds delay beyond the 50 ms target. The adaptive causal version trades a little label accuracy for zero look-ahead. (Engineering argument; the repo did not try the alternative.)

### Comparison

30. **Why not a neural VAD or phoneme classifier?**
    A neural VAD is more robust to noise but needs a pretrained model or training data and more computation per hop, and it tells you speech versus non-speech, not noisy versus vowel-like, which is the decision Module C needs. Our two features cost microseconds, need no data, are fully explainable, and run far inside the 10 ms budget. Trade-off: less robust in noise. (Engineering argument; the repo contains no neural VAD experiment.)

31. **Why ZCR and not the autocorrelation voicing strength for the vowel/consonant decision?**
    In a whisper nothing is truly voiced, so periodicity does not separate vowels from consonants; the filtered noise of whispered vowels still scores about 0.57 on the fixture versus about 0.2 for white noise, which would be an unreliable cut. ZCR measures where the energy sits, which does separate the classes. Voicing strength is computed anyway to show the whisper/normal difference, and it is O(N^2) per frame.

32. **Why RMS and not dB or spectral features?**
    RMS is the direct amplitude measure that Module C also needs for loudness matching, is cheap, and matches librosa; dB is a monotonic rescaling, so a threshold in dB would be equivalent but less convenient for a simple multiplier.

33. **Why WASAPI and not the default Windows API?**
    The default host API on Windows often adds larger buffers and latency; WASAPI lets us ask for a shared-mode stream with automatic rate conversion (the team guide's recommendation: if latency is high, select WASAPI and switch off audio enhancements). It is optional (`--wasapi`) and falls back to defaults if rejected. I did not measure the difference in this repo.

34. **What is the difference between algorithmic delay and device latency?**
    Algorithmic delay comes from the processing method even on instant hardware (block capture and playback, two hops = 20 ms in our model; a method with look-ahead would add more). Device latency is the OS and driver buffering on input and output (from `stream.latency`). They add: total = 20 ms + device.

35. **How would you extend the classifier?**
    Adaptive noise-floor tracking (minimum statistics), a spectral feature such as low-band versus high-band energy ratio next to ZCR, hysteresis (different on and off thresholds), and testing on real recordings. (Suggestions, not implemented.)

---

## 9. Cheat sheet

**Constants**

| Item | Value |
|---|---|
| Sampling rate | 16000 Hz mono float32, Nyquist 8 kHz |
| Frame / hop | 320 samples (20 ms) / 160 samples (10 ms), 50% overlap |
| Frames | $1 + \lfloor (N-320)/160 \rfloor$, last partial frame dropped |
| DC block | $(1-z^{-1})/(1-0.995z^{-1})$, cutoff about 13 Hz, time constant 12.5 ms |
| Voicing lags | 40..228 samples = 400 Hz .. 70.2 Hz; pitch = 16000 / lag |
| Spectrogram | Hann 25 ms (400 samples), hop 10 ms, nfft 400 (40 Hz bins), dB = 20 log10 |
| Calibration | first 0.3 s = 30 hops; live output is zero during it; hop 0 skipped |
| Silence threshold | max(3.0 x floor, 1e-4); offline floor = median of first 0.3 s (fallback: 10th percentile of all energies if floor > 25th percentile); live floor = in practice the 10th percentile of the 29 calibration frames (the guard fires on its own short list) |
| ZCR threshold | 70th percentile of ZCR over non-silent frames (fallback 0.5) |
| Hangover | 5 frames = 50 ms |
| Median filter | 3 frames, offline only, interior only |
| Live ZCR | init 0.12, window 3.0 s (300 hops), min 0.5 s (50 hops), update every 10th non-silent hop |
| Live frame | previous block + current block (ends now) |
| Budget | 10 ms per hop; algorithmic delay 2 x 10 = 20 ms; target total under 50 ms |
| Measured on the fixture | per hop mean about 0.3 ms (0.25 to 0.5 across runs), p95 under 1 ms, max 1 to 2 ms; never over 10 ms |
| Agreement live vs offline labels | 91% (notebook, synthetic clip, one-hop aligned) |

**Formulas**

- RMS: $\sqrt{\tfrac1N\sum x[n]^2}$
- ZCR: $\tfrac1{N-1}\sum_{n=1}^{N-1}\mathbb 1[\operatorname{sgn}x[n]\ne\operatorname{sgn}x[n-1]]$ (zero counts as positive); sine: $\approx 2f/f_s$; white noise about 0.5
- Normalised autocorrelation: $\rho[k]=\dfrac{\sum_{n=0}^{N-1-k}x[n]x[n+k]}{\sqrt{\sum_{n=0}^{N-1-k}x[n]^2\ \sum_{n=k}^{N-1}x[n]^2}}$, strength $=\max_{40\le k\le228}\rho[k]$
- Pitch from lag: $f_0 = f_s/k$
- Bin spacing: $f_s/N$
- Silence threshold in dB above floor: $20\log_{10}3 \approx 9.5$ dB

**Label rule** (per hop): `rms < silence_thr` then 0; else `zcr > zcr_thr` then 1; else 2. Then hangover (5), then (offline only) median of 3.

**Measured reference values** (320-sample frames): 120 Hz sine strength 0.9999; white noise about 0.22; 120 Hz sine ZCR 0.015; white noise ZCR 0.503; whisper fixture loud-frame strength about 0.57; fixture thresholds silence 0.00583, ZCR 0.138; label counts 117 / 47 / 92 (silence / consonant / vowel-like) on the fixture array (the walkthrough, which reads the fixture back from a 16-bit wav, shows ZCR 0.137 and 117 / 49 / 90); live silence threshold on the same clip 0.00571.

**Run it**: `python run_live.py --list-devices`; `python run_live.py --passthrough`; `python run_live.py --in 1 --out 3 [--wasapi]`; `python run_live.py --file in.wav out.wav --play`; `python -m pytest tests/test_module_a.py tests/test_live.py -q`.

**If you forget everything**: frame it (20 ms, 10 ms), measure loudness (RMS) and noisiness (ZCR), silence = 3x the first-0.3-s floor, ZCR above the 70th percentile = consonant, hold 50 ms after speech, live does the same with only the past and a running ZCR threshold, and the whole per-hop job must finish in under 10 ms inside a callback that does nothing else.
