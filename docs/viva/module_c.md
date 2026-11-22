# Viva guide: Module C (Raks) - pitch, excitation, synthesis, live processor, evaluation, app

Audience: Raks, preparing to be questioned by examiners. You should be able to learn the topic from this document alone. Every statement about behaviour or numbers was checked against the code in the repository (and the key numbers were re-run on 2026-11-22). Where a reason is not written in the code, comments or the spec, it is labelled **[standard engineering reason]** or **[my analysis]** so you do not present it as the team's documented intent.

Conventions: 16 kHz mono float32, frame 320 samples (20 ms), hop 160 samples (10 ms). "Hop" = one 10 ms block. $f_s = 16000$.

Files you own: `src/pitch.py`, `src/synth.py`, `run_offline.py`, `src/live.py` (per-hop processor; Module A owns the causal labelling parts inside it), `src/evaluate.py`, `run_eval.py`, `tools/make_dev_clip.py`, `app.py`. Tests: `tests/test_module_c.py`, `tests/test_live.py`, `tests/test_evaluate.py`, `tests/test_app.py`. Notebooks: `notebooks/c_synthesis_validation.ipynb`, sections 5 to 9 of `notebooks/walkthrough.ipynb`.

---

## 1. Your module in one paragraph, and where it sits

Module C builds the **artificial voice**. A whisper contains the vocal tract shape (the formants, captured by Module B as an LPC filter) but no vocal-fold buzz. Module C (1) invents a pitch contour $f_0(t)$ for the frames that Module A labelled vowel-like, (2) turns the labels and the contour into an **excitation** signal (glottal pulses plus a little noise for vowels, white noise for consonants, exact zeros for silence), (3) passes that excitation through Module B's all-pole filter $G/A(z)$ with the filter memory carried from hop to hop, (4) applies an **output stage** (lip radiation, de-emphasis, DC blocker, loudness matching to the input), and (5) writes the wav. It also owns the **live per-hop processor** (same step functions, causal decisions), the **evaluation** (Whisper WER, log spectral distortion, listening test, figures), and the **Gradio app**.

```
                         Module A                       Module B
 wav -> load_audio -> frame_signal -> frame_energy   lpc_all (-> warp_poles)
                          |               |                |
                          +-> classify_frames (labels 0/1/2)|   A (n,13), G (n,)
                          |               |                |
        ============== Module C starts here ================================
        labels, energy -> build_f0 -> f0 (n,)
        labels, f0     -> make_excitation -> e (n*160,)       [pulses+noise | noise | zeros]
        A, G, e        -> synthesize -> y_raw                 [lfilter, zi carried]
        y_raw, energy  -> output_stage -> y                   [radiation, de-emphasis, DC block, RMS match]
        y              -> peak_protect -> wav
        ====================================================================
 Live path (src/live.py): the same synth_step / excitation_step / output_step, one hop at a time,
 with causal labels (Module A) and f0_step instead of build_f0, soft limiter instead of peak_protect.
 Evaluation (src/evaluate.py): renders four variants, WER (Whisper), LSD, listening sheet, figures.
```

What you take from the others (the contract, README "Function contract"): labels $\in\{0,1,2\}$ and per-frame RMS energy from Module A; `A` (shape `(n_frames, order+1)`, `a[0]=1`) and gain `G` from Module B.

---

## 2. Theory from scratch

### 2.1 Source-filter model and why whisper has no pitch

Speech production is modelled as a **source** passing through a **filter**:

$$ s[n] = (e * h)[n], \qquad S(z) = E(z)\,H(z) $$

- **Voiced sound** (vowels): the vocal folds open and close quickly, 100 to 250 times per second. Air is released in puffs: a **periodic source**, with fundamental frequency $f_0$ and harmonics at $k f_0$.
- **Noisy sound** (fricatives such as "s"): turbulent air, a **noise source**, no periodicity.
- **Whisper**: the folds are held apart, so *all* sounds, even vowels, use turbulent noise as the source. The vocal tract (mouth and throat shape) still filters it, so formants (resonances) and therefore the words survive, but there is no $f_0$ and no harmonics.

Project idea: keep the filter, replace the source for vowel frames by a synthetic periodic one. That is exactly the "Module C" job.

A periodic source with period $T_0 = 1/f_0$ has a **line spectrum**: energy only at $k f_0$ (Fourier series). This is why the spec's question "why does a pulse train produce harmonics at multiples of $f_0$?" has the answer: any signal periodic in $T_0$ can be written as a sum of sinusoids at multiples of $1/T_0$. Test `test_impulse_train_harmonics` shows peaks at $120,240,\dots$ Hz.

### 2.2 Impulse train versus Rosenberg glottal flow pulse

**Impulse train** (`impulse_train`, task C1): a 1 every $f_s/f_0$ samples. Its spectrum has equal height at all harmonics (flat, I measured 41.6 dB, 41.5 dB, 40.8 dB at harmonics 1, 8, 32). Real voices are not like that, so after the vocal tract filter it sounds like a harsh buzzer. The spec lists this under "common mistakes".

**Rosenberg pulse** (`rosenberg_pulse`, task C2): a smooth model of one cycle of glottal **flow** $u_g(n)$ (volume velocity through the glottis). For a period of $L=\mathrm{round}(T_0)$ samples, with opening length $T_p = \mathrm{round}(0.40\,L)$ and closing length $T_n = \mathrm{round}(0.16\,L)$:

$$
g[n] =
\begin{cases}
\tfrac12\left(1-\cos\dfrac{\pi n}{T_p}\right), & 0 \le n < T_p \quad\text{(opening, raised cosine, 0 to 1)}\\[2mm]
\cos\dfrac{\pi m}{2 T_n}, \quad m = n - T_p, & T_p \le n < T_p+T_n \quad\text{(closing, quarter cosine, 1 towards 0)}\\[2mm]
0, & T_p+T_n \le n < L \quad\text{(closed phase)}
\end{cases}
$$

As coded: **40 % open, 16 % closing, 44 % closed**. For $L=133$ ($f_0=120$ Hz) that is 53 + 21 + 59 samples. Peak value 1 at $n=T_p$ (tests check the argmax is at $0.40L \pm 1$, the pulse starts at 0, is never negative, and is exactly 0 after the closing phase).

Note the last nonzero sample of the closing phase is $\cos(\pi (T_n-1)/(2T_n)) \approx 0.07$ (for $T_n=21$); the next sample, $m=T_n$, would be $\cos(\pi/2)=0$, which is where the closed phase starts. So the flow itself is continuous (it reaches 0), but its **slope** is steepest right at closure and then jumps to 0. This **abrupt closure** (a slope discontinuity) is deliberate in the Rosenberg model: it is the sharp event that excites the formants, and in the flow derivative it is the large negative peak at the "glottal closure instant".

Spectrum: smooth pulses have spectra that roll off. I measured the harmonic levels of the generated 120 Hz train: -6.8 dB at harmonic 2, -19.5 dB at harmonic 4, -36.1 dB at 8, -47.3 dB at 16, -58.8 dB at 32 (relative to harmonic 1). That is roughly **-12 dB per octave** at higher harmonics (I measured -11.3 and -13.9 dB/octave over two bands), the textbook slope for glottal flow. It is a big part of why it sounds more natural than a flat-spectrum impulse train.

**Flow versus differentiated flow, and lip radiation.** What leaves the lips as sound pressure is approximately the *derivative* of the glottal flow (radiation from the mouth acts like a differentiator, +6 dB/octave). The **LF model** (Liljencrants-Fant) is formulated directly on the derivative $\dot u_g$ with its negative peak at closure; the Rosenberg model is on the flow. In this code the pulse returned by `rosenberg_pulse` is the **flow** (STRUCTURE.md: "flow, not differentiated"); the differentiation is done afterwards in `output_step` as the lip-radiation filter $R(z) = 1 - 0.98 z^{-1}$ (a "leaky" first difference: gain 0 at DC would be a pure difference; the leak 0.98 makes the DC gain 0.02 instead of 0 and keeps it numerically tame). The spec's wording for C2 is "rosenberg_pulse(T0) and its difference for radiation": the same idea, implemented in the output stage instead of inside the pulse function. Because everything between the source and the output is linear, applying radiation after the vocal tract filter is equivalent (for a time-invariant filter; the filter changes each hop, so it is approximately equivalent).

### 2.3 Spectral tilt

Spectral tilt = how fast harmonic level falls with frequency. Flow pulse: about -12 dB/oct. Radiation: +6 dB/oct. Together the radiated voice source is about -6 dB/oct, close to natural speech. The LPC analysis was done after **pre-emphasis** $1 - 0.97z^{-1}$ (+6 dB/oct) which Module B uses to flatten the spectrum before fitting; so the LPC envelope $G/A(z)$ carries that extra tilt. See section 2.9 for how the output stage deals with the three tilts, including a subtlety you should know.

### 2.4 Jitter and aspiration noise

**Jitter** = small random cycle-to-cycle variation of $f_0$ (natural voices are not perfectly periodic; a perfectly periodic source sounds robotic). Here it is `jitter=0.005`: a uniform random factor $1 + 0.005\,u$, $u\sim U(-1,1)$, i.e. up to $\pm 0.5\,\%$, drawn **per 10 ms frame**, not per glottal cycle, and then smoothed. After smoothing the residual variation is small: with declination and stress off I measured a standard deviation of $f_0$ of about 0.15 % of the mean (offline and causal). Say it honestly: it is a light touch, not a physiological jitter model.

**Aspiration noise** (spec C5: "label 2 pulses plus 10 to 20 percent noise"): real voiced sounds always contain some breath noise. Each voiced hop is

$$ x = \frac{(1-a)\,p + a\,w}{\sqrt{(1-a)^2 + a^2}}, \qquad a = 0.15 $$

with $p$ the unit-RMS pulse train and $w$ unit-RMS white noise. `0.15` is an **amplitude mix weight**. Power fraction of noise: $a^2/((1-a)^2+a^2) = 0.0225/0.745 = 0.0302$, i.e. **about 3 % of the power** (verified numerically, 0.0302). The denominator rescales the mix back to unit RMS (assuming $p$ and $w$ are uncorrelated). Extra nuance [my analysis]: the unit-RMS Rosenberg train has a large mean (mean/RMS = 0.63, I measured 0.627 to 0.631), so about 40 % of its power is DC. Relative to the pulse train's AC power the noise is therefore roughly 5 %. Both numbers are small; the spec says "10 to 20 percent" in the amplitude sense, which matches 0.15.

Why it helps (what the ablation hopes to show): the noise fills the gaps between harmonics, especially at high frequency where the pulses have fallen off by 40 to 60 dB, so the output is less buzzy and, as a side effect, LPC re-analysis of the output (the LSD metric) is less biased by individual harmonics [standard engineering reason].

### 2.5 The f0 contour: design (exactly as coded, `build_f0`)

Only frames with label 2 get a nonzero $f_0$. Rule-based, **not** derived from the speaker's whisper (whisper has no pitch information).

1. **Phrases.** `_phrases(labels, gap_frames)`: a phrase break is a run of label-0 frames longer than `phrase_gap_s * fs / hop` = $0.5 \times 100 = 50$ frames (strictly greater). Each phrase runs from its first to its last non-silent frame. Silence at the ends of the clip is not a "phrase".
2. For each phrase, with `first`, `last` frame indices and `span = max(last-first, 1)`, and for each voiced frame index $i$:

$$ f_0[i] = f_{\text{base}}\,\Bigl(1 - d\,\frac{i-\text{first}}{\text{span}}\Bigr)\,\bigl(1 + s\,\delta_i\bigr)\,(1 + j\,u_i) $$

   - $f_\text{base}$ = `base_f0` = 120 Hz.
   - **Declination** $d$ = `decl` = 0.12: a linear fall of 12 % from the start of the phrase to its end (natural pitch tends to drift down over an utterance). Whatever the phrase length, the fall is 12 % over *that phrase* (offline knows the phrase end). I measured $f_0$: 120 Hz at the start, 105.6 Hz at the end of a 2 s phrase (ratio 0.88).
   - **Stress** $s$ = `stress` = 0.25: $\delta_i = \mathrm{clip}(\ln e_i - \overline{\ln e},\,-1,\,1)$ where $e_i$ is the frame RMS and the mean is over the voiced frames of the phrase. Louder than average means higher pitch (stressed syllables are louder and higher in natural speech; a rule of thumb). The clip limits it to $\pm 25\,\%$ (90 to 150 Hz around 120 Hz; verified).
   - **Jitter** $j$ = `jitter` = 0.005, $u_i\sim U(-1,1)$, from `np.random.default_rng(seed)` (seed 0, so reproducible).
3. **Smoothing** (`_smooth_runs`): for each contiguous run of voiced frames, a median filter of size 5 then a moving average of size 3 (`mode="nearest"`). The spec's mistake list: "f0 jumps between frames. Smooth the contour". Median removes outliers, moving average rounds corners.
4. **Clip** to [70, 400] Hz for voiced frames (spec C4); $f_0 = 0$ elsewhere. Returned as float32. If there is no voiced frame at all, all zeros.

Why a "per-phrase reset"? Declination restarts at the start of each phrase (after a pause longer than 0.5 s), as a speaker resets pitch after a breath. Test: `test_build_f0_declination_and_reset`, `test_build_f0_short_gap_does_not_reset`.

### 2.6 The causal f0 for live (`f0_step`)

Live cannot see the end of the phrase or the average energy of the whole phrase. `f0_step` keeps a **state** dict (`t`, `started`, `sil`, `mean`, `count`, `prev`, `rng`) and for each hop:

- label 0: silence time `sil` grows by 10 ms, `prev` reset to 0; if `sil > phrase_gap_s` (0.5 s) the phrase is reset (`t=0`, `started=False`, running energy mean cleared). While a started phrase has silence shorter than the gap, `t` keeps advancing.
- label != 0: `sil = 0`, phrase started; $t$ read before incrementing by 10 ms.
- label 1: returns 0 (no pitch), `prev = 0`.
- label 2: running mean of $\ln e$ (incremental mean with `count` capped at 200, so it behaves like a moving average over about 2 s once the cap is reached), $\delta=\mathrm{clip}(\ln e - \text{mean},-1,1)$, **declination from running time**: $\text{fall}=d\cdot\min(t/T_\text{ph},1)$ with `phrase_dur_s` $T_\text{ph}=2.0$ s (so the fall saturates at 12 % after 2 s; for a long phrase it stops falling, unlike offline). Raw value as in the offline formula, then **one-pole smoothing** $f = f_\text{prev} + 0.5\,(\text{raw} - f_\text{prev})$ (if `prev > 0`; the first voiced hop of a run is unsmoothed), clipped to [70, 400].

Offline versus causal on the same labels, 2 s phrase: offline ends at 105.6 Hz, causal ends at 105.7 Hz; on a short 0.5 s phrase offline ends at 105.7 Hz but causal only at 116.5 Hz (it has not had time to fall). Test: `test_build_f0_causal_properties`. `build_f0_causal` simply loops `f0_step` (for tests and the notebook).

### 2.7 Excitation synthesis and phase continuity (`excitation_step`)

The main problem: $f_0$ changes every hop, but the pulse train must stay smooth (no skipped or doubled pulses at hop boundaries, no loudness modulation at the 100 Hz hop rate).

**Running phase.** Keep a phase $\phi$ in cycles that increases by $f_0/f_s$ per sample. A new glottal pulse starts every time $\lfloor\phi\rfloor$ increases. If $f_0$ changes, only the *rate* of phase increase changes, so the pulse spacing glides smoothly. Per hop:

$$ f_s^{(n)} = f_\text{prev} + (f_0 - f_\text{prev})\,\frac{n}{H},\quad n = 1..H \qquad\text{(linear glide to the new f0 across the hop)} $$
$$ \phi_n = \phi_0 + \sum_{k\le n} f_s^{(k)}/f_s $$

(`H = 160`; the first voiced hop after a non-voiced one has no previous $f_0$ and uses a constant). Pulse onsets are samples where $\lfloor\phi + 10^{-9}\rfloor$ increases; at each onset a pulse of length $L=\mathrm{round}(f_s / f^{(n)})$ is added using a cached unit-RMS Rosenberg pulse (`_unit_pulse`, `lru_cache` 512 entries: pulse divided by its own RMS over the whole period, including the closed part, so a periodic train of equal pulses has unit RMS). At the end of the hop the fractional phase $\phi - \lfloor\phi\rfloor$ is stored. $f_0$ is clamped to [40, 800] Hz for the excitation (`F_MIN_EXC`, `F_MAX_EXC`; the contour itself is already 70 to 400 Hz).

**Pulse tail.** A pulse that starts near the end of a hop extends into the next. `state["tail"]` (length `TAIL_LEN = 401`, one more than the longest pulse at 40 Hz = 400 samples) stores the part beyond the hop and adds it to the next hop's start (overlap-add). So there are no cut pulses and no clicks. `PHASE_INIT = -1e-6` makes the first pulse of a voiced run start at the very first sample (the first sample is 0, the second is the first nonzero value, since the pulse starts at 0).

**Reset.** On any hop that is not voiced-with-pitch, the phase goes back to `PHASE_INIT`, `prev_f0 = 0`, the tail is cleared (a pulse cut by the end of a voiced run is dropped, so silence is exactly zero).

**Label logic** (spec C5):

| label | excitation |
|---|---|
| 2 and $f_0>0$ | Rosenberg pulses (unit RMS) mixed with aspiration noise |
| 2 and $f_0 = 0$ | white noise (fallback; test `test_label2_without_f0_is_noise`) |
| 1 | white noise, unit RMS per hop (`_unit_noise` rescales each hop to RMS exactly 1) |
| 0 | zeros, exactly |

Excitation is unit RMS in all non-silent cases. Tests: `test_excitation_unit_rms_per_label`, `test_no_hop_rate_modulation` (hop RMS stays within 0.75 to 1.25), `test_hopwise_equals_single_pass` (running the hop function over many hops equals one big call), `test_hopwise_matches_independent_train`.

Why this design and not "put an impulse every $T_0$ in each frame separately": per-frame placing would restart the grid at each frame boundary and produce irregular spacing and clicks [standard engineering reason; the spec task C3 says "make_excitation using a running phase so f0 can change smoothly"].

### 2.8 LPC synthesis filtering and why the state `zi` is carried

The vocal tract filter for hop $i$ is $H_i(z) = G_i / A_i(z)$, all-pole, with $A_i(z) = 1 + a_1 z^{-1} + \dots + a_{12} z^{-12}$. Difference equation: $y[n] = G\,e[n] - \sum_{k=1}^{12} a_k\,y[n-k]$. The filter has **memory**: the next output depends on the previous 12 outputs. `scipy.signal.lfilter(..., zi=zi)` returns the final internal state; passing it as the next call's initial state continues the filter exactly. If you restart with zero state at each hop, the ringing of the resonances is cut off every 10 ms: a discontinuity (a click or 100 Hz buzz) at every hop boundary. Spec "common mistake 1". Test: `test_synthesize_hopwise_equals_lfilter` shows the hop-by-hop result equals one `lfilter` call over the whole signal (when the coefficients are constant), and `c_synthesis_validation.ipynb` cell C6 plots the waveform across hop boundaries.

Between hops the *coefficients* change (new $A_i$) but the state (past outputs in transposed direct form) carries over, so the signal is continuous; the timbre changes in steps of 10 ms. There is no coefficient interpolation in the code (a possible limitation: tiny discontinuities in timbre, not in level).

### 2.9 Output stage (`output_step`)

Applied to each hop after synthesis, with filter memories in `state` (`zi`, `zd`, `zdc`) and the last gain:

1. **Lip radiation** $R(z) = 1 - 0.98\,z^{-1}$. Converts glottal flow into radiated pressure (differentiation).
2. **De-emphasis** $D(z) = 1 / (1 - 0.97 z^{-1})$ = inverse of pre-emphasis $P(z)=1-0.97z^{-1}$ (`DEEMPH = 0.97`). Code comment: the LPC filter was fitted on pre-emphasised speech, so its envelope carries an extra +6 dB/octave; de-emphasis removes it; without it the output is too bright and the high band hides the harmonics.
3. **DC blocker** $B(z) = (1 - z^{-1})/(1 - 0.99\,z^{-1})$ (`OUT_DC_POLE = 0.99`, cutoff about $(1-0.99) f_s / 2\pi \approx 25$ Hz; verified 25.5 Hz; gain 0.70 at 25 Hz, 0.89 at 50 Hz, 0.97 at 100 Hz). Reason in the code comment: radiation and de-emphasis together keep a DC gain of about 0.67 ($= 0.02/0.03$), and the Rosenberg flow pulse has a large mean, so without the blocker the RMS match would spend loudness on inaudible DC. Test `test_output_has_little_energy_below_60hz` (less than 5 % of power below 60 Hz).
4. **RMS match** with `volume`: measure $\text{rms}$ of the filtered hop; $g_\text{new} = \min(\text{target\_rms}\cdot\text{volume}/\text{rms},\ 50)$ (`GAIN_CAP`). The gain at the start of the hop is the previous gain clamped to within a factor 2 of the new one (`RAMP_RATIO`), and the gain **ramps linearly** across the hop. `target_rms` is the input frame RMS (`energy[i]` offline, current frame RMS live). So the output loudness follows the whisper's loudness (spec C7, viva question "why match energy": the filter gain $G$ and pulse shape do not guarantee a loudness; matching keeps the dynamics of the original, makes consonants quieter than vowels as in the input, and makes the volume knob meaningful). If a hop is digitally silent (RMS $<10^{-12}$) it is returned as is and the stored gain is not changed (test `test_output_step_digital_silence_then_onset_is_bounded`).

**A subtlety you may be asked about [my analysis from computing the filters].** $R(z)\,D(z) = (1-0.98z^{-1})/(1-0.97z^{-1})$. Its magnitude is 0.667 at DC, 0.89 at 100 Hz, 0.998 at 500 Hz, 1.003 at 1 kHz, 1.005 at 4 kHz and at Nyquist. So the radiation differentiator and the de-emphasis integrator **largely cancel each other above about 100 Hz**; the net effect of the two stages is a low-frequency shelf, not +6 dB then -6 dB at all frequencies independently. Consequences: (a) the pre-emphasis tilt that the LPC envelope inherited is *not* removed overall: it is compensated by the radiation filter, not by de-emphasis alone; (b) on the synthetic fixture, noise-labelled (label 1) output frames have an LPC envelope that is 0 dB different from the input below 1.5 kHz and about +2 dB, +5 dB, +7 dB higher in the 1.5-3, 3-6, 6-8 kHz bands (my measurement by LPC re-analysis of the output on `tests/fixtures.py` audio; approximate). That is, the output is somewhat brighter than the whisper at high frequencies. For vowel-like (label 2) frames the same re-analysis on the fixture (re-checked 2026-11-22) gives roughly 0 dB below 500 Hz, -4 to -7 dB at 0.5 to 1.5 kHz, and +12 to +18 dB above 3 kHz, for v1 (no noise) and v3 (aspiration) alike. The fixture's whispered vowels have very little energy above 3 kHz, so this number is specific to that clip, but it shows the voiced output is not spectrally matched to the input and explains much of the 10 to 13 dB LSD values (section 6.3). This is a design interaction (each filter has a separate justification in the code comments) rather than a bug, but if an examiner asks "why both radiation and de-emphasis, don't they cancel?", the honest answer is: yes, mostly; radiation converts flow to pressure for the voiced source, de-emphasis undoes pre-emphasis in the analysis filter, and what remains is a shelf below 100 Hz plus the DC blocker. See also question Q23.

**Soft limiter versus `peak_protect`.** Offline, `peak_protect(y, peak=0.99)` scales the **whole** signal down by $0.99/\max|y|$ only if the maximum exceeds 0.99 (needs the entire file). Live, `_soft_limit` in `src/live.py` works per sample, memoryless: for $|y| > 0.9$ (`LIMIT_KNEE`) it maps $|y| \to 0.9 + 0.1\tanh\bigl((|y|-0.9)/0.1\bigr)$, which approaches but never reaches 1.0; below the knee it does nothing. In the committed walkthrough run the live peak is 0.766, so the limiter is inactive there.

---

## 3. Function-by-function walkthrough

Defaults and why. "Spec" = `docs/whisper_to_voice_team_guide.pdf`.

### 3.1 `src/pitch.py`

Constants: `F_MIN_EXC=40`, `F_MAX_EXC=800` (excitation clamp), `TAIL_LEN=401` (longest pulse at 40 Hz is 400 samples), `PHASE_INIT=-1e-6` (first pulse at the first sample), `PHASE_EPS=1e-9` (guards floating-point errors when a phase lands exactly on an integer).

- **`impulse_train(f0, dur, fs=16000) -> ndarray`** (C1). `n = round(dur*fs)`; impulses at `round(k*fs/f0)`; returns zeros if `f0<=0` or `n==0`. Sum for 120 Hz over 1 s is 120 (tested within 1). Used only in tests and notebooks (the C1 check); the pipeline uses `rosenberg_pulse`.
- **`rosenberg_pulse(T0, open_frac=0.40, close_frac=0.16) -> ndarray`** (C2). Formula in 2.2. `T0` may be fractional; length `round(T0)`. Defaults 0.40/0.16: the code and spec do not cite a source; the open quotient 0.4 is in the usual range for modal voice and a short closing phase gives the sharp closure [standard engineering reason; do not claim the team fitted them to data].
- **`_unit_pulse(L)`** cached; `rosenberg_pulse(L) / rms`.
- **`_phrases(labels, gap_frames)`**: phrase boundaries as above; returns list of `(first_active, last_active)`.
- **`_smooth_runs(f0, voiced)`**: median 5 then mean 3 inside each voiced run.
- **`build_f0(labels, energy, base_f0=120, decl=0.12, stress=0.25, jitter=0.005, phrase_gap_s=0.5, hop=160, fs=16000, seed=0) -> float32 array`** (C4). Formula 2.5. Why 120 Hz: a typical male-range conversational $f_0$, inside the 70-400 Hz limits [standard]; it is a **fixed** parameter (README: base pitch is the user's choice, the app has a slider 70-250 Hz). Why 0.12 declination and 0.25 stress: not justified by data in the repo; they are chosen to be audible but mild (the walkthrough says "pitch drifts down by 12 percent across a phrase, as in natural speech"). Be honest: tuned by ear.
- **`init_f0_state(seed=0)`**, **`f0_step(label, energy, state, base_f0=120, decl=0.12, stress=0.25, jitter=0.005, phrase_gap_s=0.5, phrase_dur_s=2.0, smooth=0.5, hop=160, fs=16000) -> (f0, state)`**. Section 2.6. `phrase_dur_s=2.0`: assumed typical phrase length for the declination ramp (nothing in the repo derives it); `smooth=0.5`: one-pole coefficient (each hop moves halfway to the raw target).
- **`build_f0_causal(labels, energy, ..., seed=0)`**: loops `f0_step`.
- **`init_excitation_state(seed=0)`**, **`excitation_step(label, f0, state, hop=160, fs=16000, aspiration=0.15) -> (e float32, state)`**. Section 2.7. `aspiration` is clipped to [0, 1].
- **`_unit_noise(rng, n)`**: standard normal, rescaled to RMS exactly 1 for that hop (spec: "all at unit RMS").
- **`_voiced_hop`**: the phase/pulse/tail machinery.
- **`make_excitation(labels, f0, hop=160, fs=16000, aspiration=0.15, seed=0) -> float32 (n*hop,)`** (C5). Loops `excitation_step`. Seed 0 makes the noise reproducible (test `test_aspiration_changes_excitation_and_is_reproducible`).

### 3.2 `src/synth.py`

Constants: `GAIN_CAP=50`, `RAMP_RATIO=2`, `SILENT_RMS=1e-12`, `DEEMPH=0.97` (= the pre-emphasis coefficient, so it is exactly its inverse), `OUT_DC_POLE=0.99`, `PEAK=0.99` (just below full scale).

- **`synth_step(a, g, e_hop, zi) -> (y, zi)`** (C6). `lfilter([g], a, e_hop, zi=zi)`.
- **`synthesize(A, G, e, labels, energy, hop=160) -> float32 y`**. Loops `synth_step`; `zi` of length `order`. `labels` and `energy` are in the signature for the contract but unused (note that if asked).
- **`init_output_state()`**, **`output_step(y_hop, target_rms, state, volume=1.0)`**. Section 2.9.
- **`output_stage(y, energy_in, hop=160, volume=1.0)`** (C7). Loops `output_step`; the target for hop $i$ is `energy_in[i]`.
- **`peak_protect(y, peak=0.99)`**: as above; returns float32.
- **`_features(path_in, order=12)`**: load_audio, frame_signal, frame_energy, classify_frames, lpc_all; returns dict `fs, n, energy, labels, A, G`.
- **`_render(feats, volume=1.0, base_f0=120, decl=0.12, stress=0.25, jitter=0.005, aspiration=0.15, warp=None, hop=160)`**: optional `warp_poles` on every frame's `a`; `build_f0`; `make_excitation`; `synthesize`; `output_stage`; pads or trims to the original length `n` (the output has `n_frames*160` samples, a bit shorter than the input, so zero-padded at the end); `peak_protect`.
- **`convert(path_in, path_out, volume=1.0, base_f0=120, decl=0.12, stress=0.25, aspiration=0.15, warp=None, order=12) -> (y, fs)`** (C8). Note: `jitter` is not a parameter of `convert` (fixed at 0.005).
- **`VARIANTS`**, **`render_variant(feats, name, volume=1.0, base_f0=120)`**, **`run_ablation(path_in, results_dir="results", prefix=None, order=12)`** (C9): writes `{prefix}_{variant}.wav` for the four variants.

### 3.3 `src/live.py` (main functions)

- **`default_params()`**: dict read every hop so a GUI can change it while streaming (test `test_params_change_midstream` changes `base_f0` from 100 to 200 mid-stream and the output $f_0$ follows, ratio above 1.5). Includes calibration/labelling parameters (Module A): `calib_s=0.3`, `silence_mult=3.0`, `zcr_pct=70`, `hangover=5`, `zcr_thr=None`, `zcr_init=0.12`, `zcr_window_s=3.0`, `zcr_min_s=0.5`.
- **`init_live_state(params=None, fs=16000, hop=160)`**: all the memories (`prev` block, calibration lists, DC-block state, f0 state, excitation state, synthesis `zi`, output state, `timings`).
- **`process_hop_live(block, state) -> (out, state)`**: the per-hop chain, described in section 5.
- **`process_file_live(x, params=None, fs=16000, hop=160, state=None)`**: pushes a whole array through hop by hop (last block zero-padded); output length equals input length; stores `process_file_live.last_state` for the latency report.
- **`latency_report(state, stream_latency_s=None)`**: mean, 95th percentile, max of per-hop ms, fraction over the 10 ms `BUDGET_MS`, `algorithmic_ms = 2 * hop_ms = 20`, `device_ms` (sum of stream latencies) and `total_ms`.
- **`plot_latency(state, path)`**: per-hop time against the 10 ms budget line (spec A8).
- Helpers: `_live_zcr_threshold`, `_causal_hangover`, `_frame_features`, `_soft_limit` (Module A labelling helpers are described in module_a.md).

### 3.4 `src/evaluate.py` (main functions)

- `load_transcripts(path)`: lines "NN text" to `{"NN": text}`; a line that does not start with a number (the file has `s01 Hey, how are you?`) gives a warning and is ignored.
- `normalise_text(s)`: lower case, `-` becomes a space, remove everything that is not a word character or space, collapse spaces.
- `discover_clips(raw_dir, shared_ids)`: `shared_test.wav` (sentences 01 and 02 back to back) plus `spkXX_sNN_whisper.wav` with an optional `_normal.wav` partner.
- `load_asr(model_name="base")`: `whisper.load_model(name, device="cpu")`.
- `transcribe(model, x16k)`: float32 numpy array, `language="en"`, `temperature=0.0` (greedy, deterministic), `fp16=False` (CPU), `condition_on_previous_text=False`, `verbose=None`.
- `wer_counts(ref, hyp) -> (errors, n_ref)` using `jiwer.process_words`: errors = substitutions + deletions + insertions; if the hypothesis is empty all reference words count as errors; empty reference gives (0, 0). `wer_for` returns the ratio and hypothesis.
- `lsd_variant(x_in, y_out, order=12, mask=None)`: LPC (order 12) per frame of the input and of the output, labels of the **input** from `classify_frames`, keep frames with label != 0 (optionally AND a user mask), then `lpc.lsd_frames`.
- `pitch_stats(x, fmin=70, fmax=400)`: `librosa.pyin`, voiced and finite values only (used for the f0 histogram).
- `run_ablation_eval(clips, results_dir, asr=True, model_name="base", transcripts=None, transcripts_path=..., model=None, dev=False)`: the main routine, section 6.
- `aggregate_rows`, `ablation_markdown`, `save_ablation`: tables. `make_listening_sheet`, `aggregate_listening`, `read_listening_scores`: listening test. `make_figures` and four `fig_*` functions.

### 3.5 Entry points

`run_offline.py` (arguments `--base-f0 --decl --stress --aspiration --warp --volume --variants`), `run_eval.py`, `tools/make_dev_clip.py`, `app.py`: sections 6 and 7.

---

## 4. Ablation variants (`VARIANTS` in `src/synth.py`)

The four variants are cumulative; all use the **same labels, energy and LPC** and differ only in the source and, for v4, the filter.

| variant | `VARIANTS[...]` | what changes | expect to hear / see |
|---|---|---|---|
| `v1_robot` | `decl=0, stress=0, jitter=0, aspiration=0, warp=None` | flat 120 Hz, perfectly periodic Rosenberg pulses, no noise | monotone robotic buzz; perfectly straight harmonic lines in the spectrogram; `build_f0` contour is a flat line over each voiced run |
| `v2_contour` | `aspiration=0, warp=None` (other keys fall back to `_render` defaults: decl 0.12, stress 0.25, jitter 0.005) | adds the f0 contour: falling per phrase, rising with loudness, light jitter | pitch moves and falls; harmonic lines curve; still buzzy between harmonics |
| `v3_aspiration` | `aspiration=0.15, warp=None` | adds noise mixed into the pulses (3 % of power); **this is the default pipeline** (`convert` default and live default) | less synthetic, noise between harmonics, more breath |
| `v4_warp` | `aspiration=0.15, warp=0.92` | adds `warp_poles(a, 0.92)` on every frame: pole angles times 0.92, radius kept | formants lowered by about 8 % (e.g. a pole at 296 Hz moves to 272 Hz in the walkthrough example): a longer vocal tract, a slightly deeper "different speaker" voice; formant bands shift down in the spectrogram |

Notes: `warp` is applied also to consonant (label 1) frames, so fricative spectra shift too. In the app and `render_variant`, `base_f0` and `volume` still apply to named variants (`render_variant(feats, name, volume, base_f0)`); the sliders for decl, stress, aspiration and warp are ignored for named variants (tested by `test_f0_params_follow_variant`).

What the ablation is supposed to show (spec viva question "what does the ablation show about each addition"): v1 to v2 isolates the effect of prosody, v2 to v3 the effect of breath noise, v3 to v4 the effect of changing the filter. The spec's stated purpose for the warp is to "correct the upward formant shift of whispers" (architecture table, factor 0.90 to 0.95), i.e. to move whispered formants back toward voiced positions; whether that helps intelligibility or naturalness is exactly what the v3 to v4 comparison should test, and it has not been measured on real whispers. See section 6 for what the numbers actually say.

---

## 5. The live path

### 5.1 Causal per-hop chain (`process_hop_live`)

Input: one 160-sample block; output: one 160-sample block.

1. `dc_block` on the block with carried state `dc_zi` (pole 0.995, about 13 Hz).
2. Frame = previous block + current block (320 samples, the 20 ms **ending at the current hop**); `rms` and `zcr` of that frame (`_frame_features`).
3. **Calibration** (Module A rule): the first `int(0.3*16000//160) = 30` hops only collect noise-floor statistics (hop 0 is skipped because its previous half is zero), output is zeros; thresholds are computed at the 30th hop with `calibrate_thresholds` from the 29 collected frames (its guard nearly always fires on such a short list, so the live noise floor is the 10th percentile of the calibration RMS values rather than their median; see module_a.md 4.4). This is why the speaker must stay silent for the first 0.3 s.
4. After calibration: `silence_thr` from calibration; adaptive ZCR threshold `_live_zcr_threshold` (starts at `zcr_init=0.12`, then the 70th percentile of ZCR of the last 3 s of non-silent frames, updated every 10 non-silent hops once 0.5 s of speech is heard; a fixed `zcr_thr` param overrides it); `label_frame`; causal hangover of 5 hops (hold the last non-silent label for 50 ms; no 3-frame median because it needs the next frame).
5. `lpc_frame(frame, order=12)` (Module B) and optional `warp_poles`.
6. `f0_step` (this module) with the frame RMS as energy.
7. `excitation_step`, `synth_step` (state `zi` carried), `output_step` with target = frame RMS and `volume`.
8. `_soft_limit(np.nan_to_num(y))`, store `last_label`, `last_f0`, append the hop's wall-clock time (ms) to `timings`.

### 5.2 What is shared with the offline path

`excitation_step`, `synth_step`, `output_step` (and `lpc_frame`, `warp_poles`, `label_frame`, `calibrate_thresholds`) are literally the same functions. The offline `make_excitation`, `synthesize`, `output_stage` are just loops over them. That is why the offline chain, when fed the live path's own decisions, reproduces the live output to rounding error (5.4). The things that are *not* shared are decisions: labels, f0, which frame is analysed.

### 5.3 Latency

- **Budget**: a hop is 10 ms, so each call must finish in under 10 ms or the stream falls behind (spec A8: "processing per hop stays below 10 ms").
- **Measured processing time** (this machine, Windows, pure Python, committed notebook and my re-run on the same synthetic clip of 2.57 s = 257 hops): mean 0.25 ms (notebook: 0.273), p95 0.45 ms (0.493), max 1.1 ms (1.299), fraction over budget 0.0. Tests assert mean below 2 ms and p95 below 5 ms (`test_timing`) and that the step functions alone take less than 2 ms per hop (`test_step_functions_fast_enough_for_live`).
- **Algorithmic latency**: the code defines `algorithmic_ms = 2 * hop_ms = 20 ms`; it is a model, not a measurement, and the code gives no derivation (the notebook says "a hop is 10 ms and the frame is 20 ms, so the algorithmic delay is about two hops"; `docs/PROJECT.md` says "two hops, the frame length"). The reading used in all three viva guides **[standard reasoning for block-based duplex audio]**: the callback cannot run until the whole 160-sample input block has arrived (up to one hop, 10 ms), and the output block it returns is played during the next block period (another hop). There is no look-ahead: the analysis frame ends at the current hop, so the 20 ms frame length itself adds no waiting for future samples. The code treats it as a constant of 20 ms.
- **Total** = algorithmic + sound-device buffers: `latency_report(state, stream_latency_s)` adds the device latencies; the test uses 10 ms + 20 ms to get 50 ms. The spec's viva question: "Where does the 30 to 50 ms latency of the live system come from?" (algorithmic 20 ms plus input and output device buffers). The device part is the value the driver reports through `stream.latency` in Module A's `src/stream.py` (reported, not measured with a loopback), and it has not been recorded on the demo machine yet.
- Plotting is done outside the audio callback (spec common mistake).

### 5.4 Why live and offline are not sample identical (4 causes) and the proof

1. **Causal labels.** Live uses an adaptive ZCR threshold (offline: a percentile over the whole clip), a causal hangover, and **no 3-frame median filter**; the first 0.3 s are muted for calibration. So some labels differ. Measured: **91 %** agreement after calibration (aligned by one hop, see cause 3).
2. **`f0_step` versus `build_f0`.** Declination from running phrase time (saturating after 2 s) instead of position in the whole phrase; stress against a running mean energy instead of the phrase mean; one-pole smoothing instead of median 5 + mean 3. On the *same* labels the two $f_0$ contours differ on average by **5.7 Hz** (max 13.2 Hz) on the walkthrough clip. A few Hz of $f_0$ difference moves the pulse positions and decorrelates the waveforms completely (sample correlation live vs offline is 0.08), so waveform correlation is meaningless here; the RMS envelopes correlate at 0.952 (notebook).
3. **Frame alignment.** Offline output hop $i$ uses frame $i$, which *starts* at hop $i$ (covers hops $i$ and $i+1$, looking one hop ahead). Live output hop $j$ uses the frame *ending* at hop $j$ (hops $j-1$ and $j$). LPC filter and loudness target are one hop (10 ms) apart.
4. **Limiter.** Live uses a memoryless soft limiter (knee 0.9) since it cannot know the file's peak; offline uses `peak_protect` (whole-signal scaling above 0.99).

**The proof** (`notebooks/walkthrough.ipynb` section 8, cell "Proof"): run the live path hop by hop, **record** each hop's label, f0 and the exact 320-sample frame it analysed; then run the *offline* functions `lpc_all`, `frame_energy`, `make_excitation`, `synthesize`, `output_stage` on those recorded decisions. If the synthesis code were different between paths, the output would differ. I re-ran this code on the notebook's clip (the synthetic fixture, since no real recording exists in this repository copy): **max |difference| = 6.0e-08** (my run: 5.96e-08), live peak 0.766 (below the 0.9 knee, so the limiter did not act), `process_file_live` output equals the hop-by-hop output exactly, label agreement 91.2 %, f0 methods 5.7 Hz apart on average. So the difference comes entirely from the *inputs* (causal decisions), not from the synthesis code. If the limiter had been active the proof would show differences above the knee.

---

## 6. Evaluation

### 6.1 Data

- Real data (not in this repo copy; `data/raw/` contains only `transcripts.txt`): `data/raw/spkNN_sMM_whisper.wav` with optional `_normal.wav`, and `shared_test.wav` (sentences 01 and 02). Sentences are the Harvard sentences (spec 1.4): "The birch canoe slid on the smooth planks." etc., in `data/raw/transcripts.txt` (6 lines + one ignored line `s01 Hey, how are you?`).
- **Dev clips (`tools/make_dev_clip.py`)**: **SYNTHETIC**. Uses Windows SAPI text-to-speech (via PowerShell) to speak the transcript sentences, scales to peak 0.3, pads 0.5 s of silence each end, then `whisperize`: per 10 ms hop, an LPC (order 12) filter is fitted to the TTS speech and white noise is filtered through it (state carried), then a $1/(1-0.97z^{-1})$ filter (de-emphasis; note the variable naming), and per-hop RMS is matched to the TTS speech. A noise floor of $5\times10^{-4}$ is added. Output `results/dev/dev_sNN_whisper.wav` and `_normal.wav` (the "normal" is the TTS speech itself). Purpose: **pipeline check only, not results**: it exercises ASR, LSD, figures and the app when no recordings exist. It is an LPC-noise whisper, i.e. exactly the model the system assumes, so it says nothing about real whispers.

### 6.2 WER with Whisper

`run_ablation_eval` per clip: `_features`, load audio, render the four variants (also saved as `{stem}_{variant}.wav` in the results folder), and for each system in `[raw_whisper, v1_robot, v2_contour, v3_aspiration, v4_warp]` compute LSD and (if `asr` and a reference exists) WER; plus `normal_recording` (the clip's `_normal.wav`) as the upper reference.

- **Model**: Whisper `base` (about 74 M parameters; general knowledge), loaded on CPU (`device="cpu"`). The spec mentions a GPU for this step (only Raks's laptop), but the code forces CPU; with the base model and a few short clips CPU is fast enough [my reading]. `fp16=False` because half precision is a GPU feature.
- **Numpy input**: the waveform is passed directly as float32 at 16 kHz; Whisper accepts arrays at 16 kHz, so no ffmpeg decoding or temp files are needed [standard engineering reason].
- **Deterministic decoding**: `temperature=0.0`, `language="en"`, `condition_on_previous_text=False` (each clip independent).
- **Normalisation** (`normalise_text`): case folded, hyphens to spaces, punctuation removed; no number words or spelling variants (so "3" versus "three" counts as an error). Apostrophes are deleted ("it's" becomes "its").
- **WER** $= (S + D + I)/N$ from `jiwer.process_words`, per clip; `wer_mean` = mean of per-clip WER, `wer_corpus` = total errors / total reference words. WER can exceed 1 because of insertions.
- ASR as a judge: it is a *proxy* for intelligibility (a model trained on normal speech, with its own language model that can "fix" or hallucinate words).

### 6.3 LSD per variant

$\mathrm{LSD}=\frac1{|\mathcal F|}\sum_{i\in\mathcal F}\sqrt{\frac1K\sum_{k=1}^{K}\bigl(20\log_{10}|H_\text{in,i}(\omega_k)| - 20\log_{10}|H_\text{out,i}(\omega_k)|\bigr)^2}$ with $H = G/A(e^{j\omega})$, $K=257$ bins (`n_fft=512`). $\mathcal F$ = frames where the *input's* label is not 0 (and the optional mask). Because the gain $G$ is inside $H$, **level differences count**, not only shape. The input envelope is from the whisper; the output envelope from LPC re-analysis of the voiced output, so LSD measures how much the output's analysed spectrum deviates from the whisper's, and anything that makes the output's LPC differ from the source LPC (including harmonics biasing the LPC fit, or the deliberate warp) raises it. `raw_whisper` has LSD 0 by definition. Lower means the formant structure is retained.

### 6.4 Pitch statistics, figures

`pitch_stats` uses `librosa.pyin` (70-400 Hz). Figures (`make_figures`, in `<results>/figures/`, from the first clip with the `v4_warp` output):

- `fig_ablation_bar.png`: bars of mean WER and mean LSD per system.
- `fig_waveform_spectrogram.png`: five rows: whisper waveform, its spectrogram, label track, output waveform, output spectrogram (same dB scale; harmonic lines appear only in the output).
- `fig_envelope_overlay.png`: loudest label-2 frame, FFT magnitude against LPC envelope, for input and output, with the warped envelope dashed on the input.
- `fig_f0.png`: the built `build_f0` contour over the label track, plus a histogram of pyin f0 of our output and of the normal recording (if present).

### 6.5 Listening test

`make_listening_sheet(clips, results_dir, seed=0, n_listeners=5)`: for each clip, 5 items (raw whisper and 4 variants); items are **shuffled** with a seeded permutation and written as `item_01.wav ...` (the file name hides the system); `key.csv` maps item to clip and system; `sheet.csv` has one row per listener (L1..L5) and item with empty `clarity_1to5`, `naturalness_1to5`; `README.txt` states the ethics: voluntary, told the purpose, can stop, only scores stored, no personal data, only team recordings used with consent. Listeners do not open `key.csv`. After scoring: `python run_eval.py --aggregate-listening results/listening/sheet.csv`, which (`aggregate_listening`) skips rows with a blank or non-numeric score (a row with only one of the two scores is dropped entirely) and averages clarity and naturalness **per system** pooled over clips and listeners, writing `listening_scores.csv`; the next `run_eval.py` run puts those means in the last two columns of `ablation.md`. The normal recording is not in the listening set (no human reference score).

### 6.6 Ablation table outputs

`ablation_per_clip.csv` (per clip, system: WER, errors, n_ref, LSD, ASR hypothesis text), `ablation.csv` (means), `ablation.md` (the table, with a SYNTHETIC banner if `--dev`).

**Dev table (SYNTHETIC DEV DATA: SAPI text-to-speech sentences whisperised by LPC plus noise; 6 clips, 49 reference words in total; pipeline check only, not results about real whisper; no listening scores yet).** There are two runs, and you must know which one is on screen:

- **Current code** (2026-11-22, `python run_eval.py --dev`, in `results/dev_eval/`; three runs, identical output, so Whisper decoding is deterministic here; these are the numbers in `docs/PROJECT.md`).
- **Earlier run** (2026-11-20 21:20, *before* commit 9512509 added the input DC blocker in `load_audio` and the output DC blocker in `output_step`; kept locally in `results/dev_eval_2026-11-20/`, not committed).

| System variant | WER current code (corpus) | LSD current (dB) | WER earlier run (corpus) | LSD earlier run (dB) |
|---|---|---|---|---|
| raw_whisper | 0.328 (0.327) | 0 (reference) | 0.328 (0.327) | 0.00 |
| v1_robot | 0.782 (0.796) | 10.49 | 0.707 (0.714) | 11.23 |
| v2_contour | 0.736 (0.755) | 11.42 | 0.784 (0.796) | 12.03 |
| v3_aspiration | 0.492 (0.510) | 6.77 | 0.545 (0.551) | 6.63 |
| v4_warp | 0.750 (0.776) | 6.38 | 0.594 (0.612) | 6.57 |
| normal_recording (the TTS original) | 0.042 (0.041) | | 0.042 (0.041) | |

Error counts out of 49 words (from `ablation_per_clip.csv`): current code raw 16, v1 39, v2 37, v3 25, v4 38, normal 2; earlier run raw 16, v1 35, v2 39, v3 27, v4 30, normal 2. One word is about 2 percentage points of corpus WER.

How to read it honestly: (1) in both runs Whisper ASR does **better on the raw (pseudo-)whisper than on every voiced variant**; the conversion did not improve ASR-measured intelligibility here. That is the case the spec warns about ("An ASR trained on normal speech may not get a lower WER on our output, because rule based pitch is artificial... report it"). (2) In both runs aspiration (v3) has the lowest WER of the four variants and, with v4, much lower LSD (about 6.4 to 6.8 dB against 10.5 to 12 dB without noise); that is the only trend that survives both runs, plausibly because noise between harmonics keeps formant structure visible (and LPC re-analysis less biased by harmonics) [standard engineering reason]. (3) v1 versus v2 swaps order between the runs: no claim about the contour. (4) v4 versus v3: 0.05 WER apart in the earlier run, 0.26 apart (v4 worse, 38 versus 25 errors) with the current code. A small code change elsewhere moved v4 by 8 words, so on six synthetic sentences no claim about the warp can be made either way. (5) The synthetic whisper is, by construction, an all-pole noise model, so it is *easier* for LPC than a real whisper, and the "normal" reference is a very clean TTS voice. Before the viva, run the evaluation on real recordings if at all possible.

---

## 7. The Gradio app (`app.py`) and `run_offline.py`

`python app.py` launches a tabbed app (gradio imported only inside `build_app`; pure helper functions are tested without gradio).

- **Convert tab**: upload or record a clip, or "Load demo clip" (the shared test wav, else a generated pseudo whisper). Dropdown **Variant** (`custom`, `v1_robot` ... `v4_warp`; custom uses the sliders), sliders Base pitch 70-250 Hz, Intonation (declination) 0-0.3, Stress 0-0.5, Aspiration 0-0.3, Pole warp 0.85-1.0 (1.0 = off), Volume 0.2-2.0. **Convert** renders (`render` -> `_render` or `render_variant`) and shows: input player (A), voiced output player (B), a download button (wav), spectrograms of input and output, and a plot of frame labels with the f0 contour. **Compare all variants** renders custom plus the four named variants into five players. A note says the live mode runs from the command line.
- **Analysis tab**: "Analyse clip" then Module A figure (RMS with silence threshold, ZCR with threshold, voicing strength, labels; sliders for the silence multiplier and ZCR threshold, 0 = automatic) and Module B figure (FFT magnitude vs LPC envelope of one chosen frame, with optional pole-warped envelope; prediction gain and LPC gain text).
- **Results tab**: finds folders (`results/` and direct subfolders) with an `ablation.md`; shows the table, the figures gallery, and per-clip players for each variant. If none exist, a message tells you to run `run_eval.py`.
- Errors such as "clip shorter than one 20 ms frame" are shown as friendly messages (`friendly` wrapper). Features are cached per file.

`run_offline.py input.wav output.wav [--base-f0 150 --decl --stress --aspiration 0.2 --warp 0.92 --volume 1 --variants]`: calls `convert`; with `--variants` also `run_ablation` writing the four variant wavs into `results/`.

`run_eval.py`: `python run_eval.py` (needs `openai-whisper`, `jiwer`; writes `results/ablation.md`, `results/figures/`), `--no-asr` (LSD only), `--model base`, `--dev` (uses `results/dev`, writes to `results/dev_eval`, banner "SYNTHETIC DEV DATA"), `--listening` (blinded sheet), `--aggregate-listening SHEET`.

---

## 8. Tests that cover your module

Run: `python -m pytest tests -q`. I ran the four Module C files (65 tests) and all passed in about 22 s.

**`tests/test_module_c.py`**
- `test_impulse_train_harmonics`: 120 Hz train has line peaks at harmonics 1 to 5, much higher than between-harmonic bins; 120 impulses in one second (C1).
- `test_rosenberg_pulse_shape`: length $\mathrm{round}(T_0)$, starts at 0, peak 1 at 40 % of $L$, exactly 0 after opening+closing, never negative (C2).
- `test_excitation_f0_autocorrelation` (120 and 200 Hz): autocorrelation estimate within 3 % of the target (spec C3).
- `test_excitation_glide_tracked`: a 100-to-150 Hz glide is followed (start and end estimates within 5 %, end > 1.25 x start): the running phase works.
- `test_hopwise_equals_single_pass`, `test_hopwise_matches_independent_train`: hop-by-hop generation equals one long call and equals an independently built periodic train (no boundary artefacts; pulse tail correct).
- `test_no_hop_rate_modulation`: per-hop RMS of the excitation in [0.75, 1.25] with and without aspiration (no 100 Hz loudness ripple).
- `test_build_f0_basic_properties`: shape, zero outside voiced frames, inside 70-400 Hz, no frame-to-frame jump above 15 %.
- `test_build_f0_declination_and_reset`: falls (more than 8 %) within each phrase, resets at the second phrase, monotonic without stress and jitter.
- `test_build_f0_short_gap_does_not_reset`: a 0.3 s pause is not a phrase break.
- `test_build_f0_louder_is_higher`: correlation above 0.9 between f0 and log energy (stress works).
- `test_build_f0_flat_when_neutral`: with decl, stress, jitter 0 the contour is exactly `base_f0`.
- `test_build_f0_causal_properties`: the causal tracker has the same basic properties (range, zero outside voiced, falling, reset, flat when neutral, stress correlation above 0.5).
- `test_excitation_unit_rms_per_label`: label 0 exactly zero, label 1 unit RMS, label 2 close to unit RMS (spec C5).
- `test_label2_without_f0_is_noise`, `test_aspiration_changes_excitation_and_is_reproducible` (same seed gives identical output; aspiration changes it; RMS stays near 1).
- `test_synthesize_hopwise_equals_lfilter`: carrying `zi` across hops equals one `lfilter` call (C6, no clicks).
- `test_output_stage_follows_input_loudness`: output/input RMS ratio between 0.5 and 2 and log-RMS correlation above 0.9 (C7).
- `test_output_step_radiation_deemphasis_dc_block_applied_once`: `output_step` equals an independent `lfilter` chain times the gain, and the ramp of the second hop is correct.
- `test_output_step_digital_silence_then_onset_is_bounded`: silent hop stays zero, a sudden loud hop after silence does not explode (gain cap/ramp).
- `test_convert_end_to_end`, `test_run_ablation_writes_four_wavs`: with the pipeline patched to mocks; output length, finite, loudness correlation above 0.5; four wavs with the right names; variants differ.
- `test_step_functions_fast_enough_for_live`: under 2 ms per hop for f0, excitation, synthesis, output steps.
- `test_output_has_little_energy_below_60hz`: DC blocker works.
- `test_render_shorter_than_one_frame`: a clip with no full frame gives silence of the input length, with or without warp.

**`tests/test_live.py`**: `test_output_basic` (length, finite, peak at most 1, first 0.3 s exactly zero), `test_loudness_follows_input` (log-RMS correlation above 0.85), `test_output_is_voiced` (autocorrelation voicing strength of vowel regions rises by more than 0.1 versus the whisper), `test_timing` (mean and p95 ms; algorithmic 20 ms, total 50 ms), `test_plot_latency`, `test_params_change_midstream`, `test_live_vs_offline_envelope` (log-RMS envelope correlation above 0.8: the live and offline loudness agree), `test_imports_without_device`, `test_run_live_file_subprocess`, `test_list_devices_subprocess`, three tests of the adaptive ZCR threshold (Module A logic inside live).

**`tests/test_evaluate.py`**: transcripts parsing (warning on the odd line), `normalise_text`, `wer_counts` (0 errors on case/punctuation differences, all deletions for an empty hypothesis, one substitution), `discover_clips`, `lsd_variant` (0 for identical signals, above 0.5 against noise), `run_ablation_eval` without ASR (five systems, LSD 0 for raw and positive for variants, files written, table header) and with a monkeypatched ASR (WER 0.25 computed by hand: 1 deletion in 4 words), listening sheet (5 items per clip, blinded files exist, 5 listeners x items rows, README) and aggregation (means per system, rows with a missing score dropped, `raw_whisper` absent when it had no complete scores, `ablation.md` picks up the scores).

**`tests/test_app.py`**: convert returns audio and figures; the download file equals the output; missing audio raises `ValueError`; compare gives five players in the right order; Module A figure and thresholds (a manual ZCR threshold is honoured, a higher silence multiplier gives a higher threshold); LPC frame figure at first, middle, last frame and warps 1.0, 0.92, 0.85; results discovery (empty and populated); `build_app()` constructs; a too-short clip gives a friendly error; named variants ignore the decl/stress sliders (`f0_params`).

**Notebooks.** `notebooks/c_synthesis_validation.ipynb` (16 cells; covers C1 to C7, not C8 to C10) re-checks against the rubric: C1 120 impulses in 1 s and lines at multiples of 120 Hz; C2 pulse length 133, starts at 0, peak at 0.398 of $T_0$; C3 autocorrelation f0: 120 Hz target gives 120.3 Hz (0.25 % error), 200 Hz gives 200.0 Hz, glide 100 to 150 estimates 105.3 to 144.1 Hz, per-hop RMS 0.912 to 1.161; C4 contour plot of offline vs causal; C5 hop RMS per label (0, 1.0, 0.998); C6 waveform across hop boundaries; C7 output RMS vs input RMS (ratio 0.60 to 1.80, log-RMS correlation 0.921, `output_step` equals the `lfilter` chain). `notebooks/walkthrough.ipynb` sections 5 to 9: source, synthesis and output stage, variants, live path with proof, evaluation.

---

## 9. Weak spots and limitations: honest answers

1. **The pitch contour is synthetic, not the speaker's intent.** Whisper carries no $f_0$ information, so the contour is rule-based (declination plus loudness-driven stress plus jitter). It does not follow words, questions or emphasis. Honest answer: it is a plausible default, not recovered prosody; a real fix needs a model that predicts prosody from text or from the whisper (neural), beyond scope.
2. **Fixed base f0.** 120 Hz regardless of speaker (slider 70-250 Hz); not matched to the speaker's size or sex. A very different speaker will sound odd.
3. **Buzziness.** A parametric source with fixed pulse shape (40/16/44 %), no shimmer, tiny jitter, no turbulence modulated by the glottal cycle, no source-tract interaction: it will sound synthetic. Aspiration (3 % of power) is a modest fix.
4. **Label errors propagate.** A vowel mislabelled as consonant gets noise instead of pitch (it stays a whisper); a consonant labelled vowel gets a buzz (a voiced "s"). Labels come from two features (RMS, ZCR) with calibrated thresholds; breathy consonants and weak vowels can be misclassified. Live adds adaptive thresholds and no median filter, so live is noisier (91 % agreement).
5. **Whisper formants differ from voiced formants.** Whispering changes the spectrum (e.g. no low-frequency voice component, different bandwidths/formant frequencies) [standard phonetics], so the LPC envelope of a whisper is an approximation of the voiced tract; order 12 is below the 18 rule of thumb (walkthrough).
6. **Aspiration trade-off.** More noise: more natural/breathy and fewer harmonic artefacts, but less clear pitch and lower harmonic-to-noise ratio; 0.15 is a compromise chosen by listening, not optimised.
7. **Output stage tilt.** Radiation and de-emphasis almost cancel above 100 Hz (2.9); the output is somewhat brighter than the input at high frequencies on noise frames (up to about +7 dB) and much brighter above 3 kHz on vowel frames of the fixture (+12 to +18 dB by LPC re-analysis). Not tuned against real recordings.
8. **Time-varying filter switched each hop.** No interpolation of coefficients; the state is continuous so there are no clicks, but timbre steps every 10 ms; the all-pole filter with unit-RMS pulses includes DC that is only removed afterwards.
9. **ASR as a judge.** Whisper has a strong language model: it can recover or invent words. Greedy decoding, base model, English forced. Normalisation is simple (numbers, contractions not handled). WER measures intelligibility to a machine trained on normal speech, not to a human, and not naturalness. Pseudo-whisper ASR performance is not human performance either.
10. **Small and synthetic evaluation set.** The only results in the repo are dev numbers on 6 synthetic clips (49 words): pipeline check only. On them, every voiced variant has a **higher** WER than the raw synthetic whisper (in both runs). An earlier run, before the DC blocker changes, gave noticeably different per-variant numbers (section 6.6), which shows how sensitive six sentences are. No real WER table exists in this copy of the repo (no recordings); the listening test has no scores yet. If real recordings and 5 listeners exist by the viva, use those and say the dev table is only a check.
11. **LSD limits.** Compares LPC envelopes (includes gain), so it penalises level differences and harmonic-biased LPC fits, and v4_warp is *meant* to change formants; LSD says nothing about pitch or naturalness.
12. **Live is not identical to offline** (section 5.4); latency numbers exclude device buffers measured elsewhere; timing measured in Python on one machine.
13. **Where `jitter` is hidden.** `convert` and the app have no jitter parameter (fixed 0.005); jitter is effectively about 0.15 % after smoothing.
14. **Whisper on CPU versus the spec's GPU.** The spec mentions a GPU Whisper; the code is CPU only (fine for `base`).

---

## 10. Likely viva questions with model answers

**Conceptual**

1. *What does your module do?* It builds the artificial source and synthesises the voice: pitch contour from labels and energy, excitation (pulses plus noise for vowels, noise for consonants, zeros for silence), all-pole filtering with Module B's LPC, then an output stage (radiation, de-emphasis, DC blocker, loudness matching), and it evaluates the result. Also the live per-hop version.
2. *Why does whisper have no pitch?* In whisper the vocal folds do not vibrate; the source is turbulent noise, so there is no periodicity and no harmonics, although the tract still shapes formants, so words are intelligible.
3. *Why does a pulse train give harmonics at multiples of $f_0$?* A periodic signal with period $T_0$ has a Fourier series with components at $k/T_0$. For an impulse train all coefficients are equal; for a smooth pulse they fall off.
4. *Why is a Rosenberg pulse more natural than an impulse train?* The glottis releases a smooth flow, not an impulse. The Rosenberg pulse has a spectral roll-off of about -12 dB/octave (measured), like a real voice; an impulse has a flat spectrum and sounds like a buzzer.
5. *Explain your Rosenberg pulse.* $L=\mathrm{round}(T_0)$ samples: 40 % raised-cosine opening to 1, 16 % quarter-cosine closing, 44 % closed zero. It is the glottal *flow*; the lip-radiation differentiation is applied in the output stage.
6. *Rosenberg versus LF model?* Rosenberg is a simple piecewise-cosine model of the flow with two shape parameters (open and closing fractions). The Liljencrants-Fant model describes the flow *derivative* with an exponentially returning phase after the closure and has more parameters (e.g. return phase, asymmetry), so it can model breathier or pressed phonation and the spectral tilt more realistically. We chose Rosenberg for simplicity, few parameters, and because it is easy to generate, cache and normalise per period.
7. *How does a running phase let $f_0$ change smoothly?* The phase advances by $f_0/f_s$ per sample; a pulse starts each time it crosses an integer. Changing $f_0$ changes only the slope, so spacing glides; $f_0$ is interpolated linearly across each hop; the fractional phase is carried to the next hop, and pulse tails overlap-add across the boundary.
8. *How did you build the f0 contour and why declination and stress?* Base 120 Hz, linear 12 % fall across each phrase (phrase = region without silence longer than 0.5 s), times $(1+0.25\,\delta)$ where $\delta$ is the clipped log-energy deviation (louder means higher, as stressed syllables typically are), light jitter, median-5 and mean-3 smoothing, clip 70-400 Hz. It is rule-based because a whisper has no pitch to copy.
9. *What is jitter and how much do you use?* Cycle-to-cycle $f_0$ perturbation. Uniform $\pm0.5\,\%$ per 10 ms frame, reduced by smoothing to about 0.15 % standard deviation; it removes the perfectly periodic robotic character slightly.
10. *Why must the filter state be carried across hops?* The all-pole filter has memory; restarting at zero each hop cuts the ringing of the resonances and makes a discontinuity, heard as clicks every 10 ms. We pass `zi` from one `lfilter` call to the next; the test shows hop-wise output equals one long `lfilter`.
11. *What is lip radiation and why do you apply it?* Sound radiating from the mouth is approximately the time derivative of the volume flow, +6 dB/octave. Since the Rosenberg pulse models flow, we apply $1-0.98z^{-1}$ so the output has the radiated-pressure tilt.
12. *Why de-emphasis?* The LPC was fitted to pre-emphasised speech ($1-0.97z^{-1}$), so $G/A(z)$ carries an extra +6 dB/oct; $1/(1-0.97z^{-1})$ is its inverse. (Follow-up: see Q23.)
13. *Why match output energy to input energy?* Loudness then follows the whisper's dynamics (stressed syllables louder, consonants softer), the volume slider has a defined meaning, and the unit-RMS source plus arbitrary LPC gain would otherwise give random loudness. Per hop with a linear gain ramp to avoid steps.
14. *Why a DC blocker at the output?* The Rosenberg flow pulse has a large mean (mean/RMS about 0.63), and radiation plus de-emphasis keep a DC gain of 0.67; the RMS match would otherwise spend loudness on inaudible DC. Cutoff about 25 Hz.
15. *What does the ablation show?* Section 6.6: on the synthetic dev set the raw pseudo-whisper has lower WER (0.33) than every voiced variant (current code 0.49 to 0.78; earlier run 0.55 to 0.78); within variants aspiration gives the lowest WER (0.49 current, 0.55 earlier run) and the aspiration variants have much lower LSD (about 6.4 to 6.8 dB against 10.5 to 12); the order of v1, v2 and v4 changes between runs, so no claim about contour or warp. We state this honestly and it is not a result on real whispers.

**"Why this value"**

16. *Why 0.15 aspiration, is that 15 % noise?* It is an amplitude weight; noise power is $0.15^2/(0.85^2+0.15^2) = 3\,\%$. The spec asks for 10-20 %; it was chosen by listening within that range.
17. *Why hop 160 / frame 320?* 10 ms/20 ms: speech is roughly stationary over 20 ms; 50 % overlap; fixed by the project contract and shared by all modules.
18. *Why clip $f_0$ to 70-400 Hz?* The usual range of speaking $f_0$ (spec C4); also protects the pulse generator (longest pulse 400 samples at 40 Hz is the tail buffer size).
19. *Why a 0.5 s phrase gap?* A pause that long counts as a breath/new phrase so declination resets; shorter pauses (e.g. between words) do not (test). Chosen heuristic.
20. *Why base 120 Hz?* Typical conversational male-range pitch; a parameter, not derived from data.

**"What if"**

21. *What if you used the impulse train instead?* Flat spectrum, harsh buzzer (spec common mistake). The code still has `impulse_train` for the C1 check.
22. *What if the filter state were reset each hop?* Clicks every 10 ms (a 100 Hz buzz artefact).
23. *Don't radiation and de-emphasis cancel each other?* Largely, yes: their cascade has magnitude 0.667 at DC, 0.89 at 100 Hz and about 1.0 above 500 Hz. Together with the DC blocker they act as a low-frequency shelf. The radiation step is justified for the flow source, de-emphasis for the pre-emphasised LPC; the net result is not tuned against recordings (my measurement on a synthetic clip shows the noise frames are up to about 7 dB brighter at 6-8 kHz than the input). If you want to tune: drop or reduce the radiation or de-emphasis and listen.
24. *What if f0 is outside 70-400 Hz or label 2 has f0 = 0?* Contour is clipped; `excitation_step` treats label 2 with f0 = 0 as noise.
25. *What if the clip begins with speech (no 0.3 s of silence)?* Offline `calibrate_thresholds` falls back to the 10th percentile of all energies if the first 0.3 s are louder than the 25th percentile (Module A). Live, the same guard only sees the calibration frames and therefore uses their 10th percentile: speech that starts part way into the 0.3 s is tolerated, but speech from the very first sample gives a threshold far too high (0.106 instead of about 0.0057 on the fixture), so most speech is muted. Restart and stay quiet for 0.3 s.
26. *What if the live path is slower than 10 ms per hop?* Buffers underrun, audio glitches. Measured mean 0.25 ms, max about 1.1 ms in Python, nowhere near.
27. *What if aspiration = 0?* v2_contour; clean periodic pulses; harmonics visible but buzzy; LSD higher on the dev data.

**Code-specific**

28. *Why does `excitation_step` drop a cut pulse at the end of a voiced run?* On any non-pulse hop phase, previous f0 and tail are reset, so silence is exactly zero and the next voiced run starts with a fresh pulse (`PHASE_INIT`). Cost: the last partial pulse of a run is truncated.
29. *Why is `TAIL_LEN` 401?* One more than the longest allowed pulse (40 Hz is 400 samples), so a pulse starting at the last sample of a hop fits in `hop + TAIL_LEN`.
30. *Why `PHASE_EPS`?* Floating-point sums can leave a phase just below an integer; the epsilon avoids missing or double-counting a pulse onset.
31. *How is the per-hop loudness gain limited?* Cap at 50, ramp start clamped within a factor of 2 of the new gain, linear interpolation over the hop; digital silence passes through unchanged.
32. *Why `peak_protect` offline but a soft limiter live?* Offline knows the global peak and can scale all samples uniformly without distortion; live cannot, so it uses a memoryless tanh limiter above 0.9 that never exceeds 1.0.
33. *How do you know live and offline share the same synthesis?* Notebook section 8: offline functions fed with the live decisions reproduce the live output to 6e-8.
34. *Why is the live output not identical to offline?* Four causes (5.4): causal labels, `f0_step` vs `build_f0`, frame ending vs starting at the hop, soft limiter vs `peak_protect`.
35. *Where does 20 ms algorithmic latency come from?* It is a code constant of two hops, a model rather than a measurement: one hop to collect the 160-sample input block before the callback can run, one hop for the returned block to be played out; there is no look-ahead because the frame ends at the current hop. Total latency adds the input and output device buffers (the spec's "30 to 50 ms"); the device part has not yet been measured on the demo machine.

**Comparisons**

36. *LPC source-filter vocoder versus neural vocoder?* LPC/pitch-pulse vocoders are tiny, fast (about 0.25 ms per hop here), transparent (every parameter explained), need no training data and run in real time on a laptop CPU; but they sound buzzy and robotic. A neural vocoder (e.g. WaveNet/HiFi-GAN type, conditioned on mel spectrograms and $f_0$) sounds far more natural but needs training data and compute, is a black box, and still needs a pitch contour and an acoustic model for whisper-to-speech conversion (typically a trained mapping), which we do not have. For a signal processing course the LPC pipeline demonstrates the source-filter theory directly.
37. *Why evaluate with Whisper WER?* It is an objective, repeatable, reference-based proxy for intelligibility, runs on a laptop (base on CPU), and the spec asks for it (jiwer, Whisper). Caveat: ASR language model and training on normal speech; so we add LSD (spectral fidelity) and a blinded human listening test (clarity and naturalness).
38. *Why is raw whisper sometimes better in WER?* Whisper (ASR) is trained on a very large and varied set of web audio, so it copes with breathy and noisy input and leans on its language model; our pulses, artificial prosody and buzz add artefacts it has not seen, and noise frames come out brighter after the output stage (2.9). The spec predicts this possibility. On the synthetic dev set it happened for every variant; on real whispers it has not been measured.
39. *Offline `build_f0` vs causal `f0_step`?* Offline knows phrase length and phrase-mean energy (declination exact 12 % over the phrase, stress relative to phrase mean, median+mean smoothing); causal uses time since phrase start (saturating at 2 s), running mean energy, one-pole smoothing: lagging by a hop or two, slightly different.
40. *How is LSD computed and what are its limits?* Section 6.3; RMS over frequency of the dB difference of LPC envelopes (with gain), averaged over non-silent frames of the input; it measures envelope fidelity, not pitch or naturalness, and is inflated by harmonics-biased LPC and by deliberate warping.

---

## 11. One-page cheat sheet

**Pipeline**: wav -> frames (320/160) -> energy, labels (A) -> LPC A, G (B) -> `build_f0` -> `make_excitation` -> `synthesize` -> `output_stage` -> `peak_protect` -> wav. Live: same steps per 160-sample hop; `f0_step`; soft limiter.

**Numbers to remember**
- Frame 320 (20 ms), hop 160 (10 ms), fs 16 kHz, LPC order 12.
- Rosenberg: open 0.40, close 0.16, closed 0.44 (133 samples at 120 Hz = 53+21+59). Flow pulse; slope about -12 dB/oct; mean/RMS 0.63.
- Aspiration 0.15 amplitude = 3 % power (5 % of AC power). Spec: 10-20 %.
- f0: base 120, decl 0.12 (12 % per phrase), stress 0.25 (log-energy deviation clipped to $\pm1$, so $\pm 25\,\%$), jitter 0.005, clip 70-400 Hz, phrase gap 0.5 s, smoothing median 5 then mean 3.
- Causal f0: declination over 2 s (saturating), running mean of log energy (count cap 200), one-pole smoothing 0.5.
- Excitation clamp 40-800 Hz, tail buffer 401, unit RMS, silence exactly 0.
- Output stage: radiation $1-0.98z^{-1}$, de-emphasis $1/(1-0.97z^{-1})$, DC blocker pole 0.99 (about 25 Hz), RMS match with gain cap 50, ramp ratio 2, `peak_protect` 0.99; live soft limiter knee 0.9.
- Radiation x de-emphasis: 0.667 at DC, about 1.0 above 500 Hz.
- Variants: v1 flat, v2 + contour, v3 + aspiration (default), v4 + warp 0.92.
- Live: 30 calibration hops (0.3 s) muted; mean about 0.3 ms per hop (0.25 to 0.5 across runs), budget 10 ms; algorithmic 20 ms (capture one hop + play out one hop, a model); total adds device buffers (not yet measured).
- Live vs offline: 4 causes; proof 6e-8; labels 91 %; f0 5.7 Hz apart.
- WER: Whisper base, CPU, numpy input, greedy; jiwer S+D+I over N; normalisation lower/punct/hyphen. LSD: input non-silent frames, RMS dB difference of $G/A$ envelopes, 257 bins.
- Listening: 5 items per clip, shuffled, blinded names, key.csv, 5 listeners, scores 1-5, means per system.
- Dev table (SYNTHETIC, 49 words), current code: WER raw 0.33, v1 0.78, v2 0.74, v3 0.49, v4 0.75; TTS normal 0.04; LSD 10.5, 11.4, 6.8, 6.4. Earlier run (2026-11-20): v1 0.71, v2 0.78, v3 0.55, v4 0.59; LSD 11.2, 12.0, 6.6, 6.6. Robust in both: raw best, v3 best variant, aspiration variants lowest LSD.

**Three sentences to say first**: (1) Whisper has the filter but no source; we keep the LPC filter and invent the source. (2) Every design choice (Rosenberg pulse, running phase, carried filter state, de-emphasis, energy matching) removes a specific artefact: buzzer sound, pitch jumps, clicks, brightness, random loudness. (3) We report the ablation honestly: on synthetic data the conversion did not beat raw whisper in ASR WER; aspiration helped most.

**Commands**: `python run_offline.py in.wav out.wav --variants`, `python run_eval.py [--dev|--no-asr|--listening]`, `python app.py`, `python -m pytest tests -q`.
