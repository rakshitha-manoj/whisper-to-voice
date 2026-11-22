# Viva guide: Module B (LPC analysis), for Akhila

You own `src/lpc.py`. This guide is meant to teach you the topic from zero and then drill you on what examiners ask. Everything here was checked against the code in the repo (and by running snippets). Where a reason is my standard engineering explanation rather than something written in the code or spec, it is marked **(standard reason)** or **(standard fact)**.

Contents: 1 Overview, 2 Theory, 3 Function walkthrough, 4 Order 12, 5 Pole warping, 6 LSD, 7 Validation and tests, 8 Weak spots, 9 Viva Q&A, 10 Cheat sheet.

---

## 1. Your module in one paragraph, and where it sits

Speech is a buzz from the vocal cords shaped by the throat and mouth. Module B takes each 20 ms frame (320 samples at 16 kHz, hop 160) and estimates the **shape of the throat and mouth** as a 12 coefficient all-pole filter $1/A(z)$, plus a **gain** $G$ that sets its loudness. Whisper has no vocal cord buzz, but it still carries the filter shape (that is why you can understand a whisper), so we keep the filter and replace the noise source with a voiced one.

```
Module A:  wav -> frame_signal (N x 320) -> energy -> classify_frames (labels 0/1/2)
                                  |
                                  v
Module B:  lpc_all(frames, 12) -> A (N x 13), G (N,)     [warp_poles(a, factor) optional]
                                  |
                                  v
Module C:  excitation e (pulse train / noise) -> lfilter([G], A, e) -> output stage -> wav
Evaluation: lsd_frames(A_in, G_in, A_out, G_out, mask) -> "LSD (dB)" column of the ablation table
```

- Input: frames from Module A. Output: `A` (row i = $[1, a_1..a_{12}]$) and `G`, the handoff Module C uses.
- Offline: `src/synth.py::_features` calls `lpc_all(frames, order)`; `_render` calls `warp_poles` on every row of `A` if `warp` is set. `synth_step` then runs `scipy.signal.lfilter([g], a, e_hop, zi=zi)`.
- Live: `src/live.py::process_hop_live` calls `lpc_frame(frame, order)` once per 10 ms hop (frame = previous block + current block), optionally `warp_poles`, then `synth_step`.
- Evaluation: `src/evaluate.py::lsd_variant` runs `lpc_all` on the input and on the output wav and calls `lsd_frames`.

Spec tasks B1 to B11 are all implemented in `src/lpc.py` and checked in `notebooks/b_lpc_validation.ipynb`.

---

## 2. Theory from scratch

### 2.1 Source-filter model

Speech production is modelled as two stages:

1. **Source**: for voiced sounds, the vocal cords make a periodic pulse train (pitch $f_0$); for whispers and unvoiced sounds like "s", just turbulent noise.
2. **Filter**: the vocal tract (throat, mouth, nose cavities) has resonances called **formants**. It reshapes the source spectrum, boosting frequencies near the resonances.

The words (which vowel, which consonant) live mostly in the filter. Pitch and voicing live in the source. LPC extracts the filter. That is the whole reason we can turn a whisper into voiced speech: keep the filter, swap the source.

### 2.2 The vocal tract as an all-pole filter

A resonance is mathematically a **pole** of a filter's transfer function. We model the tract as

$$H(z)=\frac{G}{A(z)},\qquad A(z)=1+\sum_{k=1}^{p}a_k z^{-k}$$

"All-pole" means no zeros (no deliberate notches), only resonances. In the time domain: $x[n] = G\,e[n] - \sum_{k=1}^p a_k x[n-k]$, with $e$ a unit RMS excitation. This is exactly `lfilter([G], a, e)`.

### 2.3 Linear prediction and the sign convention

Predict each sample from the previous $p$ samples:

$$\hat x[n] = -\sum_{k=1}^{p} a_k\,x[n-k],\qquad e[n] = x[n]-\hat x[n] = x[n]+\sum_{k=1}^p a_k x[n-k]$$

So the prediction error is the output of $A(z)$ applied to the signal: `residual = lfilter(a, [1], x)`. **Convention (code, spec and `librosa.lpc` agree): `a[0] = 1`, and `a[1:]` appear with a plus sign in $A(z)$, hence a minus sign in the predictor.** $A(z)$ is the analysis (whitening) filter; $1/A(z)$ is the synthesis filter that puts the shape back. The spec lists mixing the sign as a common mistake.

### 2.4 Minimising the error: normal equations

Choose $a_1..a_p$ to minimise the energy $E=\sum_n e[n]^2$. Set $\partial E/\partial a_i=0$ for $i=1..p$:

$$\sum_{k=1}^p a_k \sum_n x[n-k]x[n-i] = -\sum_n x[n]x[n-i].$$

In the **autocorrelation method** the frame is windowed (zero outside) and the sums run over all $n$, so $\sum_n x[n-k]x[n-i]=r[|i-k|]$ with $r[m]=\sum_n x[n]x[n+m]$. The equations become

$$\sum_{k=1}^p a_k\, r[|i-k|] = -r[i],\quad i=1..p \quad\Longleftrightarrow\quad R\,\mathbf a=-\mathbf r$$

$$R=\begin{bmatrix} r[0]&r[1]&\cdots&r[p-1]\\ r[1]&r[0]&\cdots&r[p-2]\\ \vdots&&\ddots&\vdots\\ r[p-1]&r[p-2]&\cdots&r[0]\end{bmatrix},\quad \mathbf r=[r[1],\dots,r[p]]^T$$

**Why Toeplitz:** entry $(i,k)$ depends only on $|i-k|$ (every diagonal is constant), because the autocorrelation depends only on the lag, not on absolute position. The matrix is also symmetric and positive semi-definite. The minimum error is $E_p = r[0]+\sum_k a_k r[k]$ (the test checks exactly this identity).

`test_levinson_matches_toeplitz` solves the same system independently with `scipy.linalg.solve_toeplitz(r[:p], -r[1:p+1])`.

### 2.5 Levinson-Durbin recursion, step by step

Solving $R\mathbf a=-\mathbf r$ by general elimination costs $O(p^3)$. Because $R$ is Toeplitz, Levinson-Durbin builds the order-$i$ solution from the order-$(i-1)$ one. Start: $a^{(0)}=[1]$, $E_0=r[0]$. For $i=1..p$:

1. **Correlate the current predictor with the next lag:** $\text{acc}_i = r[i]+\sum_{j=1}^{i-1} a^{(i-1)}_j\, r[i-j]$
2. **Reflection coefficient:** $k_i = -\text{acc}_i/E_{i-1}$
3. **Update coefficients:** $a^{(i)}_j = a^{(i-1)}_j + k_i\,a^{(i-1)}_{i-j}$ for $j=1..i-1$, and $a^{(i)}_i=k_i$
4. **Update error:** $E_i = E_{i-1}\,(1-k_i^2)$

Code mapping (`levinson`): `acc = r[i] + np.dot(a[1:i], r[i-1:0:-1])`; `k = -acc/err`; `a[1:i] = prev + k*prev[::-1]` (the reversed copy is the $a_{i-j}$ term); `a[i] = k`; `err *= 1 - k*k`.

**Tiny worked example** (checked by running the code): $r=[1,\,0.5,\,0.2]$, $p=2$.
- $i=1$: acc $=0.5$, $k_1=-0.5$, $a_1=-0.5$, $E_1=1\cdot(1-0.25)=0.75$.
- $i=2$: acc $=r[2]+a_1 r[1]=0.2-0.25=-0.05$, $k_2=0.05/0.75=0.0667$. Then $a_1\leftarrow -0.5+0.0667\cdot(-0.5)=-0.5333$, $a_2=0.0667$. $E_2=0.75(1-0.00444)=0.7467$.
- Result `[1, -0.5333, 0.0667]`, error 0.7467. Solving the 2x2 system directly gives the same coefficients.

**Why $O(p^2)$:** step $i$ costs $O(i)$ (a dot product and a vector update); summing over $i=1..p$ gives about $p^2$. For $p=12$ that is about 144 multiply-adds against roughly 1700 for $p^3$. The measured cost of the full `lpc_frame` is about 0.1 ms per frame (notebook B5: 0.098 ms); a test asserts under 1 ms.

**Stability, $|k_i|<1$:** the $k_i$ are the reflection (PARCOR) coefficients. $1/A(z)$ is stable (all poles strictly inside the unit circle) if and only if every $|k_i|<1$. The error $E_i=E_{i-1}(1-k_i^2)$ must stay positive and can only decrease with order.

### 2.6 Why the autocorrelation method guarantees a stable synthesis filter

Because the frame is windowed to zero outside, $r$ is the autocorrelation of a finite-energy signal, so the Toeplitz matrix $R$ is positive definite (for a non-degenerate signal). For a positive definite Toeplitz matrix Levinson-Durbin gives $|k_i|<1$ at every stage, hence a minimum-phase $A(z)$: all poles of $1/A$ strictly inside the unit circle. Intuition: $E_i$ is an energy and cannot go negative, which forces $|k_i|\le1$; strictness comes from positive definiteness. The covariance method does not have this guarantee (its matrix is not Toeplitz and the filter can be unstable). `test_poles_inside_unit_circle` checks every fixture frame; the notebook measured a max pole radius of 0.994 (close to, but below, 1). The tiny noise correction (2.10) keeps $R$ positive definite in floating point.

### 2.7 Gain $G$ and what it matches

`lpc_frame` returns $G=\sqrt{E_p/\sum_n w[n]^2}$, where $E_p$ is Levinson's final error (the energy of the **windowed, pre-emphasised** residual) and $\sum w^2$ is the Hamming window energy (126.78 for 320 samples; I computed it).

Meaning: $E_p/\sum w^2$ is the residual power per sample after undoing the window's energy loss. The exact statement is this: the all-pole model reproduces the autocorrelation it was fitted to (the autocorrelation matching property of LPC), so the expected output power of unit-variance white noise through $G/A(z)$ is $G^2\sum_n h[n]^2 = r[0]/\sum w^2$, the **window-weighted** mean power of the pre-emphasised frame. I checked this numerically: on a synthetic /a/ frame the two sides agreed to 9 significant digits. Against the plain (unweighted) RMS of the pre-emphasised frame the match is only approximate, because the Hamming window weights the centre of the frame more than the edges: the ratio was 1.19 on one test frame and 0.89 on another, and a single white-noise realisation adds a few percent of scatter. `test_gain_matches_rms` accepts a ratio between 0.5 and 2. Say honestly in the viva:
- G matches the **pre-emphasised** frame level, not the raw one. The output stage later applies lip radiation, de-emphasis, and then **matches RMS to the input energy per frame** (the spec says "the LPC gain alone is not enough"), so absolute accuracy of G is not critical.
- G is **not** changed by `warp_poles`.

### 2.8 LPC envelope, formants, bandwidth

$|H(e^{j\omega})|=G/|A(e^{j\omega})|$ is the **LPC envelope**: a smooth curve through the spectral peaks. The FFT of a frame shows every harmonic (voiced) or random bump (whisper); the envelope shows only the vocal tract shape. It is computed with `scipy.signal.freqz([1.0], a, ...)`.

**Poles are formants.** Factor $A(z)=\prod_k (1-z_k z^{-1})$. Complex poles come in conjugate pairs $z_k=r_k e^{\pm j\theta_k}$:
- Formant frequency: $F_k=\dfrac{\theta_k f_s}{2\pi}$.
- Bandwidth (-3 dB): $B_k\approx -\dfrac{f_s}{\pi}\ln r_k$. Closer to the unit circle ($r\to1$) means a sharper, narrower peak.

Example I computed (synthetic vowel /a/ from `tests/fixtures.py`, true formants 700, 1220, 2600 Hz): order-12 LPC gave poles at 836, 1228 and 2713 Hz with radii 0.952, 0.990, 0.972 (bandwidths about 250, 53, 147 Hz). The remaining poles (radius 0.73 or less, bandwidth over 1.5 kHz) are not formants; they shape the overall tilt. The 836 and 2713 Hz estimates are 19 and 4 percent off, so formant estimates are not exact even on clean synthetic data. (The envelope includes the pre-emphasis tilt; `test_envelope_peaks_near_formants` removes it with a de-emphasis response before comparing peaks.)

### 2.9 Residual and prediction gain

The residual $e=A(z)x$ is what the model cannot predict. For voiced speech it resembles the glottal pulses; for whisper it is noise-like. So the residual approximates the source, which is why replacing the source and keeping the filter makes sense. **Prediction gain** $=10\log_{10}(\sum x^2/\sum e^2)$ dB. Notebook B7: 22.0 dB on the loudest fixture frame, mean 17.8 dB over non-silent frames (synthetic fixture, not real recordings). The spec success criterion (B7) is a residual that looks like noise and has much less energy than the frame.

Note: `prediction_gain_db(frame, a)` filters the **raw** frame with `a`, while `a` was fitted to the pre-emphasised, windowed frame, so it is a slightly mismatched measure (still positive; the test only requires $>0$).

### 2.10 Pre-emphasis 0.97, Hamming window, noise floor

- **Pre-emphasis** $y[n]=x[n]-0.97x[n-1]$ is a first-order high-pass. Speech spectra fall roughly 6 dB/octave (spectral tilt), so low frequencies dominate $r[k]$ and the fit would spend its poles on the strong low band; pre-emphasis flattens the spectrum so higher formants are modelled too. (Walkthrough text: "flattens the speech spectrum tilt so the fit is not dominated by low frequencies". 0.97 is the spec value; a typical range is 0.95 to 0.98, **standard fact**.) Synthesis undoes it with de-emphasis $1/(1-0.97z^{-1})$ in `output_stage` (`DEEMPH = 0.97`).
- **Hamming window** $w[n]=0.54-0.46\cos(2\pi n/(N-1))$. The autocorrelation method assumes the signal is zero outside the frame; a rectangular cut makes sharp edges that leak energy across the spectrum and distort $r[k]$. Tapering the edges reduces this (spec: "Forgetting the window, which gives poor LPC on frame edges"). Hamming rather than Hann because it does not go fully to zero and has lower near side lobes **(standard reason)**.
- **White noise correction** `r[0] = r[0]*(1+1e-9) + 1e-12`: a tiny diagonal loading of $R$, equivalent to adding a very small white noise to the signal. It keeps $R$ positive definite when a frame is nearly periodic or nearly zero, so Levinson cannot divide by zero. 1e-9 is a $-90$ dB relative floor, too small to change the envelope audibly. The spec asks for "a tiny floor to r[0]"; the constants are the author's choice.

---

## 3. Function by function (`src/lpc.py`)

Module constants: `_WINDOWS` (cache of Hamming windows per length, so `np.hamming(320)` is computed once) and `_EPS_ENERGY = 1e-12`.

### `_hamming(n)`
Returns the cached `np.hamming(n)`. Private helper; `evaluate.py`'s envelope-overlay figure also imports it.

### `preemphasis(x, alpha=0.97)`
$y[0]=x[0]$, $y[n]=x[n]-\alpha x[n-1]$. Converts to float64. Empty input returns an empty array. `y[0]=x[0]` is the usual choice when there is no previous sample. alpha 0.97 is spec B1. The test checks the formula with alpha 0.9 and `y[0]==x[0]`.

### `autocorr(frame, p)`
$r[k]=\sum_{n=0}^{N-1-k} x[n]x[n+k]$, $k=0..p$. "Biased": it is not divided by $N-k$ (the biased form keeps the autocorrelation sequence positive definite, **standard reason**). Implemented as a loop of dot products. Edge case: if $p\ge N$, lags beyond $N-1$ are left at 0 (`min(p, n-1)`; I checked `autocorr(ones(5), 8)` gives `[5,4,3,2,1,0,0,0,0]`). `r[0]` is the frame energy (spec B3). The test compares with `np.correlate(x, x, 'full')`.

### `levinson(r, p)`
Section 2.5. Returns `(a, err)`: `a` has length $p+1$, `a[0]=1`, `err` is the final prediction error. Edge cases coded: if `r[0] <= 0` it returns `a=[1,0,...]`, error 0. If during the recursion `err` becomes non-positive (numerically $|k|\ge1$) it clamps `err` to 0 and **breaks**, leaving higher coefficients at zero (the order is effectively truncated). Notebook agreement with `solve_toeplitz`: 8.9e-9 (spec requires 1e-8; test asserts $<10^{-8}$).

### `lpc_frame(frame, order=12)` returns `(a, gain)`
Pipeline: `preemphasis` (0.97) $\to$ multiply by Hamming $\to$ `autocorr` up to lag `order` $\to$ noise correction $\to$ `levinson` $\to$ `gain = sqrt(max(err,0)/sum(w**2))`.
Edge cases:
- empty frame, or any NaN/inf in the frame: returns `a=[1,0..0]`, gain 0.
- silent frame: if `r[0] < 1e-12`, returns `a=[1,0,...]`, gain 0 (spec: "silent frames: a=[1,0,0..], gain 0"). With gain 0 the synthesis filter outputs zeros, so silence stays silent.
- if the result contains NaN/inf it falls back to the silent result, so nothing downstream sees NaN.
- A zero-gain frame has $A=1$ (flat), which is harmless.
Default `order=12` follows the spec contract (section 4).

### `lpc_all(frames, order=12)` returns `(A, G)`
Loops `lpc_frame` over the rows. `A` shape `(n_frames, order+1)` with `A[:,0]=1`; `G` shape `(n_frames,)`. Zero frames gives shapes `(0, order+1)` and `(0,)` (tested). A plain Python loop at about 0.1 ms/frame is fine for offline and real time (a hop is 10 ms).

### `residual(frame, a)`
`scipy.signal.lfilter(a, [1.0], frame)`, i.e. $e=A(z)x$ (an FIR filter). Starts from zero state, so the first $p$ samples have an edge effect.

### `prediction_gain_db(frame, a)`
$10\log_{10}\frac{\sum x^2+10^{-20}}{\sum e^2+10^{-20}}$. The $10^{-20}$ terms avoid division by zero (a silent frame gives 0 dB).

### `lpc_envelope(a, gain=1.0, n_fft=512, fs=16000)` returns `(freqs, mag_db)`
`freqz([1], a, worN=n_fft//2+1, fs=fs)`, then $20\log_{10}(|G|\,|H|+10^{-12})$. That is 257 points from 0 to 8000 Hz (31.25 Hz spacing). The $10^{-12}$ avoids $\log 0$ (e.g. when gain is 0). `gain=1.0` default means shape only.

### `log_spectral_distortion(a_ref, a_test, gain_ref=1.0, gain_test=1.0, n_fft=512)`
RMS over the 257 frequency points of the dB difference between the two envelopes, gains included:
$$\mathrm{LSD}=\sqrt{\frac1K\sum_{k}\big(20\log_{10}|H_{ref}(\omega_k)|-20\log_{10}|H_{test}(\omega_k)|\big)^2}$$
Identical filters and gains give exactly 0 (tested). Because gains are included, a pure level difference counts as distortion.

### `lsd_frames(A_ref, G_ref, A_test, G_test, mask=None)`
Mean of `log_spectral_distortion` over the selected frames. `mask` may be boolean or an index array. If no frame is selected it returns 0.0 (not NaN). Test: masking out the only differing frame gives exactly 0.

### `envelope_lsd(frame, order=12, n_fft=512)`
A different quantity: how well one frame's LPC envelope fits the **actual FFT** of that frame. It computes LPC, then the 512-point FFT magnitude (dB) of the pre-emphasised, windowed frame, shifts the envelope so the **mean dB values are equal**, and takes the RMS of the difference. Silent frame returns 0.0. The order sweep (B9) uses this. Gain is deliberately not used: it measures shape only.

### `warp_poles(a, factor)`
Section 5.

---

## 4. Why order 12 (and the fs/1000 + 2 rule)

- **Rule of thumb** (spec glossary and walkthrough): order $\approx f_s/1000+2$, so $16+2=18$ at 16 kHz. Reasoning in the walkthrough: about one formant per kHz of bandwidth, two poles (one conjugate pair) per formant, plus two for the spectral tilt.
- **Why the project uses 12:** the spec fixes `order=12` in the contract ("order 12 to start", `lpc_frame(frame, order=12)`) and B9 asks you to choose from a sweep ("curve flattens around 10 to 16"). The notebook sweep (mean `envelope_lsd` over non-silent fixture frames):

| order | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LSD (dB) | 7.52 | 6.57 | 5.99 | 5.36 | 5.05 | 4.83 | 4.69 | 4.59 | 4.49 | 4.41 | 4.35 |

  The improvement per +2 step is 0.96, 0.58 and 0.62 dB from order 4 to 10, then 0.32 dB from 10 to 12, then 0.21 dB from 12 to 14 and 0.15 dB or less after that. The notebook rule: the smallest order whose next +2 step improves LSD by less than 0.25 dB, which gives 12. Its conclusion: "higher orders add cost and noise fitting for little gain". Caveat to admit: this sweep ran on the **synthetic fixture** (`pseudo_whisper`); the notebook falls back to it when `data/raw/shared_test.wav` is absent, and the committed output says "using tests.fixtures.pseudo_whisper()".
- **Trade-off:** LSD against the FFT keeps decreasing with order (more poles always fit better; Levinson's error never increases). So **lower LSD is not automatically better**. Higher order starts modelling individual harmonics (voiced) or random noise bumps (whisper), giving an unnatural, "unstable sounding" synthesis (spec common mistake "Order too high"). Too low (4 to 8) merges formants and loses vowel identity. Order 12 gives 6 pole pairs: roughly 4 to 5 formants plus tilt.
- Note 12 is below the 18 rule; the justification is the knee of the curve and the cost, not the rule. Another plausible reason is that whispered speech has weak, noisy high formants, so extra poles above about 4 kHz would mostly model noise **(standard reason, not tested in the repo)**.

---

## 5. Pole warping

### What the code does (exactly)
1. Return an unchanged copy if `a` has fewer than 2 entries, all coefficients beyond `a[0]` are zero, or `factor == 1.0`.
2. `roots = np.roots(a)` finds the poles.
3. For every root `z`:
   - the radius is clamped: `r = min(|z|, 0.9999)` (safety; roots of a stable LPC are already inside).
   - **real roots** (imaginary part negligible) stay real, sign preserved, with the clamped radius.
   - **complex roots**: new angle `theta = angle(z) * factor`, same radius: `r * exp(j*theta)`.
4. Rebuild: collect the upper-half-plane poles, add their conjugates and the reals, `np.poly(...)` to get coefficients, take the real part, set `res[0]=1`, return an array of the same length. If the upper and lower counts ever mismatch, unpaired poles are dropped.

So: **the angle is scaled, the radius is kept.** The frequency of each resonance becomes $F' = \text{factor}\cdot F$ (angle is proportional to frequency), while the bandwidth $-\frac{f_s}{\pi}\ln r$ is unchanged because the radius is unchanged.

Verification: on the walkthrough frame, factor 0.92 moved pole frequencies from [296, 2279, 3030, 3285, 5741, 7208] Hz to [272, 2096, 2788, 3022, 5281, 6632] Hz, each a ratio of 0.92. On my vowel test frame, 0.92 moved the formant poles to about 769, 1130, 2496 Hz and the max radius stayed 0.99 (unchanged).

### Meaning of factor < 1
Moving every resonance down by the same ratio is equivalent to a **longer vocal tract** (tube resonances scale as 1/length). Perceptually: a lower, larger-sounding, more male-like voice. Factor > 1 shortens the tract (more child/female-like). The spec asks for 0.90 to 0.95 ("shift formants down"); the ablation variant `v4_warp` uses 0.92; `warp` is also a parameter of `convert`, the live path and the app.

Why do whispered formants need it? The spec states the reason in its architecture table: the pole warp "corrects the upward formant shift of whispers" (angle factor 0.90 to 0.95). It does not quantify the shift. The literature agrees that whispered vowels have formants shifted **upward** compared with normal speech, mostly the lower formants (the open glottis changes the effective tract) **(standard fact, not measured in this project)**; shifting down moves the voiced output toward normal positions. Phrase it cautiously: the direction comes from the spec and the literature, the factor 0.92 is a fixed choice inside the spec's range, and we did not measure the shift on our own data.

### Stability preserved
Radius is kept (and capped at 0.9999), so every pole stays inside the unit circle. `test_warp_identity_and_stability` checks all roots $<1$ for factors 0.85 to 1.0, identity at 1.0 within 1e-10, `a[0]==1`, same shape, float dtype. `test_warp_degenerate`: an all-zero `a` is returned unchanged.

### Limits to know
- Real poles are not moved (they carry tilt, not formants).
- Factor > 1 is accepted by the function, and moves formants up, but it can push an angle past $\pi$ (Nyquist) and wrap around: with factor 1.5, a pole at about 7170 Hz goes to $1.5\times7170\approx10760$ Hz, which is above 8 kHz and reappears at $16000-10760\approx5240$ Hz (the conjugate pair swaps half-planes, so the rebuilt polynomial is still real and stable). So the top of the spectrum folds back. Nothing in the project uses factor > 1: `run_live.py` only passes `--warp` when it is below 1.0, the app slider stops at 1.0, and the tests cover 0.85 to 1.0. Mention only if asked.
- $G$ is not rescaled, so warped peak heights differ; the output stage RMS match corrects the level afterwards.
- Root finding plus `np.poly` per frame is slower than the Levinson step, accurate to float precision; fine at order 12.
- One uniform scale for all formants is crude; real speaker differences are not a uniform scaling **(standard remark)**.

---

## 6. LSD

### Definition as implemented
Per frame (`log_spectral_distortion`): RMS over frequency of the dB difference between two $G/A$ envelopes, 257 points from 0 to 8 kHz, gains included:
$$\mathrm{LSD}_i=\sqrt{\tfrac1K\sum_k\big(L^{ref}_i(\omega_k)-L^{test}_i(\omega_k)\big)^2}\ \text{dB},\qquad L=20\log_{10}\big(G/|A|\big)$$
Over a clip (`lsd_frames`): the arithmetic mean of $\mathrm{LSD}_i$ over the masked frames.

### How `evaluate.py` uses it
`lsd_variant(x_in, y_out, order=12, mask=None)`:
1. `lpc_all` on the input whisper (A_in, G_in) and on the synthesised output (A_out, G_out), same order and framing.
2. Labels come from Module A's `classify_frames` **on the input**; frames with label 0 (silence) are excluded: `sel = labels != 0`. An optional extra mask is ANDed.
3. Truncated to the shorter of the two lengths, then `lsd_frames(...)`.
4. The raw whisper system is defined as LSD 0 (the reference); the normal-voice recording gets no LSD.
The ablation table column "LSD (dB)" is the mean of this per-clip number over clips.

Why mask silence: in silent frames the gain is near 0 and the envelope sits at the noise floor or the $10^{-12}$ epsilon (a frame with gain exactly 0 gives $-240$ dB), so differences would be huge and meaningless.

### Interpreting dB values
- 0 dB: identical envelopes. About 1 dB is the classical rule of thumb for "transparent" envelope distortion in speech coding **(standard reference value, not from this repo)**. Our numbers are far above that because we compare input and a deliberately different output, with level included.
- Dev results (SYNTHETIC: Windows text-to-speech sentences turned into pseudo-whispers by LPC-filtered noise, `tools/make_dev_clip.py`; pipeline check only, not results). These are generated files in `results/dev_eval/` (not committed to git). Re-running `python run_eval.py --dev` with the current code (2026-11-22, the numbers also used in `docs/PROJECT.md`) gives LSD v1_robot 10.49, v2_contour 11.42, v3_aspiration 6.77, v4_warp 6.38 dB. An earlier run (2026-11-20, kept locally in `results/dev_eval_2026-11-20/`, before the input and output DC blockers were added) and shows 11.23, 12.03, 6.63, 6.57 dB. Both runs agree on the one robust pattern: the two variants with aspiration noise have much lower LSD (about 6.4 to 6.8 dB) than the two without (10.5 to 12 dB). Do not quote these as findings; rerun `run_eval.py` on real recordings first. Be careful with the warp: you would expect a deliberate formant shift to raise LSD against the input, yet v4 is not higher than v3 in either run (6.38 vs 6.77, 6.57 vs 6.63). That tells you this LSD is dominated by other differences (level and tilt from the output stage, harmonic bias of the re-analysis) rather than by formant positions, not that warping "improves" anything.
- Because the output is re-analysed with LPC (pitched pulses give a harmonic spectrum, and analysis of voiced speech finds different poles than analysis of noise), this LSD is a **resynthesis consistency** measure, not a perceptual quality score. Gains are included, so loudness errors count too.
- `envelope_lsd` values (4.3 to 7.5 dB across orders) are a different thing: envelope vs FFT of one frame. Even a perfect model cannot reach 0 on a noise-like frame, because the periodogram of white noise has a log-magnitude standard deviation of about 5.57 dB (I computed this from 2000 random frames). That explains why the sweep levels off around 4 to 5 dB. (Standard statistics, not in the repo.)

---

## 7. Validation and tests

### Notebook `notebooks/b_lpc_validation.ipynb`
Uses `data/raw/shared_test.wav` if present, else the synthetic `pseudo_whisper()` fixture. The committed run used the fixture (256 frames, 129 non-silent).

| Check | Reference | Result in notebook |
|---|---|---|
| B1 pre-emphasis | formula `x[1:]-0.97x[:-1]`, spectrum before/after | assertion passes; high frequencies lifted |
| B3 autocorr | `np.correlate(y,y,'full')` | max abs diff 0.0; r[0] equals the energy |
| B4 Levinson | `scipy.linalg.solve_toeplitz(rr[:p], -rr[1:p+1])`, orders 4, 8, 12, 16, 24 | max error 8.9e-9 (under 1e-8) |
| B5/B6 | shapes, finiteness, speed | A (256,13), G (256,), 0.098 ms/frame, all finite, max pole radius 0.994, G median 0.0027 |
| B7 | prediction gain | 22.0 dB loudest frame, mean 17.8 dB |
| B8 | `librosa.lpc` (Burg), level aligned | envelope correlation 0.990 |
| B9 | order sweep with `envelope_lsd` | table in section 4; 12 chosen |
| B10 | `warp_poles(a, 0.92)`, envelope and pole plots | identity error 0.0, a[0]=1, max radius 0.976 |

`librosa.lpc` uses the **Burg** method, so coefficients differ slightly from the autocorrelation method; the comparison is on envelope shape (correlation), as the spec says.

### Tests in `tests/test_lpc.py` (run `python -m pytest tests/test_lpc.py -q`)
- `test_preemphasis`: formula and `y[0]=x[0]`.
- `test_autocorr_matches_numpy`: equals `np.correlate`; `r[0]` is the energy. Proves lag indexing.
- `test_levinson_matches_toeplitz`: against `solve_toeplitz` to 1e-8 for orders 4 to 24 on fixture and random frames, plus the identity `err = r[0] + a.r`. Proves Levinson solves the normal equations.
- `test_poles_inside_unit_circle`: finite `a`, `g`, and all roots below 1. Proves stability.
- `test_silent_frames`: zeros give gain 0, `a=[1,0..]`, length 13.
- `test_lpc_all_shapes`: empty input, order 10, `A[:,0]==1`, finite.
- `test_speed`: under 1 ms per frame (spec B5).
- `test_prediction_gain`: positive gain and residual energy below frame energy.
- `test_gain_matches_rms`: noise through `G/A` has RMS within a factor 0.5 to 2 of the pre-emphasised frame (loose bound; I measured ratios between about 0.9 and 1.2 on different frames).
- `test_envelope_peaks_near_formants`: de-emphasised envelope's two strongest peaks within 15 percent (mean over 5 seeds) of 700 and 1220 Hz for vowel /a/. Proves it finds true formants on a controlled signal.
- `test_envelope_vs_librosa`: envelope correlation with `librosa.lpc` above 0.9 for every fixture vowel.
- `test_warp_identity_and_stability`; `test_warp_moves_formants` (peaks move down by a ratio within 0.06 of 0.9); `test_warp_degenerate`.
- `test_lsd`: 0 for identical, above 0.5 for /a/ vs /i/, `lsd_frames` mean and masking.
- `test_envelope_lsd_reasonable`: order 16 fits better than order 4; silent frame returns 0.

Indirect coverage: `tests/test_live.py`, `tests/test_module_c.py`, `tests/test_evaluate.py`.

---

## 8. Weak spots and honest answers

1. **LPC is fitted to noise excitation.** The all-pole model assumes a flat source (impulse train or white noise). Whisper noise has its own spectral shape, so the estimated "filter" is filter plus source shape. Also a noise-like 320 sample frame gives noisy $r[k]$ estimates, so formant tracks are more jittery frame to frame than in voiced speech; the roughly 4 dB LSD floor in the order sweep is partly this randomness (section 6). Whisper stays intelligible because formants survive. (Only weak evidence in the repo: on the synthetic dev set Whisper ASR got WER 0.33 on the LPC-noise pseudo-whispers against 0.04 on the text-to-speech originals. Those pseudo-whispers were made with this very LPC model, so they say nothing about real whispers.)
2. **No zeros.** Nasals and some fricatives have spectral notches an all-pole model cannot represent; it spends broad extra poles instead. Acceptable for intelligibility; a pole-zero (ARMA) model would handle it at higher cost.
3. **Order choice** is a compromise (section 4), made on synthetic data and not tuned per speaker. Higher-pitched or child voices with higher formants may want a different order.
4. **Frame-wise filter switching.** A new `a` every 10 ms with no interpolation can cause small discontinuities. How synthesis handles it: `synth_step` passes the filter memory `zi` from one hop to the next (`lfilter(..., zi=zi)`), so the signal is continuous even though the coefficients jump; there is no reset to zero at hop boundaries. Live mode keeps `state["zi"]` too, resetting it only if the order changes. Residual risk: the carried state was built under the old filter and is reinterpreted by the new one, which is not exactly right, but with 50 percent frame overlap consecutive filters are close, so the error is small. No coefficient interpolation (for example in the LSF domain) is done.
5. **Offline vs live alignment** (see the report): offline, filter $i$ comes from samples $160i$ to $160i+320$ and is applied to hop $i$, the **first** half of that window. Live, the frame is `prev + block` and the filter is applied to the current block, the **second** half. So the offline filter looks 10 ms further ahead than the live one. Probably inaudible, but know it if asked why live and offline outputs differ slightly.
6. **Gain accuracy:** $G$ exactly matches the window-weighted power of the pre-emphasised frame, so against its plain RMS it is only approximate (0.9 to 1.2 in my checks) and it ignores the de-emphasis and radiation that follow; the output stage RMS match corrects all of this.
7. **Pre-emphasis is fixed at 0.97 inside `lpc_frame`.** `lpc_envelope` plots therefore show the pre-emphasised-domain envelope (tilt flattened).
8. **`warp_poles` past Nyquist** for factor > 1, and the uniform scaling assumption (section 5).
9. **LSD includes gain** and depends on Module A's silence labels (section 6).
10. **Validation notebook ran on the synthetic fixture**, so 17.8 dB and the order sweep should be rerun on real recordings before presenting them as findings.

---

## 9. Likely viva questions and model answers

### Conceptual

**1. State the source-filter model and say what LPC extracts.**
Speech is an excitation (vocal-cord pulses or noise) passed through the vocal tract filter. LPC extracts an all-pole model $G/A(z)$ of the filter from one short frame; the leftover is the residual, an approximation of the excitation.

**2. Why can we discard the whispered excitation?**
Intelligibility is carried by the filter (formants); the source carries pitch and voicing. A whisper's source is noise; we substitute a pulse train with a pitch contour (Module C) through the same filter.

**3. What does the LPC envelope represent compared with the FFT?**
The FFT shows all detail (harmonics, noise randomness). The envelope $G/|A|$ is a smooth all-pole fit that follows the peaks: an estimate of the vocal tract frequency response with the fine structure removed.

**4. Why an all-pole model?**
Vowels are dominated by resonances; all-pole gives linear equations (the normal equations) and a simple recursive synthesis filter. The cost is that notches (nasals) cannot be modelled.

**5. What is the residual and why does it look like the source?**
$e=A(z)x$, the part not predicted from the past $p$ samples. $A(z)$ cancels the resonances, leaving pulses (voiced) or noise (whisper), i.e. the excitation.

**6. What is prediction gain?**
$10\log_{10}$ of frame energy over residual energy: how much structure the filter removes. 22 dB on the loudest fixture frame, 17.8 dB average (synthetic).

**7. What is a reflection coefficient?**
$k_i$ from the Levinson step: up to sign, the partial correlation between $x[n]$ and $x[n-i]$ after the lower-order prediction is removed (PARCOR). With our sign convention ($A(z)=1+\sum a_kz^{-k}$) the sign is flipped: $k_1=-r[1]/r[0]$, so a strongly positively correlated signal gives $k_1$ near $-1$ (in the worked example $r[1]/r[0]=0.5$ gives $k_1=-0.5$). In a lossless-tube model it is the reflection at a section boundary. $|k_i|<1$ for all $i$ means stability.

### Derivation

**8. Write the normal equations and explain why the matrix is Toeplitz.**
$\sum_k a_k r[|i-k|]=-r[i]$, $i=1..p$. Entry $(i,k)$ depends only on $|i-k|$ because the autocorrelation of a windowed signal depends only on lag.

**9. Derive them.**
Minimise $E=\sum_n(x[n]+\sum_k a_kx[n-k])^2$; differentiate with respect to each $a_i$ and set to zero; the sums become autocorrelations (section 2.4).

**10. Walk through one Levinson step.**
Given the order $i-1$ solution: acc $=r[i]+\sum_j a_j r[i-j]$; $k_i=-\text{acc}/E_{i-1}$; $a_j\leftarrow a_j+k_ia_{i-j}$; $a_i=k_i$; $E_i=E_{i-1}(1-k_i^2)$. Then give the example from 2.5.

**11. Why is Levinson $O(p^2)$ and why does it matter?**
Each step $i$ is $O(i)$, total about $p^2$, against $p^3$ for general solving. It matters for 100 frames per second live; ours takes about 0.1 ms per frame.

**12. Why does the autocorrelation method give a stable synthesis filter?**
Windowing gives a positive definite Toeplitz matrix, so every $|k_i|<1$, equivalent to all poles of $1/A$ inside the unit circle. We tested that all roots have magnitude below 1.

**13. What does $G$ match and how is it computed?**
$G=\sqrt{E_p/\sum w^2}$, so unit-variance white noise through $G/A$ has, in expectation, exactly the window-weighted power $r[0]/\sum w^2$ of the pre-emphasised frame (autocorrelation matching); against the plain RMS the ratio was 0.9 to 1.2 in my checks. The output stage matches loudness to the input anyway.

**14. Relation between poles and formants?**
A pole at $re^{j\theta}$ gives $F=\theta f_s/2\pi$ and $B\approx-(f_s/\pi)\ln r$. Not every pole is a formant; low-radius poles shape the tilt.

### Why this value

**15. Why pre-emphasis 0.97?**
It compensates the roughly $-6$ dB/octave speech tilt so high formants are not ignored by the low-frequency-dominated autocorrelation. 0.97 is the spec value; 0.95 to 0.98 would behave similarly (standard range).

**16. Why Hamming?**
The autocorrelation method needs the frame to be zero outside it; tapering the edges avoids the spectral leakage a rectangular cut causes. Hamming is the standard choice in speech; Hann would also work.

**17. Why order 12 when the rule gives 18?**
The sweep (B9) showed the LSD curve flattening: about 0.6 to 1 dB per +2 step up to order 10, 0.32 dB from 10 to 12, then 0.21 dB from 12 to 14 and less after. Rule used: the smallest order whose next step gains under 0.25 dB, which is 12. Higher orders fit harmonics and noise (spec: unstable sounding synthesis) and cost more. Honest caveat: swept on synthetic data.

**18. Why 320 sample frames with hop 160?**
20 ms is long enough for a stable estimate of 13 autocorrelation lags and short enough for the tract to be roughly stationary; 50 percent overlap. (The frame and hop are fixed by the project contract; the reason is the standard one.)

**19. Why the 1e-9 white-noise correction?**
It keeps $R$ positive definite so Levinson cannot hit $k=\pm1$ or divide by zero. It is $-90$ dB, inaudible.

### What if

**20. What if the frame is silent?**
`r[0]<1e-12` returns `a=[1,0,...]`, `G=0`, so synthesis outputs zero and nothing is NaN. Empty or non-finite input is handled the same way.

**21. What if you used order 30?**
Lower LSD, but the envelope follows individual harmonics and noise bumps, poles sit closer to the unit circle, and the result sounds buzzy and unnatural; also slower.

**22. What if pre-emphasis were omitted?**
The fit is dominated by low frequencies, so high formants are modelled poorly, and $R$ is worse conditioned. The output stage would also not need de-emphasis.

**23. What if you used the covariance method?**
No window; the matrix is symmetric but not Toeplitz, solved by Cholesky at $O(p^3)$. It can be slightly more accurate on short frames, but stability is **not** guaranteed. We chose autocorrelation for guaranteed stability and the fast recursion.

**24. What if the warp factor is above 1 or very small?**
Above 1 raises formants (shorter tract), but angles may pass $\pi$ and wrap: a pole that would land above 8 kHz folds back below it (1.5 x 7170 Hz comes back at about 5240 Hz). Very small values squash formants into low frequencies and kill intelligibility. We test 0.85 to 1.0 and use 0.92; the live command line ignores factors of 1.0 or more.

**25. Does warping change bandwidths?**
No: the radius is kept, so the bandwidth in Hz is unchanged while the centre frequencies scale.

### Code-specific

**26. What does `levinson` do if `err` goes non-positive?**
Clamps it to 0 and stops; the remaining coefficients stay zero. With the noise correction this should not happen in practice.

**27. How does synthesis cope with the filter changing every hop?**
It carries `zi` across hops, so there is no restart transient; there is no coefficient interpolation. Live mode does the same and resets only if the order changes.

**28. Where is LSD computed for the table, and over which frames?**
`evaluate.lsd_variant`: LPC order 12 of input and output, frames with label != 0 (Module A's classification of the input), `lsd_frames` mean of per-frame RMS dB differences, gains included.

**29. How does `envelope_lsd` differ from `log_spectral_distortion`?**
`envelope_lsd` compares one frame's envelope with the actual FFT of that frame, level aligned by mean dB (shape only), and drives the order sweep. `log_spectral_distortion` compares two LPC envelopes with gains.

**30. Why does `warp_poles` rebuild with `np.poly`?**
We need new filter coefficients: scale root angles, group conjugate pairs plus reals, then `np.poly` expands $\prod(1-z_kz^{-1})$. The real part is taken because the imaginary part is rounding noise.

### Comparisons

**31. LPC vs cepstrum/MFCC.**
LPC gives a parametric all-pole filter we can run directly for synthesis and manipulate (pole warping). Cepstral smoothing also gives an envelope but not a recursive filter. MFCCs use a mel filterbank and are lossy, built for recognition, not resynthesis. We must resynthesise, so LPC.

**32. Autocorrelation method vs Burg (`librosa.lpc`).**
Burg estimates reflection coefficients from forward and backward errors without an explicit window; it is also stable and often better on short frames. Coefficients differ slightly, so we compared envelope shapes (correlation 0.990), not coefficients.

**33. Does the LPC envelope fit peaks or valleys better?**
Peaks: the error criterion penalises under-estimating peaks, so envelopes hug peaks and sit above valleys (standard property).

---

## 10. One-page cheat sheet

**Fixed numbers (code/spec)**: fs 16 kHz, frame 320 (20 ms), hop 160 (10 ms), order 12, `a` has 13 entries with `a[0]=1`, pre-emphasis 0.97, Hamming ($\sum w^2$ = 126.78), noise correction `r[0]*(1+1e-9)+1e-12`, silent threshold `r[0]<1e-12`, envelope 512-point FFT gives 257 points (31.25 Hz spacing), dB floor $10^{-12}$, warp variant 0.92 (spec range 0.90 to 0.95), speed under 1 ms/frame.

**Formulas**
- $A(z)=1+\sum_{k=1}^p a_kz^{-k}$; synthesis $H(z)=G/A(z)$ via `lfilter([G], a, e)`; residual via `lfilter(a, [1], x)`.
- Predictor $\hat x[n]=-\sum a_kx[n-k]$.
- Normal equations $R\mathbf a=-\mathbf r$, $R_{ik}=r[|i-k|]$, $r[k]=\sum_nx[n]x[n+k]$.
- Levinson: $k_i=-\frac{r[i]+\sum_{j<i}a_jr[i-j]}{E_{i-1}}$; $a_j\leftarrow a_j+k_ia_{i-j}$; $a_i=k_i$; $E_i=E_{i-1}(1-k_i^2)$. $O(p^2)$. Stable iff all $|k_i|<1$.
- Gain $G=\sqrt{E_p/\sum w^2}$.
- Envelope $20\log_{10}(G/|A(e^{j\omega})|)$.
- Pole $re^{j\theta}$: $F=\theta f_s/2\pi$, $B\approx-\frac{f_s}{\pi}\ln r$.
- Prediction gain $10\log_{10}(\sum x^2/\sum e^2)$ dB.
- LSD $=\sqrt{\text{mean}_k(\Delta_{dB})^2}$; clip LSD = mean over frames with label != 0.
- Warp: $\theta\to\text{factor}\cdot\theta$, $r$ unchanged, real poles unchanged; $F'=\text{factor}\cdot F$, bandwidth same.
- Order rule $f_s/1000+2=18$; project uses 12.

**Measured numbers (synthetic fixture, notebook)**: Levinson vs `solve_toeplitz` max error 8.9e-9; 0.098 ms/frame; max pole radius 0.994; prediction gain 22.0 dB (loudest), 17.8 dB (mean); envelope correlation with librosa 0.990; order sweep LSD 7.52 (p=4), 5.05 (p=12), 4.83 (p=14), 4.35 (p=24); gain: output power of unit white noise through $G/A$ = $r[0]/\sum w^2$ exactly; ratio to plain RMS 0.9 to 1.2 (my checks).

**One-line answers**
- Why stable: windowed autocorrelation gives positive definite Toeplitz $R$, so $|k_i|<1$.
- Why order 12: knee of the LSD curve, cost, avoid fitting harmonics and noise.
- Why warp below 1: the spec's reason is to correct the upward formant shift of whispers; it lowers formants (like a longer tract, deeper voice); radius kept so it stays stable.
- Why LPC not MFCC: we need a runnable, adjustable filter for resynthesis.
- Main limitations: all-pole, fitted to noisy whisper, no nasal zeros, hop-wise filter switching without interpolation.
