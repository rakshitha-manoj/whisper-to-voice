"""
Self-contained test for Module C (pitch.py + synth.py).
Uses only numpy/scipy and mocks -- no audio file required.
Run with:  python test_module_c.py
"""
import sys
import numpy as np
import scipy.signal

sys.path.insert(0, ".")

from src.mocks import mock_frames, mock_energy, mock_labels, mock_lpc
from src.pitch import build_f0, make_excitation, _rosenberg_pulse
from src.synth import synthesize, output_stage

FS = 16000
HOP = 160
FRAME_LEN = 320

rng = np.random.default_rng(1)
x = rng.standard_normal(FS * 2).astype(np.float32) * 0.05


def _make_mixed_labels(n):
    labels = np.zeros(n, dtype=int)
    labels[: n // 4] = 0
    labels[n // 4: n // 2] = 1
    labels[n // 2:] = 2
    return labels


frames = mock_frames(x, FRAME_LEN, HOP)
energy = mock_energy(frames)
n_frames = len(frames)
labels = _make_mixed_labels(n_frames)
A, G = mock_lpc(n_frames, order=12)

PASS = []
FAIL = []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {detail}")


print("\n=== Step 1: impulse_train ===")
f0_target = 120.0
n_samples = int(1.0 * FS)
e_imp = np.zeros(n_samples, dtype=np.float32)
period = FS / f0_target
t = period
while t < n_samples:
    idx = int(round(t))
    if idx < n_samples:
        e_imp[idx] = 1.0
    t += period
n_impulses = int(np.sum(e_imp))
expected = int(f0_target * 1.0)
check("impulse_count", abs(n_impulses - expected) <= 1,
      f"got {n_impulses}, want ~{expected}")

E = np.abs(np.fft.rfft(e_imp))
freqs = np.fft.rfftfreq(n_samples, 1 / FS)
for h in range(1, 5):
    target_hz = h * f0_target
    idx_bin = np.argmin(np.abs(freqs - target_hz))
    local = E[max(0, idx_bin - 2):idx_bin + 3]
    check(f"harmonic_{h}x120", np.argmax(local) in (1, 2, 3),
          f"peak not at {target_hz} Hz")

print("\n=== Step 2: rosenberg_pulse ===")
T0 = int(FS / 120)
pulse = _rosenberg_pulse(T0)
check("pulse_length", len(pulse) == T0, f"len={len(pulse)}")
check("pulse_unit_peak", abs(np.max(np.abs(pulse)) - 1.0) < 0.01,
      f"peak={np.max(np.abs(pulse)):.4f}")
check("pulse_nonzero_start", pulse[0] == 0.0, f"pulse[0]={pulse[0]}")

print("\n=== Step 3: make_excitation + f0 autocorr check ===")
f0_const = np.where(labels == 2, 120.0, 0.0).astype(np.float32)
e = make_excitation(labels, f0_const, hop=HOP, fs=FS)
check("excitation_length", len(e) == n_frames * HOP,
      f"{len(e)} vs {n_frames * HOP}")

voiced_start = (n_frames // 2) * HOP
voiced_seg = e[voiced_start: voiced_start + FS]
acf = np.correlate(voiced_seg, voiced_seg, mode="full")
acf = acf[len(acf) // 2:]
min_lag = int(FS / 400)
max_lag = int(FS / 70)
peak_lag = np.argmax(acf[min_lag:max_lag]) + min_lag
est_f0 = FS / peak_lag
err_pct = abs(est_f0 - 120.0) / 120.0 * 100
check("f0_within_3pct", err_pct <= 3.0, f"est={est_f0:.1f} Hz, err={err_pct:.1f}%")

print("\n=== Step 4: build_f0 contour ===")
f0_cont = build_f0(labels, energy, base_f0=120, decl=0.12, stress=0.25)
check("f0_shape", f0_cont.shape == (n_frames,), str(f0_cont.shape))
voiced_mask = labels == 2
check("f0_zero_silence", np.all(f0_cont[labels == 0] == 0), "")
check("f0_range_voiced",
      np.all((f0_cont[voiced_mask] >= 70) & (f0_cont[voiced_mask] <= 400)),
      f"min={f0_cont[voiced_mask].min():.1f} max={f0_cont[voiced_mask].max():.1f}")
voiced_f0_vals = f0_cont[voiced_mask]
mean_first_half = np.mean(voiced_f0_vals[: len(voiced_f0_vals) // 2])
mean_second_half = np.mean(voiced_f0_vals[len(voiced_f0_vals) // 2 :])
check("f0_declination",
      mean_first_half >= mean_second_half - 5,
      f"first_half_mean={mean_first_half:.1f} second_half_mean={mean_second_half:.1f}")

print("\n=== Step 5: excitation mixing label check ===")
e_mixed = make_excitation(labels, f0_cont, hop=HOP, fs=FS)
for i in range(n_frames):
    s, en = i * HOP, (i + 1) * HOP
    seg = e_mixed[s:en]
    rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))
    if labels[i] == 0:
        check(f"silence_zero_frame{i}", np.all(seg == 0), f"rms={rms:.4f}")
        break

voiced_i = np.where(labels == 2)[0][0]
seg_v = e_mixed[voiced_i * HOP:(voiced_i + 1) * HOP]
rms_v = float(np.sqrt(np.mean(seg_v.astype(np.float64) ** 2)))
check("voiced_unit_rms", abs(rms_v - 1.0) < 0.15, f"rms={rms_v:.4f}")

noise_i = np.where(labels == 1)[0][0]
seg_n = e_mixed[noise_i * HOP:(noise_i + 1) * HOP]
rms_n = float(np.sqrt(np.mean(seg_n.astype(np.float64) ** 2)))
check("noise_unit_rms", abs(rms_n - 1.0) < 0.15, f"rms={rms_n:.4f}")

print("\n=== Step 6: synthesize (lfilter with zi) ===")
y = synthesize(A, G, e_mixed, labels, energy, hop=HOP)
check("synth_length", len(y) == n_frames * HOP, f"{len(y)}")
check("synth_no_nan", not np.any(np.isnan(y)), "")
check("synth_no_inf", not np.any(np.isinf(y)), "")

max_val = np.max(np.abs(y))
check("synth_audible_range", max_val > 1e-6, f"max={max_val:.2e}")

diff = np.diff(y.astype(np.float64))
hop_boundaries = [i * HOP - 1 for i in range(1, n_frames) if i * HOP < len(diff)]
interior = np.delete(np.arange(len(diff)), hop_boundaries)
if len(interior) > 0 and len(hop_boundaries) > 0:
    rms_interior = np.sqrt(np.mean(diff[interior] ** 2))
    rms_hops = np.sqrt(np.mean(diff[hop_boundaries] ** 2))
    check("no_hop_clicks", rms_hops < rms_interior * 20,
          f"hop_rms={rms_hops:.4f} interior={rms_interior:.4f}")

print("\n=== Step 7: output_stage (radiation + RMS match) ===")
y_out = output_stage(y, energy, hop=HOP, volume=1.0)
check("output_length", len(y_out) == len(y), "")
check("output_no_nan", not np.any(np.isnan(y_out)), "")

for i in range(0, n_frames, max(1, n_frames // 4)):
    if labels[i] == 0:
        continue
    s, en2 = i * HOP, (i + 1) * HOP
    seg_in_rms = float(energy[i])
    seg_out_rms = float(np.sqrt(np.mean(y_out[s:en2].astype(np.float64) ** 2)))
    if seg_in_rms > 1e-6:
        ratio = seg_out_rms / seg_in_rms
        check(f"rms_match_frame{i}", 0.5 < ratio < 2.0,
              f"out_rms={seg_out_rms:.4f} in_rms={seg_in_rms:.4f} ratio={ratio:.2f}")

print(f"\n{'='*50}")
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
else:
    print("All checks passed!")
