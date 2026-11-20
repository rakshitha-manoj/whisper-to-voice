import numpy as np
from pathlib import Path

# Import the utilities we just implemented
from audio_utils import (
    load_audio,
    frame_signal,
    frame_energy,
    frame_zcr,
    voicing_strength,
    classify_frames,
    plot_spectrogram,
)

def main():
    """Run a quick sanity‑check of all Module A utilities.

    - Generates a 1 s 440 Hz tone if ``test.wav`` is missing.
    - Loads and resamples to 16 kHz.
    - Frames, computes RMS, ZCR, voicing strength, and frame labels.
    - Prints a few representative values and label distribution.
    - Plots and saves a spectrogram.
    """
    wav_path = Path('test.wav')

    # --------------------------------------------------------------
    # Create a placeholder wav if none exists
    # --------------------------------------------------------------
    if not wav_path.exists():
        try:
            from scipy.io import wavfile
            fs_src = 44100
            t = np.arange(fs_src) / fs_src
            tone = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
            wavfile.write(str(wav_path), fs_src, tone)
            print(f'Generated placeholder wav: {wav_path}')
        except Exception as e:
            print('Failed to generate placeholder wav:', e)
            return

    # --------------------------------------------------------------
    # Load audio (always 16 kHz mono)
    # --------------------------------------------------------------
    x, fs = load_audio(wav_path)
    print('Loaded audio – shape:', x.shape, 'sample rate:', fs)

    # --------------------------------------------------------------
    # Frame the audio
    # --------------------------------------------------------------
    frames = frame_signal(x)
    print('Framed – shape (n_frames, frame_len):', frames.shape)

    # --------------------------------------------------------------
    # Compute RMS energy and zero‑crossing rate per frame
    # --------------------------------------------------------------
    rms = frame_energy(frames)
    zcr = frame_zcr(frames)
    print('RMS – first 5 values:', rms[:5])
    print('ZCR – first 5 values:', zcr[:5])

    # --------------------------------------------------------------
    # Voicing strength (autocorrelation peak)
    # --------------------------------------------------------------
    vs = voicing_strength(frames)
    print('Voicing strength – first 5 values:', vs[:5])

    # --------------------------------------------------------------
    # Frame classification (0 = silence, 1 = noisy consonant, 2 = vowel‑like)
    # --------------------------------------------------------------
    labels = classify_frames(frames, rms, zcr)
    # classify_frames now returns (labels, zcr_thr, silence_thr)
    if isinstance(labels, tuple):
        labels, zcr_thr, silence_thr = labels
    print('Labels – first 20 values:', labels[:20])
    uniq, cnt = np.unique(labels, return_counts=True)
    print('Label distribution:', dict(zip(map(int, uniq), map(int, cnt))))

    # --------------------------------------------------------------
    # Spectrogram test (now inside main, so x and fs are defined)
    # --------------------------------------------------------------
    S_db, freqs, times = plot_spectrogram(x, fs, output_path='spectrogram.png')
    print('Spectrogram saved as spectrogram.png')
    print('Shape of S_db:', S_db.shape)
    print('Freq range: %.1f‑%.1f Hz' % (freqs.min(), freqs.max()))
    print('Time range: %.2f‑%.2f s' % (times.min(), times.max()))


if __name__ == '__main__':
    main()