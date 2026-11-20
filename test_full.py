# /Users/hafsanavas/Documents/GitHub/whisper-to-voice/test_full.py
import numpy as np
from pathlib import Path
import time

# Import everything from audio_utils
from audio_utils import (
    load_audio,
    frame_signal,
    frame_energy,
    frame_zcr,
    voicing_strength,
    classify_frames,
    lpc_frame,
    lpc_all,
    warp_poles,
    build_f0,
    make_excitation,
    synthesize,
    plot_spectrogram,
    start_duplex_stream,
)

def main():
    wav_path = Path('test.wav')

    # create placeholder wav if missing (1 s 440 Hz sine)
    if not wav_path.exists():
        from scipy.io import wavfile
        fs_src = 44100
        t = np.arange(fs_src) / fs_src
        tone = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
        wavfile.write(str(wav_path), fs_src, tone)
        print('Generated placeholder wav')

    # 1. load audio
    x, fs = load_audio(wav_path)
    print('1 – Loaded shape:', x.shape, 'fs =', fs)

    # 2. frame
    frames = frame_signal(x)
    print('2 – Frames shape:', frames.shape)

    # 3. RMS & ZCR
    rms = frame_energy(frames)
    zcr = frame_zcr(frames)
    print('3 – RMS first 5:', rms[:5])
    print('3 – ZCR first 5:', zcr[:5])

    # 4. voicing strength
    vs = voicing_strength(frames)
    print('4 – Voicing strength first 5:', vs[:5])

    # 5. classification
# 5. classification
    labels = classify_frames(frames, rms)
    print('5 – Labels first 20:', labels[:20])
    uniq, cnt = np.unique(labels, return_counts=True)
    print('   distribution:', dict(zip(map(int, uniq), map(int, cnt))))

    # Plot waveform with labels for threshold tuning
    from audio_utils import plot_labels
    plot_labels(x, fs, frames, labels, output_path='labels_plot.png')
    print('   Labels plot saved as labels_plot.png')


    # 6. LPC and warping example
    A, G = lpc_all(frames, order=12)
    A_warped = np.apply_along_axis(warp_poles, 1, A, factor=0.1)
    print('6 – LPC coeffs shape:', A.shape, 'gain shape:', G.shape)

    # 7. pitch contour, excitation, synthesis
    f0 = build_f0(labels, rms)
    e = make_excitation(labels, f0)
    y = synthesize(A, G, e, labels, rms)
    print('7 – Synthesized length:', y.shape)

    # 8. spectrogram
    S_db, f, t = plot_spectrogram(x, fs, output_path='spectrogram.png')
    print('8 – Spectrogram saved, shape:', S_db.shape)

    # 9. timing (voicing_strength as example)
    start = time.time()
    for _ in range(100):
        _ = voicing_strength(frames)
    avg = (time.time() - start) / 100.0
    print('9 – Avg voicing_strength time per call: {:.3f} ms'.format(avg*1000))
    # Plot timing vs budget
    from audio_utils import plot_timing
    plot_timing(avg*1000)
    print('   Timing plot saved as timing_plot.png')


    # 10. real‑time stream (uncomment to use)
    # stream = start_duplex_stream()
    # while True: pass

if __name__ == '__main__':
    main()
