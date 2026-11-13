import numpy as np


def mock_frames(x, frame_len=320, hop=160):
    n = len(x) // hop
    x = np.pad(x, (0, frame_len))
    return np.stack([x[i * hop:i * hop + frame_len] for i in range(n)])


def mock_energy(frames):
    return np.sqrt(np.mean(frames ** 2, axis=1))


def mock_labels(n_frames):
    return np.full(n_frames, 2)


def mock_lpc(n_frames, order=12):
    A = np.zeros((n_frames, order + 1))
    A[:, 0] = 1
    G = np.ones(n_frames)
    return A, G
