import numpy as np
import matplotlib.pyplot as plt


def frame_zcr(frames):
    raise NotImplementedError


def voicing_strength(frames, fs=16000):
    raise NotImplementedError


def classify_frames(frames, energy):
    raise NotImplementedError


def plot_spectrogram(x, fs=16000):
    raise NotImplementedError
