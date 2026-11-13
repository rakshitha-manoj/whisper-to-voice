import numpy as np
import soundfile


def load_audio(path):
    raise NotImplementedError


def frame_signal(x, frame_len=320, hop=160):
    raise NotImplementedError


def frame_energy(frames):
    raise NotImplementedError
