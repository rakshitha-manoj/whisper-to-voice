import numpy as np
import scipy.signal


def preemphasis(x, alpha=0.97):
    raise NotImplementedError


def autocorr(frame, p):
    raise NotImplementedError


def levinson(r, p):
    raise NotImplementedError


def lpc_frame(frame, order=12):
    raise NotImplementedError


def lpc_all(frames, order=12):
    raise NotImplementedError


def warp_poles(a, factor):
    raise NotImplementedError
