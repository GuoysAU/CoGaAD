import math
import numpy as np
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf


def entropy_guided_snr(values, gamma_min=25, gamma_max=35):
    """Entropy-guided target SNR in dB, Eq. (4): gamma_min + (gamma_max - gamma_min) * H(X) / log2(k), where H(X) is
    the Shannon entropy of the value histogram with k bins by Scott's rule."""
    counts, _ = np.histogram(values, bins="scott")
    if len(counts) == 1:   # a constant series has no spread
        return int(gamma_min)
    p = counts[counts > 0] / counts.sum()
    entropy = -np.sum(p * np.log2(p)) / np.log2(len(counts))
    return int(gamma_min + (gamma_max - gamma_min) * entropy)


def quantization_snr(values, eps):
    """SNR in dB of the series rounded to multiples of eps."""
    noise = np.sum((values - np.round(values / eps) * eps) ** 2)
    return 10 * math.log10(np.sum(values ** 2) / noise) if noise > 0 else math.inf


def quantization_step(values, target_snr):
    """Largest power of two eps that keeps the SNR at or above the target: start from Eq. (3), then double."""
    eps = 2.0 ** math.floor(math.log2(2 * math.sqrt(10 ** (-target_snr / 10) * np.mean(values ** 2))))
    while quantization_snr(values, 2 * eps) >= target_snr:
        eps *= 2
    return eps


def estimate_window(values):
    """Window length by the rule of find_length in TSB-UAD (github.com/TheDatumOrg/TSB-UAD,
    TSB_UAD/utils/slidingWindows.py): the lag of the highest local maximum of the autocorrelation of the first
    20,000 samples, between lags 6 and 303, otherwise 100."""
    rho = acf(values[:20000], nlags=400, fft=True)
    peaks = argrelextrema(rho[3:], np.greater)[0] + 3      # lags of the local maxima beyond lag 3
    if len(peaks) == 0:
        return 100
    best = int(peaks[np.argmax(rho[peaks])])
    return best if 6 <= best <= 303 else 100
