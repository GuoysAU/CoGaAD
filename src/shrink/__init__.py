"""Step 1 of CoGa: time series -> Base.

`encode` sets the target SNR from the entropy of the series, derives the quantization step from it and runs the
SHRINK compressor (shrinking cones, cone merging). The Base is the time-ordered list of the resulting segments.
"""
import gzip
import numpy as np
from dataclasses import dataclass
from .Point import Point
from .Shrink import Shrink
from .SNRQuantization import entropy_guided_snr, estimate_window, quantization_snr, quantization_step


def encode(values, gamma_min=25, gamma_max=35, residual_split=True, local_origin=False):
    """Compress a series into its Base.

    values:          one-dimensional series.
    gamma_min/max:   range of the entropy-guided target SNR in dB; equal bounds give a fixed target.
    residual_split:  end a cone when its residuals fail the Ljung-Box white-noise test.
    local_origin:    quantize cone origins with the local error bound, as the original SHRINK (S2D).
    """
    values = np.asarray(values, dtype=float)
    target_snr = entropy_guided_snr(values, gamma_min, gamma_max)
    epsilon = quantization_step(values, target_snr)
    window = estimate_window(values)
    points = [Point(i, float(v)) for i, v in enumerate(values)]
    shrink = Shrink(points=points, epsilon=epsilon, window=window, residual_split=residual_split, local_origin=local_origin)
    return Base.from_shrink(shrink, n_points=len(values), gamma=target_snr,
                            snr=quantization_snr(values, epsilon), epsilon=epsilon, window=window)


@dataclass
class Base:
    """Compressed representation of one series."""
    segments: list      # [[t_start, t_end], b, a] per segment, in time order
    n_points: int       # length of the series
    gamma: int          # target SNR in dB
    snr: float          # SNR reached by the chosen quantization step
    epsilon: float      # quantization step, also the error bound of the cones
    window: int         # window length estimated from the autocorrelation of the series
    shrink: Shrink      # compressor state, kept for the byte representation

    @classmethod
    def from_shrink(cls, shrink, **fields):
        """The compressor keeps its segments grouped by pattern; the Base lists them in time order with their end."""
        ordered = sorted(shrink.segments, key=lambda s: (s.get_init_timestamp, s.get_b, s.get_a))
        starts = [s.get_init_timestamp for s in ordered] + [shrink.lastTimeStamp + 1]
        segments = [[[starts[i], starts[i + 1] - 1], s.b, s.a] for i, s in enumerate(ordered)]
        return cls(segments=segments, shrink=shrink, **fields)

    def __len__(self):
        return len(self.segments)

    def to_bytes(self):
        """Byte representation written by SHRINK: epsilon, then for every pattern its intercept, its slope and
        the start times of its segments, then the last time stamp."""
        return self.shrink.toByteArray(variableByte=False, zstd=False)

    def footprint(self):
        """Semantic footprint in bytes: the byte representation of the Base after gzip compression."""
        return len(gzip.compress(self.to_bytes(), compresslevel=9))
