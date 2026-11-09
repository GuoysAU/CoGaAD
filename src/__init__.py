"""CoGa: anomaly detection on a compressed time series through a transition graph.

Three steps, one module each:
    shrink/   series -> Base
    graph.py  Base   -> transition graph
    score.py  graph  -> anomaly scores

`detect` runs the three steps on one series.
"""
import numpy as np
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from .shrink import Base, encode
from .graph import KB2Graph, build_graph
from .score import TERMS, query_length, score_points


@dataclass
class Detection:
    scores: np.ndarray       # anomaly score per time point, scaled to [0, 1]
    base: Base
    graph: KB2Graph
    query_length: int        # window length in points used for scoring


def detect(values, gamma_min=25, gamma_max=35, residual_split=True, alpha=1.0, terms=TERMS):
    """Run CoGa on one series.

    values:          one-dimensional series.
    gamma_min/max:   range of the entropy-guided target SNR in dB; equal bounds give a fixed target.
    residual_split:  end a cone when its residuals fail the white-noise test.
    alpha:           Laplace smoothing constant of the transition probabilities.
    terms:           terms of the path score, any non-empty subset of ("node", "edge").
    """
    base = encode(values, gamma_min=gamma_min, gamma_max=gamma_max, residual_split=residual_split)
    graph = build_graph(base, alpha=alpha)
    q = query_length(graph)
    raw = score_points(graph, q, terms=terms)
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(raw.reshape(-1, 1)).ravel()
    return Detection(scores=scores, base=base, graph=graph, query_length=q)
