"""Step 3 of CoGa: transition graph -> anomaly scores.

A window of the series maps to a path of consecutive states. Its anomaly score is the negated mean of
log2(d_hat(u) * w_hat(u, v)) over the transitions (u, v) on the path, where d_hat is the normalized degree of
u (node term) and w_hat the smoothed transition probability (edge term). The path of a window changes only
when one of its ends crosses a segment boundary, so prefix sums over the transitions score each path in O(1).
"""
import math
from statistics import NormalDist

import numpy as np

EPS = 1e-12
TERMS = ("node", "edge")
MAX_LAG = 400                     # a fixed number of lags keeps the period estimation O(n)
Z_95 = NormalDist().inv_cdf(0.95)  # one-sided 95% quantile of the standard normal, Eq. (13)


def log_terms(graph):
    """Per-transition terms of the score: (log2 d_hat(z_m), log2 w_hat(z_m, z_m+1)) for m = 0..M-2."""
    if graph.graph_ is None:
        raise RuntimeError("Build the graph before scoring.")

    transition_prob = graph.transition_prob_

    degree_dict = dict(graph.graph_.degree())
    total_degree = max(sum(degree_dict.values()), 1)

    degree_log_weight = {
        node: math.log2(max(degree / total_degree, EPS))
        for node, degree in degree_dict.items()
    }
    edge_log_weight = {
        edge: math.log2(max(prob, EPS))
        for edge, prob in transition_prob.items()
    }

    z = graph.state_sequence_
    node = [degree_log_weight[z[i]] for i in range(len(z) - 1)]
    edge = [edge_log_weight[(z[i], z[i + 1])] for i in range(len(z) - 1)]
    return node, edge


def _prefix_sums(graph, terms):
    """Running sums of the selected terms over the transitions, accumulated left to right."""
    unknown = set(terms) - set(TERMS)
    if unknown or not terms:
        raise ValueError(f"terms must be a non-empty subset of {TERMS}, got {terms}")
    use_node, use_edge = "node" in terms, "edge" in terms

    node, edge = log_terms(graph)
    prefix = np.zeros(len(node) + 1, dtype=float)
    for i in range(len(node)):
        if use_node and use_edge:
            prefix[i + 1] = prefix[i] + node[i] + edge[i]
        elif use_node:
            prefix[i + 1] = prefix[i] + node[i]
        else:
            prefix[i + 1] = prefix[i] + edge[i]
    return prefix


def _path_normality(prefix, start_idx, end_idx):
    """Mean of the terms over the transitions between segments start_idx and end_idx; (value, path length)."""
    path_len = end_idx - start_idx
    if path_len <= 0:
        return 0.0, 0
    total = prefix[end_idx] - prefix[start_idx]
    return total / path_len, path_len


def _segment_of_point(graph):
    """Index of the segment that contains each time point."""
    min_t = int(min(seg.t_start for seg in graph.segments))
    max_t = int(max(seg.t_end for seg in graph.segments))
    total_points = max_t - min_t + 1

    edge_in_time = np.empty(total_points, dtype=np.intp)
    for seg_idx, seg in enumerate(graph.segments):
        s = int(seg.t_start) - min_t
        e = int(seg.t_end) - min_t + 1
        edge_in_time[s:e] = seg_idx

    return edge_in_time


def score_points(graph, query_length, terms=TERMS):
    """One anomaly score per time point.

    A window of `query_length` points slides over the series. A window that starts in segment i and ends in
    segment j is scored by the path of transitions from i to j. The score is recomputed only where i or j
    changes, and the result is shifted by half a window so that it is centred on the window.

    terms: which terms of the score to use, ("node", "edge") by default.
    """
    if graph.graph_ is None:
        raise RuntimeError("Build the graph before scoring.")
    if query_length < 1:
        raise ValueError("query_length must be at least 1 point.")

    edge_in_time = _segment_of_point(graph)
    total_points = len(edge_in_time)
    prefix = _prefix_sums(graph, terms)

    n_windows = total_points - query_length
    starts = edge_in_time[:n_windows]
    ends = edge_in_time[query_length: query_length + n_windows]

    change_mask = np.empty(n_windows, dtype=bool)
    change_mask[0] = True
    change_mask[1:] = (starts[1:] != starts[:-1]) | (ends[1:] != ends[:-1])
    change_pts = np.where(change_mask)[0]

    raw = np.empty(n_windows)
    prev_score = 0.0

    for k, t in enumerate(change_pts):
        seg_start = int(starts[t])
        seg_end = int(ends[t])

        normality_score, path_len = _path_normality(prefix, seg_start, seg_end)
        if path_len == 0:
            score = prev_score
        else:
            score = -normality_score
            prev_score = score
        next_t = int(change_pts[k + 1]) if k + 1 < len(change_pts) else n_windows
        raw[t:next_t] = score

    pad = query_length // 2
    return np.array([raw[0]] * pad + raw.tolist() + [raw[-1]] * pad, dtype=float)


def query_length(graph):
    """Detection window length q in points (Sec. IV-C): the first significant period of the node sequence of the
    samples, Eqs. (12)-(15)."""
    lengths = np.array([segment.length for segment in graph.segments])
    z = np.repeat(np.unique(graph.state_sequence_, return_inverse=True)[1], lengths)   # node of every sample
    n = len(z)
    lags = np.arange(1, min(n // 2, MAX_LAG) + 1)
    rho = np.array([np.mean(z[:-k] == z[k:]) for k in lags])                          # Eq. (12)
    rho0 = np.sum((np.bincount(z) / n) ** 2)
    significant = rho >= rho0 + Z_95 * np.sqrt(rho0 * (1 - rho0) / (n - lags))         # Eq. (13)
    peaks = [i for i in range(1, len(lags) - 1) if rho[i] > rho[i - 1] and rho[i] > rho[i + 1]]
    first_significant = [i for i in peaks if significant[i]]
    if first_significant:                                                                # Eq. (14)
        return int(lags[first_significant[0]])                                           # Eq. (15): q = k*
    if peaks:
        return int(lags[max(peaks, key=lambda i: rho[i])])
    return int(lags[np.argmax(rho)])
