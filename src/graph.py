"""Step 2 of CoGa: Base -> transition graph.

A node is one pattern of the Base, i.e. one distinct (b, a) pair. A directed edge links the patterns of two
consecutive segments, weighted by how often that transition occurs. Self-loops are kept.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Hashable, List, Optional, Tuple

import networkx as nx


@dataclass(frozen=True)
class Segment:
    """One segment of the Base."""
    t_start: float
    t_end: float
    b: Hashable     # intercept (quantized cone origin)
    a: float        # slope

    @property
    def length(self) -> int:
        """Number of points covered, both ends included."""
        return int(self.t_end - self.t_start + 1)


class KB2Graph:
    """Transition graph over the patterns of a Base."""

    def __init__(self, sort_by_time: bool = True):
        self.sort_by_time = sort_by_time

        self.segments: List[Segment] = []
        self.cluster_values_: List[Tuple[Hashable, float]] = []     # distinct (b, a) pairs
        self.cluster_map_: Dict[Tuple[Hashable, float], str] = {}   # (b, a) -> node label
        self.state_sequence_: List[str] = []                        # node label of every segment, in time order
        self.node_support_: Dict[str, int] = {}                     # segments per node
        self.edge_weight_: Dict[Tuple[str, str], int] = {}          # transition counts
        self.transition_prob_: Dict[Tuple[str, str], float] = {}   # smoothed transition probabilities w_hat
        self.graph_: Optional[nx.DiGraph] = None

        self.cluster_segments_: Dict[str, List[int]] = {}           # node -> segment indices
        self.cluster_members_: Dict[str, List[Segment]] = {}

    def fit(self, segments: List[List[Any]], alpha: float = 1.0) -> "KB2Graph":
        """Build the graph from segments given as [[t_start, t_end], b, a].

        alpha: Laplace smoothing constant of the transition probabilities.
        """
        self.segments = self._parse_segments(segments)

        if self.sort_by_time:
            self.segments.sort(key=lambda s: (s.t_start, s.t_end))

        # 1) Nodes: one per distinct (b, a) pair
        self._build_clusters_from_b()

        # 2) State sequence
        self.state_sequence_ = [self.cluster_map_[(seg.b, seg.a)] for seg in self.segments]

        # 3) Node support
        self.node_support_ = dict(Counter(self.state_sequence_))

        # 4) Edge weights from consecutive segments (self-loops kept)
        edge_counter = Counter()
        for i in range(len(self.state_sequence_) - 1):
            u = self.state_sequence_[i]
            v = self.state_sequence_[i + 1]
            edge_counter[(u, v)] += 1
        self.edge_weight_ = dict(edge_counter)

        # 5) Graph
        self.graph_ = nx.DiGraph()
        self.graph_.add_nodes_from(self.cluster_map_.values())

        label_to_key = {v: k for k, v in self.cluster_map_.items()}
        for node, supp in self.node_support_.items():
            self.graph_.nodes[node]["support"] = supp
            self.graph_.nodes[node]["size"] = supp
            b_val, a_val = label_to_key[node]
            self.graph_.nodes[node]["b_value"] = b_val
            self.graph_.nodes[node]["a_value"] = a_val

        for (u, v), w in self.edge_weight_.items():
            self.graph_.add_edge(u, v, weight=w)

        self._build_cluster_membership()

        # 6) Transition probabilities, used by the score
        self.transition_prob_ = self.compute_transition_probabilities(alpha=alpha)

        return self

    def compute_transition_probabilities(self, alpha: float = 1.0) -> Dict[Tuple[str, str], float]:
        """Laplace-smoothed P(v | u) = (W(u, v) + alpha) / (out(u) + alpha * K) for the observed edges.

        K is the number of nodes. Scoring only looks up consecutive states, which are exactly the edges of
        the graph, so the probabilities of unobserved pairs are never needed.
        """
        if self.graph_ is None:
            raise RuntimeError("Call fit() before computing transition probabilities.")

        num_nodes = len(list(self.graph_.nodes()))

        outgoing_sum = defaultdict(int)
        for (u, v), w in self.edge_weight_.items():
            outgoing_sum[u] += w

        return {(u, v): (w + alpha) / (outgoing_sum[u] + alpha * num_nodes)
                for (u, v), w in self.edge_weight_.items()}

    def summary(self) -> Dict[str, Any]:
        if self.graph_ is None:
            raise RuntimeError("Call fit() first.")
        return {
            "num_segments": len(self.segments),
            "num_nodes": self.graph_.number_of_nodes(),
            "num_edges": self.graph_.number_of_edges(),
        }

    def _parse_segments(self, segments: List[List[Any]]) -> List[Segment]:
        parsed: List[Segment] = []

        for idx, seg in enumerate(segments):
            if not isinstance(seg, (list, tuple)) or len(seg) != 3:
                raise ValueError(
                    f"Segment at index {idx} must have format [[t_start, t_end], b, a], got: {seg}"
                )

            time_span, b, a = seg

            if not isinstance(time_span, (list, tuple)) or len(time_span) != 2:
                raise ValueError(
                    f"Segment at index {idx} has invalid time span: {time_span}"
                )

            t_start, t_end = time_span
            parsed.append(Segment(float(t_start), float(t_end), b, float(a)))

        return parsed

    def _build_clusters_from_b(self) -> None:
        """One node per distinct (b, a) pair, labelled in order of first appearance."""
        unique_pairs = []
        seen: set = set()

        for seg in self.segments:
            key = (seg.b, seg.a)
            if key not in seen:
                seen.add(key)
                unique_pairs.append(key)

        self.cluster_values_ = unique_pairs
        self.cluster_map_ = {key: f"C{i}" for i, key in enumerate(unique_pairs)}

    def _build_cluster_membership(self) -> None:
        cluster_segments = defaultdict(list)
        cluster_members = defaultdict(list)

        for idx, seg in enumerate(self.segments):
            c = self.cluster_map_[(seg.b, seg.a)]
            cluster_segments[c].append(idx)
            cluster_members[c].append(seg)

        self.cluster_segments_ = dict(cluster_segments)
        self.cluster_members_ = dict(cluster_members)


def build_graph(base, alpha: float = 1.0) -> KB2Graph:
    """Transition graph of a Base."""
    return KB2Graph(sort_by_time=True).fit(base.segments, alpha=alpha)
