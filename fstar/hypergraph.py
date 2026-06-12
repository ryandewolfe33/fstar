import scipy.sparse as sp
import numpy as np

from .fstar_ import find_best_matches


def vague_distance(h1, h2, node_map, node_mismatch_penalty=1.0):
    """
    Compute a score as a vague hypergraph distance

    An edge m in h1 contributes (1-best_jaccard_similarity)*size(m) to the distance

    Each node in h1 that is not in h2 (or vice versa) contributes node_mismatch_penalty

    Parameters:
    --------------
    h1: first hypergraph, node x edge incidence matrix
    h2: second hypergraph, node x edge incidence matrix
    node_map: list of tuples where we match (h1_node, h2_node)
    """
    h1 = h1.tocsc()
    h2 = h2.tocsc()

    h1 = h1.tocsr()
    h2 = h2.tocsr()

    # Pad with empty rows to make union of nodes
    n_union = h1.shape[0] + h2.shape[0] - len(node_map)
    
    n_to_pad_h1 = n_union - h1.shape[0]
    pad = type(h1)((n_to_pad_h1, h1.shape[1]), dtype="bool")
    h1 = sp.vstack([h1, pad])

    n_to_pad_h2 = n_union - h2.shape[0]
    pad = type(h2)((n_to_pad_h2, h2.shape[1]), dtype="bool")
    h2 = sp.vstack([h2, pad])

    # Order rows to put shared nodes first (and matched)
    paired_nodes = np.empty((len(node_map), 2), dtype="int32")
    for i, pair in enumerate(node_map):
        paired_nodes[i] = pair
    paired_nodes = paired_nodes.transpose()
    h1_nodes = np.arange(h1.shape[0], dtype="int64")
    h1_node_no_map = np.setdiff1d(h1_nodes, paired_nodes[0])
    h1_node_order = np.concatenate([paired_nodes[0], h1_node_no_map])
    h1 = h1[h1_node_order]

    h2_nodes = np.arange(h2.shape[0], dtype="int64")
    h2_node_no_map = np.setdiff1d(h2_nodes, paired_nodes[1])
    h2_node_order = np.concatenate([paired_nodes[1], h2_node_no_map])
    h2 = h2[h2_node_order]

    h1_fs, h2_fs = find_best_matches(h1, h2)

    h1_edge_sizes =np.asarray(h1.sum(axis=1)).reshape(-1)
    h2_edge_sizes =np.asarray(h2.sum(axis=1)).reshape(-1)

    score = np.sum((1-h1_fs) * h1_edge_sizes) + np.sum((1-h2_fs) * h2_edge_sizes) + (n_union - len(node_map)) * node_mismatch_penalty
    return score




