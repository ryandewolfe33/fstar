import numpy as np
import pytest
import scipy.sparse as sp

import fstar


@pytest.fixture
def adjacency():
    return sp.csr_array([
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ])


def test_clustering_array_to_sparse():
    array = np.array([0, 0, 1, 1, -1, 1])
    sparse = fstar.clustering_array_to_sparse(array)
    assert sparse.shape == (2, 6)
    assert np.all(sparse.indptr == np.array([0,2,5]))
    assert np.all(sparse.indices == np.array([0,1,2,3,5]))
    assert np.all(sparse.data)


def test_node_clustering_to_edge_clustering(adjacency):
    node_clustering = sp.csr_array([
        [True, True, False, True, False],
        [False, False, True, True, False]
    ])
    edge_clustering = fstar.node_clustering_to_edge_clustering(adjacency, node_clustering)
    assert edge_clustering.shape == (2, 4)
    assert np.all(edge_clustering.indptr == np.array([0,2,3]))
    assert np.all(edge_clustering.indices == np.array([0,1,3]))
    assert np.all(edge_clustering.data)


def test_edge_clustering_to_node_clustering(adjacency):
    edge_clustering = sp.csr_array([
        [True, True, False, False],
        [False, False, True, False]
    ])
    node_clustering = fstar.edge_clustering_to_node_clustering(adjacency, edge_clustering)
    assert node_clustering.shape == (2, 5)
    assert np.all(node_clustering.indptr == np.array([0,3,5]))
    assert np.all(node_clustering.indices == np.array([0,1,3,1,2]))
    assert np.all(node_clustering.data)
