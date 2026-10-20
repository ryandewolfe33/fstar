# Fstar
A pragmatic method for comparing clusterings with overlaps and outliers.

## Getting Started
The package is currently pip installable but is not on pypi. For now, please clone this
repository and install locally by following the instructions below.

```{shell}
git clone https://github.com/ryandewolfe33/fstar.git
cd fstar
pip install .
```

## Example
For examples of using fstar, please see the notebooks in the `experiments` folder.

## API
The main function is also named `fstar`, and computes a similarity score between two clusterings.
```python
def fstar.fstar(c1, c2, outliers=True,drop_outliers=False, alpha=0.5) -> float:
    """ Compute F*_wo between c1 and c2.

    Parameters
    ----------
    c1:scipy.sparse.array or np.ndarray - the first clustering
    c2:scipy.sparse_array or np.ndarray - the second clustering
    outliers:bool (defualt=True) - Flag to add outlier comparison term. If True compute F*_wo,
    if False compute F*_w
    drop_outliers:[False, 'c1', 'c2', 'either', 'both'] (default=False) - Flag to drop objects
    that consider outliers before computing score. Can be helpful for determining the quality
    of the clusters when extra outliers are not a concern.
    alpha:float (default=0.5) - A value between 0 and 1 to control the importance of matching
    in each direction. The default 0.5 is a symmetric measure, alpha=1 puts all the weight on
    the ability of each cluster in c1 to find a match and alpha=0 puts all the weight on the
    ability of each cluster in c2 to find a match.
    
    Returns
    -------
    float: the similarity scoree between c1 and c2.
    """
```
The clusterings may be input as either a sparse cluster x object membership matrix (entry (i,j)
is true if object j is in cluster i), or a 1d numpy where the value at index i is the cluster id
of object i (and -1 denotes no cluster, i.e, object i is an outlier).

If you want the cluster level fstar scores, or the ids of the best matching cluster, these
intermediate results are exposed as part of the `find_matches` function.
```python
def find_best_matches(c1, c2, return_match_ids=False):
    """ Find the best match for each cluster. Returns either
    two arrays of the F* scores for each cluster, or if
    `return_match_ids=True` then return four arrays, the first
    two the same as above and the second two are the cluster ids
    of the best match in the other clustering.

    Parameters
    ----------
    c1: scipy.sparse_array - the first clustering
    c2: scipy.sparse_array - the second clustering
    return_match_ids:bool (default=False) - Flag to include match ids in return
    """
```

There are also two helper function for working with graph aware comparisons. To compute the closed
edge clustering of an edge clustering `e1`, run
`node_clustering_to_edge_clustering(adjacency, edge_clustering_to_node_clustering(adjacency, e1))`.
```python
def fstar.node_clustering_to_edge_clustering(adjacency, node_clustering)
    """ Project a node clustering onto the edges. An edge is in a cluster if both endpoints are in
    that cluster. Edge ids are computed by the data_index of a csr representation of the upper
    triangular adjacency matrix.

    Parameters
    ----------
    adjacency:scipy.sparse.array - the adjacency matrix of the graph.
    node_clustering:scipy.sparse_array or np.ndarray - the node clustering. 
    
    Returns
    -------
    scipy.sparse.csr_array - the induced edge clustering. Edge ids are the data_index of a csr
    representation of the upper triangular adjacency matrix.
    """
```
```python
def fstar.edge_clustering_to_node_clustering(adjacency, edge_clustering):
    """
    Project an edge clustering onto the nodes. An node is in a cluster if it has at least one
    incident edge in that cluster. Edge ids are assumed to be the data_index of a csr
    representation of the upper triangular adjacency matrix.

    Parameters
    ----------
    adjacency:scipy.sparse.array - the adjacency matrix of the graph.
    node_clustering:scipy.sparse_array or np.ndarray - the edge clustering. Edge ids are assumed
    to be the data_index of a csr representation of the upper triangular adjacency matrix.
    
    Returns
    -------
    scipy.sparse.csr_array - the induced node clustering.
    """
```


## Reference
Ryan DeWolfe, Paweł Prałat, and François Théberge. A Pragmatic Method for Comparing Clusterings with
Overlaps and Outliers. arxiv preprint (2026).
[https://doi.org/10.48550/arXiv.2602.14855]([https://doi.org/10.48550/arXiv.2602.14855)

```bibtex
@article{dewolfe2026fstar,
    author    = {Ryan DeWolfe and Paweł Prałat and and François Théberge},
    title     = {A Pragmatic Method for Comparing Clusterings with Overlaps and Outliers},
    journal   = {arxiv preprint},
    year      = {2026},
    doi       = {10.48550/arXiv.2602.14855}
}
```
