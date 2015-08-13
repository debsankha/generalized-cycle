#!/usr/bin/env python

"""
    Contains helper functions for returning
    sparse versions of incidence matrix, laplacian
    and dual laplacian.
"""

import networkx as nx
import numpy as np
import scipy

def delete_row_lil(mat, i):
    if not isinstance(mat, scipy.sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])

def sparse_incidence_matrix(G, nodelist=None, edgelist=None,
                     oriented=False, weight=None):
    """Return incidence matrix of G.

    The incidence matrix assigns each row to a node and each column to an edge.
    For a standard incidence matrix a 1 appears wherever a row's node is
    incident on the column's edge.  For an oriented incidence matrix each
    edge is assigned an orientation (arbitrarily for undirected and aligning to
    direction for directed).  A -1 appears for the tail of an edge and 1
    for the head of the edge.  The elements are zero otherwise.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional   (default= all nodes in G)
       The rows are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    edgelist : list, optional (default= all edges in G)
       The columns are ordered according to the edges in edgelist.
       If edgelist is None, then the ordering is produced by G.edges().

    oriented: bool, optional (default=False)
       If True, matrix elements are +1 or -1 for the head or tail node
       respectively of each edge.  If False, +1 occurs at both nodes.

    weight : string or None, optional (default=None)
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.  Edge weights, if used,
       should be positive so that the orientation can provide the sign.

    Returns
    -------
    A : SciPy sparse matrix
      The incidence matrix of G.

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges in edgelist should be
    (u,v,key) 3-tuples.

    "Networks are the best discrete model for so many problems in
    applied mathematics" [1]_.

    References
    ----------
    .. [1] Gil Strang, Network applications: A = incidence matrix,
       http://academicearth.org/lectures/network-applications-incidence-matrix
    """
    import scipy.sparse
    if nodelist is None:
        nodelist = G.nodes()
    if edgelist is None:
        if G.is_multigraph():
            edgelist = G.edges(keys=True)
        else:
            edgelist = G.edges()
    A = scipy.sparse.lil_matrix((len(nodelist),len(edgelist)))
    node_index = dict( (node,i) for i,node in enumerate(nodelist) )
    for ei,e in enumerate(edgelist):
        (u,v) = e[:2]
        if u == v: continue  # self loops give zero column
        try:
            ui = node_index[u]
            vi = node_index[v]
        except KeyError:
            raise NetworkXError('node %s or %s in edgelist '
                                'but not in nodelist"%(u,v)')
        if weight is None:
            wt = 1
        else:
            if G.is_multigraph():
                ekey = e[2]
                wt = G[u][v][ekey].get(weight,1)
            else:
                wt = G[u][v].get(weight,1)
        if oriented:
            A[ui,ei] = -wt
            A[vi,ei] = wt
        else:
            A[ui,ei] = wt
            A[vi,ei] = wt

    return A

def edge_weight(G, u, v):
    """ Return the default edge weight of edge u, v in graph G.
    edge weight is 1/effective conductivity.
    """
    return G[u][v]['weight']/G[u][v]['conductivity']**4

def cycle_laplacian(G_pruned, cycles, neighbor_cycles):
    """ Take pruned graph, cycles, and cycle adjacency list to
    construct sparse cycle laplacian.
    """
    # construct cycle Laplacian matrix
    iis = []
    jjs = []
    data = []
    
    # off diagonal entries
    for n in neighbor_cycles:
        if len(n) != 2:
            print "Warning!! Cycles have wrong number of neighbors!!"
            continue

        a, b = n
        edgeset_a, edgeset_b = cycles[a].edgeset, cycles[b].edgeset
        intersection = edgeset_a.intersection(edgeset_b)

        # add up weights
        weight = np.sum([edge_weight(G_pruned, u, v) 
            for u, v in intersection])
        weight *= -cycles[a].orientation()*cycles[b].orientation()

        iis.append(a)
        iis.append(b)
        jjs.append(b)
        jjs.append(a)
        data.append(weight)
        data.append(weight)
    
    # diagonal entries
    for i, cy in enumerate(cycles):
        weight = np.sum([edge_weight(G_pruned, u, v) for u, v in cy.edges])

        iis.append(i)
        jjs.append(i)
        data.append(weight)

    # laplacian
    n = len(cycles)
    L = scipy.sparse.coo_matrix((data, (iis, jjs)), shape=(n, n))
    
    return L.tocsc()

def sparse_laplacian(G, weight='weight'):
    import scipy.sparse
    nodelist = G.nodes()
    A = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight,
                                  format='csr')
    n,m = A.shape
    diags = A.sum(axis=1)
    D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format='csr')
    return D - A

