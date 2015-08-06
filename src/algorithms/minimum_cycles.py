from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for

@not_implemented_for('directed')
@not_implemented_for('multigraph')
def minimum_cycle_basis(G, weight=None):
    """
    Returns a minimum weight cycle basis for G
    It is defined as a cycle basis for which the total weight (length 
    for unweighted graphs) of all the cycles is minimum. 

    Parameters
    ----------
    G : NetworkX Graph
    weight: edge attribute to use

    Returns
    -------
    A list of cycle lists.  Each cycle list is a list of nodes
    which forms a cycle (loop) in G. Note that the nodes are not
    necessarily returned in a order by which they appear in the cycle

    Examples
    --------
    >>> G=nx.Graph()
    >>> G.add_cycle([0,1,2,3])
    >>> G.add_cycle([0,3,4,5])
    >>> print(nx.minimum_cycle_basis(G))
    [[0, 1, 2, 3], [0, 3, 4, 5]]

    References:    
        [1] Kavitha, Telikepalli, et al. "An O(m^2n) Algorithm for Minimum Cycle Basis of Graphs."
        http://link.springer.com/article/10.1007/s00453-007-9064-z
        [2] de Pina, J. 1995. Applications of shortest path methods. Ph.D. thesis, University of Amsterdam, Netherlands 

    See Also
    --------
    simple_cycles, cycle_basis

    """

    # We first split the graph in commected subgraphs
    return sum((_min_cycle_basis(c, weight) for c in
                nx.connected_component_subgraphs(G)), [])

def _min_cycle_basis(comp, weight):
    cb = []
    # We  extract the edges not in this spanning tree
    spanning_tree_edges = list(
        nx.minimum_spanning_edges(comp, weight=None, data=False))
    edges_excl = [
        frozenset(e) for e in comp.edges() if e not in spanning_tree_edges]
    N = len(edges_excl)

    # We maintain a set of vectors orthogonal to sofar found cycles
    set_orth = [set([edge]) for edge in edges_excl]
    for k in range(N):
        # kth cycle is "parallel" to kth vector in set_orth
        new_cycle = _min_cycle(comp, set_orth[k], weight=weight)
        cb.append(list(set().union(*new_cycle)))
        # now update set_orth so that k+1,k+2... th elements are
        # orthogonal to the newly found cycle, as per [p. 336, 1]
        base = set_orth[k]
        set_orth[k + 1:] = [orth ^ base if len(orth & new_cycle) % 2\
                            else orth for orth in set_orth[k + 1:]]

    return cb


def _min_cycle(G, orth, weight=None):
    """
    Computes the minimum weight cycle in G, orthogonal to the vector orth
    as per [p. 338, 1]
    """
    T = nx.Graph()

    nodes_idx = {node: idx for idx, node in enumerate(G.nodes())}
    idx_nodes = {idx: node for node, idx in nodes_idx.items()}

    nnodes = len(nodes_idx)

    # Add 2 copies of each edge in G to T. If edge is in orth, add cross edge;
    # otherwise in-plane edge
    if weight is not None:
        for u, v in G.edges():
            uidx, vidx = nodes_idx[u], nodes_idx[v]
            edge_w = G[u][v][weight]
            if frozenset((u, v)) in orth:
                T.add_edges_from(
                    [(uidx, nnodes + vidx), (nnodes + uidx, vidx)], weight=edge_w)

            else:
                T.add_edges_from(
                    [(uidx, vidx), (nnodes + uidx, nnodes + vidx)], weight=edge_w)

        all_shortest_pathlens = nx.shortest_path_length(T, weight='weight')

    else:
        for u, v in G.edges():
            uidx, vidx = nodes_idx[u], nodes_idx[v]
            if frozenset((u, v)) in orth:
                T.add_edges_from(
                    [(uidx, nnodes + vidx), (nnodes + uidx, vidx)])

            else:
                T.add_edges_from(
                    [(uidx, vidx), (nnodes + uidx, nnodes + vidx)])

        all_shortest_pathlens = nx.shortest_path_length(T)

    cross_paths_w_lens = {
        n: all_shortest_pathlens[n][nnodes + n] for n in range(nnodes)}

    # Now compute shortest paths in T, which translates to cyles in G
    min_path_startpoint = min(cross_paths_w_lens, key=cross_paths_w_lens.get)
    min_path = nx.shortest_path(
        T, source=min_path_startpoint, target=nnodes + min_path_startpoint, weight='weight')

    # Now we obtain the actual path, re-map nodes in T to those in G
    min_path_nodes = [
        node if node < nnodes else node - nnodes for node in min_path]
    # Now remove the edges that occur two times
    mcycle_pruned = _path_to_cycle(min_path_nodes)

    return {frozenset((idx_nodes[u], idx_nodes[v])) for u, v in mcycle_pruned}


def _path_to_cycle(path):
    """
    Removes the edges from path that occur even number of times.
    Returns a set of edges
    """
    edges = set()
    start = path[0]

    for elem in path[1:]:
        edge = frozenset((start, elem))
        if edge in edges:
            edges.remove(edge)
        else:
            edges.add(edge)
        start = elem

    return edges
