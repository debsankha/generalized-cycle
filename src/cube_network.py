#!/usr/bin/env python

"""
    cube_network.py

    Creates a 3d cubic lattice graph, calculates the cycle basis
    and then checks perturbations
"""

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import networkx as nx

from algorithms.minimum_cycles import minimum_cycle_basis

import seaborn as sns

class CubicLattice:
    def __init__(self, n):
        """ Create a new cubic lattice with given linear dimension.
        Final number of nodes will be n**3.
        """
        self.G = nx.grid_graph([n, n, n])
        self.G = nx.convert_node_labels_to_integers(self.G)
        self._find_cycle_basis()
        self._find_cycle_dual()

    def _find_cycle_basis(self):
        print "finding cycle basis"
        self.cycle_basis = minimum_cycle_basis(self.G)
        print "done"

        # construct cycles as sets of edges
        def to_edgeset(cycle):
            es = zip(cycle[:-1], cycle[1:]) + [(cycle[-1], cycle[0])]
            # sort ascending
            return set([tuple(sorted(e)) for e in es])

        self.edge_cycle_basis = [to_edgeset(c) for c in self.cycle_basis]

    def _find_cycle_dual(self):
        """ Construct the cycle dual graph. This is the graph G*
        in which each cycle of G is associated with a node of G*.
        Two nodes in G* are connected if their respective cycles share
        at least one edge.
        """
        # dict in which each key is an edge and each value is
        # a list of indices in the edge_cycle_basis array 
        # the edge belongs to.

        edges_cycles = defaultdict(list)
        for i, c in enumerate(self.edge_cycle_basis):
            for e in c:
                edges_cycles[e].append(i)
        
    def perturb_edge(self, e):
        """ Perturb the edge e, return 
        """

if __name__ == '__main__':
    # test
    n = 2
    cl = CubicLattice(n)
    
    print "Number of cycles in cycle basis:"
    print cl.G.number_of_edges() - cl.G.number_of_nodes() + 1

    print "Number of faces:"
    
    pos = nx.spring_layout(cl.G)
    nx.draw_networkx_edges(cl.G, pos=pos)
    nx.draw_networkx_labels(cl.G, pos=pos)

    print cl.cycle_basis
    
    # draw edges in cycle basis
    colors = sns.color_palette('hls', len(cl.edge_cycle_basis))
    for cy, col in zip(cl.edge_cycle_basis, colors):
        nx.draw_networkx_edges(cl.G, pos=pos,
                edgelist=cy, edge_color=len(cy)*[col], width=2)
        print cy
    plt.show()
