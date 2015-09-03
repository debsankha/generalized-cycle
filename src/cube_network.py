#!/usr/bin/env python

"""
    cube_network.py

    Creates a 3d cubic lattice graph, calculates the cycle basis
    and then checks perturbations
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import networkx as nx

from algorithms.minimum_cycles import minimum_cycle_basis
from cycle_flows.cycle_flows import simulate_single_edge_flow
from cycle_flows.cycle_flow_sim import distance_means

from autosave import asv

import seaborn as sns

def fix_cycle_basis(G, cycle_basis):
    """ take a cycle basis from minimum_cycle_basis and
    order the nodes such that they are in the same order as in the
    real cycle.
    """

    fixed_basis = []
    for cycle in cycle_basis:
        cycle_graph = G.subgraph(cycle).copy()
        n0, n1 = cycle_graph.edges()[0]
        cycle_graph.remove_edge(n0, n1)

        fixed_basis.append(nx.shortest_path(cycle_graph, n0, n1))

    return fixed_basis

class CubicLattice:
    def __init__(self, n):
        """ Create a new cubic lattice with given linear dimension.
        Final number of nodes will be n**3.
        """
        self.G = nx.grid_graph([n, n, n])
        self.G = nx.convert_node_labels_to_integers(self.G,
                label_attribute='pos')
        self._find_cycle_basis()
        self._find_cycle_dual()

    def _find_cycle_basis(self):
        print "finding cycle basis"
        self.cycle_basis = minimum_cycle_basis(self.G)
        print "done"
        
        print "fixing cycle basis"
        self.cycle_basis = fix_cycle_basis(self.G, self.cycle_basis)

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
        # the edge belongs to, i.e. a list of adjacent cycles

        edges_cycles = defaultdict(list)
        for i, c in enumerate(self.edge_cycle_basis):
            for e in c:
                edges_cycles[e].append(i)
        
        # sort
        for e in self.G.edges_iter():
            edges_cycles[e] = sorted(edges_cycles[e])

        # construct dual
        cycle_dual = nx.Graph()
        for v in edges_cycles.values():
            # connect up all adjacent cycles
            for pair in combinations(v, 2):
                cycle_dual.add_edge(*pair)

        # add virtual cycles for all those edges that are only contained
        # in one cycle (so we can later calculate distances using
        # the line graph)
        j = len(self.edge_cycle_basis)
        for k, v in edges_cycles.iteritems():
            if len(v) == 1:
                v.append(j)
                cycle_dual.add_edge(v[0], j)
                j += 1

        self.cycle_dual = cycle_dual
        self.edge_in_cycles = edges_cycles
        
    def perturb_edge(self, e):
        """ Perturb the edge e, return vector of perturbation
        currents.
        """
        return simulate_single_edge_flow(self.G, e, weighted=False)[0]

    def edge_distances(self, e):
        """ Return vector of shortest path edge distances between e
        and all other edges
        """
        L = nx.line_graph(self.G)
        path_lens = nx.shortest_path_length(L, source=e)

        return np.array([path_lens[f] for f in self.G.edges_iter()])
    
    def cycle_distances(self, e):
        """ Return vector of shortest path cycle distances between
        e and all other edges
        """
        L = nx.line_graph(self.cycle_dual)

        # find all nodes of the line graph that e belongs to

        line_nodes = [list(combinations(self.edge_in_cycles[f], 2))
                for f in self.G.edges_iter()]

        e_ind = self.G.edges().index(e)
        dists_from_all_sources = [nx.shortest_path_length(L, source=src) 
                for src in line_nodes[e_ind]]

        cycle_dists = np.array([[min([d[h] for h in line_nodes[i]])
            for i in xrange(self.G.number_of_edges())]
                for d in dists_from_all_sources]).min(axis=0)

        return cycle_dists

    def central_edge(self):
        """ Return the edge index of the edge
        closest to the center of gravity of the network.
        """
        coords = np.array([0.5*(np.array(self.G.node[u]['pos']) 
            + np.array(self.G.node[v]['pos']))
                for u, v in self.G.edges_iter()])

        cog = coords.mean(axis=0)        
        dists = np.linalg.norm(coords - cog, axis=1)
        
        return np.argmax(dists)

class PeriodicLattice(CubicLattice):
    def __init__(self, n):
        """ Create a new periodic lattice with given linear dimension.
        Final number of nodes will be n**2.
        """
        self.G = nx.grid_graph([n, n], periodic=True)
        self.G = nx.convert_node_labels_to_integers(self.G,
                label_attribute='pos')
        #self._find_cycle_basis()
        #self._find_cycle_dual()

def cube_network_analysis():
    # test
    n = 5
    cl = CubicLattice(n)
    
    print "Number of cycles in cycle basis:"
    print cl.G.number_of_edges() - cl.G.number_of_nodes() + 1

    #pos = nx.spring_layout(cl.G)
    #nx.draw_networkx_edges(cl.G, pos=pos)
    #nx.draw_networkx_labels(cl.G, pos=pos)
 
    ## draw edges in cycle basis
    #colors = sns.color_palette('hls', len(cl.edge_cycle_basis))
    #for cy, col in zip(cl.edge_cycle_basis, colors):
    #    nx.draw_networkx_edges(cl.G, pos=pos,
    #            edgelist=cy, edge_color=len(cy)*[col], width=2)
    ##plt.show()
    
    e = cl.G.edges()[cl.central_edge()]
    DeltaF = np.abs(cl.perturb_edge(e))
    edge_dists = cl.edge_distances(e)
    cycl_dists = cl.cycle_distances(e)

    # plot distance dependence
    plt.figure()
    
    asv('edge_dists_{}_DeltaF.txt'.format(n))(plt.loglog)(
            edge_dists, DeltaF, 'o', label='edge distances')
   
    plt.xlabel('edge distance')
    plt.ylabel('$\Delta F$')

    plt.figure()

    asv('cycle_dists_{}_DeltaF.txt'.format(n))(plt.loglog)(
            cycl_dists, DeltaF, 'o', label='cycle distances')
   
    plt.xlabel('cycle distance')
    plt.ylabel('$\Delta F$')
    
    plt.figure()

    x, m, std = distance_means(edge_dists, DeltaF)
    asv('edge_dists_{}_mean_DeltaF.txt'.format(n))(plt.loglog)(
            x, m, 'o')
    plt.xlabel('edge distance')
    plt.ylabel('mean $\Delta F$')
    
    x = np.array(x)
    m = np.array(m)
    gt0 = (x > 0) & (x < 0.8*x.max())
    print np.polyfit(np.log(x[gt0]), np.log(m[gt0]), 1)

    plt.figure()

    x, m, std = distance_means(cycl_dists, DeltaF)
    x = np.array(x)
    m = np.array(m)

    asv('cycle_dists_{}_mean_DeltaF.txt'.format(n))(plt.loglog)(
            x, m, 'o')
    plt.xlabel('cycle distance')
    plt.ylabel('mean $\Delta F$')

    gt0 = (x > 0) & (x < 0.8*x.max())
    print np.polyfit(np.log(x[gt0]), np.log(m[gt0]), 1)
    
    plt.show()

if __name__ == '__main__':
    # nice plotting
    sns.set(style='ticks', font_scale=1.2)
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    params = {'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)

    # analyze periodic lattice
    n = 50

    cl = PeriodicLattice(n)

    print "Number of cycles in cycle basis:"
    print cl.G.number_of_edges() - cl.G.number_of_nodes() + 1
    print cl.G.number_of_edges()
    print cl.G.number_of_nodes()

    pos = dict((n, d['pos']) for n, d in cl.G.nodes_iter(data=True))
    #nx.draw_networkx_edges(cl.G, pos=pos)
    #nx.draw_networkx_labels(cl.G, pos=pos)
 
    ## draw edges in cycle basis
    #colors = sns.color_palette('hls', len(cl.edge_cycle_basis))
    #for cy, col in zip(cl.edge_cycle_basis, colors):
    #    nx.draw_networkx_edges(cl.G, pos=pos,
    #            edgelist=cy, edge_color=len(cy)*[col], width=2)
    #plt.show()

    e = cl.G.edges()[cl.central_edge()]
    e = cl.G.edges()[0]
    DeltaF = np.abs(cl.perturb_edge(e))
    DeltaF /= DeltaF.max()
    #edge_dists = cl.edge_distances(e)
    #cycl_dists = cl.cycle_distances(e)
    
    # plot perturbation strength
    cmap = plt.get_cmap('jet')
    norm = colors.LogNorm(vmin=DeltaF.min(), vmax=1)
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)
    
    plt.figure()
    ax = plt.gca()
    
    # unweighted
    scalar_map.set_array(DeltaF)
    cols = list(scalar_map.to_rgba(DeltaF))

    nx.draw_networkx_edges(cl.G, pos=pos, edge_color=cols,
            width=5*DeltaF**0.2)

    print e
    nx.draw_networkx_edges(cl.G, pos=pos, edgelist=[e],
            width=10)

    plt.show()
