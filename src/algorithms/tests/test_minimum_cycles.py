#!/usr/bin/env python
from nose.tools import *
import networkx
import networkx as nx

from minimum_cycles import minimum_cycle_basis



class TestMinimumCycles:

    def setUp(self):
        T = nx.Graph()
        T.add_cycle([1, 2, 3, 4], weight=1)
        T.add_edge(2, 4, weight=5)
        self.diamond_graph = T

    def basis_equal(self, a, b):
        assert_list_equal(sorted(a), sorted(b))

    def test_unweighted_diamond(self):
        mcb = minimum_cycle_basis(self.diamond_graph)
        self.basis_equal(mcb, [[1, 2, 4], [2, 3, 4]])

    def test_weighted_diamond(self):
        mcb = minimum_cycle_basis(self.diamond_graph, weight='weight')
        self.basis_equal(mcb, [[1, 2, 4], [1, 2, 3, 4]])

    def test_dimentionality(self):
        #checks |MCB|=|E|-|V|+|NC|
        ntrial = 10
        for _ in range(ntrial):
            rg = nx.erdos_renyi_graph(10, 0.3)
            nnodes = rg.number_of_nodes()
            nedges = rg.number_of_edges()
            ncomp = nx.number_connected_components(rg)

            dim_mcb = len(minimum_cycle_basis(rg))
            assert(dim_mcb == nedges - nnodes + ncomp)

    def test_complete_graph(self):
        cg = nx.complete_graph(5)
        mcb = minimum_cycle_basis(cg)
        all([len(cycle) == 3 for cycle in mcb])

    def test_tree_graph(self):
        tg = nx.balanced_tree(3, 3)
        assert_false(minimum_cycle_basis(tg))
