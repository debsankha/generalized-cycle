#!/usr/bin/env python

"""
    cycle_basis.py

    functions for calculating the cycle basis of a graph
"""

from numpy import *
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path

if matplotlib.__version__ >= '1.3.0':
    from matplotlib.path import Path
else:
    from matplotlib import nxutils

from itertools import chain
from itertools import ifilterfalse
from itertools import izip
from itertools import tee

from collections import defaultdict

import plot

class Cycle():
    """ Represents a set of nodes that make up a cycle in some
    graph. Is hashable and does not care about orientation or things
    like that, two cycles are equal if they share the same nodes.
    A cycle can be compared to a set or frozenset of nodes.
    path is a list of vertices describing a closed path in the cycle.
    if it is absent, a closed path will be calculated together with
    coordinates.
    coords is an array of x-y pairs representing the coordinates of
    the cycle path elements.
    """
    def __init__(self, graph, edges, is_ordered=True, coords=None):
        """ Initializes the Cycle with an edge list representing the
        cycle.
        All edges should be ordered such that a cycle is represented
        as

        (1,2)(2,3)(3,4)...(n-2,n-1)(n-1,1)

        Parameters:
            graph: The underlying graph object

            edges: The edge list making up the cycle.
            
            is_ordered: If set to false, will use the neighborhood
                information from graph to construct ordered edge set
                from unordered one.
                In case the unordered edge set is not a connected graph,
                e.g. when removing one cycle splits the surrounding
                one in half, the smaller connected component in terms
                of total length is thrown away. Since our cycles are
                typically convex, this means we use the outermost 
                component.
        """
        self.graph = graph

        if not is_ordered:
            edges = self.ordered_edges(edges)
        
        self.path = zip(*edges)[0]
        if coords == None:
            self.coords = array([[graph.node[n]['x'], graph.node[n]['y']]
                    for n in self.path])
        else:
            self.coords = coords
        self.edges = edges
        
        # This allows comparisons
        self.edgeset = set([tuple(sorted(e)) for e in edges])
        self.com = mean(self.coords, axis=0)

        # This frozenset is used to compare/hash cycles.
        self._nodeset = frozenset(self.path)

    def ordered_edges(self, edges):
        """ Uses the graph associated to this cycle to order
        the unordered edge set.
        """
        # construct subgraph consisting of only the specified edges
        edge_graph = nx.Graph(edges)

        con = nx.connected_component_subgraphs(edge_graph)
        
        # Calculate sorted edge list for each connected component
        # of the cycle
        component_sorted_edges = []
        for comp in con:
            # get ordered list of edges
            component_edges = comp.edges()
            n_edges = len(component_edges)
            sorted_edges = []
            start = component_edges[0][0]
            cur = start
            prev = None

            for i in xrange(n_edges):
                nextn = [n for n in comp.neighbors(cur)
                        if n != prev][0]
                sorted_edges.append((cur, nextn))

                prev = cur
                cur = nextn

            component_sorted_edges.append(sorted_edges)
        
        # if there is more than one component, use the one with
        # larger area
        if len(con) > 1:
            print "Disconnected edge set, using larger area one."
            nodelists = [array([e[0] for e in edges] + [edges[-1][1]])
                    for edges in component_sorted_edges]
            coords = [array([[self.graph.node[n]['x'], 
                self.graph.node[n]['y']] for n in nodelist])
                    for nodelist in nodelists]

            areas = [polygon_area(coord) for coord in coords]
            
            sorted_edges = component_sorted_edges[argmax(areas)]

        return sorted_edges

    def intersection(self, other):
        """ Returns an edge set representing the intersection of
        the two cycles.
        """
        inters = self.edgeset.intersection(other.edgeset)

        return inters

    def union(self, other, data=True):
        """ Returns the edge set corresponding to the union of two cycles.
        Will overwrite edge/vertex attributes from other to this,
        so only use if both cycle graphs are the same graph!
        """
        union = self.edgeset.union(other.edgeset)
        return union

    def symmetric_difference(self, other, intersection=None):
        """ Returns a Cycle corresponding to the symmetric difference of
        the Cycle and other. This is defined as the set of edges which
        is present in either cycle but not in both.
        If the intersection has been pre-calculated it can be used.
        This will fail on non-adjacent loops.
        """
        new_edgeset = list(self.edgeset.symmetric_difference(
            other.edgeset))

        return Cycle(self.graph, new_edgeset, is_ordered=False)

    def area(self):
        """ Returns the area enclosed by the polygon defined by the
        Cycle. Warning: This will fail if the cycle is self-intersecting
        (this should not happen as long as nothing fundamental goes wrong).

        Formula adapted from 
        http://www.seas.upenn.edu/~sys502/extra_materials/Polygon%20Area%20and%20Centroid.pdf
        """
        Xs = self.coords[:,0]
        Ys = self.coords[:,1]

        # Close polygon, our node path is only implicitly closed
        Xs = append(Xs, Xs[0])
        Ys = append(Ys, Ys[0])
        
        # Ignore orientation
        return 0.5*abs(sum(Xs[:-1]*Ys[1:] - Xs[1:]*Ys[:-1]))

    def orientation(self):
        """ return orientation of cycle, +1 if counterclockwise,
        -1 if clockwise. same code as area calculation, but we
        use only the sign.
        """
        Xs = self.coords[:,0]
        Ys = self.coords[:,1]

        # Close polygon, our node path is only implicitly closed
        Xs = append(Xs, Xs[0])
        Ys = append(Ys, Ys[0])
        
        # Ignore orientation
        return sign(sum(Xs[:-1]*Ys[1:] - Xs[1:]*Ys[:-1]))

    def radii(self):
        """ Return the radii of all edges in this cycle.
        """
        return array([self.graph[u][v]['conductivity'] 
            for u, v in self.edgeset])

    def __hash__(self):
        """ Implements hashing by using the internal set description's hash
        """
        return self._nodeset.__hash__()

    def __eq__(self, other):
        """ Implements comparison using the internal set description
        """
        if isinstance(other, Cycle):
            return self._nodeset.__eq__(other._nodeset)
        elif isinstance(other, frozenset) or isinstance(other, set):
            return self._nodeset.__eq__(other)
        else:
            return -1

    def __repr__(self):
        return repr(self._nodeset)

def polygon_area(coords):
    """ Return the area of a closed polygon
    """
    Xs = coords[:,0]
    Ys = coords[:,1]

    # Ignore orientation
    return 0.5*abs(sum(Xs[:-1]*Ys[1:] - Xs[1:]*Ys[:-1]))
   
def traverse_graph(G, start, nextn):
    """ Traverses the pruned (i.e. ONLY LOOPS) graph G counter-clockwise 
    in the direction of nextn until start is hit again.
    If G has treelike components this will fail and get stuck, there
    is no backtracking.

    Returns a list of nodes visited, a list of edges visited and
    an array of node coordinates.
    This will find (a) all internal
    smallest loops (faces of the planar graph) and (b) one maximal
    outer loop
    """
    start_coords = array([G.node[start]['x'], G.node[start]['y']])
    nodes_visited = [start]
    nodes_visited_set = set()
    edges_visited = []
    coords = [start_coords]
    
    prev = start
    cur = nextn

    while cur != start:
        cur_coords = array([G.node[cur]['x'], G.node[cur]['y']])
        # We ignore all neighbors we alreay visited to avoid multiple loops
        neighs = [n for n in G.neighbors(cur) if n != prev and n != cur]
        
        edges_visited.append((prev, cur))
        nodes_visited.append(cur)
        coords.append(cur_coords)
        
        n_neighs = len(neighs)
        if n_neighs > 1:
            # Choose path that keeps the loop closest on the left hand side
            prev_coords = array([G.node[prev]['x'], G.node[prev]['y']])
            neigh_coords = array([[G.node[n]['x'], G.node[n]['y']] \
                for n in neighs])
             
            ## Construct vectors and normalize
            u = cur_coords - prev_coords
            vs = neigh_coords - cur_coords
            
            # calculate cos and sin between direction vector and neighbors
            u /= sqrt((u*u).sum(-1))
            vs /= sqrt((vs*vs).sum(-1))[...,newaxis]
            
            coss = dot(u, vs.T)
            sins = cross(u, vs)
            
            # this is a function between -2 and +2, where the
            # leftmost path corresponds to -2, rightmost to +2
            # sgn(alpha)(cos(alpha) - 1)
            ranked = copysign(coss - 1., sins)

            prev = cur
            cur = neighs[argmin(ranked)]
        else:
            # No choice to make
            prev = cur
            cur = neighs[0]
        
        # Remove pathological protruding loops
        if prev in nodes_visited_set:
            n_ind = nodes_visited.index(prev)
            
            del nodes_visited[n_ind+1:]
            del coords[n_ind+1:]
            del edges_visited[n_ind:]
    
        nodes_visited_set.add(prev)

    edges_visited.append((nodes_visited[-1], nodes_visited[0]))
     
    return nodes_visited, edges_visited, array(coords)

def cycle_mtp_path(cycle):
    """ Returns a matplotlib Path object describing the cycle.
    """
    # Set up polygon
    verts = zeros((cycle.coords.shape[0] + 1, cycle.coords.shape[1]))
    verts[:-1,:] = cycle.coords
    verts[-1,:] = cycle.coords[0,:]

    codes = Path.LINETO*ones(verts.shape[0])
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY

    return Path(verts, codes)

# We need to be backwards compatible.
if matplotlib.__version__ >= '1.3.0':
    def outer_loop(G, cycles):
        """ Detects the outermost loop in the set of fundamental cycles
        including the outermost loop by testing whether more than one
        centroid lies on the inside (this happens *only* for the outermost
        loop)
        """
        coms = array([c.com for c in cycles])
        for c in cycles:
            path = cycle_mtp_path(c)
            if path.contains_points(coms).all():
                # Only outer loop contains all centers of mass
                return c
else:
    # Backwards compatibility
    def outer_loop(G, cycles):
        coms = array([c.com for c in cycles])

        for c in cycles:
            xyverts = zeros((c.coords.shape[0] + 1, c.coords.shape[1]))
            xyverts[:-1,:] = c.coords
            xyverts[-1,:] = c.coords[0,:]
            if nxutils.points_inside_poly(coms, xyverts).all():
                return c

def shortest_cycles(G, raise_exception=False):
    """ Returns a list of lists of Cycle objects belonging to the
    fundamental cycles of the pruned (i.e. there are no treelike
    components) graph G by traversing the graph counter-clockwise
    for each node until the starting node has been found.
    Also returns the outer loop.
    """
    cycleset = set()
    # Betti number counts interior loops, this algorithm finds
    # exterior loop as well!
    #print nx.number_connected_components(G)
    n_cycles = G.number_of_edges() - G.number_of_nodes() + 1
    
    # Count outer loop as well
    if n_cycles >= 2:
        n_cycles += 1

    print "Number of cycles including boundary: {}.".format(n_cycles)
    
    mst = nx.minimum_spanning_tree(G, weight=None)

    for u, v in G.edges_iter():
        if not mst.has_edge(u, v):
            # traverse cycle in both directions
            path, edges, coords = traverse_graph(G, u, v)
            cycleset.add(Cycle(G, edges, coords=coords))

            path, edges, coords = traverse_graph(G, v, u)
            cycleset.add(Cycle(G, edges, coords=coords))
        
    if len(cycleset) != n_cycles:
        print "WARNING: Found incorrect number of cycles,", len(cycleset), \
                "instead of", n_cycles
        if raise_exception:
            raise Exception('Wrong number of cycles. {} instead of {}'.format(len(cycleset), n_cycles))

    
    #print "Number of detected facets:", len(cycleset)
    return list(cycleset)

def find_neighbor_cycles(G, cycles):
    """ Returns a set of tuples of cycle indices describing
    which cycles share edges
    """
    n_c = len(cycles)

    # Construct edge dictionary
    edges = defaultdict(list)

    for i, c in enumerate(cycles):
        for e in c.edges:
            edges[tuple(sorted(e))].append(i)
    
    # Find all neighboring cycles
    neighbor_cycles = set()

    for k, n in edges.iteritems():
        neighs = tuple(sorted(n))

        if len(neighs) == 2:
            neighbor_cycles.add(neighs)
        elif len(neighs) == 1:
            print "Warning!! Some edge has only one neighboring cycle."
            print k, n
            print neighs[0]
            print cycles[neighs[0]].edges
            print cycles[neighs[0]].com
            #plt.figure()
            #plot.draw_leaf(G)
            #plot.draw_leaf(G, edge_list=cycles[neighs[0]].edges, color='r')
            #plot.draw_leaf(G, edge_list=[k], color='g')
            #plt.show()
        else:
            print "Warning!! Some edge has {} neighboring cycles.".format(
                    len(neighs))
            print k, n
            print neighs[0]
            print cycles[neighs[0]].edges
            print cycles[neighs[0]].com
            #plt.figure()

            #plot.draw_leaf(G)

            #for nn in neighs:
            #    plot.draw_leaf(G, edge_list=cycles[nn].edges, 
            #            color='r')

            #plot.draw_leaf(G, edge_list=[k], color='g')

            #for nn in neighs:
            #    plt.figure()
            #    plot.draw_leaf(G, edge_list=cycles[nn].edges, 
            #            color='r')

            #plt.show()

    return neighbor_cycles, edges

def prune_graph(G):
    """
        Return a graph describing the loopy part of G, which is
        implicitly described by the list of cycles.
        The loopy part does not contain any

        (a) tree subgraphs of G
        (b) bridges of G

        Thus pruning may disconnect the graph into several 
        connected components.
    """
    cycles = nx.cycle_basis(G)
    pruned = G.copy()
    cycle_nodes = set(chain.from_iterable(cycles))
    cycle_edges = []

    for c in cycles:
        cycle = c + [c[0]]
        a, b = tee(cycle)
        next(b, None)
        edges = izip(a, b)

        cycle_edges.append(edges)
    
    all_cycle_edges = set(tuple(sorted(e)) \
            for e in chain.from_iterable(cycle_edges))
    # remove treelike components and bridges by removing all
    # edges not belonging to loops and then all nodes not
    # belonging to loops.
    pruned.remove_edges_from(e for e in G.edges_iter() \
            if (not tuple(sorted(e)) in all_cycle_edges))
    pruned.remove_nodes_from(n for n in G if not n in cycle_nodes)

    return pruned

def apply_workaround(G, thr=1e-10):
    """ Applies a workaround to the graph which removes all
    exactly (up to given threshold) collinear edges.
    """

    removed_edges = []
    for n in G.nodes_iter():
        nei = G.neighbors(n)

        p1 = array([[G.node[m]['x'], G.node[m]['y']] \
                for m in nei])
        p0 = array([G.node[n]['x'], G.node[n]['y']])
        
        dp = p1 - p0
        dp_l = sqrt((dp*dp).sum(axis=1))
        dp_n = dp/dp_l[...,newaxis]

        coss = dot(dp_n, dp_n.T)
        
        tril_i = tril_indices(coss.shape[0])
        coss[tril_i] = 0.
        
        coll = abs(coss - 1.) < thr
        for i in xrange(len(nei)):
            c = where(coll[:,i])[0]
            if len(c) > 0:
                edges = tuple((n, nei[cc]) for cc in c)
                dp_c = zip(dp_l[c], edges) + [(dp_l[i], (n, nei[i]))]
                max_v, max_e = max(dp_c)
                
                #print "Found collinear edges:"
                #print dp_c
                #print "Removing offending edge {}.".format(max_e)
                G.remove_edge(*max_e)

                removed_edges.append(max_e)

    return removed_edges

# cython version of this is about 4x faster
#from traverse_graph import traverse_graph as trav_c
#traverse_graph = trav_c

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A, B, C, D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def knbrs(G, start, k):
    """ Return the k-neighborhood of node start in G.
    """
    nbrs = set([start])
    for l in xrange(k):
        nbrs = set((nbr for n in nbrs for nbr in G[n]))
    return nbrs

def remove_intersecting_edges(G):
    """ Remove any two edges that intersect from G,
    correcting planarity errors.
    Since we cannot tell which one of the edges is the "correct" one,
    we remove both.
    """
    
    edges_to_rem = []
    edges = G.edges()
    for i in xrange(len(edges)):
        u1, v1 = edges[i]

        u1_x = G.node[u1]['x']
        u1_y = G.node[u1]['y']

        v1_x = G.node[v1]['x']
        v1_y = G.node[v1]['y']

        u1_vec = [u1_x, u1_y]
        v1_vec = [v1_x, v1_y]
        
        # look at order 5 neighbors subgraph (this is an approximation,
        # not guaranteed to work every single time! It is fast though.)
        neighs = knbrs(G, u1, 5)
        neighs.update(knbrs(G, v1, 5))
        sg = G.subgraph(neighs)

        for u2, v2 in sg.edges_iter():
            # If the edges have a node in common, disregard.
            if u2 == u1 or u2 == v1 or v2 == u1 or v2 == u2:
                continue

            u2_x = G.node[u2]['x']
            u2_y = G.node[u2]['y']

            v2_x = G.node[v2]['x']
            v2_y = G.node[v2]['y']
            
            u2_vec = [u2_x, u2_y]
            v2_vec = [v2_x, v2_y]
            
            if intersect(u1_vec, v1_vec, u2_vec, v2_vec):
                    edges_to_rem.append((u1, v1))
                    edges_to_rem.append((u2, v2))
                    #print (u1, v1), (u2, v2)
    
    G.remove_edges_from(edges_to_rem) 
    return edges_to_rem
