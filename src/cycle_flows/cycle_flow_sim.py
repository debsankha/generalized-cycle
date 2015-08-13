#!/usr/bin/env python

"""
    cycle_flow_sim.py

    simulates a perturbation on a real leaf network and makes a cutesy
    little plot of the cycles.
"""

from collections import defaultdict
from time import time
import cPickle as pickle
import bz2
from glob import glob

import sys
import os

import networkx as nx
import scipy.sparse
import scipy.spatial
from scipy.sparse.linalg import spsolve, factorized, lsqr, lsmr
from scipy.optimize import curve_fit
from scipy.stats import linregress

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
from matplotlib.patches import Arc
from matplotlib.path import Path

from joblib import Parallel, delayed

from cycle_basis import *
from sparse_networks import *
from cycle_flows import *
from cycle_plots import *
import plot
import correlations

import seaborn as sns

def rotate_network(G):
    """ Rotate the graph coordinates by 90 degrees.
    """
    for n, d in G.nodes_iter(data=True):
        d['x'], d['y'] = -d['y'], d['x']

def exp_decay(x, A, B):
    return A*np.exp(-np.abs(x)/B)

def power_law(x, A, B):
    return A*np.abs(x)**(-B)

def cutoff(x, a, b, c):
    return a*x**(-b)*np.exp(-x/c)

def filter_data(x, y, x_min, x_max, remove_y_zeros=False):
    x = np.asarray(x)
    y = np.asarray(y)

    gt0 = x >= x_min

    x = x[gt0]
    y = y[gt0]

    ltm = x <= x_max

    x = x[ltm]
    y = y[ltm]

    # remove zeros
    if remove_y_zeros:
        y_nonzero = y > 0

        x = x[y_nonzero]
        y = y[y_nonzero]

    return x, y

def fit_exponential(x, y, x_min=1, linear_fit=False):
    """ fit an exponential decay function to the cycle density
    data using nonlinear least squares.
    The function we fit is f(x) = A exp(-B |x|)
    """
    x, y = filter_data(x, y, x_min, 0.5*np.max(x), 
            remove_y_zeros=linear_fit)
    
    if linear_fit:
        y = np.log(y)
        b, a, r, p, err = linregress(x, y)

        popt = np.array([np.exp(a), -1./b])
        pcov = np.array([r**2, p**2])

        error = np.sum(np.square(a + b*x - y))
    else:
        popt, pcov = curve_fit(exp_decay, x, y)
        error = np.sum(np.square((exp_decay(x, *popt) - y)))

    return popt, pcov, error

def fit_power_law(x, y, x_min=1, linear_fit=False):
    """ Fit a power law. The function is
    f(x) = A (x+1)**(-B)
    """
    x, y = filter_data(x, y, x_min, 0.5*np.max(x),
            remove_y_zeros=linear_fit)

    if linear_fit:
        x = np.log(x)
        y = np.log(y)

        b, a, r, p, err = linregress(x, y)

        popt = np.array([np.exp(a), -b])
        pcov = np.array([r**2, p**2])

        error = np.sum(np.square(a + b*x - y))
    else:
        popt, pcov = curve_fit(power_law, x, y)
        error = np.sum(np.square((power_law(x, *popt) - y)))
    
    return popt, pcov, error


def fit_cutoff(x, y, x_min=1):
    """ Fit a power law. The function is
    f(x) = A (x+1)**(-B)
    """
    x, y = filter_data(x, y, x_min, 0.5*np.max(x))

    popt, pcov = curve_fit(cutoff, x, y)
    
    error = np.sum(np.square((cutoff(x, *popt) - y)))
    
    return popt, pcov, error

def minmax_values(distances, perts):
    """ return the minima and maxima for each distance.
    """
    maxs = defaultdict(int)
    mins = defaultdict(int)
    
    perts = perts/np.max(np.abs(perts))

    for d, p in zip(distances, perts):
        if p > maxs[d]:
            maxs[d] = p

        if p < mins[d]:
            mins[d] = p

    return maxs, mins

def means_values(distances, perts):
    """ return the means of cycle currents
    """
    all_values = defaultdict(list)
    perts = np.abs(perts)/np.max(np.abs(perts))

    for d, p in zip(distances, perts):
        all_values[d].append(p)
    
    dists = np.array(sorted(all_values.keys()))
    means = np.array([np.mean(all_values[d]) for d in dists])

    return dists, means

def line_distance(x0, angle, p):
    """ Assumes a line staring at point x0 and extending in direction
    angle. Returns the shortest distances of the points p to the line.
    """
    e = np.array([np.cos(angle), np.sin(angle)])
    
    # distance from origin in direction of angle
    u = np.dot(p - x0, e)
    u[u < 0] = 0
    
    projections = x0 + u[:,np.newaxis]*e

    dx = projections - p

    dist = np.linalg.norm(dx, axis=1)

    return dist

def plot_cycle_currents_dual_distance_direction(cycles, pert_currents, 
        pert_cycles, G_dual, directions=5):
    """ Plot a graph of perturbation strength vs shortest path distance
    on the dual graph measured from the point of perturbation in a number
    of directions. We find all cycles who contain the direction line and
    plot their cycle strengths.
    """
    c_a = pert_cycles[0]

    path_lengths = nx.shortest_path_length(G_dual, source=c_a)
    distances = np.array([path_lengths[c] for c in xrange(len(cycles))])
    
    angles = np.linspace(0, 2*np.pi, directions)[:-1]
    
    cycle_coms = np.array([c.com for c in cycles])

    plt.figure()
    for phi in angles:
        # find all cycles which are close to the line
        dists = line_distance(cycles[c_a].com, phi, cycle_coms)
        close_cycles = np.where(dists < 50)[0]

        dual_distances = distances[close_cycles]
        cycle_currents = np.abs(pert_currents[close_cycles])/np.max(np.abs(pert_currents))

        # plot stuff
        plt.semilogy(dual_distances, cycle_currents, 'o',
                label="$\phi = {:.2}\pi$".format(phi/np.pi))

    plt.xlabel('cycle distance $d_c$')
    plt.ylabel('cycle current $|f_c| / |f|_\mathrm{max}$')
    
    plt.legend(loc='best')
    plt.savefig('plots/current_strength_cycle_distance_directions.svg', 
            dpi=300, bbox_inches='tight')   


def remove_bad_edges(G):
    """ Remove edges which have zero or very small radius/length
    """
    edges_to_rem = [(u, v) for u, v, d in G.edges_iter(data=True)
            if d['conductivity'] < 0.2 or d['weight'] < 0.2]

    G.remove_edges_from(edges_to_rem)

def shortest_path_lengths_between_edges(G, source):
    """ Uses Dijkstra's algorithm to calculate the
    shortest path length between edges, defined as 
    the minimum distance between nodes belonging to the edges.
    """
    print "Calculating edge shortest paths"
    edge_lens = nx.shortest_path_length(nx.line_graph(G), 
            source=source)

    lens = [edge_lens[(u, v)] for u, v in G.edges_iter()]
    
    return lens

def remove_outer_from_dual(dual, outer, outer_n, edges, cycles):
    """ Removes the outermost loop from the dual graph
    and creates new nodes for each loop bordering it.
    Also updates the edges dict.
    """
    # Only necessary if there is more than one loop
    if dual.number_of_nodes() <= 1:
        return

    # Find boundary nodes in dual
    boundary = [n for n in dual.nodes_iter() 
            if outer_n in dual.neighbors(n)]
    
    # add new edges in the dual
    max_nodes = max(dual.nodes())
    k = 1
    for b in boundary:
        new = max_nodes + k

        # add to dual graph
        dual.add_edge(b, new)

        # update edges dict
        for e in cycles[b].intersection(cycles[outer_n]):
            se = tuple(sorted(e))
            edges[se].remove(outer_n)
            edges[se].append(new)
     
        k = k + 1
    
    # Remove original boundary node
    dual.remove_node(outer_n)

def shortest_cycle_path_lengths_between_edges(G, source, graph_fname=''):
    """ Uses Dijkstra's algorithm to calculate the
    shortest cycle path length between edges, defined as 
    the minimum distance between nodes belonging to the edges.
    """
    # check if computation result is cached
    cache_file = 'cache/' + os.path.basename(graph_fname) + \
            '_cycles_cache.pkl.bz2'
    if graph_fname != '' and os.path.exists(cache_file):
        print "Computation results found in cache, using cached data."
        cached = pickle.load(bz2.BZ2File(cache_file, 'rb'))

        G_dual = cached['G_dual']
        G_line = cached['G_line']
        cycles = cached['cycles']
        boundary = cached['boundary']
        boundary_ind = cached['boundary_ind']
        neighbor_cycles = cached['neighbor_cycles']
        edges = cached['edges']
    else:
        G_prun = prune_graph(G)
        G_pruned = nx.connected_component_subgraphs(G_prun)[0]

        # detect cycles
        #plt.figure()
        #plot.draw_leaf(G_pruned, fixed_width=True)
        #plt.show()
        cycles = shortest_cycles(G_pruned, raise_exception=True)

        boundary = outer_loop(G_pruned, cycles)
        boundary_ind = cycles.index(boundary)

        neighbor_cycles, edges = find_neighbor_cycles(G_pruned, cycles)
        neighbor_cycles = list(neighbor_cycles)    

        # construct cycle dual graph
        G_dual = nx.Graph()
        G_dual.add_edges_from(neighbor_cycles)
        
        # remove outer loop
        remove_outer_from_dual(G_dual, boundary, boundary_ind, 
                edges, cycles)

        # compute line graph and calculate distances
        G_line = nx.line_graph(G_dual)

        # save into cache
        pickle.dump({
            'cycles': cycles, 
            'boundary': boundary, 
            'boundary_ind': boundary_ind, 
            'neighbor_cycles': neighbor_cycles, 
            'edges': edges,
            'G_line': G_line,
            'G_dual': G_dual}, bz2.BZ2File(cache_file, 'wb'))

    # source edge in dual and shortest paths
    dual_source = tuple(edges[tuple(sorted(source))])
    lens = nx.shortest_path_length(G_line, source=dual_source)

    elens = []
    for e in G.edges_iter():
        cy = tuple(sorted(edges[tuple(sorted(e))]))

        if cy in lens:
            elens.append(lens[cy])
        elif cy[::-1] in lens:
            elens.append(lens[cy[::-1]])
            print "Reverse order tuple??"
        else:
            print dual_source, e
            print "Warning, cycle distance undefined. Setting to inf"
            elens.append(float('inf'))

    return elens

def random_loopy_network(n, weights='uniform'):
    """ Constructs a random loopy network with n loops by calculating
    a Voronoi tesselation of the plane for n points.
    
    Returns a NetworkX graph representing the random loopy network.
    """
    pts = np.random.random((n, 2))
    vor = scipy.spatial.Voronoi(pts)
    vor.close()

    G = nx.Graph()

    in_region = lambda v: v[0] <= 1 and v[0] >= 0 and v[1] <= 1 and v[1] >= 0
    
    # Add vertices of Voronoi tesselation
    for i in xrange(len(vor.vertices)):
        v = vor.vertices[i]
        # Only use what's within our region of interest
        if in_region(v):
            G.add_node(i, x=v[0], y=v[1])
    
    # Add ridges which do not go to infinity
    for r in vor.ridge_vertices:
        v0 = vor.vertices[r[0]]
        v1 = vor.vertices[r[1]]
        if r[0] >= 0 and r[1] >= 0 and in_region(v0) and in_region(v1):
            if weights == 'uniform':
                cond = 1.0
            else:
                cond = np.random.random()
            length = linalg.norm(v0 - v1)
            G.add_edge(r[0], r[1], conductivity=cond, weight=length)
    
    # rescale to 1000x1000
    for n, d in G.nodes_iter(data=True):
        d['x'] *= 1000
        d['y'] *= 1000

    return prune_graph(G)

def random_loopy_network_holes(n, n_holes=50, weights='uniform'):
    """ Construct a random loopy network with large holes such that
    edge distance and cycle distance become more uncorrelated.
    """
    G = random_loopy_network(n, weights=weights)
    
    holes = 900*np.random.random((n_holes, 2)) + 50
    hole_radii = 10*np.random.randn(n_holes) + 50
    
    nodes_to_rem = []
    for (x, y), r in izip(holes, hole_radii):

        r_sqr = r**2
        for n, d in G.nodes_iter(data=True):
            xn = d['x']
            yn = d['y']

            if (xn - x)**2 + (yn - y)**2 < r_sqr:
                nodes_to_rem.append(n)

    G.remove_nodes_from(nodes_to_rem)

    #plt.figure()
    #plot.draw_leaf(G)
    #for hole, r in izip(holes, hole_radii):
    #    c = plt.Circle((hole[0], hole[1]), r, color='b')
    #    plt.gca().add_artist(c)
    #plt.show()
    
    G = nx.connected_component_subgraphs(prune_graph(G))[0]
    G = nx.convert_node_labels_to_integers(G)
    return G

def affected_edge_density(G, DeltaF_normalized, threshold, ax=None):
    """ Calculate the number of edges for which DeltaF is
    above threshold. Take the convex hull of the associated
    nodes and count the total number of edges inside.
    The ratio of these numbers is the affected edge density.
    """
    points = [((G.node[u]['x'], G.node[u]['y']),
        (G.node[v]['x'], G.node[v]['y']))
            for i, (u, v) in enumerate(G.edges_iter())
            if DeltaF_normalized[i] > threshold]

    a, b = zip(*points)
    points = np.array(a + b)

    ch = scipy.spatial.ConvexHull(points)
    path = Path(points[ch.vertices], closed=True)

    nodes_in_path = [n for n, d in G.nodes_iter(data=True)
            if path.contains_point((d['x'], d['y']), radius=10)]

    sg = G.subgraph(nodes_in_path)

    #print "Edges above threshold", len(points)
    #print "Edges in area:", sg.number_of_edges()

    if ax != None:
        for s in ch.simplices:
            ax.plot(points[s,0], points[s,1], 'r', lw=1.5)

    return float(len(points))/(2*sg.number_of_edges())

def total_edge_density(G, DeltaF_normalized, threshold):
    """ Calculate the number of edges for which DeltaF is
    above threshold. 
    The ratio of this number to the total number of
    edges is the total affected edge density.
    """
    points = [((G.node[u]['x'], G.node[u]['y']),
        (G.node[v]['x'], G.node[v]['y']))
            for i, (u, v) in enumerate(G.edges_iter())
            if DeltaF_normalized[i] > threshold]

    a, b = zip(*points)
    points = np.array(a + b)


    return float(len(points))/(2*G.number_of_edges())

###
### Functions for doing analysis/making figures etc...
###


def random_graph_analysis():
    print "Loading graph file"
    G = random_loopy_network(5000)
        
    cy, perts, G_pruned, G_dual, pert_cycless, edges = \
            simulate_cycle_flow(G)
        
    # plot the decay behavior
    for pert, pert_cycles in zip(perts, pert_cycless):

        # plot whole graph
        #plot_cycle_currents(G, cy, pert, pert_cycles)
        #plot_cycle_currents_distance(cy, pert, pert_cycles)

        plot_cycle_currents_dual_distance(cy, pert, pert_cycles, G_dual)
        plot_cycle_currents_dual_distance_direction(cy, pert, 
                pert_cycles, G_dual)

    plt.show()

def distance_dependence_analysis(G, show_plot=False):
    G = nx.convert_node_labels_to_integers(G)
    
    # perturb network
    print "Calculating edge flows"
    #t0 = time()
    #DeltaF2 = simulate_edge_flows(G, weighted=False)
    #t1 = time()
    DeltaF = simulate_edge_flows2(G, weighted=False)
    #t2 = time()

    #print "INV:", t1-t0, 
    #print "QR:", t2-t1
    #print np.allclose(DeltaF, DeltaF2)
    #print DeltaF[0,:5]
    #print DeltaF2[0,:5]
    #raw_input()

    # distances
    print "Measuring distances"
    all_edge_dists = []
    all_cycle_dists = []
    all_DeltaF = []
    
    DeltaF_edge = defaultdict(list)
    DeltaF_cycle = defaultdict(list)

    for i, (u, v) in enumerate(G.edges_iter()):
        edge_dists = shortest_path_lengths_between_edges(G, (u, v))
        cycle_dists = shortest_cycle_path_lengths_between_edges(G, 
                (u, v))

        all_edge_dists.append(edge_dists)
        all_cycle_dists.append(cycle_dists)
        all_DeltaF.append(np.abs(DeltaF[i,:]/DeltaF[i,i]))
        
        #print edge_dists
        #print cycle_dists
        #print DeltaF[i,:]
        #raw_input()
    
    all_edge_dists = np.array(all_edge_dists)
    all_cycle_dists = np.array(all_cycle_dists)
    all_DeltaF = np.array(all_DeltaF)
    
    for de, dc, DF in izip(all_edge_dists.flatten(), 
            all_cycle_dists.flatten(), all_DeltaF.flatten()):
        DeltaF_edge[de].append(DF)
        DeltaF_cycle[dc].append(DF)

    # correlation ratios and mutual information
    #print "Calculating correlations"
    #cats_edges = correlations.categorize(all_edge_dists.flatten(), 
    #        all_DeltaF.flatten())
    #cats_cycles = correlations.categorize(all_cycle_dists.flatten(), 
    #        all_DeltaF.flatten())
    #
    #print "Edge distance correlation ratio:"
    #print correlations.correlation_ratio(cats_edges[1])

    #print "Cycle distance correlation ratio:"
    #print correlations.correlation_ratio(cats_cycles[1])
    
    eds = all_edge_dists.flatten()
    cds = all_cycle_dists.flatten()
    DFs = all_DeltaF.flatten()

    print "Edge distance MI:"
    #print correlations.mutual_information(all_edge_dists.flatten(),
    #        all_DeltaF.flatten())
    edge_mi = correlations.mutual_information2(eds,
            DFs, k=3)    
    print edge_mi

    print "Cycle distance MI:"
    #print correlations.mutual_information(all_cycle_dists.flatten(),
    #        all_DeltaF.flatten())
    cycle_mi = correlations.mutual_information2(cds,
            DFs, k=3)
    print cycle_mi
  
    # plots
    if show_plot:
        # plot of edge and cycle distance total statistics
        f, (ax1, ax2) = plt.subplots(2)
        
        ax1.plot(all_edge_dists, all_DeltaF, 'o')
        ax1.set_xlabel('edge distance $d_e$')
        ax1.set_ylabel('flow change $\Delta F_e/\Delta F_{e_0}$')
        
        ax2.plot(all_cycle_dists, all_DeltaF, 'o')
        ax2.set_xlabel('cycle distance $d_c$')
        ax2.set_ylabel('flow change $\Delta F_e/\Delta F_{e_0}$')
        
        plt.tight_layout()
        plt.savefig('plots/distance_statistics.png', dpi=600,
                bbox_inches='tight')
        
        # plot of some histograms for the statistics
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        
        # histograms for several edge and cycle distances
        for i in xrange(1, 4):
            ax1.hist((all_DeltaF[all_edge_dists == i]), bins=30, 
                    range=(0, 1), label='$d_e = {}$'.format(i), 
                    normed=True, alpha=0.7)
        ax1.legend(loc='best')
        
        ax1.set_xlabel('perturbation current $|\Delta F_e/\Delta F_{e_0}|$')
        ax1.set_ylabel('probability density')
        
        for i in xrange(3):
            ax2.hist((all_DeltaF[all_cycle_dists == i]), bins=30, 
                    range=(0, 1), label='$d_c = {}$'.format(i), 
                    normed=True, alpha=0.7)
        ax2.legend(loc='best')
        
        ax2.set_xlabel('perturbation current $|\Delta F_e/\Delta F_{e_0}|$')
        ax2.set_ylabel('probability density')
        
        # decay of mean deltaf with edge/cycle distance
        eds = np.array(sorted(list(set(all_edge_dists.flatten()))))
        DFes = np.array([np.mean(np.abs(DeltaF_edge[de])) for de in eds])
        ax3.loglog(eds, DFes, 'o')

        # exponential fit
        #def func(x, a, c, d):
        #    return a*np.exp(-c*x) + d

        #popt, pcov = curve_fit(func, eds, DFes, p0=(1, 1, 0))
        #xx = np.linspace(eds.min(), eds.max(), 100)
        #ax3.plot(xx, func(xx, *popt))
        
        ze = np.polyfit(np.log(eds[eds > 0]), np.log(DFes[eds > 0]), 1)
        x = np.linspace(1, np.max(eds), 100)
        ax3.loglog(x, np.exp(np.polyval(ze, np.log(x))))

        ax3.set_xlabel('edge distance $d_e$')
        ax3.set_ylabel('perturbation current $|\Delta F_e/\Delta F_{e_0}|$')
        #ax3.set_xlim(0, 8)

        cds = np.array(sorted(list(set(all_cycle_dists.flatten()))))
        DFcs = np.array([np.mean(np.abs(DeltaF_cycle[de])) for de in cds])
        ax4.loglog(cds, DFcs, 'o')

        #popt, pcov = curve_fit(func, cds, DFcs, p0=(1, 1, 0))
        #xx = np.linspace(cds.min(), cds.max(), 100)
        #ax4.plot(xx, func(xx, *popt))
        
        zc = np.polyfit(np.log(cds[cds > 0]), np.log(DFcs[cds > 0]), 1)
        x = np.linspace(1, np.max(cds))
        ax4.loglog(x, np.exp(np.polyval(zc, np.log(x))))

        ax4.set_xlabel('cycle distance $d_c$')
        ax4.set_ylabel('perturbation current $|\Delta F_e/\Delta F_{e_0}|$')    
        print ze
        print zc
        #ax4.set_xlim(0, 8)
        
        # save as npz file
        np.savez('cycle_data.npz', 
                all_edge_dists, all_cycle_dists, all_DeltaF)

        plt.tight_layout()
        plt.savefig('plots/network_statistics.svg', bbox_inches='tight')
        
        # network
        plt.figure()
        plot.draw_leaf(G)

        #plt.figure()
        #for i, x in enumerate(cats_edges[1]):
        #    plt.plot(len(x)*[i], x, 'o')
        
        plt.savefig('plots/network.svg')
        plt.show()

    return edge_mi, cycle_mi, G.graph['dual'].number_of_nodes()
        
def distance_dependence_analysis_multi(n_per_param=5, n_holes=30, 
        seed=123456):
    """ Create multiple random graphs with various sizes,
    calculate their edge and cycle mutual information and plot
    """
    np.random.seed(seed)
    
    sizes = []
    edge_mis = []
    cycle_mis = []

    edges_per_size = defaultdict(list)
    cycles_per_size = defaultdict(list)

    for size in range(50, 500, 50):
        print "XXXXX"
        print "Size:", size
        print "XXXXX"
        for i in xrange(n_per_param):
            G = random_loopy_network_holes(size, n_holes=n_holes)
            edge_mi, cycle_mi, n_cycles = distance_dependence_analysis(G)

            sizes.append(n_cycles)
            edge_mis.append(edge_mi)
            cycle_mis.append(cycle_mi)
            
            edges_per_size[size].append(edge_mi)
            cycles_per_size[size].append(cycle_mi)

    edge_mis = np.array(edge_mis)
    cycle_mis = np.array(cycle_mis)
    sizes = np.array(sizes)

    # save as text file
    sav = np.hstack((sizes[:,np.newaxis], edge_mis.T[:,np.newaxis], 
        cycle_mis[:,np.newaxis]))
    np.savetxt('plots/mutual_informations_{}_holes.txt'.format(n_holes), 
            sav)

    plt.figure()
    plt.plot(sizes, edge_mis, 'o', label='edge distance $d_e$')
    plt.plot(sizes, cycle_mis, 'o', label='cycle distance $d_c$')
    
    # plot lines with error bars for the means
    edge_mean = edge_mis.mean()
    cycle_mean = cycle_mis.mean()

    edge_std = edge_mis.std()
    cycle_std = cycle_mis.std()

    print "Mean Edge MI:", edge_mean
    print "Mean Cycle MI:", cycle_mean
    
    xmin, xmax = plt.gca().get_xlim()

    plt.axhline(edge_mean, color=sns.color_palette()[0], ls='--')
    plt.axhline(cycle_mean, color=sns.color_palette()[1], ls='--')

    plt.errorbar(0.5*(xmin + xmax) + 10, edge_mean, yerr=1.96*edge_std,
            color=sns.color_palette()[0])
    plt.errorbar(0.5*(xmin + xmax) - 10, cycle_mean, yerr=1.96*cycle_std,
            color=sns.color_palette()[1])

    plt.xlabel('network size (cycles)')
    plt.ylabel('mutual information $I(\\frac{\Delta F_e}{\Delta F_{e_0}} : d)$')
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('plots/mutual_informations_{}_holes.svg'.format(n_holes), 
            bbox_inches='tight')
    plt.show()


def calculate_cache_DeltaF(G, edge, j, graph_fname, cache=True):
    """ Calculate DeltaF vector for given graph, edge and graph
    file name. Use cached data if available and desired.
    """

    cache_file = 'cache/' + os.path.basename(graph_fname) + '_edge_{}_perturbations_cache.npz'.format(j)
    if cache and os.path.exists(cache_file):
        print "Computation results found in cache, using cached data."
        cache = np.load(cache_file)
        DeltaF_wt = cache['DeltaF_wt']
        DeltaF_uw = cache['DeltaF_uw']
        conds = cache['conds']
    else:
        print "Computing flow"
        DeltaF_wt, conds = simulate_single_edge_flow(G, edge, 
                weighted=True, sparse=True)
        DeltaF_uw, _ = simulate_single_edge_flow(G, edge, 
                weighted=False, sparse=True)
        
        #DeltaF_wt_old = simulate_single_edge_flow_old(G, edge, 
        #        weighted=True)

        #print DeltaF_wt/DeltaF_wt_old


        # save into cache
        if cache:
            np.savez(cache_file, DeltaF_wt=DeltaF_wt, 
                    DeltaF_uw=DeltaF_uw, conds=conds)
    
    return DeltaF_uw, DeltaF_wt, conds

###
### Figures for the paper
###


def distance_bins(x, y, dists=None):
    """ return a defaultdict with lists containing all data points
    measured at x. If dists is not None, use that as
    the defaultdict.
    """
    if dists == None:
        dists = defaultdict(list)
    
    for xx, yy in izip(x, y):
        dists[xx].append(yy)
    
    return dists

def distance_means(x, y):
    """ Take a set of measurements x, y and
    return x, <y>(x), std(y)(x)
    """
    xs = sorted(list(set(x)))

    ys = defaultdict(list)
    for xx, yy in izip(x, y):
        ys[xx].append(yy)
    

    means = np.array([np.mean(ys[xx]) for xx in xs])
    stds = np.array([np.std(ys[xx], ddof=1) for xx in xs])

    stds[np.isnan(stds)] = 0
    
    return xs, means, stds

def distance_minmax(x, y):
    """ Take a set of measurements x, y and
    return x, max(y)(x), min(y)(x)
    """
    xs = sorted(list(set(x)))

    ys = defaultdict(list)
    for xx, yy in izip(x, y):
        ys[xx].append(yy)

    mins = np.array([np.min(ys[xx]) for xx in xs])
    maxs = np.array([np.max(ys[xx]) for xx in xs])
    
    return xs, maxs, mins

def fit_data(eds, cds, DF, name, x_min=1, savename=None, plot=True):
    """ Fit power law and exponential functions to data.
    print errors and parameters
    """
    # fit power law and exponential to data
    popt_exp_ed, pcov_exp_ed, error_exp_ed = \
            fit_exponential(eds, DF, x_min=x_min, linear_fit=True)

    popt_exp_cd, pcov_exp_cd, error_exp_cd = \
            fit_exponential(cds, DF, x_min=x_min, linear_fit=True)

    popt_pwl_ed, pcov_pwl_ed, error_pwl_ed = \
            fit_power_law(eds, DF, x_min=x_min, linear_fit=True)

    popt_pwl_cd, pcov_pwl_cd, error_pwl_cd = \
            fit_power_law(cds, DF, x_min=x_min, linear_fit=True)

    #popt_cut_ed, pcov_cut_ed, error_cut_ed = \
    #        fit_cutoff(eds, DF, x_min=x_min)

    #popt_cut_cd, pcov_cut_cd, error_cut_cd = \
    #        fit_cutoff(cds, DF, x_min=x_min)
    
    print ""
    print "Exponential Fit:"
    print "edge distance:"
    print "Params:", popt_exp_ed
    print "Std:", np.sqrt(np.diag(pcov_exp_ed))
    print "Error:", error_exp_ed

    print "cycle distance:"
    print "Params:", popt_exp_cd
    print "Std:", np.sqrt(np.diag(pcov_exp_cd))
    print "Error:", error_exp_cd
    
    print ""
    print "Power Law Fit:"
    print "edge distance:"
    print "Params:", popt_pwl_ed
    print "Std:", np.sqrt(np.diag(pcov_pwl_ed))
    print "Error:", error_pwl_ed

    print "cycle distance:"
    print "Params:", popt_pwl_cd
    print "Std:", np.sqrt(np.diag(pcov_pwl_cd))
    print "Error:", error_pwl_cd

    #print ""
    #print "Cutoff Power Law Fit:"
    #print "edge distance:"
    #print "Params:", popt_cut_ed
    #print "Std:", np.sqrt(np.diag(pcov_cut_ed))
    #print "Error:", error_cut_ed

    #print "cycle distance:"
    #print "Params:", popt_cut_cd
    #print "Std:", np.sqrt(np.diag(pcov_cut_cd))
    #print "Error:", error_cut_cd
    
    if plot:
        f, (ax1, ax2) = plt.subplots(2)
        
        ax1.plot(eds, DF, 'o')

        x = linspace(x_min, np.max(eds), 1000)
        ax1.plot(x, exp_decay(x, *popt_exp_ed), label='exponential')
        ax1.plot(x, power_law(x, *popt_pwl_ed), label='power law')
        #ax1.plot(x, cutoff(x, *popt_cut_ed), label='cutoff power law')
        ax1.legend(loc='best')

        ax1.set_xlabel('edge distance')
        ax1.set_ylabel('normalized $\Delta F$')

        ax2.plot(cds, DF, 'o')

        x = linspace(x_min, np.max(cds), 1000)
        ax2.plot(x, exp_decay(x, *popt_exp_cd), label='exponential')
        ax2.plot(x, power_law(x, *popt_pwl_cd), label='power law')
        #ax2.plot(x, cutoff(x, *popt_cut_cd), label='cutoff power law')
        ax2.legend(loc='best')
        
        ax2.set_ylabel('normalized $\Delta F$')
        ax2.set_xlabel('cycle distance')
        
        plt.tight_layout()
        plt.show()

    mi_edge = correlations.mutual_information2(eds, DF)
    mi_cycle = correlations.mutual_information2(cds, DF)
    
    if savename != None:
        sn = os.path.splitext(savename)
        savename = sn[0] + '_' + name + sn[1]

        with open(savename, 'a') as f:
            print >>f, ("\t".join(11*['{}'])).format(name, 
                    popt_exp_ed[1], pcov_exp_ed[0],
                    popt_exp_cd[1], pcov_exp_cd[0],
                    popt_pwl_ed[1], pcov_pwl_ed[0],
                    popt_pwl_cd[1], pcov_pwl_cd[0], 
                    mi_edge, mi_cycle)


def fit_data_means(eds, DF_eds, cds, DF_cds, name, 
        x_min=1, savename=False, plot=True):
    """ Fit power law and exponential functions to data.
    print errors and parameters
    """
    # fit power law and exponential to data
    popt_exp_ed, pcov_exp_ed, error_exp_ed = \
            fit_exponential(eds, DF_eds, x_min=x_min, linear_fit=True)

    popt_exp_cd, pcov_exp_cd, error_exp_cd = \
            fit_exponential(cds, DF_cds, x_min=x_min, linear_fit=True)

    popt_pwl_ed, pcov_pwl_ed, error_pwl_ed = \
            fit_power_law(eds, DF_eds, x_min=x_min, linear_fit=True)

    popt_pwl_cd, pcov_pwl_cd, error_pwl_cd = \
            fit_power_law(cds, DF_cds, x_min=x_min, linear_fit=True)

    #popt_cut_ed, pcov_cut_ed, error_cut_ed = \
    #        fit_cutoff(eds, DF_eds, x_min=x_min)

    #popt_cut_cd, pcov_cut_cd, error_cut_cd = \
    #        fit_cutoff(cds, DF_cds, x_min=x_min)
    
    print ""
    print "Exponential Fit:"
    print "edge distance:"
    print "Params:", popt_exp_ed
    print "Std:", np.sqrt(np.diag(pcov_exp_ed))
    print "Error:", error_exp_ed

    print "cycle distance:"
    print "Params:", popt_exp_cd
    print "Std:", np.sqrt(np.diag(pcov_exp_cd))
    print "Error:", error_exp_cd
    
    print ""
    print "Power Law Fit:"
    print "edge distance:"
    print "Params:", popt_pwl_ed
    print "Std:", np.sqrt(np.diag(pcov_pwl_ed))
    print "Error:", error_pwl_ed

    print "cycle distance:"
    print "Params:", popt_pwl_cd
    print "Std:", np.sqrt(np.diag(pcov_pwl_cd))
    print "Error:", error_pwl_cd

    #print ""
    #print "Cutoff Power Law Fit:"
    #print "edge distance:"
    #print "Params:", popt_cut_ed
    #print "Std:", np.sqrt(np.diag(pcov_cut_ed))
    #print "Error:", error_cut_ed

    #print "cycle distance:"
    #print "Params:", popt_cut_cd
    #print "Std:", np.sqrt(np.diag(pcov_cut_cd))
    #print "Error:", error_cut_cd
    
    if plot:
        f, (ax1, ax2) = plt.subplots(2)
        
        ax1.plot(eds, DF_eds, 'o')

        x = linspace(x_min, np.max(eds), 1000)
        ax1.plot(x, exp_decay(x, *popt_exp_ed), label='exponential')
        ax1.plot(x, power_law(x, *popt_pwl_ed), label='power law')
        #ax1.plot(x, cutoff(x, *popt_cut_ed), label='cutoff power law')
        ax1.legend(loc='best')

        ax1.set_xlabel('edge distance')
        ax1.set_ylabel('normalized $\Delta F$')

        ax2.plot(cds, DF_cds, 'o')

        x = linspace(x_min, np.max(cds), 1000)
        ax2.plot(x, exp_decay(x, *popt_exp_cd), label='exponential')
        ax2.plot(x, power_law(x, *popt_pwl_cd), label='power law')
        #ax2.plot(x, cutoff(x, *popt_cut_cd), label='cutoff power law')
        ax2.legend(loc='best')
        
        ax2.set_ylabel('normalized $\Delta F$')
        ax2.set_xlabel('cycle distance')
        
        plt.tight_layout()
        plt.show()

    if savename != None:
        sn = os.path.splitext(savename)
        savename = sn[0] + '_' + name + sn[1]

        with open(savename, 'a') as f:
            print >>f, ("\t".join(9*['{}'])).format(name,
                    popt_exp_ed[1], pcov_exp_ed[0],
                    popt_exp_cd[1], pcov_exp_cd[0],
                    popt_pwl_ed[1], pcov_pwl_ed[0],
                    popt_pwl_cd[1], pcov_pwl_cd[0])


def grid_graph(n_grid, random_weights=True):
    """ Make a rectangular grid graph to test decay fits
    """
    G = nx.grid_2d_graph(n_grid, n_grid)

    for n, d in G.nodes_iter(data=True):
        d['x'] = n[0] + 0.01*np.random.random()
        d['y'] = n[1] + 0.01*np.random.random()

    for u, v, d in G.edges_iter(data=True):
        if random_weights:
            d['conductivity'] = np.random.random()
        else:
            d['conductivity'] = 1.0

        d['weight'] = 1.0

    graph_fname = "grid_" + str(n_grid)
    GG = nx.convert_node_labels_to_integers(G)

    return graph_fname, GG

class NetworkLoader():
    """ Loads a network and applies some sanitizers
    """
    def __init__(self, graph_fname, random_edge=False, shift=0,
            homogeneous=False, cache=False):
        self.graph_fname = graph_fname
        self.name = os.path.splitext(os.path.split(graph_fname)[-1])[0]
        self.homogeneous = homogeneous
        self.cache = cache
        
        print "graph name:"
        print self.name
        
        if graph_fname == 'grid':
            print "Generating grid"
            self.graph_fname, self.G = grid_graph(n)
            self.name = self.graph_fname
        elif graph_fname[-3:] == 'dot':
            print "Loading dot file"

            G = nx.Graph(nx.read_dot(self.graph_fname))
            GG = nx.connected_component_subgraphs(G)[0]

            self.G = nx.convert_node_labels_to_integers(GG)

            # add weights
            for u, v, d in self.G.edges_iter(data=True):
                d['conductivity'] = 1
                d['weight'] = 1

        else:
            # Load graph
            print "Loading graph file"
            G = nx.read_gpickle(self.graph_fname)

            print "Removing intersecting edges"
            print len(remove_intersecting_edges(G))
            
            G = nx.connected_component_subgraphs(G)[0]

            G = prune_graph(G)

            print "Applying collinearity workaround"
            print len(apply_workaround(G, thr=1e-3))
            
            print "Removing bad edges"
            remove_bad_edges(G)
            
            print "Cycle pruning"
            pruned = prune_graph(G)

            graph = nx.connected_component_subgraphs(pruned)[0]
            self.G = nx.convert_node_labels_to_integers(graph)

        if homogeneous:
            # all conductivities are equal to 10
            for u, v, d in self.G.edges_iter(data=True):
                d['conductivity'] = 1
                d['weight'] = 1
        
        ## make grid graph
        #self.graph_fname, self.G = grid_graph(400)
        #self.name = self.graph_fname
        self.init_edge(random_edge, shift=shift)

        self.calc_edge()

    def init_edge(self, random_edge, shift=0):
        # initialization
        #G = random_loopy_network_holes(200, n_holes=0)
        if random_edge:
            print "Finding random edge"
            n = self.G.number_of_edges()
            i = np.random.randint(n)
            self.edge = self.G.edges()[i]
        else:
            print "Finding center edge"
            self.edge = get_center_edge(self.G, weighted=True, 
                    shift=shift)

        self.G.graph['name'] = self.name

        self.j = self.G.edges().index(self.edge)
    
    def calc_edge(self):
        pass

class EdgeFlowAnalyzer(NetworkLoader):
    """ Contains functions that make figures. Can re-use information
    """
    def calc_edge(self):
        DeltaF_uw, DeltaF_wt, conds = calculate_cache_DeltaF(self.G, 
                self.edge, self.j, self.graph_fname, cache=self.cache)
        
        self.DeltaF_uw, self.DeltaF_wt = DeltaF_uw, DeltaF_wt
        self.conds = conds

    def threshold_affected_edges(self, thr=1e-3):
        """ Load a real leaf network, plot
        perturbation flow for full and homogeneous network.
        Return the network graph as well as the perturbation vectors
        for subsequent figures.
        """
        G = self.G
        edge = self.edge
        j = self.j
        DeltaF_uw, DeltaF_wt = self.DeltaF_uw, self.DeltaF_wt

        print "Plotting"
        cmap = plt.get_cmap('jet')
        norm = colors.LogNorm(vmin=1e-8, vmax=1)
        scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        
        # unweighted
        DeltaF_uw_normalized = np.abs(DeltaF_uw/DeltaF_uw[j])
        scalar_map.set_array(DeltaF_uw_normalized)
        colors_uw = list(enumerate(scalar_map.to_rgba(DeltaF_uw_normalized)))

        plot.draw_leaf(G, ax=ax1, mark_edges=colors_uw, fixed_width=True)
        
        # threshold for perturbation strength plot
        print "Unweighted"
        print "Density:", affected_edge_density(G, DeltaF_uw_normalized, thr,
                ax=ax1)

        # all points above theshold
        above_thr = np.where(DeltaF_uw_normalized > thr)[0]
        edges_above_thr = [e for i, e in enumerate(G.edges_iter())
                if i in above_thr]
        plot.draw_leaf(G, ax=ax1, edge_list=edges_above_thr, color='k')

        #print "Edges above threshold", len(above_thr)
        #print "Power dissipated by perturbation", np.linalg.norm(DeltaF_uw_normalized)

        ax1.set_title('homogeneous network')
        ax1.autoscale(tight=True)
        
        # weighted
        DeltaF_wt_normalized = np.abs(DeltaF_wt/DeltaF_wt[j])
        colors_wt = list(enumerate(scalar_map.to_rgba(DeltaF_wt_normalized)))

        plot.draw_leaf(G, ax=ax2, mark_edges=colors_wt, fixed_width=True)

        print "Weighted"
        print "Density:", affected_edge_density(G, DeltaF_wt_normalized, thr,
                ax=ax2)

        # all points above theshold
        above_thr = np.where(DeltaF_wt_normalized > thr)[0]
        edges_above_thr = [e for i, e in enumerate(G.edges_iter())
                if i in above_thr]
        plot.draw_leaf(G, ax=ax2, edge_list=edges_above_thr, color='k')
        
        #print "Edges above threshold:", len(above_thr)

        #conds = np.mean([d['conductivity']**4/d['weight']
        #    for u, v, d in G.edges_iter(data=True)])
        #conds /= conds.mean()
        #print "Power dissipated by perturbation", np.linalg.norm(DeltaF_wt_normalized/np.sqrt(conds))

        ax2.set_title('hierarchical network')
        ax2.autoscale(tight=True)

        # add colorbar
        f.colorbar(scalar_map, ax=ax1)
        f.colorbar(scalar_map, ax=ax2)

        # plot affected edge densities
        ax3.set_xlabel('threshold perturbation strength')
        ax3.set_ylabel('edge density in affected region')

        thresholds = np.logspace(-1, -8, 25)
        edge_dens_uw = np.array([
            affected_edge_density(G, DeltaF_uw_normalized, t) 
            for t in thresholds])

        edge_dens_wt = np.array([
            affected_edge_density(G, DeltaF_wt_normalized, t) 
            for t in thresholds])
        
        ax3.semilogx(thresholds, edge_dens_uw, label='homogeneous')
        ax3.semilogx(thresholds, edge_dens_wt, label='hierarchical')

        ax3.legend(loc='best')

        # plot total edge densities
        ax4.set_xlabel('threshold perturbation strength')
        ax4.set_ylabel('total edge density')

        tot_edge_dens_uw = np.array([
            total_edge_density(G, DeltaF_uw_normalized, t) 
            for t in thresholds])

        tot_edge_dens_wt = np.array([
            total_edge_density(G, DeltaF_wt_normalized, t) 
            for t in thresholds])
        
        ax4.semilogx(thresholds, tot_edge_dens_uw, label='homogeneous')
        ax4.semilogx(thresholds, tot_edge_dens_wt, label='hierarchical')

        ax4.legend(loc='best')
        
        plt.tight_layout()
        plt.savefig('plots/figure1_' + self.name +
                '.png', dpi=300, bbox_inches='tight')
        #plt.show()

    def scaling_single_edge_no_embedding(self, plot=False):
        """ Look at the scaling behavior of the graph if there
        is no embedding and no weights (i.e. there is no well-defined
        cycle distance)
        """
        G = self.G
        edge = self.edge
        j = self.j

        # check if computation result is cached
        cache_file = 'cache/' + os.path.basename(self.graph_fname) + '_edge_{}_distances_cache.npz'.format(j)
        if self.cache and os.path.exists(cache_file):
            print "Computation results found in cache, using cached data."
            cache = np.load(cache_file)
            edge_dists = cache['edge_dists']
        else:
            edge_dists = shortest_path_lengths_between_edges(G, edge)
            
            # save into cache
            np.savez(cache_file, 
                    edge_dists=edge_dists)

        # normalize perturbation currents
        DeltaF_uw_norm = np.abs(self.DeltaF_uw/self.DeltaF_uw[j])
        #DeltaF_wt_norm = np.abs(self.DeltaF_wt/self.DeltaF_wt[j])

        # mean delta f at given distance
        eds, mean_DF_uw_ed, std_DF_uw_ed = distance_means(edge_dists,
                DeltaF_uw_norm)

        #eds, mean_DF_wt_ed, std_DF_wt_ed = distance_means(edge_dists,
        #        DeltaF_wt_norm)

        if plot:
            # plot
            f, (ax1, ax2) = plt.subplots(2)
            
            # plot DF against cycle distance
            #ax1.loglog(eds, mean_DF_uw_ed, '-', label='homogeneous')
            #ax1.loglog(eds, mean_DF_wt_ed, '-', label='hierarchical')

            ax1.loglog(edge_dists, DeltaF_uw_norm, 'o', label='hom')

            ax1.set_xlabel('edge distance $d_e$')
            ax1.set_ylabel('$\langle\Delta F_e/\Delta F_{e_0}\\rangle$')
            ax1.legend(loc='best')
           

            # plot DF against cycle distance
            ax2.semilogy(edge_dists, DeltaF_uw_norm, 'o', label='hom')
            #ax2.semilogy(eds, mean_DF_wt_ed, '-', label='hierarchical')

            ax2.set_xlabel('edge distance $d_e$')
            ax2.set_ylabel('$\langle\Delta F_e/\Delta F_{e_0}\\rangle$')
            ax2.legend(loc='best')
           

            plt.tight_layout()
            #plt.savefig('plots/figure3_' + self.name +
            #        '.png', dpi=300, bbox_inches='tight')

            plt.show()

    def scaling_single_edge(self, save=False, plot=False,
            cond_normalize=False):
        """ We show power law decay when damaging a single edge,
        perturbation data of which was taken from figure 1
        """
        G = self.G
        edge = self.edge
        j = self.j
        # distances between edge and all other edges
            
        # check if computation result is cached
        cache_file = 'cache/' + os.path.basename(self.graph_fname) + '_edge_{}_distances_cache.npz'.format(j)
        if os.path.exists(cache_file):
            print "Computation results found in cache, using cached data."
            cache = np.load(cache_file)
            edge_dists = cache['edge_dists']
            cycle_dists = cache['cycle_dists']
        else:
            edge_dists = shortest_path_lengths_between_edges(G, edge)
            cycle_dists = shortest_cycle_path_lengths_between_edges(G, edge,
                    self.graph_fname)
            
            # save into cache
            np.savez(cache_file, 
                    edge_dists=edge_dists, cycle_dists=cycle_dists)
        
        # normalize by conductivities to get bare decay behavior
        if cond_normalize:
            print "Normalizing weighted flows by conductivities"
            DeltaF_wt = self.DeltaF_wt / self.conds

            pert_term = '$\langle\Delta F_e/(K_e \Delta F_{e_0}\\rangle$'
        else:
            DeltaF_wt = self.DeltaF_wt

            pert_term = '$\langle\Delta F_e/\Delta F_{e_0}\\rangle$'

        # normalize perturbation currents
        DeltaF_uw_norm = np.abs(self.DeltaF_uw/self.DeltaF_uw[j])
        DeltaF_wt_norm = np.abs(DeltaF_wt/DeltaF_wt[j])

        # mean delta f at given distance
        eds, mean_DF_uw_ed, std_DF_uw_ed = distance_means(edge_dists,
                DeltaF_uw_norm)

        cds, mean_DF_uw_cd, std_DF_uw_cd = distance_means(cycle_dists,
                DeltaF_uw_norm)
     
        eds, mean_DF_wt_ed, std_DF_wt_ed = distance_means(edge_dists,
                DeltaF_wt_norm)

        cds, mean_DF_wt_cd, std_DF_wt_cd = distance_means(cycle_dists,
                DeltaF_wt_norm)
        
        if plot:
            # plot
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
            
            # plot DF against cycle distance
            ax1.loglog(eds, mean_DF_uw_ed, '-', label='homogeneous')
            ax1.loglog(eds, mean_DF_wt_ed, '-', label='hierarchical')

            ax1.set_xlabel('edge distance $d_e$')
            ax1.set_ylabel(pert_term)
            ax1.legend(loc='best')
           
            ax2.loglog(cds, mean_DF_uw_cd, '-', label='homogeneous')
            ax2.loglog(cds, mean_DF_wt_cd, '-', label='hierarchical')

            ax2.set_xlabel('cycle distance $d_c$')
            ax2.set_ylabel(pert_term)
            ax2.legend(loc='best')

            # plot DF against cycle distance
            ax3.semilogy(eds, mean_DF_uw_ed, '-', label='homogeneous')
            ax3.semilogy(eds, mean_DF_wt_ed, '-', label='hierarchical')

            ax3.set_xlabel('edge distance $d_e$')
            ax3.set_ylabel(pert_term)
            ax3.legend(loc='best')
           
            ax4.semilogy(cds, mean_DF_uw_cd, '-', label='homogeneous')
            ax4.semilogy(cds, mean_DF_wt_cd, '-', label='hierarchical')

            ax4.set_xlabel('cycle distance $d_c$')
            ax4.set_ylabel(pert_term)
            ax4.legend(loc='best')

            plt.tight_layout()
            plt.savefig('plots/figure3_' + self.name +
                    '.png', dpi=300, bbox_inches='tight')

        #edge_dists += 1
        #cycle_dists += 1

        #eds = np.asarray(eds) + 1
        #cds = np.asarray(cds) + 1
        
        if save:
            print "---"
            print "Unweighted Fits"
            print "---"
            fit_data(edge_dists, cycle_dists, DeltaF_uw_norm, self.name,
                    x_min=1, savename='data/scaling_full_data_uw.txt',
                    plot=False)

            print "---"
            print "Weighted Fits"
            print "---"
            fit_data(edge_dists, cycle_dists, DeltaF_wt_norm, self.name,
                    x_min=1, savename='data/scaling_full_data_wt.txt',
                    plot=False)

            print "---"
            print "Unweighted Fits (means)"
            print "---"
            fit_data_means(eds, mean_DF_uw_ed, cds, mean_DF_uw_cd, self.name,
                    x_min=1, savename='data/scaling_means_uw.txt',
                    plot=False)

            print "---"
            print "Weighted Fits (means)"
            print "---"
            fit_data_means(eds, mean_DF_wt_ed, cds, mean_DF_wt_cd, self.name,
                    x_min=1, savename='data/scaling_means_wt.txt',
                    plot=False)

class CycleFlowAnalyzer(NetworkLoader):
    """ This class calculates the cycle flows for a given graph
    and perturbed edge and allows to calculate some measures.
    """
    def calc_edge(self):
        edges_to_perturb = [self.edge]
        
        # caculate cycle flows
        r = simulate_cycle_flow(self.G, edges_to_perturb=edges_to_perturb)

        self.cycles, self.perturbation_currents, self.G_pruned, \
                self.G_dual, self.all_pert_cycles, \
                self.bordering_cycles, self.edges_to_pert = r
        
        # calculate actual flow perturbations
        self.DeltaF_abs = DeltaF_from_cycle_flows(self.G_pruned,
                self.perturbation_currents[0], self.bordering_cycles,
                self.cycles)
        
        # normalize by conductivity
        conds = np.array([d['conductivity']**4/d['weight'] 
                for u, v, d in self.G_pruned.edges_iter(data=True)])

        self.DeltaF_abs_normalized = self.DeltaF_abs / conds

    def find_domain_boundary(self):
        """ Detect the domain boundary induced by the cycle flows
        """
        boundary_edges = []
        boundary_edge_inds = []
        boundary_neighborhoods = []
        boundary_neighborhood_inds = []

        fs = self.perturbation_currents[0]

        edge_list = self.G_pruned.edges()

        for i, e in enumerate(edge_list):
            bord = self.bordering_cycles[tuple(sorted(e))]

            if len(bord) != 2:
                #print "# bordering cycles:", len(bord), "!?"
                continue

            c1, c2 = self.cycles[bord[0]], self.cycles[bord[1]]
            f1, f2 = fs[bord[0]], fs[bord[1]]
            
            if np.sign(c1.orientation()*f1) \
                    != np.sign(c2.orientation()*f2):
                boundary_edges.append(e)
                boundary_edge_inds.append(i)

                local_neighborhood_1 = [g for g in c1.edges
                        if tuple(sorted(g)) != e]
                local_neighborhood_2 = [g for g in c2.edges
                        if tuple(sorted(g)) != e]

                neighs = local_neighborhood_1 + local_neighborhood_2
                neigh_inds = [edge_list.index(tuple(sorted(n)))
                        for n in neighs]
                
                boundary_neighborhoods.append(neighs)
                boundary_neighborhood_inds.append(neigh_inds)

        # edge tuples
        self.domain_boundary = boundary_edges

        # indices of the edges in the edge list
        self.domain_boundary_inds = boundary_edge_inds
        
        # edges that are in the same cycle as the boundary edge
        self.boundary_neighborhoods = boundary_neighborhoods
        self.boundary_neighborhood_inds = boundary_neighborhood_inds

    def plot_domain_boundary_flows(self):
        """ Compare flows on the domain boundary with rest.
        """
        flows_on_boundary = [self.DeltaF_abs[i]
                for i in self.domain_boundary_inds]

        neighbor_flows = [[self.DeltaF_abs[j] for j in
            self.boundary_neighborhood_inds[i]]
            for i, e in enumerate(self.domain_boundary)]
        
        rest_flows = [df for i, df in enumerate(self.DeltaF_abs)
                if not i in self.domain_boundary_inds]

        neighbor_ratios = np.array([boundary_flow/neighbor_flow
                for i, boundary_flow in enumerate(flows_on_boundary)
            for neighbor_flow in neighbor_flows[i]])

        plt.figure()
        plt.hist(neighbor_ratios, range=(0, 10), bins=50)
        plt.axvline(1, lw=2, ls='--', color='k')
        plt.xlabel('$\Delta F_\mathrm{boundary}/\Delta F_\mathrm{neighbor}$')
        plt.ylabel('frequency')

        print "relative number larger flows:", (neighbor_ratios >= 1).sum()/float(len(neighbor_ratios))
            
def analyze_scaling_Bronx():
    """ Analyze the scaling behavior of the perturbation flows
    in the Bronx leaf network database.
    """
    graphs = glob('/scratch.local/henrik/BronxGraphsFinal/BronxA/*.gpickle')
    
    # remove one which doesn't work
    #graphs = [g for g in graphs if '042' in g]
     
    #for graph_fname in graphs:
    #    analyze_fname(graph_fname)

    Parallel(n_jobs=5)(delayed(analyze_fname)(graph_fname) 
            for graph_fname in graphs)

def analyze_fname(graph_fname):
    fm = EdgeFlowAnalyzer(graph_fname, random_edge=True)

    # analyze 200 random edges
    for i in xrange(100):
        fm.scaling_single_edge(save=True, plot=False,
                cond_normalize=True)
        fm.init_edge(True)
        fm.calc_edge()

def analyze_cycle_flows():
    graph_fname = 'graphs/9797_9797_graph.gpickle'
    #graph_fname = 'graphs/2000_2000.gpickle'
    #graph_fname = 'graphs/500_500.gpickle'
    #graph_fname = 'graphs/mapleb_80_graph.gpickle'
    #graph_fname = 'graphs/BronxA_001_binary_graph.gpickle'
    #graph_fname = 'graphs/BronxA_004_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/BronxA_009_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/BronxA_015_a_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/BronxA_030_a_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/BronxB_005_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/BronxB_015_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/net_final_ds14.dot'
    #graph_fname = 'grid'

    ## Analyze the domain boundary 
    cfa = CycleFlowAnalyzer(graph_fname, random_edge=False, shift=0,
            homogeneous=False)
    cfa.find_domain_boundary()

    #plt.figure()

    #plot.draw_leaf(cfa.G_pruned)
    
    plot_DeltaF_in_leaf(cfa.DeltaF_abs_normalized, 
            cfa.G_pruned, fixed_width=True)

    plot.draw_leaf(cfa.G_pruned, edge_list=cfa.domain_boundary, color='r',
            fixed_width=True, width=3)

    plot.draw_leaf(cfa.G_pruned, edge_list=[cfa.edge], color='b',
            fixed_width=True, width=4)
    
    #plot_cycle_currents(cfa.G_pruned, cfa.cycles,
    #        cfa.perturbation_currents[0], cfa.all_pert_cycles[0])
    
    cfa.plot_domain_boundary_flows()

    plt.show()

def analyze_scaling_single():
    """ Analyze a single graph's scaling behavior
    """
    #graph_fname = 'graphs/9797_9797_graph.gpickle'
    #graph_fname = 'graphs/2000_2000.gpickle'
    #graph_fname = 'graphs/500_500.gpickle'
    #graph_fname = 'graphs/mapleb_80_graph.gpickle'
    graph_fname = 'graphs/BronxA_001_binary_graph.gpickle'
    #graph_fname = 'graphs/BronxA_004_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/BronxA_009_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/BronxA_015_a_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/BronxA_030_a_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/BronxB_005_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/BronxB_015_binary_corrected_graph.gpickle'
    #graph_fname = 'graphs/net_final_ds14.dot'
    #graph_fname = 'grid'
    efa = EdgeFlowAnalyzer(graph_fname, random_edge=True)
    
    efa.scaling_single_edge(save=False, plot=True, 
            cond_normalize=True)
    
    norm = efa.DeltaF_wt/efa.conds
    norm /= norm[efa.j]

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    plot_DeltaF_in_leaf(np.abs(norm), efa.G, fixed_width=True, ax=ax1)
    ax1.set_title('$\Delta F_e/(K_e \Delta F_{e_0})$ (hierarchical)')

    plot_DeltaF_in_leaf(np.abs(efa.DeltaF_wt), efa.G, 
            fixed_width=True, ax=ax2)
    ax2.set_title('$\Delta F_e/\Delta F_{e_0}$ (hierarchical)')

    plot_DeltaF_in_leaf(np.abs(efa.DeltaF_uw), efa.G, 
            fixed_width=True, ax=ax3)
    ax3.set_title('$\Delta F_e/\Delta F_{e_0}$ (homogeneous)')

    plt.show()


if __name__ == '__main__':
    sns.set(style='ticks', font_scale=1.)
    params = {'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)
    
    #real_graph_analysis()
    #raw_input()
    #lattice_analysis()
    #random_graph_analysis()
    #np.random.seed(15)
    
    #analyze_scaling_single()

    analyze_scaling_Bronx()
    #analyze_cycle_flows()
