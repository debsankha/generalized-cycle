#!/usr/bin/env python

"""
    cycle_flows.py

    Contains functions that numerically simulate cycle flows
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
import plot
import correlations

def get_center_edge(G, shift=0, weighted=False):
    """ return the edge closest to the center of the graph.
    If weighted, the distances are weighted by 1/edge thickness
    to favor thick edges.
    """
    coords = np.array([((u, v), 
        0.5*(G.node[u]['x'] + G.node[v]['x']), 
        0.5*(G.node[u]['y'] + G.node[v]['y'])) 
        for u, v in G.edges_iter()])

    es, xs, ys = zip(*coords)

    xs = np.array(xs)
    ys = np.array(ys)

    xmean = xs.mean()
    ymean = ys.mean()

    if weighted:
        widths = np.array([d['conductivity']**2
            for u, v, d in G.edges_iter(data=True)])
        dists_squared = ((xs - xmean)**2 + (ys - ymean + shift)**2)/widths
    else:
        dists_squared = (xs - xmean)**2 + (ys - ymean + shift)**2

    closest_node = np.argmin(dists_squared)

    return tuple(sorted(es[closest_node]))

def simulate_cycle_flow(G, edges_to_perturb=[]):
    """ Simulate cycle flow from damaging the edge closest to
    given (x, y) coordinate in graph G.
    return a list of cycles and associated cycle flow strengths,
    the cycle pruned graph and the two cycles that have been perturbed
    """
    G_pruned = G
    # detect cycles
    try:
        cycles = shortest_cycles(G_pruned, raise_exception=True)
    except:
        sys.exit(0)

    boundary = outer_loop(G_pruned, cycles)
    boundary_ind = cycles.index(boundary)

    neighbor_cycles, edges = find_neighbor_cycles(G_pruned, cycles)
    neighbor_cycles = list(neighbor_cycles)
    
    # remove neighborhood relationships with boundary and update indices
    update_ind = lambda i: i if i < boundary_ind else i - 1

    neighbor_cycles = [(update_ind(u), update_ind(v)) 
            for u, v in neighbor_cycles 
            if u != boundary_ind and v != boundary_ind]

    del cycles[boundary_ind]
    
    # update cycle indices
    edges_new = {}
    for e, c in edges.iteritems():
        edges_new[e] = [update_ind(cc) for cc in c
                if cc != boundary_ind]
    bordering_cycles = edges_new

    # construct cycle dual graph
    G_dual = nx.Graph()
    G_dual.add_edges_from(neighbor_cycles)
    
    # Laplacian Matrix
    L = cycle_laplacian(G_pruned, cycles, neighbor_cycles)
            
    # take a number of random edges
    u, v = edges_to_perturb[0]

    perturbation_currents = []
    all_pert_cycles = []
   
    for pedge in edges_to_perturb:
        print "Perturbed edge:", pedge
        # rhs with perturbation
        q = np.zeros(L.shape[0])
    
        perturbed_cycles = bordering_cycles[pedge]

        print "Bordering cycles:", perturbed_cycles
        if len(perturbed_cycles) == 0:
            continue

        q[perturbed_cycles[0]] = \
                1.*cycles[perturbed_cycles[0]].orientation()

        q[perturbed_cycles[1]] = \
                -1.*cycles[perturbed_cycles[1]].orientation()

        #perturbation_currents = spsolve(L, q, use_umfpack=False)
        perturbation_currents.append(spsolve(L, q, use_umfpack=False))
        #perturbation_currents.append(np.linalg.solve(L.todense(), q))
        all_pert_cycles.append(perturbed_cycles)

    return cycles, perturbation_currents, G_pruned, G_dual, \
            all_pert_cycles, bordering_cycles, edges_to_perturb

def simulate_edge_flows2(G, weighted=False):
    """ Like simulate_edge_flows but uses QR decomposition
    instead of matrix inversion
    """
    if weighted:
        wts = []
        for u, v, d in G.edges_iter(data=True):
            d['cond'] = np.sqrt(d['conductivity']**4/d['weight'])
            wts.append(d['cond'])

        wts = np.array(wts)
        wts_sqr = wts**2
    else:
        for u, v, d in G.edges_iter(data=True):
            d['cond'] = 1.
    
    # construct reduced incidence matrix
    d1 = np.array(nx.incidence_matrix(G, oriented=True, 
        weight='cond'))[1:,:]
    
    Q, _ = np.linalg.qr(d1.T)
    A = np.dot(Q, Q.T)
    
    if weighted:
        A = (wts)*(A*wts.T).T

    diag_inds = np.diag_indices(A.shape[0])

    if weighted:
        A[diag_inds] = wts_sqr - A[diag_inds]
    else:
        A[diag_inds] = 1 - A[diag_inds]
    return A

def simulate_single_edge_flow(G, edge, weighted=False, sparse=False):
    """ Calculate the same as simulate_edge_flows but for one
    single damaged edge.
    Uses preconditioning
    """
    # conductivity weights
    wts = []
    if weighted:
        for u, v, d in G.edges_iter(data=True):
            d['cond'] = np.sqrt(d['conductivity']**4/d['weight'])
            wts.append(d['cond'])
    else:
        for u, v, d in G.edges_iter(data=True):
            d['cond'] = 1.
            wts.append(1.)
    
    wts = np.array(wts)
    wts_sqr = wts**2

    edge_ind = G.edges().index(edge)
    
    # least squares method (can be sparsified!)
    d1 = sparse_incidence_matrix(G, oriented=True, weight='cond')

    delete_row_lil(d1, 0)
    d1t = d1.transpose()

    # rhs
    y = np.zeros(d1t.shape[0])
    y[edge_ind] = wts[edge_ind]

    #ret = scipy.sparse.linalg.lsqr(d1t, y)
    
    if sparse:
        # precondition
        d = 1./np.array(np.abs(d1t).sum(axis=0))[0]
        D = scipy.sparse.diags([d], [0])
        d1t = d1t.dot(D)
        
        # solve
        ret = lsmr(d1t, y, atol=1e-8, btol=1e-8,
                show=False, maxiter=10*d1t.shape[1])
    else:
        d1t = np.array(d1t.todense())
        ret = np.linalg.lstsq(d1t, y)
        
    x = ret[0]
    DeltaF = d1t.dot(x)*wts/wts_sqr[edge_ind]    

    # correct DeltaF at the damaged edge
    DeltaF[edge_ind] = DeltaF[edge_ind] - 1
    #DeltaF[edge_ind] = 1 - DeltaF[edge_ind]
    
    return DeltaF, wts_sqr

def simulate_single_edge_flow_old(G, edge, weighted=False):
    """ Use the old formula (pressures) to get the perturbation.
    """
    # conductivity weights
    wts = []
    if weighted:
        for u, v, d in G.edges_iter(data=True):
            d['cond'] = d['conductivity']**4/d['weight']
            wts.append(d['cond'])
    else:
        for u, v, d in G.edges_iter(data=True):
            d['cond'] = 1.
            wts.append(1.)

    L = np.array(sparse_laplacian(G, weight='cond').todense())

    Li = np.linalg.pinv(L)
    u, v = edge

    DeltaF = np.array([Li[w,u] - Li[w,v] - Li[x,u] + Li[x,v]
            for w, x in G.edges_iter()])*wts

    j = G.edges().index(edge)

    DeltaF[j] -= 1

    return DeltaF

def simulate_edge_flows(G, weighted=False):
    """ Simulate the edge flows to first order in kappa.
    Return a matrix in which each column contains DeltaF
    for the corresponding perturbed edge in G.edges_iter().
    Actual perturbation current is proportional to F^0_e,
    but that information is not in this matrix.
    The result is the symmetric matrix B. If the network is
    weighted it is necessary to divide either rows or columns
    by K_e^2 in order to get the factor proportional to
    \kappa F_e^0 to order O(kappa).
    """
    # conductivity weights
    if weighted:
        for u, v, d in G.edges_iter(data=True):
            d['cond'] = d['conductivity']**4/d['weight']
    else:
        for u, v, d in G.edges_iter(data=True):
            d['cond'] = 1.
    
    L = sparse_laplacian(G, weight='cond').tolil()
    
    # newfangled method
    #t0 = time()
    Li = np.linalg.pinv(L.todense())
    d1 = np.array(nx.incidence_matrix(G, oriented=True))
    A = np.array(np.dot(np.dot(d1.T, Li), d1))
            
    if weighted:
        weights = np.array([d['cond'] 
            for u, v, d in G.edges_iter(data=True)])
        A = weights*(A*weights.T).T

    #Re = A[np.diag_indices(A.shape[0])]/weights
    #Re2 = np.array([Li[u,u] - 2*Li[u,v] + Li[v,v]
    #    for u, v in G.edges_iter()])
    #print np.allclose(Re, Re2)
    
    # first order in kappa
    diag_inds = np.diag_indices(A.shape[0])

    if weighted:
        A[diag_inds] = weights - A[diag_inds]
    else:
        A[diag_inds] = 1 - A[diag_inds]

    return A

def simulate_edge_flows2(G, weighted=False):
    """ Like simulate_edge_flows but uses QR decomposition
    instead of matrix inversion
    """
    if weighted:
        wts = []
        for u, v, d in G.edges_iter(data=True):
            d['cond'] = np.sqrt(d['conductivity']**4/d['weight'])
            wts.append(d['cond'])

        wts = np.array(wts)
        wts_sqr = wts**2
    else:
        for u, v, d in G.edges_iter(data=True):
            d['cond'] = 1.
    
    # construct reduced incidence matrix
    d1 = np.array(nx.incidence_matrix(G, oriented=True, 
        weight='cond'))[1:,:]
    
    Q, _ = np.linalg.qr(d1.T)
    A = np.dot(Q, Q.T)
    
    if weighted:
        A = (wts)*(A*wts.T).T

    diag_inds = np.diag_indices(A.shape[0])

    if weighted:
        A[diag_inds] = wts_sqr - A[diag_inds]
    else:
        A[diag_inds] = 1 - A[diag_inds]
    return A

def DeltaF_from_cycle_flows(G, pert, bordering_cycles, cycles):
    """ Return DeltaF by appropriately adding up cycle flows.
    G needs to be the pruned graph!
    Signs in DeltaF may be (are!) wrong, so returns only
    absolute value.
    """
    DF = []
    for e in G.edges_iter():
        adjacent_cycles = bordering_cycles[tuple(sorted(e))]
        
        al = np.empty((len(adjacent_cycles),), float)
        al[::2] = 1.0
        al[1::2] = -1.0
        
        flows = np.array([pert[c]*cycles[c].orientation()
                for c in adjacent_cycles])*al

        DF.append(flows.sum())
    
    return np.abs(DF)

