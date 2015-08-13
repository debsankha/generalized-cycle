#!/usr/bin/env python

"""
    cycle_plots.py

    contains functions to draw and plot cycle flows
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
import plot
import correlations

import seaborn as sns

def draw_oriented_circle(x, y, o, size, color):
    """ Draw a circle with an orientation o at position x, y
    """
    # draw circle
    #circle = Wedge((x, y), size, 0, 180,
    #        color=color, fill=False, lw=2)
    
    if o < 0:
        circle = Arc((x, y), 2*size, 2*size, angle=-160, theta2=270,
                color=color, lw=2.5)
    else:
        circle = Arc((x, y), 2*size, 2*size, angle=80, theta2=270,
                color=color, lw=2.5)

    plt.gca().add_artist(circle)

    # draw arrow
    try:
        ax = x
        ay = y + size
        plt.arrow(ax, ay, o, -0.3, width=0.04*size, color=color)
    except:
        print "Exception drawing arrow"
        print ax, ay, size

def plot_cycle_currents(G, cycles, pert_currents, pert_cycles):
    """ make a plot of perturbation currents in the leaf graph.
    """
    pert_currents = 20*pert_currents/np.max(np.abs(pert_currents))
    
    pal = sns.color_palette()
    blue = pal[0]
    red = pal[2]

    for cy, pert in izip(cycles, pert_currents):
        pert *= cy.orientation()
        col = blue if pert > 0 else red
        #plt.plot(cy.com[0], cy.com[1], 'o', markersize=np.abs(pert),
        #        color=col)

        draw_oriented_circle(cy.com[0], cy.com[1], np.sign(pert),
                np.abs(pert), col)
    
    edges = cycles[pert_cycles[0]].edgeset.intersection(
            cycles[pert_cycles[1]].edgeset)

    #print "edges in intersection:", edges
    #plot.draw_leaf(G, edge_list=edges, color='r')

    #plt.savefig('plots/cycle_currents.png', dpi=600, bbox_inches='tight')

def plot_cycle_currents_distance(cycles, pert_currents, pert_cycles):
    """ Plot a graph of perturbation strength vs distance from
    perturbation.
    """
    c_a_com = cycles[pert_cycles[0]].com
    c_b_com = cycles[pert_cycles[1]].com

    pos_of_pert = 0.5*(c_a_com + c_b_com)

    coms = np.array([c.com for c in cycles])
    distances = np.sqrt(np.sum((coms - pos_of_pert)**2, axis=1))

    plt.figure()
    plt.semilogy(distances, 
            np.abs(pert_currents)/np.max(np.abs(pert_currents)), 'o')

    plt.xlabel('distance from perturbation (px)')
    plt.ylabel('relative cycle current $|f_c| / |f_\mathrm{max}|$')
    
    plt.locator_params(axis='x', nbins=3)

    plt.savefig('plots/current_strength_distance.svg', dpi=300, 
            bbox_inches='tight')

def plot_cycle_currents_dual_distance(cycles, pert_currents, pert_cycles,
        G_dual):
    """ Plot a graph of perturbation strength vs shortest path distance
    on the dual graph measured from the point of perturbation
    """
    c_a = pert_cycles[0]

    path_lengths = nx.shortest_path_length(G_dual, source=c_a)
    distances = [path_lengths[c] for c in xrange(len(cycles))]

    #plt.plot(distances,
    #    (pert_currents)/np.max(np.abs(pert_currents)), 'o')

    maxs, mins = minmax_values(distances, pert_currents)

    max_d = np.array(maxs.keys())
    max_v = np.array(maxs.values())

    min_d = np.array(mins.keys())
    min_v = -np.array(mins.values())

    mean_d, mean_v = means_values(distances, pert_currents)

    # fit an exponential decay
    #popt_max, pcov_max = fit_exponential(max_d[40:], max_v[40:])
    #popt_min, pcov_min = fit_exponential(min_d[40:], min_v[40:])

    #print popt_max
    #print popt_min
    
    #print "fitting line"
    #z_max = fit_log_line(max_d[10:-10], max_v[10:-10])
    #z_min = fit_log_line(min_d[10:], min_v[10:])
    #
    #if z_max[0] != 0:
    #    print "Decay length f_max:", -1./z_max[0]
    #else:
    #    print "Decay length f_max is infinite (error?)"

    #if z_min[0] != 0:
    #    print "Decay length -f_min:", -1./z_min[0]
    #else:
    #    print "Decay length -f_min is infinite (error?)"
    
    # plot stuff
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.semilogy(max_d, max_v, '-', 
            label='$f_\mathrm{max}(d_c)$', markersize=20)
    ax1.semilogy(min_d, min_v, '-', 
            label='$-f_\mathrm{min}(d_c)$', markersize=20)
    ax1.semilogy(mean_d, mean_v, '-', 
            label='$f_\mathrm{mean}(d_c)$', markersize=20)
    
    ax1.set_xlabel('cycle distance $d_c$')
    ax1.set_ylabel('cycle current $|f_c| / |f|_\mathrm{max}$')
    
    ax1.legend(loc='best')

    ax2.loglog(max_d, max_v, '-', 
            label='$f_\mathrm{max}(d_c)$', markersize=20)
    ax2.loglog(min_d, min_v, '-', 
            label='$-f_\mathrm{min}(d_c)$', markersize=20)
    ax2.loglog(mean_d, mean_v, '-', 
            label='$f_\mathrm{mean}(d_c)$', markersize=20)
    
    ax2.set_xlabel('cycle distance $d_c$')
    ax2.set_ylabel('cycle current $|f_c| / |f|_\mathrm{max}$')
    
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('plots/current_strength_cycle_distance.svg', dpi=300, 
            bbox_inches='tight')

    #plt.figure()
    #plt.semilogy(distances,
    #    np.abs(pert_currents)/np.max(np.abs(pert_currents)), 'o')

    #plt.xlabel('cycle distance from perturbation')
    #plt.ylabel('relative cycle current $|f_c| / |f_\mathrm{max}|$')

    #plt.savefig('plots/current_strength_cycle_distance_semilog.png', 
    #        dpi=300, bbox_inches='tight')


def plot_DeltaF_cycle_dist(cycle_dists, DeltaF):
    cd, DF_mean, DF_std = distance_means(cycle_dists, DeltaF)

    f, (ax1, ax2) = plt.subplots(1, 2)
    ax2.loglog(cd, DF_mean)
    ax2.set_xlabel('cycle distance $d_c$')
    ax2.set_ylabel('$\\langle\Delta F_e / \Delta F_{e_0} \\rangle$')

    ax1.semilogy(cd, DF_mean)
    ax1.set_xlabel('cycle distance $d_c$')
    ax1.set_ylabel('$\\langle\Delta F_e / \Delta F_{e_0} \\rangle$')
    
    plt.tight_layout()
    plt.savefig('flow_changes_from_cycle_flows.svg', 
            bbox_inches='tight')

def plot_DeltaF_in_leaf(DeltaF, G_pruned, ax=None, fixed_width=False):
    # plot DeltaF
    cmap = plt.get_cmap('jet')
    norm = colors.LogNorm(vmin=1e-8, vmax=1)
    scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)
    
    if ax == None:
        plt.figure()
        ax = plt.gca()
    
    # unweighted
    scalar_map.set_array(DeltaF)
    cols = list(enumerate(scalar_map.to_rgba(DeltaF)))

    plot.draw_leaf(G_pruned, ax=ax, mark_edges=cols, 
            fixed_width=fixed_width)
    
    ax.autoscale(tight='true')
    #plt.savefig('perturbation_current_from_cycle_flows.png',
    #        dpi=600)


