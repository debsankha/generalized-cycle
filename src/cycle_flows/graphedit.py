#!/usr/bin/env python

"""
graphedit.py

Can manipulate tree graphs.

Henrik Ronellenfitsch 2013
"""

import os
import os.path

import argparse
from itertools import izip

import networkx as nx

import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.collections

from numpy import *
import numpy.random

import storage
import plot

from cycle_basis import *

class GraphEditor(object):
    def __init__(self, fname, interactive=True):
        self.fname = fname
        self.graph = nx.read_gpickle(fname)
        
        #apply_workaround(self.graph, thr=1e-3)
        #remove_intersecting_edges(self.graph)

        print "Number of connected components:", \
                nx.number_connected_components(self.graph)

        self.selected_path_verts = []
        
        if interactive:
            self.fig = plt.figure()
            self.path_patch = None
            
            G_p = nx.connected_component_subgraphs(self.graph)[0]
            #G_p = nx.connected_component_subgraphs(prune_graph(self.graph))[0]

            plot.draw_leaf(G_p, fixed_width=True)

            plt.ion()
            plt.show()

            self.edit_loop()    

    def points_to_path(self, close=False):
        points = self.selected_path_verts + [[0,0]]
        codes = Path.LINETO*ones(len(self.selected_path_verts) + 1)
        codes[0] = Path.MOVETO

        if close:
            codes[-1] = Path.CLOSEPOLY
        else:
            codes[-1] = Path.STOP

        path = Path(points, codes)
        return path

    def draw_poly(self, close=False):
        path = self.points_to_path(close=close)
        ax = plt.gca()
        
        if self.path_patch != None:
            self.path_patch.remove()

        self.path_patch = patches.PathPatch(path, facecolor='None',
                edgecolor='red')
        ax.add_patch(self.path_patch)
        plt.show()

    def onclick_polyselect(self, event):
        if event.button == 1 and \
                event.xdata != None and event.ydata != None:
            print "Selected point ({}, {})".format(event.xdata,
                    event.ydata)

            self.selected_path_verts.append([event.xdata, event.ydata])

            self.draw_poly()

    def select_polygon(self):
        print "Polygon selection mode."
        print """Click on the points you want to select.
        Available commands:

        r: Remove last selected point
        c: Close polygon and finish
        a: Abort.
        """

        self.selected_path_verts = []
        
        self.cid_click = self.fig.canvas.mpl_connect(
                'button_press_event', self.onclick_polyselect)

        while True:
            cmd = raw_input("SE> ")

            if cmd == 'r':
                if len(self.selected_path_verts) > 0:
                    del self.selected_path_verts[-1]
                    self.draw_poly()
                else:
                    print "Nothing selected."
            elif cmd == 'c':
                if len(self.selected_path_verts) > 2:
                    self.fig.canvas.mpl_disconnect(self.cid_click)
                    
                    self.draw_poly(close=True)
                    print "Polygon selected."
                    return
                else:
                    print "Not enough points selected to form polygon!"
                    print "Exiting."
                    self.fig.canvas.mpl_disconnect(self.cid_click)
                    return
            elif cmd == 'a':
                print "Aborted."
                self.selected_path_verts = []
                self.fig.canvas.mpl_disconnect(self.cid_click)
                
                if self.path_patch != None:
                    self.path_patch.remove()
                    self.path_patch = None

                    plt.show()
                return
            else:
                print "Command {} not recognized".format(cmd)
    
    def suggested_name(self, fname, extension="_edited_"):
        i = 0
        while True:
            bname, ext = os.path.splitext(fname)
            suggested_name = bname + extension + str(i) + ext

            if not os.path.exists(suggested_name):
                return suggested_name
            else:
                i += 1

    def export_selection(self):
        if len(self.selected_path_verts) == 0:
            print "Nothing selected!"
            return

        path = self.points_to_path(close=True)
        
        nodes = [n for n in self.graph.nodes_iter() if
                path.contains_point((self.graph.node[n]['x'], 
                    self.graph.node[n]['y']))]

        subgraph = nx.Graph(self.graph.subgraph(nodes))
        
        suggested_name = self.suggested_name(self.fname,
                extension="_selection_")

        savname = raw_input("Save to [{}]: ".format(suggested_name))

        if savname == '':
            savname = suggested_name

        nx.write_gpickle(subgraph, savname)
        print "Done."
    
    def deselect_polygon(self):
        self.selected_path_verts = []

        if self.path_patch != None:
            self.path_patch.remove()
            self.path_patch = None
            plt.show()

        print "Deselected polygon."
    
    def distance(self, a, b):
        return sqrt((a[0] - b[0])**2  + (a[1] - b[1])**2)

    def is_between(self, a, c, b, tol=1):
        return abs(self.distance(a, c) + self.distance(c, b) 
            - self.distance(a, b)) < tol

    def draw_selected_edges(self):
        for l in self.drawn_edges:
            l.remove()

        self.drawn_edges = []

        for e in self.selected_edges:
            ed = self.edges[e]
            xs = [self.graph.node[ed[0]]['x'], self.graph.node[ed[1]]['x']]
            ys = [self.graph.node[ed[0]]['y'], self.graph.node[ed[1]]['y']]

            line, = plt.plot(xs, ys, color='r', lw=3)
            self.drawn_edges.append(line)

    def onclick_straighten_path(self, event):
        x, y = event.xdata, event.ydata
        
        cont = [i for i in xrange(len(self.edges)) 
                if self.is_between(self.edge_paths[i][0], (x,y), 
                    self.edge_paths[i][1])]
        
        if len(cont) == 1:
            pick = cont[0]
            if pick in self.selected_edges:
                self.selected_edges.remove(pick)
            else:
                self.selected_edges.append(pick)

            self.draw_selected_edges()

    def fit_selected_edges_linear(self):
        es = [self.edges[i] for i in self.selected_edges]
        ns = list(set([e[0] for e in es] + [e[1] for e in es]))
        
        xs = [self.graph.node[n]['x'] for n in ns]
        ys = [self.graph.node[n]['y'] for n in ns]

        a, b = polyfit(xs, ys, 1)
        
        # Calculate closest points
        for n, x, y in izip(ns, xs, ys):
            xnew = (x + a*y - a*b)/(1 + a**2)
            ynew = a*xnew + b

            self.graph.node[n]['x'] = xnew
            self.graph.node[n]['y'] = ynew
           
    def straighten_path(self):
        print """Please select the edges you would like to straighten out.
        Available commands:

        l: Linear curve fit
        p: Print edge list
        x: Exit
        """
        self.edges = self.graph.edges()
        self.edge_paths = [[(self.graph.node[e[0]]['x'], 
            self.graph.node[e[0]]['y']),
            (self.graph.node[e[1]]['x'],
            self.graph.node[e[1]]['y'])] for e in self.edges]
        self.selected_edges = []
        self.drawn_edges = []
        
        self.cid_path_click = self.fig.canvas.mpl_connect(
                'button_press_event',
                self.onclick_straighten_path)

        while True:
            cmd = raw_input("SP> ")

            if cmd == 'l':
                self.fit_selected_edges_linear()
                print "Linear fit concluded. Exiting."
            
                plt.clf()
                plot.draw_leaf(self.graph)
                self.fig.canvas.mpl_disconnect(self.cid_path_click)
                return
            elif cmd == 'x':
                # Clean up
                self.fig.canvas.mpl_disconnect(self.cid_path_click)
                return
            elif cmd == 'p':
                print [self.edges[e] for e in self.selected_edges]
            else:
                print "Command not recognized."

    def save_graph(self):
        suggested_name = self.suggested_name(self.fname,
                extension="_edited_")

        savname = raw_input("Save to [{}]: ".format(suggested_name))

        if savname == '':
            savname = suggested_name

        nx.write_gpickle(self.graph, savname)
        print "Done."

    def fragment_graph(self, path, x=3, y=3, mode='normal'):
        """ Generate fragments of the graph and save them
        individually in path.

        mode == 'pixels': x, y are dimensions of fragments
        otherwise: x, y are number of fragments in resp. axis
        """
        print "Fragmenting."
        G = nx.connected_component_subgraphs(self.graph)[0]

        # bounding box
        xs = [d['x'] for n, d in G.nodes_iter(data=True)]
        ys = [d['y'] for n, d in G.nodes_iter(data=True)]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)
        
        # equal sized tiles. otherwise x, y mean number of tiles in
        # respective axis
        if mode == 'pixels':
            x = float(x)
            y = float(y)
            
            x_fragments = int((x_max - x_min)/x)
            y_fragments = int((y_max - y_min)/y)

            print "Tiling into {}x{} fragments of size {}x{}.".format(
                    x_fragments, y_fragments, x, y)

        # fragment into pieces
        fragments = []
        for i in xrange(x_fragments):
            for j in xrange(y_fragments):
                x0 = x_min + i/float(x_fragments)*(x_max - x_min)
                x1 = x0 + 1./float(x_fragments)*(x_max - x_min)

                y0 = y_min + j/float(y_fragments)*(y_max - y_min)
                y1 = y0 + 1./float(y_fragments)*(y_max - y_min)
                
                nodes = [n for n, d in G.nodes_iter(data=True)
                        if d['x'] >= x0 and d['x'] <= x1
                        and d['y'] >= y0 and d['y'] <= y1]

                fragments.append(G.subgraph(nodes))
        
        # save fragments as individual graphs
        if not os.path.exists(path):
            os.makedirs(path)
        
        print "Saving fragments."
        name, ext = os.path.splitext(os.path.basename(self.fname))
        for i, fragment in enumerate(fragments):
            nx.write_gpickle(fragment, 
                    os.path.join(path, 
                        name + '_fragment_{}.gpickle'.format(i)))
                        
    def randomize(self):
        #for n, d in self.graph.nodes_iter(data=True):
        #    d['x'] += 10*(numpy.random.random() - 0.5)
        #    d['y'] += 10*(numpy.random.random() - 0.5)

        #    d['x'] = 1.3*d['x'] + 0.5*d['y']
        #    d['y'] = 1*d['y']

        for u, v, d in self.graph.edges_iter(data=True):
            d['conductivity'] += 5 + 0.25*(numpy.random.random()-0.5)        
        plt.clf()
        plot.draw_leaf(self.graph)
        plt.show()

    def cycle_basis(self):
        G_p = nx.connected_component_subgraphs(prune_graph(self.graph))[0]

        shortest_cycles(G_p)

    def print_help_message(self):
        print """GraphEdit command prompt. The following commands are
        available:

        s: Select polygon in graph
        d: Deselect polygon
        e: Extract selected polygon from graph and save
        t: Straighten path
        v: Save current graph
        r: Add randomness to nodes and bonds and do a shear transformation.
        c: Calculate cycle basis

        h: Print this help message
        x: Exit GraphEdit
        """

    def edit_loop(self):
        self.print_help_message()
        while True:
            cmd = raw_input("GE> ")

            if cmd == 'h':
                self.print_help_message()
            elif cmd == 'x':
                print "Exiting. Have a nice day!"
                return
            elif cmd == 's':
                self.select_polygon()
            elif cmd == 'e':
                self.export_selection()
            elif cmd == 'd':
                self.deselect_polygon()
            elif cmd == 't':
                self.straighten_path()
            elif cmd == 'v':
                self.save_graph()
            elif cmd == 'r':
                self.randomize()
            elif cmd == 'c':
                self.cycle_basis()
            else:
                print "Command {} not recognized.".format(cmd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("graphedit.py")
    parser.add_argument('INPUT', help="Input file in .gpickle format")
    parser.add_argument('-f', '--fragment', help='Fragement graph'
            'and output the parts as separate files. '
            'Argument is the path where to save.', default=None,
            type=str)
    parser.add_argument('-e', '--equal-size', action='store_true',
            help='Use 2500x2500px fragments instead of exactly'
            ' tiling into 3x3 tiles.')

    args = parser.parse_args()
    
    interactive = True
    if args.fragment != None:
        interactive = False

    edt = GraphEditor(args.INPUT, interactive=interactive)

    if args.fragment != None:
        if args.equal_size:
            edt.fragment_graph(args.fragment, mode='pixels', x=2500, y=2500)
        else:
            edt.fragment_graph(args.fragment)
