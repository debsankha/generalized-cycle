#!/usr/bin/env python

"""
    This script analyzes fits of the scaling data in leaf networks.
"""
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy

import seaborn as sns

def compare_errors(exponential, power_law):
    pl_better = np.where(power_law > exponential)[0].size
    ex_better = np.where(power_law < exponential)[0].size
    
    print "Power law better fit:", pl_better
    print "Exp. better fit:", ex_better

    return pl_better, ex_better

def compare_dataset_errors(data, ax=None, ax2=None, ax3=None):
    # clean up (remove infs)
    not_nan_inds = -np.isnan(data).any(axis=1)
    data = data[not_nan_inds,:]

    print "Edge distances:"
    print "---"
    pl_better_edge, ex_better_edge = compare_errors(data[:,1], data[:,5])
    
    print "---"
    print "Cycle distances:"
    print "---"
    pl_better_cycl, ex_better_cycl = compare_errors(data[:,3], data[:,7])
    
    print "Scaling exponent edge dist:", data[:,4].mean(), "+-", \
            data[:,4].std(ddof=1)

    print "Scaling exponent cycle dist:", data[:,6].mean(), "+-", \
            data[:,6].std(ddof=1)

    if ax != None:
        ind = np.arange(2)
        width = 0.35
        cols = sns.color_palette()

        rects1 = ax.bar([0, 0+width], 
                [pl_better_edge, ex_better_edge], width,
                color=[cols[0], cols[1]])
        rects2 = ax.bar([1, 1+width], 
                [pl_better_cycl, ex_better_cycl], 
                width, color=[cols[0], cols[1]])

        ax.set_xticks(ind + width)
        ax.set_xticklabels(('edge distance', 'cycle distance'))
        
        ax.set_ylabel('counts')
        ax.legend((rects1[0], rects1[1]), 
                ('power law fits better', 'exponential fits better'),
                loc='best')

    if ax2 != None:
        ax2.hist(data[:,5] - data[:,1], range=(-0.5, 0.5), bins=24, 
                alpha=0.8,
                label='edge dist')
        ax2.hist(data[:,7] - data[:,3], range=(-0.5, 0.5), bins=24, 
                alpha=0.8,
                label='cycle dist')

        ax2.legend(loc='best')
        
        ax2.set_xlabel('$R^2_\mathrm{power law} - R^2_\mathrm{exponential}$')
        ax2.set_ylabel('frequency')

    if ax3 != None:
        ax3.hist(data[:,4], range=(1, 5), bins=24, alpha=0.8,
                label='edge dist')
        ax3.hist(data[:,6], range=(1, 5), bins=24, alpha=0.8,
                label='cycle dist')

        ax3.legend(loc='best')

        ax3.set_xlabel('scaling power')
        ax3.set_ylabel('frequency')
    
    # cycle distance scaling exponent
    return data[:,6].mean(), data[:,6].std(ddof=1)

def compare_mutual_informations(mi_edge, mi_cycle, ax=None):
    cycle_better = np.where(mi_cycle > mi_edge)[0].size
    edge_better = mi_edge.size - cycle_better

    print "Cycle distance better MI:", cycle_better
    print "Edge distance better MI:", mi_edge.size - cycle_better
    print "Mean difference (MI_c - MI_e)", -np.mean(mi_edge - mi_cycle)
    print "Std difference", np.std(mi_edge - mi_cycle, ddof=1)
    
    if ax != None:
        width = 0.35
        ind = (0, 0+width)
        cols = sns.color_palette()

        rects = ax.bar(ind, [edge_better, cycle_better], width,
                color=cols[:2])

        ax.set_xticks([])
        ax.set_ylabel('counts')

        ax.legend((rects[0], rects[1]),
                ('edge distance better', 'cycle distance better'),
                loc='best')

def analyze_file(gname, plot=True):
    # load data files
    full_uw = np.loadtxt('data/scaling_full_data_uw_' + gname + '.txt', 
            usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    full_wt = np.loadtxt('data/scaling_full_data_wt_' + gname +'.txt',
            usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

    mean_uw = np.loadtxt('data/scaling_means_uw_' + gname + '.txt', 
            usecols=(1, 2, 3, 4, 5, 6, 7, 8))
    mean_wt = np.loadtxt('data/scaling_means_wt_' + gname +'.txt', 
            usecols=(1, 2, 3, 4, 5, 6, 7, 8))
    
    # setup plots
    if plot:
        f, axs = plt.subplots(2, 2)
        f2, axs2 = plt.subplots(2)
        f3, axs3 = plt.subplots(2, 2)
        f4, axs4 = plt.subplots(2, 2)

        print ""
        print "---"
        print "Full Data Sets (unweighted)"
        print "---"
        
        f_uw, _ = compare_dataset_errors(full_uw, ax=axs[0][0], ax2=axs3[0][0], 
                ax3=axs4[0][0])
        axs[0][0].set_title('full data (homogeneous)')
        axs3[0][0].set_title('full data (homogeneous)')
        axs4[0][0].set_title('full data (homogeneous)')
        
        print ""
        print "Mutual informations"
        
        compare_mutual_informations(full_uw[:,-2], full_uw[:,-1],
                ax=axs2[0])
        axs2[0].set_title('homogeneous')

        print ""
        print "---"
        print "Full Data Sets (weighted)"
        print "---"
        
        f_wt, _ = compare_dataset_errors(full_wt, ax=axs[0][1], ax2=axs3[0][1],
                ax3=axs4[0][1])
        axs[0][1].set_title('full data (hierarchical)')
        axs3[0][1].set_title('full data (hierarchical)')
        axs4[0][1].set_title('full data (hierarchical)')

        print ""
        print "Mutual informations"
        
        compare_mutual_informations(full_wt[:,-2], full_wt[:,-1],
                ax=axs2[1])
        axs2[1].set_title('hierarchical')
        
        print ""
        print "---"
        print "Mean Data Sets (unweighted)"
        print "---"
        
        m_uw, _ = compare_dataset_errors(mean_uw, ax=axs[1][0], ax2=axs3[1][0],
                ax3=axs4[1][0])
        axs[1][0].set_title('means (homogeneous)')
        axs3[1][0].set_title('means (homogeneous)')
        axs4[1][0].set_title('means (homogeneous)')

        print ""
        print "---"
        print "Mean Data Sets (weighted)"
        print "---"
        
        m_wt, _ = compare_dataset_errors(mean_wt, ax=axs[1][1], ax2=axs3[1][1],
                ax3=axs4[1][1])
        axs[1][1].set_title('means (hierarchical)')
        axs3[1][1].set_title('means (hierarchical)')
        axs4[1][1].set_title('means (hierarchical)')

        f2.suptitle('Mutual information $I(\Delta F_e/\Delta F_{e_0}:d)$')
        
        f.tight_layout()
        f2.tight_layout()
        f3.tight_layout()
        f4.tight_layout()

        f.savefig('plots/scaling_comparison_' + gname +'.png', dpi=300)
        f2.savefig('plots/mi_comparison_' + gname + '.png', dpi=300)
        f3.savefig('plots/correlation_comparison_' + gname + '.png', dpi=300)
        f4.savefig('plots/scaling_estimation_' + gname + '.png', dpi=300)

        plt.show()

        return f_uw, f_wt, m_uw, m_wt, 0, 0, 0, 0
    else:
        print ""
        print "---"
        print "Full Data Sets (unweighted)"
        print "---"
        
        full_uw_power, full_uw_std = compare_dataset_errors(full_uw)        

        print ""
        print "---"
        print "Full Data Sets (weighted)"
        print "---"
        
        full_wt_power, full_wt_std = compare_dataset_errors(full_wt)
        
        print ""
        print "---"
        print "Mean Data Sets (unweighted)"
        print "---"
        
        mean_uw_power, mean_uw_std = compare_dataset_errors(mean_uw)
        print ""
        print "---"
        print "Mean Data Sets (weighted)"
        print "---"
        
        mean_wt_power, mean_wt_std = compare_dataset_errors(mean_wt)

        return full_uw_power, full_wt_power, mean_uw_power, mean_wt_power, \
                full_uw_std, full_wt_std, mean_uw_std, mean_wt_std

if __name__ == '__main__':
    # set plotting style
    sns.set(style='ticks', font_scale=1.2)
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    params = {'mathtext.fontset': 'stixsans'}
    plt.rcParams.update(params)
    
    # name of the graph we want to analyze
    files = glob('data/scaling_full_data_uw*.txt')
    names = [f[26:-4] for f in files]

    #names = ['grid_BronxA_001_binary_graph',
    #'BronxA_004_binary_corrected_graph',
    #'BronxA_009_binary_corrected_graph',
    #'BronxA_015_a_binary_corrected_graph',
    #'BronxA_030_a_binary_corrected_graph',
    #'mapleb_80_graph']

    gname = 'grid_200'
    
    # analyze cycle distance scaling powers 
    full_uws = []
    full_wts = []
    mean_uws = []
    mean_wts = []
    full_uw_stds = []
    full_wt_stds = []
    mean_uw_stds = []
    mean_wt_stds = []

    for gname in names:
        print gname
        f_uw, f_wt, m_uw, m_wt, \
                f_uw_s, f_wt_s, m_uw_s, m_wt_s = \
                analyze_file(gname, plot=False)

        if not np.isnan(f_uw):
            full_uws.append(f_uw)
            full_uw_stds.append(f_uw_s)

        if not np.isnan(f_wt):
            full_wts.append(f_wt)
            full_wt_stds.append(f_wt_s)

        if not np.isnan(m_uw):
            mean_uws.append(m_uw)
            mean_uw_stds.append(m_uw_s)

        if not np.isnan(m_wt):
            mean_wts.append(m_wt)
            mean_wt_stds.append(m_wt_s)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    def sorted_plot(ax, data, std):
        inds = np.argsort(data)
        ax.errorbar(range(len(data)), np.array(data)[inds],
            yerr=np.array(std)[inds], fmt='o',
            ecolor=sns.color_palette()[1])

    sorted_plot(ax1, full_uws, full_uw_stds)
    ax1.set_ylabel('cycle dist. scaling power')
    ax1.set_xlabel('specimen')
    ax1.set_title('full data (hom)')
    ax1.set_ylim(1.8, 3.4)

    sorted_plot(ax2, full_wts, full_wt_stds)
    ax2.set_ylabel('cycle dist. scaling power')
    ax2.set_xlabel('specimen')
    ax2.set_title('full data (hier)')
    ax2.set_ylim(2, 7)

    sorted_plot(ax3, mean_wts, mean_wt_stds)
    ax3.set_ylabel('cycle dist. scaling power')
    ax3.set_xlabel('specimen')
    ax3.set_title('means (hom)')
    ax3.set_ylim(1.5, 4)

    sorted_plot(ax4, mean_uws, mean_uw_stds)
    ax4.set_ylabel('cycle dist. scaling power')
    ax4.set_xlabel('specimen')
    ax4.set_title('means (hier)')
    ax4.set_ylim(2, 3.5)
    
    plt.tight_layout()
    plt.savefig('plots/leaf_dataset_scaling.png', dpi=300,
            bbox_inches='tight')
    
    # correlation plot between mean and full fits

    f, (ax1, ax2) = plt.subplots(2)

    ax1.plot(full_uws, mean_uws, 'o')
    ax1.set_xlabel('full data')
    ax1.set_ylabel('means')
    ax1.set_title('homogeneous ' + str(scipy.stats.spearmanr(full_uws,
        mean_uws)))

    ax2.plot(full_wts, mean_wts, 'o')
    ax2.set_xlabel('full data')
    ax2.set_ylabel('means')
    ax2.set_title('hierarchical ' + str(scipy.stats.spearmanr(full_wts,
        mean_wts)))
    
    plt.tight_layout()

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    ax1.hist(full_uws, bins=20)
    ax1.set_xlabel('scaling exponent')
    ax1.set_ylabel('frequency')
    ax1.set_title('full data (hom)')

    ax2.hist(full_wts, bins=20)
    ax2.set_xlabel('scaling exponent')
    ax2.set_ylabel('frequency')
    ax2.set_title('full data (hier)')

    ax3.hist(mean_uws, bins=20)
    ax3.set_xlabel('scaling exponent')
    ax3.set_ylabel('frequency')
    ax3.set_title('means (hom)')

    ax4.hist(mean_wts, bins=20)
    ax4.set_xlabel('scaling exponent')
    ax4.set_ylabel('frequency')
    ax4.set_title('means (hier)')

    plt.tight_layout()
    plt.savefig('plots/leaf_dataset_scaling_histo.png', dpi=300,
            bbox_inches='tight')
    
    # normality test
    print "Shapiro-Wilk normality tests (large p means normal distribution)"
    print scipy.stats.shapiro(full_uws)
    print scipy.stats.shapiro(full_wts)
    print scipy.stats.shapiro(mean_uws)
    print scipy.stats.shapiro(mean_wts)

    plt.show()
    print len(names)
