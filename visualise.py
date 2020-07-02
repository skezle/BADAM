import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # for server
import matplotlib.pyplot as plt

import sys
import os.path
sys.path.insert(1, os.path.join(sys.path[0], '..'))

def predictive_dist_plot(x_train, y_train, xx, mus, sigmas, tag, root=''):
    mus = [float(d) for d in mus]
    data = {'xx': xx, 'mus': mus, 'sigmas': sigmas}
    dist = pd.DataFrame(data)
    dist = dist.sort_values(by=['xx'])  # Sorts

    fig_size = (4, 3)
    set_y_axis = False

    lw = 2
    grid_color = '0.7'
    grid_lw = 0.2
    title_size = 16
    label_size = 16
    tick_size = 14

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.scatter(x_train, y_train, alpha=0.20, color='k', s=2)
    ax.plot(dist.xx.values, dist.mus.values, linewidth=lw)
    ax.fill_between(dist.xx.values,
                    [x - y for x, y in zip(dist.mus.values, dist.sigmas.values)],
                    [x + y for x, y in zip(dist.mus.values, dist.sigmas.values)],
                    alpha=0.2,
                    color='b')

    ax.fill_between(dist.xx.values,
                    [x - y for x, y in zip(dist.mus.values, dist.sigmas.values)],
                    [x + y for x, y in zip(dist.mus.values, dist.sigmas.values)],
                    alpha=0.05,
                    color='b')

    plt.grid(True, which="both", color=grid_color, linewidth=0.2, alpha=0.2)
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-2.8, 2.8)
    x_ticks = np.arange(-0.5, 1.5, step=0.5)
    y_ticks = np.arange(-2.0, 3.0, step=1.0)
    plt.xticks(x_ticks, fontsize=tick_size)
    plt.yticks(y_ticks, fontsize=tick_size)
    plt.tight_layout()
    plt.savefig("{1}plots/pred_dist_{0}.pdf".format(tag, root), bbox_inches='tight')
    plt.clf()
    return np.mean([2*s for s in dist.sigmas.values])

def predictive_dist_plot_sampling(x_train, y_train, xx, yy, preds, tag, root=''):
    dist = pd.DataFrame({'x': xx,
                         'y': yy,
                         'preds': preds})
    dist = dist.groupby('x', as_index=False).agg([np.mean, np.std])
    dist = dist.reset_index()
    dist.columns = ['x', 'y_mean', 'y_std', 'preds_mean', 'preds_std']

    fig_size = (4, 3)
    set_y_axis = False

    lw = 2

    grid_color = '0.7'
    grid_lw = 0.2

    title_size = 16
    label_size = 16
    tick_size = 14

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.scatter(x_train, y_train, alpha=0.20, color='k', s=2)
    #ax.scatter(xx, yy, alpha=0.05, color='k', s=2)
    ax.plot(dist.x.values, dist.preds_mean.values,linewidth=lw)
    ax.fill_between(dist.x.values,
                   [x - y for x, y in zip(dist.preds_mean.values, dist.preds_std.values)],
                   [x + y for x, y in zip(dist.preds_mean.values, dist.preds_std.values)],
                   alpha=0.2,
                   color='b')

    # ax.fill_between(dist.x.values,
    #                [x - 1.96*y for x, y in zip(dist.preds_mean.values, dist.preds_std.values)],
    #                [x + 1.96*y for x, y in zip(dist.preds_mean.values, dist.preds_std.values)],
    #                alpha=0.05,
    #                color='b')

    plt.grid(True, which="both",color=grid_color, linewidth=0.2, alpha=0.2)
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-2.8, 2.8)
    x_ticks = np.arange(-0.5, 1.5, step=0.5)
    y_ticks = np.arange(-2.0, 3.0, step=1.0)
    plt.xticks(x_ticks, fontsize=tick_size)
    plt.yticks(y_ticks, fontsize=tick_size)
    plt.tight_layout()
    plt.savefig("{1}plots/pred_dist_{0}.pdf".format(tag, root), bbox_inches='tight')
    plt.clf()
    return np.mean([2*s for s in dist.preds_std.values])

if __name__ == '__main__':
    pass
