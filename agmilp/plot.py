from __future__ import division, absolute_import, print_function

import logging
import itertools as it

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from stlmilp import stl
from . import agmilp

class PlotAssumptionMinining(object):
    def __init__(self, lims, fig=None, ax=None):
        if fig is None and ax is None:
            self.fig, self.ax = plt.subplots(1)
        else:
            self.fig = fig
            self.ax = ax
        self.ax.set_xlim(lims[0])
        self.ax.set_ylim(lims[1])

    def plot_step(self, traces, sat_for):
        self.clear()
        cs = {-1: 'r', 1: 'g'}
        plot_formula(self.ax, sat_for)
        x, y = traces.signals[:, :-1, 0].T
        self.ax.scatter(x, y, c=[cs[l] for l in traces.labels])
        sc = self.ax.scatter(x[-1], y[-1],  marker='s',
                             s=(plt.rcParams['lines.markersize'] ** 2) * 2)
        sc.set_facecolor('none')
        sc.set_edgecolor(cs[traces.labels[-1]])
        plt.draw()
        plt.pause(0.1)

    def set_interactive(self):
        plt.ion()
        plt.show()

    def render(self):
        plt.show()

    def clear(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.clear()
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)


def plot_formula(ax, formula):
    rects = compute_rectangles(formula)
    for rect in rects:
        plot_box(ax, rect, alpha=0.2, color='b', zorder=0)
    return ax

def compute_rectangles(formula):
    if formula.op == stl.EXPR:
        expr = formula.args[0]
        if not isinstance(expr, agmilp.MILPSignal):
            raise ValueError("Expression is not MILPSignal, given {}".format(expr.__class__))
        if expr.index > 1:
            raise ValueError("Can only compute 2D rectangles, given dimension index {}".format(expr.index))

        rect = np.empty((2,2))
        p = expr.op * expr.f([0])
        j = 1 - (expr.op + 1) // 2
        rect[:, j] = - expr.op * np.array([np.inf, np.inf])
        rect[expr.index, 1 - j] = p
        rect[1 - expr.index, 1 - j] = expr.op * np.inf
        return np.array([rect])

    elif formula.op == stl.NOT:
        rects = compute_rectangles(formula.args[0])
        if len(rects) != 1:
            raise ValueError("Cannot compute rectangles for formula: NOT can only precede a half space")
        rect = -np.c_[rects[0][:, 1], rects[0][:, 0]]
        pindex = np.abs(rect) < np.inf
        rect[pindex] = - rect[pindex]
        return np.array([rect])

    elif formula.op == stl.AND:
        rects_list = [compute_rectangles(f) for f in formula.args]
        return np.array(
            [intersect_rects(np.array(rects)) for rects in it.product(*rects_list)])

    elif formula.op == stl.OR:
        return np.array(
            [rect for rects in [compute_rectangles(f) for f in formula.args]
             for rect in rects])

    else:
        return compute_rectangles(formula.args[0])

def intersect_rects(rects):
    return np.c_[np.amax(rects[:, :, 0], axis=0), np.amin(rects[:, :, 1], axis=0)]


def plot_box(ax, box, **kwargs):
    """
    box = [[x1, x2], [y1, y2]]
    """
    if box.shape != (2, 2):
        raise ValueError("box should have (2, 2) shape, given {}".format(box.shape))
    box2 = box.copy()
    _crop_box(ax, box2)
    x, y = box2[:2,0]
    w, h = box2[:2,1] - box2[:2,0]
    ax.add_patch(patches.Rectangle((x,y), w, h, **kwargs))

def _crop_box(ax, box):
    ylims = ax.get_ylim()
    xlims = ax.get_xlim()
    box[0,0] = _x_or_lim(box[0,0], xlims)
    box[0,1] = _x_or_lim(box[0,1], xlims)
    box[1,0] = _x_or_lim(box[1,0], ylims)
    box[1,1] = _x_or_lim(box[1,1], ylims)

def _x_or_lim(x, lims):
    if np.abs(x) == np.inf:
        return lims[(int(np.sign(x)) + 1) // 2]
    else:
        return x
