from __future__ import division, absolute_import, print_function

import logging
import unittest

import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt

from templogic.stlmilp import stl
from agmilp import agmine, plot


class TestPlot(unittest.TestCase):
    def test_intersect_rectangles(self):
        rs = np.array([[[0, 1], [0, 1]], [[0.5, 0.75], [0.5, 0.75]]])
        npt.assert_array_equal(
            plot.intersect_rects(rs), np.array([[0.5, 0.75], [0.5, 0.75]])
        )

    def test_compute_rectangles_expr(self):
        form1 = stl.STLPred(agmine.MILPSignal(lambda x: x - 7, 1, 0, False))
        npt.assert_array_equal(
            plot.compute_rectangles(form1)[0],
            np.array([[-np.inf, 7], [-np.inf, np.inf]]),
        )

        form2 = stl.STLPred(agmine.MILPSignal(lambda x: x - 5, 1, 1, False))
        npt.assert_array_equal(
            plot.compute_rectangles(form2)[0],
            np.array([[-np.inf, np.inf], [-np.inf, 5]]),
        )

        form3 = stl.STLPred(agmine.MILPSignal(lambda x: x - 3, -1, 1, False))
        npt.assert_array_equal(
            plot.compute_rectangles(form3)[0],
            np.array([[-np.inf, np.inf], [3, np.inf]]),
        )

        form4 = stl.STLPred(agmine.MILPSignal(lambda x: x - 10, 1, 0, False))
        npt.assert_array_equal(
            plot.compute_rectangles(form4)[0],
            np.array([[-np.inf, 10], [-np.inf, np.inf]]),
        )

    @unittest.skip("Plot test")
    def test_plot_formula(self):
        form = stl.Formula(
            stl.OR,
            [
                stl.Formula(
                    stl.AND,
                    [
                        stl.Formula(
                            stl.NOT,
                            [
                                stl.Formula(
                                    stl.EXPR,
                                    [agmine.MILPSignal(lambda x: x - 5, -1, 0)],
                                )
                            ],
                        ),
                        stl.Formula(
                            stl.EXPR, [agmine.MILPSignal(lambda x: x - 5, -1, 1)]
                        ),
                    ],
                ),
                stl.Formula(
                    stl.AND,
                    [
                        stl.Formula(
                            stl.NOT,
                            [
                                stl.Formula(
                                    stl.EXPR, [agmine.MILPSignal(lambda x: x - 7, 1, 0)]
                                )
                            ],
                        ),
                        stl.Formula(
                            stl.EXPR, [agmine.MILPSignal(lambda x: x - 5, 1, 1)]
                        ),
                        stl.Formula(
                            stl.EXPR, [agmine.MILPSignal(lambda x: x - 3, -1, 1)]
                        ),
                        stl.Formula(
                            stl.EXPR, [agmine.MILPSignal(lambda x: x - 10, 1, 0)]
                        ),
                    ],
                ),
            ],
        )

        fig, ax = plt.subplots(1)
        ax.set_ylim([-10, 10])
        ax.set_xlim([-10, 10])
        plot.plot_formula(ax, form)
        plt.show()
