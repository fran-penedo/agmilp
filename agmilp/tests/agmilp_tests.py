from __future__ import division, absolute_import, print_function

import logging
import unittest

import numpy as np

from femformal.core import system as sys
from stlmilp import stl

from agmilp import agmilp, plot

def integrate_trapez(system, x0, args):
    x = sys.trapez_integrate(system, x0, args[0], args[1], log=False)
    return np.hstack([x, np.arange(0, args[0] + args[1] / 2, args[1])[None]]).T

class TestAGMilp(unittest.TestCase):
    def setUp(self):
        self.sys = sys.FOSystem(-np.identity(2), np.zeros((2,2)), np.array([1, 1]), dt=1.0)
        self.sys2 = sys.FOSystem(np.identity(2), np.array([[-.3, 0.1], [-0.1, -.3]]), np.array([0, 0]), dt=1.0)
        self.bounds = [[-10, -10], [10, 10]]
        self.integrate = lambda system, x0, args: sys.trapez_integrate(
            system, x0, args[0], args[1], log=False)

    def test_agmilp_simple(self):
        formula = stl.Formula(stl.OR, [
            stl.Formula(stl.ALWAYS, [
                stl.Formula(stl.EXPR, [
                    agmilp.MILPSignal(lambda x: x - (-4), 1, 0)
                ])
            ], [2, 4]),
            stl.Formula(stl.ALWAYS, [
                stl.Formula(stl.EXPR, [
                    agmilp.MILPSignal(lambda x: x - 4, -1, 0)
                ])
            ], [2, 4])
        ])
        args = [10.0, self.sys.dt]
        plotter=plot.PlotAssumptionMinining([[-10, 10], [-10, 10]])
        plotter.set_interactive()
        formula = agmilp.mine_assumptions(
            self.sys, self.bounds, formula, self.integrate, args,
            tol_min=0.5, tol_init=1.0, alpha=0.5, num_init_samples=10,
            plotter=plotter
        )
        print(formula)
        x = input()
        raise Exception()

    def test_agmilp_simple2(self):
        formula = stl.Formula(stl.OR, [
            stl.Formula(stl.ALWAYS, [
                stl.Formula(stl.EXPR, [
                    agmilp.MILPSignal(lambda x: x - (-4), 1, 0)
                ])
            ], [2, 4]),
            stl.Formula(stl.ALWAYS, [
                stl.Formula(stl.EXPR, [
                    agmilp.MILPSignal(lambda x: x - 4, -1, 0)
                ])
            ], [2, 4])
        ])
        args = [10.0, self.sys2.dt]
        plotter=plot.PlotAssumptionMinining([[-10, 10], [-10, 10]])
        plotter.set_interactive()
        formula = agmilp.mine_assumptions(
            self.sys2, self.bounds, formula, self.integrate, args,
            tol_min=0.25, tol_init=1.0, alpha=0.5, num_init_samples=200,
            plotter = plotter
        )
        print(formula)
        x = input()
        raise Exception()
