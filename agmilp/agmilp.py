from __future__ import division, absolute_import, print_function

import logging
from bisect import bisect_right

import numpy as np

from lltinf import inference, llt
from stlmilp import stl, milp_util as milp, stl_milp_encode as stl_milp
from femformal.core import system as sys, system_milp_encode as milp_encode

logger = logging.getLogger(__name__)

SEED = 1

def mine_assumptions(system, bounds, formula, integrate, args,
                     tol_min=1.0, tol_init=5.0, alpha=.9, num_init_samples=1000,
                     plotter=None):
    signals = []
    labels = []
    stl.scale_time(formula, system.dt)
    logger.debug("Starting mining")
    logger.debug("Doing initial sampling")
    for pars in sample_init(bounds, num_init_samples):
        x = integrate(system, pars, args)
        prep_pars = _prepare_x(pars[:system.n], pars[system.n:],
                          getattr(system, 'control_f', None))
        signals.append(prep_pars)
        model = Model(x, system.dt)
        # maybe >= tol_min
        if stl.robustness(formula, model) >= 0:
            labels.append(1)
        else:
            labels.append(-1)
    logger.debug("Initial sampling completed")

    nsamples = len(labels)
    tol_cur = tol_init
    lltinf = inference.LLTInf(
        0, primitive_factory=llt.make_llt_d1_primitives,
        stop_condition=[inference.perfect_stop],
        redo_after_failed=50, optimizer_args={'maxiter': 10})
    logger.debug("Starting directed sampling with tol = {}".format(tol_cur))
    while tol_cur >= tol_min:
        if nsamples % 20 == 0:
            logger.debug("Current number of samples = {}".format(nsamples))
        traces = inference.Traces(signals, labels)
        lltinf.fit_partial(traces, disp=True)
        for sig, lab in lltinf.tree.traces.zipped():
            assert lltinf.predict([sig])[0] == lab

        sat_for = lltform_to_milpform(lltinf.get_formula(), [0], system.n)
        if plotter:
            plotter.plot_step(lltinf.tree.traces, sat_for)

        opt_res = _min_robustness(system, bounds, formula, sat_for, tol_cur)
        if opt_res.f < 0:
            # logger.debug("Adding negative sample")
            assert lltinf.predict([opt_res.x])[0] == 1
            signals = [opt_res.x]
            labels = [-1]
            nsamples +=1
        else:
            unsat_for = stl.Formula(stl.NOT, [sat_for])
            opt_res = _max_robustness(system, bounds, formula, unsat_for, tol_cur)
            if opt_res.f >= 0:
                assert lltinf.predict([opt_res.x])[0] == -1
                # logger.debug("Adding positive sample")
                signals = [opt_res.x]
                labels = [1]
                nsamples +=1
            else:
                tol_cur *= alpha
                logger.debug("Reduced tolerance to tol = {}".format(tol_cur))

    return sat_for

def contract_verif():
    is_covered = False
    while not is_covered:
        try:
            x0 = sample(assumption, covered)
            rho = max_robustness(system, formula, x0)
            if rho > 0:
                funnel = make_funnel(system, formula, x0, rho)
                covered = covered + funnel
            else:
                return UNSAT
        except EmptySampleSpace:
            is_covered = True
    return SAT

def make_funnel(system, formula, x0):
    pass



class Model(object):
    def __init__(self, signal, dt):
        self.signal = signal
        self.tinter = 1

    def getVarByName(self, var_t):
        l, i, j = milp_encode.unlabel(var_t)
        return self.signal[j, i]


class MILPSignal(stl.Signal):
    def __init__(self, f, op, index, isstate, system_n=0, bounds=None):
        if isstate:
            self.labels = [lambda t: milp_encode.label('d', index, t)]
        else:
            #FIXME add correct index
            self.labels = [lambda t: milp_encode.label('f', 0, t)]
        self.f = lambda vs: -op * f(vs[0])
        self.op = op
        self.index = index
        self.isstate = isstate
        if bounds is None:
            self.bounds = [-1000, 1000]
        else:
            self.bounds = bounds

    @classmethod
    def from_lltsignal(cls, lltsignal, system_n):
        i = lltsignal.index
        isstate = i < system_n
        if not isstate:
            i = i - system_n
        sig = cls(None, None, i, isstate)
        sig.f = lltsignal.f
        sig.op = -1 if sig.f([0]) < sig.f([1]) else 1
        return sig

    def __str__(self):
        return "{}_{:d} {} {:.2f}".format("x" if self.isstate else 'f', self.index, "<=" if self.op == 1 else ">", -self.f([0]))


def lltform_to_milpform(form, t, system_n):
    bounds = [bisect_right(t, b) - 1 for b in form.bounds]
    if form.op == stl.EXPR:
        args = [MILPSignal.from_lltsignal(form.args[0], system_n)]
    else:
        args = [lltform_to_milpform(arg, t, system_n) for arg in form.args]
    return stl.Formula(form.op, args, bounds)


# From scipy
class OptRes(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"


def sample_init(bounds, num):
    np.random.seed(SEED)
    return np.random.uniform(bounds[0], bounds[1], (num, len(bounds[0])))


def _min_max_robustness(system, bounds, formula, x_init_form, tol, obj):
    xs_milp = []
    m = stl_milp.build_and_solve(
        formula, _encode(system, bounds, xs_milp, x_init_form, tol), obj,
        log_files=False, outputflag=0)
    if m.status == stl_milp.GRB.status.INFEASIBLE:
        logger.warning("MILP infeasible, logging IIS")
        m.computeIIS()
        m.write("out.ilp")
        raise Exception("MILP Infeasible")
        return OptRes({'x': None, 'f': obj * np.Inf})
    else:
        if hasattr(system, 'control_f'):
            x0 = [x_milp.getAttr('x') for x_milp in xs_milp[:-1]]
            z = [z_milp.getAttr('x') for z_milp in xs_milp[-1]]
        else:
            x0 = [x_milp.getAttr('x') for x_milp in xs_milp]
            z = None
        f = m.getVarByName("spec").getAttr("x")
        return OptRes({'x': _prepare_x(x0, z, getattr(system, 'control_f', None)), 'f': f})

def _min_robustness(system, bounds, formula, x_init_form, tol):
    return _min_max_robustness(system, bounds, formula, x_init_form, tol, 1.0)

def _max_robustness(system, bounds, formula, x_init_form, tol):
    return _min_max_robustness(system, bounds, formula, x_init_form, tol, -1.0)

def _encode(system, bounds, xs_milp, x_init_form, tol):
    def encode(m, hd):
        x = milp_encode._add_trapez_constr(m, "d", system, hd)
        for i in range(system.n):
            xs_milp.append(x[milp_encode.label("d", i, 0)])
            m.addConstr(x[milp_encode.label("d", i, 0)] >= bounds[0][i])
            m.addConstr(x[milp_encode.label("d", i, 0)] <= bounds[1][i])



        if hasattr(system, 'control_f'):
            pf = [m.addVar(obj=0, lb=-milp_encode.g.GRB.INFINITY,
                        ub=milp_encode.g.GRB.INFINITY,
                        name=milp_encode.label('pf', i, 0))
                  for i in range(len(bounds[0]) - system.n)]
            xs_milp.append(pf)
            for i in range(system.n):
                for j in range(hd):
                    m.addConstr(x[milp_encode.label('f', i, j)] ==
                                system.control_f(j * system.dt, pf, i))
        else:
            for j in range(hd):
                m.addConstr(x[milp_encode.label('f', i, j)] == 0)
        fvar, vbds = stl_milp.add_stl_constr(m, "init", x_init_form)
        m.addConstr(fvar >= tol)
        return x

    return encode

def _prepare_x_init(x0):
    return np.hstack([x0, np.zeros(1)])[None].T

def _prepare_z(z, f):
    return np.array([z, f.ts])

def _prepare_x(x0, z, f):
    x0 = _prepare_x_init(x0)
    if f is None:
        return x0
    else:
        z = _prepare_z(z, f)
        x = np.vstack([np.ones(z.shape[1]) * x0[i, 0] for i in range(x0.shape[0] - 1)])
        return np.vstack([x, z])
