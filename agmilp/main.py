import argparse
import importlib
import sys
from argparse import Namespace
from typing import Any, Callable, Optional, Sequence, Tuple

import attr
import numpy as np
from templogic.stlmilp import stl

from agmilp.agmine import mine_assumptions
from agmilp.plot import PlotAssumptionMinining
from agmilp.system.system import FOSystem


@attr.s(auto_attribs=True)
class Scenario(object):
    system: FOSystem
    bounds: list[list[float]]
    formula: stl.STLTerm
    integrate: Callable[[FOSystem, np.ndarray, Any], np.ndarray]
    args: Sequence[Any]
    tol_min: float
    tol_init: float
    alpha: float
    num_init_samples: int
    plotter: Optional[PlotAssumptionMinining]


def process_options() -> Scenario:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--tol-min", metavar="t", type=float, default=1.0)
    parser.add_argument("--tol-init", metavar="t", type=float, default=2.0)
    parser.add_argument("--alpha", metavar="a", type=float, default=0.5)
    parser.add_argument("--num-init-samples", metavar="n", type=int, default=10)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("scenario")

    options = parser.parse_args()

    try:
        spec = importlib.util.spec_from_file_location("module", options.scenario)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module"] = module
        spec.loader.exec_module(module)
    except Exception as s:
        raise s

    try:
        system = module.system
        bounds = module.bounds
        formula = module.formula
        integrate = module.integrate
        args = module.args
    except Exception as e:
        raise Exception("Scenario module missing required object", e)

    plotter = None
    if options.plot:
        plotter = PlotAssumptionMinining(bounds)
        plotter.set_interactive()
    scenario = Scenario(
        system,
        bounds,
        formula,
        integrate,
        args,
        options.tol_min,
        options.tol_init,
        options.alpha,
        options.num_init_samples,
        plotter,
    )

    return scenario


def main() -> None:
    scenario = process_options()

    formula = mine_assumptions(
        scenario.system,
        scenario.bounds,
        scenario.formula,
        scenario.integrate,
        scenario.args,
        tol_min=scenario.tol_min,
        tol_init=scenario.tol_init,
        alpha=scenario.alpha,
        num_init_samples=scenario.num_init_samples,
        plotter=scenario.plotter,
    )

    print(f"Assumption formula: {formula}")

    if scenario.plotter is not None:
        scenario.plotter.pause()
    sys.exit(0)
