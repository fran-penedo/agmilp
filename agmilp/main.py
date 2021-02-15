import argparse
import importlib
import sys
from typing import Any, Callable, Optional, Sequence
import logging
import logging.config

import attr
import numpy as np
from templogic.stlmilp import stl

from agmilp import LOG_CONFIG
from agmilp.agmine import mine_assumptions
from agmilp.plot import PlotAssumptionMinining
from agmilp.system.system import FOSystem


logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class Scenario(object):
    system: Optional[FOSystem] = None
    bounds: Optional[list[list[float]]] = None
    formula: Optional[stl.STLTerm] = None
    integrate: Optional[Callable[[FOSystem, np.ndarray, Any], np.ndarray]] = None
    args: Optional[Sequence[Any]] = None
    tol_min: float = 1.0
    tol_init: float = 2.0
    alpha: float = 0.5
    num_init_samples: int = 10
    plotter: Optional[PlotAssumptionMinining] = None


def process_options() -> Scenario:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    scenario = Scenario()

    parser.add_argument("--tol-min", metavar="t", type=float)
    parser.add_argument("--tol-init", metavar="t", type=float)
    parser.add_argument("--alpha", metavar="a", type=float)
    parser.add_argument("--num-init-samples", metavar="n", type=int)
    parser.add_argument("--plot", action="store_true", default=False)
    parser.add_argument("--verbose", "-v", action="store_true", default=False)
    parser.add_argument("scenario")

    options = parser.parse_args()

    if options.verbose:
        logging.config.dictConfig(LOG_CONFIG)

    try:
        spec = importlib.util.spec_from_file_location("module", options.scenario)
        module = importlib.util.module_from_spec(spec)
        sys.modules["module"] = module
        spec.loader.exec_module(module)  # type: ignore # Don't need to handle this
    except Exception as s:
        raise s

    for k in vars(scenario).keys():
        if (value := getattr(module, k, None)) is not None:
            setattr(scenario, k, value)
        if (value := getattr(options, k, None)) is not None:
            setattr(scenario, k, value)

    if any(
        x is None
        for x in [
            scenario.system,
            scenario.bounds,
            scenario.formula,
            scenario.integrate,
            scenario.args,
        ]
    ):
        raise Exception(f"Scenario module missing required object\n{scenario}")

    plotter = None
    if options.plot:
        assert scenario.bounds is not None
        plotter = PlotAssumptionMinining(list(zip(*scenario.bounds)))
        plotter.set_interactive()
    scenario.plotter = plotter

    logger.debug(f"{scenario}")

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
