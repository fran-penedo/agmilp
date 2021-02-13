import numpy as np

from agmilp.system.system import FOSystem, trapez_integrate
from agmilp.agmine import MILPSignal
from templogic.stlmilp import stl

system = FOSystem(
    np.identity(2),
    np.array([[-0.3, 0.1], [-0.1, -0.3]]),
    np.array([0, 0]),
    dt=1.0,
)
bounds = [[-10, -10], [10, 10]]
integrate = lambda system, x0, args: trapez_integrate(
    system, x0, args[0], args[1], log=False
)

isstate = True
system_n = 2

formula = stl.STLOr(
    args=[
        stl.STLAlways(
            bounds=[2, 4],
            arg=stl.STLPred(MILPSignal(lambda x: x - (-4), 1, 0, isstate, system_n)),
        ),
        stl.STLAlways(
            bounds=[2, 4],
            arg=stl.STLPred(MILPSignal(lambda x: x - 4, -1, 0, isstate, system_n)),
        ),
    ]
)

args = [10.0, system.dt]
