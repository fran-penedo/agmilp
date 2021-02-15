import numpy as np

from agmilp.system.system import FOSystem, trapez_integrate
from agmilp.agmine import MILPSignal
from templogic.stlmilp import stl

# System definition (M dx = Ax + b, with time interval dt)
system = FOSystem(
    np.identity(2),  # M
    np.array([[-0.3, 0.1], [-0.1, -0.3]]),  # A
    np.array([0, 0]),  # b
    dt=1.0,
)

# Initial state bounds
bounds = [[-10, -10], [10, 10]]

# Integration function. You should probably leave this as it is
integrate = lambda system, x0, args: trapez_integrate(
    system, x0, args[0], args[1], log=False
)

# Signals are over system states, as opposed to external inputs (not supported yet)
isstate = True
# System dimensions
system_n = 2

# Specification
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

# [Final time, time interval]
args = [10.0, system.dt]

# Assumption mining parameters
tol_min = 0.5
tol_init = 1.0
alpha = 0.5
num_init_samples = 200
