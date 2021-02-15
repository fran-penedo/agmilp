# AGMilp: Assume-Guarantee STL Contract Mining Using MILP

## What is AGMilp

AGMilp is a tool for mining Assume-Guarantee (AG) Signal Temporal Logic (STL) contracts
for linear systems. It leverages MILP encodings of STL and STL inference in order to
obtain contracts for general (not necessarily monotone) linear systems. Currently it
only supports obtaining assume contracts over the initial conditions of the system,
with most of the logic for assume contracts over time-variant
piece-wise affine external inputs implemented but not in working condition yet.

## Requirements

You need Python3.8 or newer, [Gurobi 9.1](https://www.gurobi.com/) or newer, and the use
of virtualenv's or similar is encouraged.

## Quickstart

Clone the repository with:

    $ git clone https://github.com/franpenedo/agmilp.git

Install with PIP:

    $ pip install agmilp
    
## Sampling Based Assumption Mining

The assumption mining problem has the following form: suppose we have a linear system
S(x0, z), with x0 the initial state and z(t) an external input. We wish this system to
satisfy a given specification written as an STL formula `f_spec` over the state x(t).
The problem is to find an STL formula over x0 and z(t), `f_assumption`, such that a) if
x0 and z(t) satisfy `f_assumption`, then S(x0, z) satisfies `f_spec`, and b) we cannot
find a stricter assumption formula.

An overview of our sampling based algorithm is as follows:

1. Sample the space of x0 and z(t).
2. Classify each sample as producing a satisfying system trajectory or not.
3. Use STL inference (implemented in
   [templogic](https://github.com/fran-penedo/templogic)) to obtain an approximation to
   `f_assumption` from the satisfying samples.
4. Find a sample (x0, z(t)) that satisfies `f_assumption` but produces a not satisfying
   trajectory using an MILP encoding of the system, `f_spec` and `f_assumption`. If
   there is one, go to 3.
5. Find a sample (x0, z(t)) that does not satisfy `f_assumption` but produces a
   satisfying trajectory. If there is one, go to 3.

In order to solve the assumption mining problem with AGMilp, write a python module with
the system definition, the specification `f_spec` and the integration parameters
following this example (taken from `examples/simple.py`, note that assumptions over z(t)
are not supported yet):

```python
import numpy as np

from agmilp.system.system import FOSystem, trapez_integrate
from agmilp.agmine import MILPSignal
from templogic.stlmilp import stl

# System definition (M dx = Ax + b, with time interval dt), required
system = FOSystem(
    np.identity(2),  # M
    np.array([[-0.3, 0.1], [-0.1, -0.3]]),  # A
    np.array([0, 0]),  # b
    dt=1.0,
)

# Initial state bounds, required
bounds = [[-10, -10], [10, 10]]

# Integration function. You should probably leave this as it is, required
integrate = lambda system, x0, args: trapez_integrate(
    system, x0, args[0], args[1], log=False
)

# Signals are over system states, as opposed to external inputs (not supported yet)
isstate = True
# System dimensions
system_n = 2

# Specification, required
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

# [Final time, time interval], required
args = [10.0, system.dt]

# Assumption mining parameters, optional
tol_min = 0.5
tol_init = 1.0
alpha = 0.5
num_init_samples = 200
```

Then, you can obtain `f_assumption` by running:

```shell
$ agmilp --plot examples/simple.py
Assumption formula: (((G_[0.00, 0.00] (x_0 > 4.06)) & ...
```
    
Note that our sampling based algorithm stops at a given tolerance `tol_min` after
progressively reducing it from the initial value `tol_init` by factors of `alpha`. This
tolerance must be understood as the following: our algorithm guarantees that the
produced `f_assumption` is such that every initial state x0 with STL robustness degree
greater than `tol_min` produces a system trajectory that satisfies `f_spec`.

## Publications

A full description of AG mining in the context of formal methods for partial
differential equations, along with our assumption mining algorithm can be found in
chapter 3 of my PhD thesis [Penedo Alvarez, Francisco. “Formal Methods for Partial
Differential Equations,” 2020.
https://open.bu.edu/handle/2144/41039.](https://open.bu.edu/handle/2144/41039)

## Copyright and Warranty Information

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Copyright (C) 2018-2021, Francisco Penedo Alvarez (contact@franpenedo.com)
