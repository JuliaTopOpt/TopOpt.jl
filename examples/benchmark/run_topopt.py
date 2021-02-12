# pip install cvxpy nlopt topopt
# https://github.com/zfergus/topopt

import time
import numpy
from topopt.boundary_conditions import MBBBeamBoundaryConditions, CantileverBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.solvers import TopOptSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI

nelx, nely = 180, 60  # Number of elements in the x and y
volfrac = 0.3  # Volume fraction for constraints
penal = 3.0  # Penalty for SIMP
rmin = 4.0  # Filter radius

start_time = time.time()
# Initial solution
x = volfrac * numpy.ones(nely * nelx, dtype=float)

# Boundary conditions defining the loads and fixed points
# bc = MBBBeamBoundaryConditions(nelx, nely)
bc = CantileverBoundaryConditions(nelx, nely)

# Problem to optimize given objective and constraints
problem = ComplianceProblem(bc, penal)
gui = GUI(problem, "Topology Optimization Example")
topopt_filter = DensityBasedFilter(nelx, nely, rmin)
solver = TopOptSolver(problem, volfrac, topopt_filter, gui, maxeval=2000, ftol_rel=1e-3)
x_opt = solver.optimize(x)
print("#of evals: {}".format(solver.opt.get_numevals()))

print('Total time: {}'.format(time.time() - start_time))

input("Press enter...")