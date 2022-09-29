# # Topological optimization of binary structures (TOBS)

# ### Description

# The method of topological optimization of binary structures ([TOBS](https://www.sciencedirect.com/science/article/abs/pii/S0168874X17305619?via%3Dihub)) was originally developed in the context of optimal distribution of material in mechanical components. In its core, is a heuristic to solve binary optimization problems by first linearizing the objective and constraints. Then, a binary nonlinear program is solved (default solver is [Cbc](https://github.com/jump-dev/Cbc.jl)) to determine which binary variables must be flipped in the current iteration.

# ### Packages and parameters

using NonconvexTOBS, TopOpt
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 6.0 # filter radius
xmin = 0.001 # minimum density
V = 0.5 # maximum volume fraction
p = 3.0 # topological optimization penalty

# ### Define FEA problem

problem_size = (160, 100); # size of rectangular mesh
x0 = fill(1.0, prod(problem_size)); # initial design
problem = PointLoadCantilever(Val{:Linear}, problem_size, (1.0, 1.0), E, v, f);

# FEA solver and auxiliary functions need to be defined as well:

solver = FEASolver(Direct, problem; xmin = xmin);
cheqfilter = DensityFilter(solver; rmin = rmin); # filter function
comp = TopOpt.Compliance(solver); # compliance function

# The usual topology optimization problem adresses compliance minimization under volume restriction. Therefore, the objective and the constraint are:

obj(x) = comp(cheqfilter(PseudoDensities(x))); # compliance objective
constr(x) = sum(cheqfilter(PseudoDensities(x))) / length(x) - V; # volume fraction constraint

# ### Optimization setup

# Finally, the optimization problem is defined and solved:

m = Model(obj); # create optimization model
addvar!(m, zeros(length(x0)), ones(length(x0))); # setup optimization variables
Nonconvex.add_ineq_constraint!(m, constr); # setup volume inequality constraint
options = TOBSOptions(); # optimization options with default values
TopOpt.setpenalty!(solver, p);

# Perform TOBS optimization
@time r = Nonconvex.optimize(m, TOBSAlg(), x0; options = options)

# ### Results
@show obj(r.minimizer)
@show constr(r.minimizer)
topology = r.minimizer;
