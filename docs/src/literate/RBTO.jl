# # Multimaterial RBTO example 

# [Reliability-based topology optimization](https://link.springer.com/article/10.1007/s00158-010-0518-6) (RBTO) allows the search for an optimal design while considering uncertainty in the problem, such as material properties. In the following, an example of RBTO with two materials is developed. First, packages and basic problem definitions are needed.

# Packages and definitions.

using ReliabilityOptimization, Test, ChainRulesCore
using TopOpt, Zygote, FiniteDifferences, Nonconvex

Nonconvex.@load MMA

const v = 0.3 # Poisson’s ratio
const densities = [0.0, 1.0] # for mass calc
const f = 1.0 # downward force
const nmats = 2
const V = 0.4 # volume fraction
const problemSize = (4, 4) # size of rectangular mesh
const elSize = (1.0, 1.0); # size of QUAD4 elements
# Point load cantilever problem to be solved
problem = PointLoadCantilever(Val{:Linear}, problemSize, elSize, 1.0, v, f)
ncells = TopOpt.getncells(problem) # Number of elements in mesh
solver = FEASolver(Direct, problem; xmin = 0.0)
filter = DensityFilter(solver; rmin = 3.0) # filter to avoid checkerboarding
comp = Compliance(solver) # function that returns compliance
penalty = TopOpt.PowerPenalty(3.0) # SIMP penalty
const avgEs = [1e-6, 0.5]
logEs = MvNormal(log.(avgEs), Matrix(Diagonal(abs.(log.(avgEs) * 0.1))))
# 'Original' function. At least one input is random. In this example, Es is the random input.
function compObj(y, logEs)
    penalty1 = TopOpt.PowerPenalty(3.0)
    interp1 = MaterialInterpolation(exp.(logEs), penalty1)
    x = tounit(MultiMaterialVariables(y, nmats))
    _E = interp1(filter(x))
    dsd = comp(_E)
    return dsd
end
penalty2 = TopOpt.PowerPenalty(1.0)
interp2 = MaterialInterpolation(densities, penalty2)
rf = RandomFunction(compObj, logEs, FORM(RIA())) # wrap original function in RandomFunction struct
x0 = fill(V, ncells * (length(logEs) - 1)) # initial homogeneous distribution of pseudo-densities
# call wrapper with example input. This returns a propability distribution of the objective for current point
d = rf(x0)

constr = y -> begin
    _rhos = interp2(MultiMaterialVariables(y, nmats))
    return sum(_rhos.x) / ncells - 0.4 # elements have unit volumes
end
function obj(x) # objective for TO problem
    dist = rf(x)
    mean(dist)[1] + 2 * sqrt(cov(dist)[1, 1])
end

m = Model(obj) # create optimization model
addvar!(m, zeros(length(x0)), ones(length(x0))) # setup optimization variables
Nonconvex.add_ineq_constraint!(m, constr) # setup volume inequality constraint
@time r = Nonconvex.optimize(m, MMA87(), x0; options = MMAOptions())