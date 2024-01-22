using ReliabilityOptimization, Test, NonconvexTOBS,
    ChainRulesCore, TopOpt, Zygote, FiniteDifferences

const v = 0.3 # Poisson’s ratio
const f = 1.0 # downward force
const V = 0.4 # volume fraction
const problemSize = (4, 4) # size of rectangular mesh
const elSize = (1.0, 1.0) # size of QUAD4 elements
# Point load cantilever problem to be solved
problem = PointLoadCantilever(Val{:Linear}, problemSize, elSize, 1.0, v, f)
ncells = TopOpt.getncells(problem) # Number of elements in mesh
solver = FEASolver(Direct, problem; xmin = 0.0)
filter = DensityFilter(solver; rmin = 3.0) # filter to avoid checkerboarding
comp = Compliance(solver) # function that returns compliance
penalty = TopOpt.PowerPenalty(3.0) # SIMP penalty
const avgEs = [1e-6, 0.5]
logEs = MvNormal(log.(avgEs), Matrix(Diagonal(0.1 .* avgEs .|> log .|> abs))) 
# 'Original' function. At least one input is random.
# In this example, Es is the random input.
function compObj(x, E)
    # comp(filter(PseudoDensities(x)))
    interp = MaterialInterpolation(E, penalty)
    MultiMaterialVariables(x, 2) |> interp |> filter |> comp
end
# wrap original function in RandomFunction struct
rf = RandomFunction(compObj, logEs, FORM(RIA()))
# initial homogeneous distribution of pseudo-densities
x0 = fill(V, ncells * (length(logEs) - 1))
# call wrapper with example input
# (Returns propability distribution of the objective for current point)
d = rf(x0)
constr = x -> volfrac(filter(PseudoDensities(x))) - V
function obj(x) # objective for TO problem
    dist = rf(x)
    mean(dist)[1] + 2 * sqrt(cov(dist)[1, 1])
end
obj(x0)
Zygote.gradient(obj, x0)
FiniteDifferences.grad(central_fdm(5, 1), obj, x0)[1]

m = Model(obj) # create optimization model
addvar!(m, zeros(length(x0)), ones(length(x0))) # setup optimization variables
Nonconvex.add_ineq_constraint!(m, constr) # setup volume inequality constraint
@time r = Nonconvex.optimize(m, TOBSAlg(), x0; options = TOBSOptions())