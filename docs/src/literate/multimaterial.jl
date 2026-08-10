# # Multi-material topology optimization example
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`multimaterial.ipynb`](@__NBVIEWER_ROOT_URL__/examples/multimaterial.ipynb)
#-
# ## Commented Program
#
# This example optimizes the distribution of *several* candidate materials
# (plus void) over a cantilever beam. Each cell holds a softmax over
# `nmats - 1` decision variables; `MaterialInterpolation` maps the
# resulting densities to the corresponding Young's moduli (for compliance)
# and physical densities (for the mass constraint).
#md # The full program, without comments, can be found in the next [section].

using TopOpt, Zygote, Test

# ### Define the problem
Es = [1e-5, 1.0, 4.0]  # Young's moduli of 3 materials (incl. void)
densities = [0.0, 0.5, 1.0] # physical densities, for the mass constraint
nmats = 3

nu = 0.3 # Poisson's ratio
f = 1.0 # downward force

problem = PointLoadCantilever(
    Val{:Linear},    # order of basis functions
    (160, 40),       # number of cells
    (1.0, 1.0),      # cell dimensions
    1.0,             # base Young's modulus
    nu,              # Poisson's ratio
    f,               # load
)
ncells = TopOpt.getncells(problem)

# ### FEA solver and filter
solver = FEASolver(DirectSolver, problem; xmin=0.0)
filter = DensityFilter(solver; rmin=4.0)
comp = Compliance(solver)

# ### Material interpolation
# Two interpolations are used: one maps the softmax densities to Young's
# moduli for the compliance evaluation, the other maps them to physical
# densities for the mass constraint.
penalty1 = TopOpt.PowerPenalty(3.0)
interp1 = MaterialInterpolation(Es, penalty1)

penalty2 = TopOpt.PowerPenalty(1.0)
interp2 = MaterialInterpolation(densities, penalty2)

# ### Objective and constraint
obj = y -> begin
    x = tounit(MultiMaterialVariables(y, nmats))
    _E = interp1(filter(x))
    return comp(_E)
end

y0 = zeros(ncells * (nmats - 1))

# Sanity-check the objective and its gradient before optimizing.
obj(y0)
Zygote.gradient(obj, y0)

constr = y -> begin
    _rhos = interp2(MultiMaterialVariables(y, nmats))
    return sum(_rhos.x) / ncells - 0.4 # elements have unit volumes
end

constr(y0)
Zygote.gradient(constr, y0)

# ### Optimization
model = Model(obj)
addvar!(model, fill(-10.0, length(y0)), fill(10.0, length(y0)))
add_ineq_constraint!(model, constr)

alg = MMA87()
options = MMAOptions(; s_init=0.1, tol=Tolerance(; kkt=1e-3))

y0 = zeros(ncells * (nmats - 1))
res = optimize(model, alg, y0; options)
y = res.minimizer

# ### Verify the result
# The mass constraint should be satisfied within tolerance, and every
# cell's softmax must sum to one (the unit-density property of
# `MultiMaterialVariables`).
println("Constraint value: $(constr(y))")
x = TopOpt.tounit(reshape(y, ncells, nmats - 1))
println("Non-void fraction: $(sum(x[:, 2:3]) / size(x, 1))")

@test constr(y) < 1e-6
@test all(x -> isapprox(x, 1), sum(x; dims=2))

#md # ## Plain Program
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [multimaterial.jl](multimaterial.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```