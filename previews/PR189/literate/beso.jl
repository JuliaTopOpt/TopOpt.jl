# # BESO example: HalfMBB Beam 
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`beso.ipynb`](@__NBVIEWER_ROOT_URL__/examples/beso.ipynb)
#-
# ## Commented Program
#
# What follows is a program spliced with comments.
#md # The full program, without comments, can be found in the next [section](@ref beso-plain-program).

using TopOpt

# ### Define the problem
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0; # downward force

nels = (160, 40)
problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), E, v, f)

# ### Define the FEA Solver and penalty functions
solver = FEASolver(Direct, problem; xmin=0.01, penalty=TopOpt.PowerPenalty(3.0))

# ### Define the compliance objective function and volume fraction constraint
comp = Compliance(solver)
volfrac = Volume(solver)
sensfilter = SensFilter(solver; rmin=4.0)
beso = BESO(comp, volfrac, 0.5, sensfilter)

# ### Run optimization
x0 = ones(length(solver.vars))
result = beso(x0)

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add("Makie")` first and either `Pkg.add("CairoMakie")` or `Pkg.add("GLMakie")`
using Makie
using CairoMakie
# alternatively, `using GLMakie`
fig = visualize(problem; topology=result.topology)
Makie.display(fig)

#md # ## [Plain Program](@id beso-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [beso.jl](beso.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
