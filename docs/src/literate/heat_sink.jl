# # Heat sink topology optimization example
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`heat_sink.ipynb`](@__NBVIEWER_ROOT_URL__/examples/heat_sink.ipynb)
#-
# ## Commented Program
#
# This example minimizes the thermal compliance of a heat-conduction problem.
# A distributed heat flux enters the top edge, the left edge is held at
# `T = 100` and the right edge at `T = 0` (the heat-sink fins), and the
# remaining boundaries are insulated. The optimizer redistributes a fixed
# volume of high-conductivity material to dissipate the heat most efficiently.
#md # The full program, without comments, can be found in the next.

using TopOpt, Ferrite, LinearAlgebra, Zygote

# ### Define the problem
nels = (60, 30)        # mesh resolution
sizes = (1.0, 1.0)     # element sizes
k = 1.0               # thermal conductivity
heatflux = Dict{String,Float64}("top" => 100.0) # heat flux on the top boundary (W/m²)
V = 0.5               # volume fraction constraint

# `Tleft=100` and `Tright=0` fix the temperature at the left and right edges,
# playing the role of the heat-sink fins, while the flux enters from the top.
problem = HeatConductionProblem(
    Val{:Linear}, nels, sizes, k;
    Tleft=100.0, Tright=0.0, heatflux=heatflux
)

println("Created heat conduction problem with $(Ferrite.getncells(problem)) elements")

# ### Define the FEA solver and objective
solver = FEASolver(DirectSolver, problem; xmin=0.001)

# `ThermalCompliance` measures `J = Qᵀ T`, the work done by the boundary heat
# flux against the temperature field. Minimizing it drives heat toward the
# cold (right) boundary.
comp = ThermalCompliance(solver)
vol = TopOpt.Volume(solver; fraction=true)
filter = DensityFilter(solver; rmin=2.0)

f = x -> comp(filter(PseudoDensities(x)))
g = x -> [vol(filter(PseudoDensities(x))) - V]

# ### Run optimization
x0 = fill(V, length(solver.vars))

println("Starting topology optimization...")
println("Volume constraint: $V")
println("Number of design variables: $(length(x0))")
println("Initial objective value (thermal compliance): $(f(x0))")

model = Model(f)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, g)
alg = MMA87()
options = MMAOptions(; tol=Tolerance(; kkt=1e-4))
result = optimize(model, alg, x0; options)

println("\nOptimization complete!")
println("Final objective value (thermal compliance): $(result.minimum)")
println("Iterations: $(result.iter)")

# ### Inspect the result
x_opt = result.minimizer

solver.vars .= x_opt
solver()
T_max = maximum(solver.u)
println("Maximum temperature: $T_max")

println("\nDesign statistics:")
println("  Maximum density: $(maximum(x_opt))")
println("  Minimum density: $(minimum(x_opt))")
println("  Mean density: $(sum(x_opt) / length(x_opt))")

# Verify that the gradient at the optimum is finite.
grad = Zygote.gradient(f, x_opt)[1]
println("\nGradient check:")
println("  Gradient norm: $(norm(grad))")
println("  All gradients finite: $(all(isfinite, grad))")

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add("Makie")` first and either `Pkg.add("CairoMakie")` or `Pkg.add("GLMakie")`
using Makie
using CairoMakie
# alternatively, `using GLMakie`
fig = visualize(problem; topology=result.minimizer)
Makie.display(fig)

#md # ## Plain Program
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [heat_sink.jl](heat_sink.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```