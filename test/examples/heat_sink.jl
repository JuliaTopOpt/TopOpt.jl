# Heat Sink Topology Optimization Example
# This script demonstrates topology optimization of a heat sink
# for maximizing heat dissipation (minimizing thermal compliance)

using TopOpt, Ferrite, LinearAlgebra

# Problem parameters
nels = (60, 30)  # Mesh resolution
sizes = (1.0, 1.0)  # Element sizes
k = 1.0  # Thermal conductivity
heat_source = 1.0  # Volumetric heat generation
V = 0.5  # Volume fraction constraint

# Create heat conduction problem
# Temperature is fixed at top and bottom edges (heat sink fins)
problem = HeatConductionProblem(
    Val{:Linear}, nels, sizes, k, heat_source;
    Tleft=100.0, Tright=0.0
)

println("Created heat conduction problem with $(Ferrite.getncells(problem)) elements")

# Create solver
solver = FEASolver(DirectSolver, problem; xmin=0.001)

# Create thermal compliance objective (heat dissipation)
comp = ThermalCompliance(solver)

# Create volume constraint
vol = TopOpt.Volume(solver; fraction=true)

# Create density filter for mesh-independent solution
filter = DensityFilter(solver; rmin=2.0)

# Set up optimization problem
f = x -> comp(filter(PseudoDensities(x)))
g = x -> [vol(filter(PseudoDensities(x))) - V]

# Initial design (uniform density)
x0 = fill(V, length(solver.vars))

println("Starting topology optimization...")
println("Volume constraint: $V")
println("Number of design variables: $(length(x0))")
println("Objective value (thermal compliance): $(f(x0))")

# Run optimization using MMA (Method of Moving Asymptotes)
model = Model(f)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, g)
alg = MMA87()
options = MMAOptions(; tol=Tolerance(; kkt=1e-4))
result = optimize(model, alg, x0; options)

println("\nOptimization complete!")
println("Objective value (thermal compliance): $(result.minimum)")
println("Iterations: $(result.iter)")

# Get optimal design
x_opt = result.minimizer

# Save result
# Using TopOpt's built-in save_mesh function
# save_mesh("heat_sink_optimal", problem, x_opt)

println("\nDesign saved. Maximum density: $(maximum(x_opt))")
println("Minimum density: $(minimum(x_opt))")

# Compute final temperature field
solver.vars .= x_opt
solver()
T_max = maximum(solver.u)
T_avg = mean(solver.u)
println("Maximum temperature: $T_max")
println("Average temperature: $T_avg")

# Visualize (if Makie is available)
# using GLMakie
# visualize(problem, topology = x_opt)
