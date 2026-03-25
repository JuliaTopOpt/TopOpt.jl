# Heat Sink Topology Optimization Example
# Demonstrates thermal compliance minimization for heat dissipation

using TopOpt, Ferrite, LinearAlgebra

println("Heat Sink Topology Optimization Example")
println("=" ^ 50)

# Problem parameters
nels = (60, 30)  # Mesh resolution
sizes = (1.0, 1.0)  # Element sizes
k = 1.0  # Thermal conductivity
heat_source = 1.0  # Volumetric heat generation
V = 0.5  # Volume fraction constraint

# Create heat conduction problem
# Temperature is fixed at left and right edges (heat sink fins)
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
println("Initial objective value (thermal compliance): $(f(x0))")

# Run optimization using MMA (Method of Moving Asymptotes)
model = Model(f)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, g)
alg = MMA87()
options = MMAOptions(; tol=Tolerance(; kkt=1e-4))
result = optimize(model, alg, x0; options)

println("\nOptimization complete!")
println("Final objective value (thermal compliance): $(result.minimum)")
println("Iterations: $(result.iter)")

# Get optimal design
x_opt = result.minimizer

# Compute final temperature field
solver.vars .= x_opt
solver()
T_max = maximum(solver.u)
println("Maximum temperature: $T_max")

# Print density statistics
println("\nDesign statistics:")
println("  Maximum density: $(maximum(x_opt))")
println("  Minimum density: $(minimum(x_opt))")
println("  Mean density: $(sum(x_opt) / length(x_opt))")

# Verify gradient after optimization
grad = Zygote.gradient(f, x_opt)[1]
println("\nGradient check:")
println("  Gradient norm: $(norm(grad))")
println("  All gradients negative (expected): $(all(grad .< 0))")

println("\nHeat sink optimization example completed successfully!")