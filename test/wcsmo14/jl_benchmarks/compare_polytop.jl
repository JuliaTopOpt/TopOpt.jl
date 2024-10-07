using TopOpt
using Makie
using CairoMakie
# using GLMakie

using TimerOutputs

println("Start running.")
# https://github.com/KristofferC/TimerOutputs.jl
to = TimerOutput()
reset_timer!(to)

# Define the problem
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 0.5; # downward force

# Parameter settings
V = 0.5 # volume fraction
# xmin = 0.001 # minimum density
xmin = 1e-6 # minimum density
rmin = 0.04; # density filter radius

nels = (720, 240) # (720, 240) # (360, 120) # | (720, 240) 
sizes = (3.0 / nels[1], 1.0 / nels[2])
@timeit to "problem def" problem = HalfMBB(Val{:Linear}, nels, sizes, E, v, f);

# Define a finite element solver
@timeit to "penalty def" penalty = TopOpt.PowerPenalty(3.0)
@timeit to "solver def" solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty);

# Define compliance objective
@timeit to "objective def" begin
    # Define compliance objective
    comp = Compliance(solver)
    filter = DensityFilter(solver; rmin=rmin)
    obj = x -> comp(filter(PseudoDensities(x)))
end

# Define volume constraint
@timeit to "constraint def" begin
    volfrac = TopOpt.Volume(solver)
    constr = x -> volfrac(filter(PseudoDensities(x))) - V
end

@timeit to "define problem" begin
    x0 = fill(V, length(solver.vars))
    model = Model(obj)
    addvar!(model, zeros(length(x0)), ones(length(x0)))
    add_ineq_constraint!(model, constr)
    alg = MMA87()
    convcriteria = GenericCriteria()
    options = MMAOptions(;
        maxiter=1000, tol=Tolerance(; x=1e-3, fabs=1e-3, frel=0.0, kkt=1e-3), convcriteria
    )
end

# Solve
# initial solution, critical to set it to volfrac! (blame non-convexity :)
@timeit to "simp run" r = optimize(model, alg, x0; options)

# Print the timings in the default way
println()
show(to)

@show obj(r.minimizer)

# Visualize the result using Makie.jl
fig = visualize(
    problem;
    topology=r.minimizer,
    default_exagg_scale=0.07,
    scale_range=10.0,
    vector_linewidth=3,
    vector_arrowsize=0.005,
    default_support_scale=0.01,
    default_load_scale=0.01,
)
Makie.display(fig)
