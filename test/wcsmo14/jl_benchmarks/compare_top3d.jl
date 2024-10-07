using TopOpt
Nonconvex.@load Ipopt
using Makie
using CairoMakie
# using GLMakie

using Suppressor

using FromFile
@from "new_problems.jl" import NewPointLoadCantilever

using TimerOutputs

println("Start running.")
# https://github.com/KristofferC/TimerOutputs.jl
to = TimerOutput()
reset_timer!(to)

# Define the problem
E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0; # downward force

# Parameter settings
V = 0.3 # volume fraction
# xmin = 0.001 # minimum density
xmin = 1e-6 # minimum density
rmin = 2.0; # density filter radius

nels = (96, 48, 48)
sizes = (1.0, 1.0, 1.0)
@timeit to "problem def" problem = NewPointLoadCantilever(
    Val{:Linear}, nels, sizes, E, v, f
);

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

@timeit to "problem definition" begin
    x0 = fill(V, length(solver.vars))
    model = Model(obj)
    addvar!(model, zeros(length(x0)), ones(length(x0)))
    add_ineq_constraint!(model, constr)
    convcriteria = GenericCriteria()
    options = MMAOptions(;
        maxiter=300, tol=Tolerance(; x=1e-3, fabs=1e-3, frel=0.0), convcriteria
    )
end

@timeit to "simp run" r = optimize(model, alg, x0; options)

# Print the timings in the default way
show(to)
println()

output = @capture_out begin
    show(to)
    @show obj(r.minimizer)
    @show constr(r.minimizer)
end;

open("jl-top3d.matlab_$(nels).txt", "w") do io
    write(io, output)
end

# Visualize the result using Makie.jl
fig = visualize(
    problem;
    topology=r.minimizer,
    default_exagg_scale=0.07,
    scale_range=10.0,
    vector_linewidth=3,
    vector_arrowsize=0.5,
)
Makie.display(fig)

Makie.save("jl-top3d.matlab__$(nels).png", fig)
