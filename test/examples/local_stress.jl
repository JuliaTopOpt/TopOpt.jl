# using Revise
using TopOpt, LinearAlgebra, StatsFuns, Test
using StatsFuns: logsumexp
using Makie
using CairoMakie
# using GLMakie

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 3.0

problems = Any[
    PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f),
    HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
    LBeam(Val{:Linear}, Float64),
]
problem_names = ["Cantilever beam", "Half MBB beam", "L-beam"]

i = 1
# for i in 1:length(problems)
println(problem_names[i])
problem = problems[i]
# Parameter settings
V = 0.5 # volume fraction
xmin = 0.001 # minimum density
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1

penalty = TopOpt.PowerPenalty(3.0)
# Define a finite element solver
solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
# Define compliance objective
stress = TopOpt.von_mises_stress_function(solver)
filter = DensityFilter(solver; rmin=rmin)
volfrac = TopOpt.Volume(solver)
nvars = length(solver.vars)
x0 = fill(1.0, nvars)
threshold = 3 * maximum(stress(filter(PseudoDensities(x0))))

x = copy(x0)
x .= 1
for p in 1.0:1.0:3.0
    global x, solver, stress, filter, volfrac, obj, constr, alg, options, model, r
    penalty = TopOpt.PowerPenalty(p)
    # Define a finite element solver
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
    # Define compliance objective
    stress = TopOpt.von_mises_stress_function(solver)
    filter = DensityFilter(solver; rmin=rmin)
    volfrac = TopOpt.Volume(solver)
    obj = x -> 2volfrac(filter(PseudoDensities(x)))
    constr =
        x -> begin
            s = stress(filter(PseudoDensities(x)))
            return [
                (s .- threshold) / threshold
                (logsumexp(s) - log(length(s)) - threshold) / threshold
            ]
        end
    alg = PercivalAlg()
    options = PercivalOptions(; max_iter=5, subsolver_max_eval=200)
    model = Nonconvex.Model(obj)
    Nonconvex.addvar!(model, zeros(nvars), ones(nvars))
    Nonconvex.add_ineq_constraint!(model, constr)
    r = Nonconvex.optimize(model, alg, x; options)
    x = r.minimizer
    @show obj(x)
end

s = stress(filter(PseudoDensities(x)))
@test (maximum(s) - threshold) / threshold < 0.01
fig = visualize(
    problem;
    topology=filter(PseudoDensities(x)),
    default_exagg_scale=0.07,
    scale_range=10.0,
    vector_linewidth=3,
    vector_arrowsize=0.5,
)
save("result.png", fig)
