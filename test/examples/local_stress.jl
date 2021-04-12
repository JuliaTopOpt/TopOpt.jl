# using Revise
using TopOpt, LinearAlgebra
# using Makie
# using TopOpt.TopOptProblems.Visualization: visualize

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 3.0

problems = Any[
    PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f), 
    HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f), 
    LBeam(Val{:Linear}, Float64), 
    TieBeam(Val{:Quadratic}, Float64),
]
problem_names = [
    "Cantilever beam",
    "Half MBB beam",
    "L-beam",
    "Tie-beam",
]

for i in 1:length(problems)
    println(problem_names[i])
    problem = problems[i]
    # Parameter settings
    V = 0.5 # volume fraction
    xmin = 0.0001 # minimum density
    steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
    convcriteria = Nonconvex.KKTCriteria()
    solver = FEASolver(Displacement, Direct, problem, xmin = xmin)
    x0 = fill(1.0, length(solver.vars))
    for p in [1.0, 2.0, 3.0]
        #penalty = TopOpt.PowerPenalty(1.0)
        global penalty = TopOpt.PowerPenalty(p)
        # Define a finite element solver
        solver = FEASolver(
            Displacement, Direct, problem, xmin = xmin, penalty = penalty,
        )
        # Define compliance objective
        global stress = TopOpt.MicroVonMisesStress(solver)
        global filter = if problem isa TopOptProblems.TieBeam
            identity
        else
            DensityFilter(solver, rmin = rmin)
        end
        global volfrac = Volume(problem, solver)
        global comp = Compliance(problem, solver)

        obj = x -> volfrac(filter(x)) - V
        constr = x -> begin
            s = stress(filter(x))
            thr = 10
            vcat(
                (s .- thr) / 100,
                logsumexp(s) - log(length(s)) - thr,
                #comp(filter(x)) - 2000.0,
            )
        end
        alg = Nonconvex.PercivalAlg()
        options = Nonconvex.PercivalOptions()
        optimizer = Optimizer(
            obj, constr, x0, alg,
            options = options, convcriteria = convcriteria,
        )
        # Define continuation SIMP optimizer
        simp = SIMP(optimizer, solver, p)
        # Solve
        global result = simp(x0)
        x0 = result.topology
    end
    #visualize(
    #    problem; topology = penalty.(filter(result.topology)), default_exagg_scale = 0.07,
    #    scale_range = 10.0, vector_linewidth = 3, vector_arrowsize = 0.5,
    #)
end
