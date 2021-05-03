using TopOpt, LinearAlgebra, StatsFuns

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 3.0

problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f)

V = 0.5 # volume fraction
xmin = 0.0001 # minimum density
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1

x0 = fill(1.0, 160*40) # initial design
x = copy(x0)
for p in [1.0, 2.0, 3.0]
    global penalty, stress, filter, result, stress, x
    penalty = TopOpt.PowerPenalty(p)
    solver = FEASolver(
        Displacement, Direct, problem, xmin = xmin, penalty = penalty,
    )
    stress = TopOpt.MicroVonMisesStress(solver)
    filter = DensityFilter(solver, rmin = rmin)
    volfrac = TopOpt.Volume(problem, solver)

    obj = x -> volfrac(filter(x)) - V
    thr = 10 # stress threshold
    constr = x -> begin
        s = stress(filter(x))
        vcat(
            (s .- thr) / 100,
            logsumexp(s) - log(length(s)) - thr,
        )
    end
    alg = Nonconvex.PercivalAlg()
    options = Nonconvex.PercivalOptions()
    optimizer = Optimizer(
        obj, constr, x, alg,
        options = options,
    )
    simp = SIMP(optimizer, solver, p)
    result = simp(x)
    x = result.topology
end

maximum(stress(filter(x0))) # 0.51
maximum(stress(filter(x))) # 10.01

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

