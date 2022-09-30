using TopOpt, LinearAlgebra, StatsFuns
using StatsFuns: logsumexp

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 3.0

problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f)

V = 0.5 # volume fraction
xmin = 0.0001 # minimum density
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1

x0 = fill(0.5, 160 * 40) # initial design
N = length(x0)
penalty = TopOpt.PowerPenalty(1.0)
solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
stress = TopOpt.von_mises_stress_function(solver)
filter = DensityFilter(solver; rmin=rmin)
volfrac = TopOpt.Volume(solver)

obj = x -> volfrac(filter(PseudoDensities(x))) - V
thr = 150 # stress threshold
constr = x -> begin
    s = stress(filter(PseudoDensities(x)))
    return (s .- thr) / length(s)
end
alg = PercivalAlg()
options = PercivalOptions()
model = Model(obj)
addvar!(model, zeros(N), ones(N))
add_ineq_constraint!(model, constr)

x = copy(x0)
for p in [1.0, 2.0, 3.0]
    TopOpt.setpenalty!(solver, p)
    global r = optimize(model, alg, x; options)
    global x = r.minimizer
end

maximum(stress(filter(PseudoDensities(x0))))
maximum(stress(filter(PseudoDensities(x))))

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

