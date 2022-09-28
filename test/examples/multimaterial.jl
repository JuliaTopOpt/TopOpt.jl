using TopOpt, Test

Es = [1e-4, 1.0, 2.0] # Young’s modulii of base material + 2 materials
densities = [0.0, 1.0, 2.0] # for mass calculation
nmats = 3

v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problem = PointLoadCantilever(
    Val{:Linear},
    (160, 40),
    (1.0, 1.0),
    1.0, v, f,
)
ncells = 160 * 40
penalty = TopOpt.PowerPenalty(3.0)
interp = MaterialInterpolation(Es, penalty)

# Parameter settings

rmin = 3.0
filter = DensityFilter(solver; rmin=rmin)

x0 = fill(M, TopOpt.getncells(problem) * (length(Es) - 1))

solver = FEASolver(Direct, problem; xmin = 0.0)
comp = Compliance(solver)
obj = x -> comp(filter(PseudoDensities(x)))

M = 0.5 # mass fraction
function mass_constr(x::Matrix)
    return sum(x * densities) / ncells - M
end
function mass_constr(x::Vector)
    return mass_constr(reshape(x, ncells, nmats))
end
function multimat_constr(x::Vector)
end

# Define volume constraint
    global volfrac = Volume(solver)
    global constr = x -> volfrac(filter(PseudoDensities(x))) - V
    model = Model(obj)
    addvar!(model, zeros(length(x0)), ones(length(x0)))
    add_ineq_constraint!(model, constr)
    alg = MMA87()

    nsteps = 8
    ps = range(1.0, 5.0; length=nsteps + 1)
    tols = exp10.(range(-1, -3; length=nsteps + 1))
    global x = x0
    for j in 1:(nsteps + 1)
        p = ps[j]
        tol = tols[j]
        TopOpt.setpenalty!(solver, p)
        options = MMAOptions(; tol=Tolerance(; kkt=tol), maxiter=1000)
        res = optimize(model, alg, x; options, convcriteria)
        global x = res.minimizer
    end
end
