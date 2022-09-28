using TopOpt, Test, Zygote

Es = [1e-4, 1.0, 2.0] # Young’s modulii of base material + 2 materials
densities = [1.0, 2.0] # for mass calculation
nmats = 2

v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problem = PointLoadCantilever(
    Val{:Linear},
    (160, 40),
    (1.0, 1.0),
    1.0, v, f,
)
ncells = TopOpt.getncells(problem)

# Parameter settings

rmin = 3.0
solver = FEASolver(Direct, problem; xmin = 0.0)
filter = DensityFilter(solver; rmin=rmin)

M = 0.5 # mass fraction
x0 = fill(M / nmats, ncells * (length(Es) - 1))

comp = Compliance(solver)
penalty = TopOpt.PowerPenalty(3.0)
interp = MaterialInterpolation(Es, penalty)
obj = x -> begin
    return MultiMaterialPseudoDensities(x, nmats) |> interp |> filter |> comp
end
obj(x0)
Zygote.gradient(obj, x0)

# mass constraint
constr1 = x -> begin
    ρs = MultiMaterialPseudoDensities(x, nmats)
    return sum(element_densities(ρs, densities)) / ncells - 1.0 # elements have unit volumes
end
constr1(x0)
Zygote.gradient(constr1, x0)

# material selection constraint
constr2 = x -> begin
    ρs = MultiMaterialPseudoDensities(x, nmats)
    # aggregation
    return sum(ρs, dims = 2) .- 1
end
constr2(x0)
Zygote.jacobian(constr2, x0)

model = Model(obj)
addvar!(model, zeros(length(x0)), ones(length(x0)))
add_ineq_constraint!(model, constr1)
add_ineq_constraint!(model, constr2)
alg = MMA87()

tol = 1e-5
options = MMAOptions(; tol=Tolerance(; kkt=tol), maxiter=1000)
Nonconvex.NonconvexCore.show_residuals[] = true
res = optimize(model, alg, x0; options)
x = res.minimizer
