using TopOpt, Test, Zygote, Test

Es = [1e-6, 0.5, 2.0] # Young’s modulii of base material + 2 materials
densities = [0.0, 0.5, 1.0] # for mass calculation
nmats = 3

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

M = 1/nmats/2 # mass fraction
x0 = fill(M, ncells * (length(Es) - 1))

comp = Compliance(solver)
penalty = TopOpt.PowerPenalty(3.0)
interp = MaterialInterpolation(Es, penalty)
obj = x -> begin
    return MultiMaterialVariables(x, nmats) |> interp |> filter |> comp
end
obj(x0)
Zygote.gradient(obj, x0)

# mass constraint
constr = x -> begin
    ρs = PseudoDensities(MultiMaterialVariables(x, nmats))
    return sum(element_densities(ρs, densities)) / ncells - 0.3 # elements have unit volumes
end
constr(x0)
Zygote.gradient(constr1, x0)

model = Model(obj)
addvar!(model, fill(-10.0, length(x0)), fill(10.0, length(x0)))
add_ineq_constraint!(model, constr)

tol = 1e-3
alg = MMA87()
options = MMAOptions(; tol=Tolerance(; kkt=tol), maxiter=1000)

res = optimize(model, alg, x0; options)
x = res.minimizer
ρs = PseudoDensities(MultiMaterialVariables(x, nmats))
@test constr(x) < 1e-6
@test constr(x0) > 0
@test all(==(1), sum(ρs, dims = 2))
sum(ρs[:,2:3]) / size(ρs, 1) # the material elements as a ratio

for i in 1:3
    @test minimum(abs, ρs[:,i] .- 0.5) > 0.48 # mostly binary design
end
