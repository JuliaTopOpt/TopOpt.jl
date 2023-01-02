using TopOpt, Test, Zygote, Test

Es = [1e-4, 1.0, 4.0] # Young's moduli of 3 materials (incl. void)

densities = [0.0, 0.5, 1.0] # for mass calc
nmats = 3

nu = 0.3 # Poisson's ratio
f = 1.0 # downward force

# problem definition
problem = PointLoadCantilever(
  Val{:Linear}, # order of bases functions
  (160, 40), # number of cells
  (1.0, 1.0), # cell dimensions
  1.0, # base Young's modulus
  nu, # Poisson's ratio
  f, # load
)
ncells = TopOpt.getncells(problem)

# FEA solver
solver = FEASolver(Direct, problem; xmin=0.0)

# density filter definition
filter = DensityFilter(solver; rmin=3.0)

# compliance function
comp = Compliance(solver)

# Young's modulus interpolation for compliance
penalty1 = TopOpt.PowerPenalty(4.0)
interp1 = MaterialInterpolation(Es, penalty1)

# density interpolation for mass constraint
penalty2 = TopOpt.PowerPenalty(1.0)
interp2 = MaterialInterpolation(densities, penalty2)

# objective function
obj = y -> begin
  x = tounit(MultiMaterialVariables(y, nmats))
  _E = interp1(filter(x))
  return comp(filter(_E))
end

# initial decision variables as a vector
y0 = zeros(ncells * (nmats - 1))

# testing the objective function
obj(y0)
# testing the gradient
Zygote.gradient(obj, y0)

# mass constraint
constr = y -> begin
  _rhos = interp2(MultiMaterialVariables(y, nmats))
  return sum(_rhos.x) / ncells - 0.4 # elements have unit volumes
end

# testing the mass constraint
constr(y0)
# testing the gradient
Zygote.gradient(constr, y0)

# building the optimization problem
model = Model(obj)
addvar!(
  model,
  fill(-10.0, length(y0)),
  fill(10.0, length(y0)),
)
add_ineq_constraint!(model, constr)

# optimization settings
alg = MMA87()
options = MMAOptions(;
  s_init = 0.1,
  tol=Tolerance(; kkt=1e-3),
)

y0 = zeros(ncells * (nmats - 1))

# solving the optimization problem
res = optimize(model, alg, y0; options)
y = res.minimizer

# testing the solution
@test constr(y) < 1e-6

x = TopOpt.tounit(reshape(y, ncells, nmats - 1))
sum(x[:, 2:3]) / size(x, 1) # the non-void elements as a ratio
@test all(==(1), sum(x; dims=2))
