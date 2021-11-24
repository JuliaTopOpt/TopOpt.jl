using TopOpt

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f)

V = 0.5
xmin = 0.001
rmin = 4.0
p = 3.0
convcriteria = KKTCriteria()
solver = FEASolver(Direct, problem; xmin)
nvars = length(solver.vars)
x0 = fill(V, nvars)

penalty = TopOpt.PowerPenalty(p)
solver = FEASolver(Direct, problem; xmin, penalty)
filter = DensityFilter(solver; rmin)
comp = Compliance(problem, solver)
volfrac = Volume(problem, solver)

m = 10
nn = Chain(Dense(2, m, sigmoid), Dense(m, m, sigmoid), Dense(m, 1, sigmoid))
nn_model = NeuralNetwork(nn, problem)
tf = TrainFunction(nn_model)
p0 = nn_model.init_params
tf(p0)

obj = p -> comp(filter(tf(p)))
constr = p -> volfrac(filter(tf(p))) - V

alg = IpoptAlg()
options = IpoptOptions()

##
model1 = Model()
nparams = length(p0)
addvar!(model1, fill(-10.0, nparams), fill(10.0, nparams))
set_objective!(model1, constr)
res1 = optimize(model1, alg, p0; options)
##

model2 = Model()
nparams = length(p0)
addvar!(model2, fill(-10.0, nparams), fill(10.0, nparams))
set_objective!(model2, obj)
add_ineq_constraint!(model2, constr)

Zygote.gradient(obj, p0)
Zygote.gradient(constr, p0)

res2 = optimize(model2, alg, res1.minimizer; options)

using Makie, CairoMakie

topology = filter(res.minimizer)
visualize(problem; topology)
