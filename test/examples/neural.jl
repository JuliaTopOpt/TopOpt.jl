using TopOpt, Zygote, ChainRulesCore
Nonconvex.@load Ipopt

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f)

V = 0.5
xmin = 0.0001
rmin = 4.0
p = 3.0
convcriteria = KKTCriteria()
solver = FEASolver(Direct, problem; xmin)
nvars = length(solver.vars)
x0 = fill(V, nvars)

penalty = TopOpt.PowerPenalty(p)
solver = FEASolver(Direct, problem; xmin, penalty)
filter = DensityFilter(solver; rmin)
comp = Compliance(solver)
volfrac = Volume(solver)

m = 20
act = leakyrelu
nn = Chain(Dense(2, m, act), Dense(m, m, act), Dense(m, m, act), Dense(m, 1, sigmoid))
nn_model = NeuralNetwork(nn, problem)
tf = TrainFunction(nn_model)
p0 = nn_model.init_params
tf(p0)

obj = p -> comp(filter(tf(p)))
constr = p -> volfrac(filter(tf(p))) - V

alg = MMA()
options = MMAOptions()

##
alg = IpoptAlg()
options = IpoptOptions(; max_iter=200)
model1 = Model()
nparams = length(p0)
addvar!(model1, fill(-100.0, nparams), fill(100.0, nparams))
set_objective!(model1, p -> (constr(p) + 0.1)^2)
res1 = optimize(model1, alg, p0; options)
##

struct LoggerFunction{F} <: Function
    f::F
end
function (lf::LoggerFunction)(x)
    @info "Solution extrema: $(extrema(x))"
    val = lf.f(x)
    @info "Objective value: $(val)"
    return val
end
function ChainRulesCore.rrule(lf::LoggerFunction, x)
    @info "Solution extrema: $(extrema(x))"
    val, pb = Zygote.pullback(lf.f, x)
    @info "Objective value: $(val)"
    return val, Δ -> begin
        temp = pb(Δ)
        return (NoTangent(), temp[1])
    end
end

# alg = IpoptAlg()
# nparams = length(p0)
# res2 = res1
# for _ in 1:10
#     global res2
#     # Incremental optimization
#     model2 = Model()
#     addvar!(model2, res2.minimizer .- 1.0, res2.minimizer .+ 1.0)
#     noise = randn(nparams)
#     set_objective!(model2, LoggerFunction(p -> obj(p) + 0.1 * norm(p - res2.minimizer) + dot(noise, p)))
#     # set_objective!(model2, LoggerFunction(obj))
#     add_ineq_constraint!(model2, p -> constr(p))
#     # res2 = optimize(model2, alg, res1.minimizer; options)
#     options = IpoptOptions(max_iter = 10)
#     res2 = optimize(model2, alg, res2.minimizer; options)

#     # Feasibility restoration
#     # model2 = Model()
#     # addvar!(model2, res2.minimizer .- 2.0, res2.minimizer .+ 2.0)
#     # set_objective!(model2, p -> norm(p - res2.minimizer))
#     # add_ineq_constraint!(model2, p -> 100*constr(p))
#     # options = IpoptOptions(max_iter = 5)
#     # res2 = optimize(model2, alg, res2.minimizer; options)
# end

# model2 = Model()
# addvar!(model2, fill(-10.0, nparams), fill(10.0, nparams))
# noise = randn(nparams)
# # set_objective!(model2, obj)
# set_objective!(model2, p -> obj(p) + 0.1 * norm(p))
# add_ineq_constraint!(model2, constr)
# # add_ineq_constraint!(model2, p -> 10*constr(p))
# options = IpoptOptions(max_iter = 300)
# res2 = optimize(model2, alg, res1.minimizer; options)

μ = 1.0
res2 = res1
# Increase the number of iterations to get a good design
for _ in 1:3
    global μ *= 2
    global res2
    model2 = Model()
    addvar!(model2, fill(-100.0, nparams), fill(100.0, nparams))
    noise = randn(nparams)
    # set_objective!(model2, obj)
    set_objective!(model2, p -> μ * obj(p) - log(max(0, -constr(p))))
    # Increase the following number e.g. to 100 for a better design
    options = IpoptOptions(; max_iter=20)
    res2 = optimize(model2, alg, res2.minimizer; options)
    @show extrema(tf(res2.minimizer))
    @show obj(res2.minimizer)
    @show constr(res2.minimizer)
end

Zygote.gradient(obj, p0)
Zygote.gradient(constr, p0)

tf(res2.minimizer)

using Makie
using CairoMakie
# using GLMakie

topology = filter(tf(res2.minimizer))
visualize(problem; topology)
