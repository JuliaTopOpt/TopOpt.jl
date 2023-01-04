using TopOpt, Zygote, Flux

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
els = (160, 40)

problem = PointLoadCantilever(Val{:Linear}, els, (1.0, 1.0), E, v, f)

# problem settings
V = 0.5
xmin = 1e-6
rmin = 3.0

# SIMP penalty
p = 1.0
delta_p = 0.01
p_max = 5.0

penalty = TopOpt.PowerPenalty(p)
solver = FEASolver(Direct, problem; xmin, penalty)
cheqfilter = DensityFilter(solver; rmin)
comp = Compliance(solver)
volfrac = Volume(solver)

# constraint aggregation penalty
alpha = 0.1
delta_alpha = 0.05
alpha_max = 100

# neural network
m = 20
act = leakyrelu
nn = NeuralNetwork(
    Chain(
        Dense(2, m, act, init = Flux.glorot_normal),
        Dense(m, m, act, init = Flux.glorot_normal),
        Dense(m, m, act, init = Flux.glorot_normal),
        Dense(m, m, act, init = Flux.glorot_normal),
        Dense(m, m, act, init = Flux.glorot_normal),
        Dense(m, m, act, init = Flux.glorot_normal),
        softmax,
        x -> [x[1]],
    ),
    problem,
    scale = true,
)
w0 = nn.init_params
# w0 ./= 10

C0 = comp(cheqfilter(PseudoDensities(fill(V, TopOpt.getncells(problem)))))

# optimization
alg = Flux.Optimise.Adam(0.1)
clip_alg = Flux.Optimise.ClipValue(1.0)
w = copy(w0)
Δ = copy(w)
proj = HeavisideProjection(0.0)

# termination criteria
eps = Inf
eps_star = 0.05
maxiter = 100
epoch = 1
constr_tol = 0.01
violation = Inf
todensities(w; filter = true) = filter ? PseudoDensities(proj.(cheqfilter(nn(NNParams(w))).x)) : PseudoDensities(proj.(nn(NNParams(w)).x))
while true
    epoch > maxiter && break
    eps < eps_star && violation < constr_tol && break
    global penalty = TopOpt.PowerPenalty(p)
    global solver = FEASolver(Direct, problem; xmin, penalty)
    global cheqfilter = DensityFilter(solver; rmin)
    global comp = Compliance(solver)
    global volfrac = Volume(solver)

    global obj = w -> comp(todensities(w, filter = true)) / C0
    global constr = w -> volfrac(todensities(w, filter = false)) / V - 1
    global combined_obj = w -> obj(w) + alpha * constr(w)^2

    global Δ = Zygote.gradient(combined_obj, w)[1]
    @info "grad norm: $(norm(Δ))"
    Flux.Optimise.apply!(clip_alg, w, Δ)
    Flux.Optimise.apply!(alg, w, Δ)
    global w = w - Δ
    violation = constr(w)
    global alpha = min(alpha_max, alpha + delta_alpha)
    global p = min(p_max, p + delta_p)
    global epoch += 1
    global x = todensities(w; filter = false)
    global eps = sum(0.05 .< x.x .< 0.95) / length(x.x)
    @info "eps = $eps"
    @info "obj = $(comp(todensities(w; filter = true)))"
    @info "constr = $(volfrac(x) - V)"
    @show alpha
end

using Images, ImageInTerminal
reshape(Gray.(1 .- x), els...)'
