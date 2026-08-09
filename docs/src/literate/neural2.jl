# # Neural-network parametrized topology optimization (Adam optimizer)
#-
#md # !!! tip
#md #     This example is also available as a Jupyter notebook:
#md #     [`neural2.ipynb`](@__NBVIEWER_ROOT_URL__/examples/neural2.ipynb)
#-
# ## Commented Program
#
# Like the [neural example](@ref neural-plain-program), this script
# parametrizes the design with a feed-forward network from `Flux.jl`. The
# difference is the optimizer: instead of IPOPT, the network weights are
# updated with `Flux.Optimise.Adam` inside a continuation loop that
# progressively stiffens the SIMP penalty `p` and the constraint-aggregation
# weight `α`. The loop terminates when the design is sufficiently binary
# (`eps < eps_star`) and the volume violation is below tolerance.
#md # The full program, without comments, can be found in the next [section](@id neural2-plain-program).

using TopOpt, Zygote, Flux

# ### Define the problem
E = 1.0 # Young's modulus
v = 0.3 # Poisson's ratio
f = 1.0 # downward force
els = (160, 40)

problem = PointLoadCantilever(Val{:Linear}, els, (1.0, 1.0), E, v, f)

# ### Problem settings
V = 0.5       # volume fraction
xmin = 1e-6   # minimum density
rmin = 3.0    # filter radius

# SIMP penalty continuation: start at `p = 1`, increase by `delta_p` each
# epoch up to `p_max`.
p = 1.0
delta_p = 0.01
p_max = 5.0

penalty = TopOpt.PowerPenalty(p)
solver = FEASolver(DirectSolver, problem; xmin, penalty)
cheqfilter = DensityFilter(solver; rmin)
comp = Compliance(solver)
volfrac = Volume(solver)

# Constraint aggregation penalty `α` is increased each epoch up to `alpha_max`.
alpha = 0.1
delta_alpha = 0.05
alpha_max = 100

# ### Neural-network parametrization
# A deeper MLP than in the [`neural`](@ref neural-plain-program) example: six
# hidden layers followed by `softmax` and a slice that picks the first
# component as the cell density.
m = 20
act = leakyrelu
nn = NeuralNetwork(
    Chain(
        Dense(2, m, act; init=Flux.glorot_normal),
        Dense(m, m, act; init=Flux.glorot_normal),
        Dense(m, m, act; init=Flux.glorot_normal),
        Dense(m, m, act; init=Flux.glorot_normal),
        Dense(m, m, act; init=Flux.glorot_normal),
        Dense(m, m, act; init=Flux.glorot_normal),
        softmax,
        x -> [x[1]],
    ),
    problem;
    scale=true,
)
w0 = nn.init_params

# Normalize the initial compliance by the uniform design so the objective is
# scale-free.
C0 = comp(cheqfilter(PseudoDensities(fill(V, TopOpt.getncells(problem)))))

# ### Optimizer
alg = Flux.Optimise.Adam(0.1)
clip_alg = Flux.Optimise.ClipValue(1.0)
w = copy(w0)
Δ = copy(w)
proj = HeavisideProjection(0.0)

# ### Termination criteria
eps = Inf          # current fraction of "intermediate" densities (0.05 < x < 0.95)
eps_star = 0.05    # target intermediate-density fraction
maxiter = 100
epoch = 1
constr_tol = 0.01
violation = Inf

# `todensities` runs the network, projects to `[0, 1]`, and optionally
# applies the density filter. The filter is used for the compliance
# evaluation but skipped for the volume constraint, so the constraint acts
# on the raw network output.
function todensities(w; filter=true)
    if filter
        PseudoDensities(proj.(cheqfilter(nn(NNParams(w))).x))
    else
        PseudoDensities(proj.(nn(NNParams(w)).x))
    end
end

# ### Optimization loop
# Each iteration rebuilds the solver with the current penalty, evaluates the
# normalized compliance objective and the volume constraint, takes an Adam
# step on the combined objective `obj + α·constr²`, and tightens `p`/`α`.
while true
    epoch > maxiter && break
    eps < eps_star && violation < constr_tol && break
    global penalty = TopOpt.PowerPenalty(p)
    global solver = FEASolver(DirectSolver, problem; xmin, penalty)
    global cheqfilter = DensityFilter(solver; rmin)
    global comp = Compliance(solver)
    global volfrac = Volume(solver)

    global obj = w -> comp(todensities(w; filter=true)) / C0
    global constr = w -> volfrac(todensities(w; filter=false)) / V - 1
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
    global x = todensities(w; filter=false)
    global eps = sum(0.05 .< x.x .< 0.95) / length(x.x)
    @info "eps = $eps"
    @info "obj = $(comp(todensities(w; filter = true)))"
    @info "constr = $(volfrac(x) - V)"
    @show alpha
end

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add("Makie")` first and either `Pkg.add("CairoMakie")` or `Pkg.add("GLMakie")`
# (The original example displayed the design in the terminal with
# `Images.jl`/`ImageInTerminal.jl`; here we use Makie for consistency with
# the other examples.)
using Makie
using CairoMakie
# alternatively, `using GLMakie`
fig = visualize(problem; topology=cheqfilter(nn(NNParams(w))).x)
Makie.display(fig)

#md # ## [Plain Program](@id neural2-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [neural2.jl](neural2.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```