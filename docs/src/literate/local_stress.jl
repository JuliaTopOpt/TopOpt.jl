# using Revise
using TopOpt, LinearAlgebra, StatsFuns
using StatsFuns: logsumexp

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force
rmin = 3.0

# ### Define the problem
problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f)

# ### Parameter settings
V = 0.5 # volume fraction
xmin = 0.0001 # minimum density
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1

# ### Continuation SIMP
x0 = fill(0.5, 160 * 40) # initial design
x = copy(x0)
for p in [1.0, 2.0, 3.0]
    global penalty, stress, filter, result, stress, x
    penalty = TopOpt.PowerPenalty(p)
    solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)
    stress = TopOpt.von_mises_stress_function(solver)
    filter = DensityFilter(solver; rmin=rmin)
    volfrac = TopOpt.Volume(problem, solver)

    obj = x -> volfrac(filter(x)) - V
    thr = 150 # stress threshold
    constr = x -> begin
        s = stress(filter(x))
        return (s .- thr) / length(s)
    end
    alg = PercivalAlg()
    options = PercivalOptions()
    optimizer = Optimizer(obj, constr, x, alg; options=options)
    simp = SIMP(optimizer, solver, p)
    result = simp(x)
    x = result.topology
end

maximum(stress(filter(x0)))
maximum(stress(filter(x)))

# ### (Optional) Visualize the result using Makie.jl
# Need to run `using Pkg; Pkg.add(Makie)` first
# ```julia
# using Makie
# using TopOpt.TopOptProblems.Visualization: visualize
# fig = visualize(
#    problem; topology = penalty.(filter(result.topology)), default_exagg_scale = 0.07,
#    scale_range = 10.0, vector_linewidth = 3, vector_arrowsize = 0.5,
# )
# Makie.display(fig)
# ```

#md # ## [Plain Program](@id local-stress-plain-program)
#md #
#md # Below follows a version of the program without any comments.
#md # The file is also available here: [local-stress.jl](local_stress.jl)
#md #
#md # ```julia
#md # @__CODE__
#md # ```
