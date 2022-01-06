using TopOpt

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problems = Any[
    PointLoadCantilever(Val{:Linear}, (60, 20, 20), (1.0, 1.0, 1.0), E, v, f),
    PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), E, v, f),
    HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f),
    LBeam(Val{:Linear}, Float64; force=f),
    TieBeam(Val{:Quadratic}, Float64),
]
problem_names = [
    "3d cantilever beam", "cantilever beam", "half MBB beam", "L-beam", "tie-beam"
]

i = 2
println(problem_names[i])
problem = problems[i]

V = 0.5 # volume fraction
xmin = 0.001 # minimum density
rmin = 3.0
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
reuse = true # adaptive penalty flag

convcriteria = Nonconvex.GenericCriteria()
penalty = TopOpt.PowerPenalty(1.0)
pcont = Continuation(penalty; steps=steps, xmin=xmin, pmax=5.0)

mma_options = options = MMAOptions(; maxiter=1000)
maxtol = 0.01 # maximum tolerance
mintol = 0.0001 # minimum tolerance
b = log(mintol / maxtol) / steps
a = maxtol / exp(b)
mma_options_gen = TopOpt.MMAOptionsGen(;
    steps=steps,
    initial_options=mma_options,
    ftol_gen=ExponentialContinuation(a, b, 0.0, steps + 1, mintol),
)
csimp_options = TopOpt.CSIMPOptions(;
    steps=steps, options_gen=mma_options_gen, p_gen=pcont, reuse=reuse
)

solver = FEASolver(Direct, problem; xmin=xmin, penalty=penalty)

comp = Compliance(problem, solver)
filter = if problem isa TopOptProblems.TieBeam
    identity
else
    DensityFilter(solver; rmin=rmin)
end
obj = x -> comp(filter(x))

volfrac = TopOpt.Volume(problem, solver)
constr = x -> volfrac(filter(x)) - V

x0 = fill(V, length(solver.vars))
optimizer = Optimizer(
    obj, constr, x0, MMA87(); options=mma_options, convcriteria=convcriteria
)

simp = SIMP(optimizer, solver, penalty.p)
cont_simp = ContinuationSIMP(simp, steps, csimp_options)

result = cont_simp(x0)

@show result.convstate
@show optimizer.workspace.iter
@show result.objval

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

