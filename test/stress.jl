#using Revise
using TopOpt

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0 # downward force

problems = Any[PointLoadCantilever(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f), 
            HalfMBB(Val{:Linear}, (60, 20), (1.0, 1.0), E, v, f), 
            LBeam(Val{:Linear}, Float64), 
            TieBeam(Val{:Quadratic}, Float64)]
problem_names = ["cantilever beam", "half MBB beam", "L-beam", "tie-beam"]
approx_objvals = [330.0, 175.0, 65.0, 1413.0]

i = 1

problem = problems[i]
# Parameter settings
V = 0.5 # volume fraction
xmin = 0.001 # minimum density
steps = 40 # maximum number of penalty steps, delta_p0 = 0.1
reuse = false # adaptive penalty flag
convcriteria = MMA.KKTCriteria()
#penalty = TopOpt.PowerPenalty(1.0)
penalty = TopOpt.PowerPenalty(1.0)
pcont = Continuation(penalty, steps = steps, xmin = xmin, pmax = 5.0)

mma_options = options = MMA.Options(maxiter=1000, tol = MMA.Tolerances(kkttol = 1e-4))
if convcriteria isa MMA.KKTCriteria
    maxtol = 0.01 # maximum tolerance
    mintol = 0.001 # minimum tolerance

    b = log(mintol / maxtol) / steps
    a = maxtol / exp(b)
    mma_options_gen = TopOpt.MMAOptionsGen(steps = steps, initial_options = mma_options, kkttol_gen = ExponentialContinuation(a, b, 0.0, steps + 1, mintol))
else
    maxtol = 0.01 # maximum tolerance
    mintol = 0.0001 # minimum tolerance

    b = log(mintol / maxtol) / steps
    a = maxtol / exp(b)
    mma_options_gen = TopOpt.MMAOptionsGen(steps = steps, initial_options = mma_options, ftol_gen = ExponentialContinuation(a, b, 0.0, steps + 1, mintol))
end
csimp_options = TopOpt.CSIMPOptions(steps = steps, 
                                    options_gen = mma_options_gen, 
                                    p_gen = pcont, 
                                    reuse = reuse
                                    )

# Define a finite element solver
solver = FEASolver(Displacement, Direct, problem, xmin = xmin, penalty = penalty)
# Define compliance objective
filterT = problem isa TopOptProblems.TieBeam ? nothing : DensityFilter
obj = Objective(Volume(problem, solver))
constr = Constraint(TopOpt.GlobalStress(solver), 1.0 + eps(0.0))
#cu_obj = TopOpt.cu(obj)
# Define volume constraint
# Define subproblem optimizer
#optimizer = MMAOptimizer{CPU}(cu_obj, constr, MMA.MMA87(),
#    ConjugateGradient(), maxiter=1000); optimizer.obj.fevals = 0
optimizer = MMAOptimizer{CPU}(obj, constr, MMA.MMA87(),
    ConjugateGradient(), options = mma_options, convcriteria = convcriteria); optimizer.obj.f.fevals = 0

# Define continuation SIMP optimizer
simp = SIMP(optimizer, penalty.p)

cont_simp = ContinuationSIMP(simp, steps, csimp_options) 

# Solve
x0 = fill(1.0, length(solver.vars))
result = cont_simp(x0)
