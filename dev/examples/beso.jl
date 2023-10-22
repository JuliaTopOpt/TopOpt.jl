using TopOpt

E = 1.0 # Young’s modulus
v = 0.3 # Poisson’s ratio
f = 1.0; # downward force

nels = (160, 40)
problem = HalfMBB(Val{:Linear}, nels, (1.0, 1.0), E, v, f)

solver = FEASolver(Direct, problem; xmin=0.01, penalty=TopOpt.PowerPenalty(3.0))

comp = Compliance(solver)
volfrac = Volume(solver)
sensfilter = SensFilter(solver; rmin=4.0)
beso = BESO(comp, volfrac, 0.5, sensfilter)

x0 = ones(length(solver.vars))
result = beso(x0)

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
