using TopOpt

nels = (20, 10, 10)
sizes = (1.0, 1.0, 1.0)
E = 1.0
ν = 0.3
force = -1.0
problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force)
solver1 = FEASolver(Direct, problem)
# Takes forever to compile
solver2 = FEASolver(CG, MatrixFree, problem, abstol = 1e-7)
solver3 = FEASolver(CG, Assembly, problem, abstol = 1e-7)

x0 = rand(length(solver1.vars))
solver1.vars .= x0
solver2.vars .= x0
solver3.vars .= x0

solver1()
solver2()
solver3()

@test solver1.u ≈ solver2.u
@test solver1.u ≈ solver3.u
