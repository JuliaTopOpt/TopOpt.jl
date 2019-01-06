using TimerOutputs

struct LinearElasticityResult{T, TV}
    comp::T
    u::TV
end
GPUUtils.whichdevice(r::LinearElasticityResult) = whichdevice(r.u)

function simulate(problem::StiffnessTopOptProblem, topology = ones(getncells(TopOptProblems.getdh(problem).grid)); round = true, hard = true, xmin = 0.001)
    if round 
        if hard
            solver = FEASolver(Displacement, Direct, problem, xmin = 0.0)
        else
            solver = FEASolver(Displacement, Direct, problem, xmin = xmin)
        end
        if solver.vars isa GPUArray
            solver.vars = map(round, typeof(solver.vars)(topology))
        else
            solver.vars .= Base.round.(topology)
        end
    else
        solver = FEASolver(Displacement, Direct, problem, xmin = xmin)
        copyto!(solver.vars, topology)
    end

    solver(Val{true})
    comp = dot(solver.u, solver.globalinfo.f)

    return LinearElasticityResult(comp, copy(solver.u))
end
