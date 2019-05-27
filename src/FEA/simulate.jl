using TimerOutputs

@params struct LinearElasticityResult
    comp
    u
end

function simulate(problem::StiffnessTopOptProblem, topology = ones(getncells(TopOptProblems.getdh(problem).grid)); round = true, hard = true, xmin = 0.001)
    if round 
        if hard
            solver = FEASolver(Displacement, Direct, problem, xmin = 0.0)
        else
            solver = FEASolver(Displacement, Direct, problem, xmin = xmin)
        end
    else
        solver = FEASolver(Displacement, Direct, problem, xmin = xmin)
    end
    fill_vars!(vars, problem, topology; round = round)
    solver(Val{true})
    comp = dot(solver.u, solver.globalinfo.f)
    return LinearElasticityResult(comp, copy(solver.u))    
end

function fill_vars!(vars::Array, topology; round)
    if round 
        vars .= Base.round.(topology)
    else
        copyto!(vars, topology)
    end
    return vars
end
