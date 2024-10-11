using TimerOutputs

struct LinearElasticityResult{Tc,Tu}
    comp::Tc
    u::Tu
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, ::LinearElasticityResult)
    return println("TopOpt linear elasticity result")
end

function simulate(
    problem::StiffnessTopOptProblem,
    topology=ones(getncells(TopOptProblems.getdh(problem).grid));
    round=true,
    hard=true,
    xmin=0.001,
)
    if round
        if hard
            solver = FEASolver(Direct, problem; xmin=0.0)
        else
            solver = FEASolver(Direct, problem; xmin=xmin)
        end
    else
        solver = FEASolver(Direct, problem; xmin=xmin)
    end
    vars = solver.vars
    fill_vars!(vars, topology; round=round)
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
