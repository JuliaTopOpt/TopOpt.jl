using TimerOutputs

function simulate(problem::StiffnessTopOptProblem; round = true, hard = true, xmin = 0.001)
    if round 
        problem.vars .= round.(problem.vars)
        if hard
            solver = DirectDisplacementSolver(problem, xmin = 0.)
        else
            solver = DirectDisplacementSolver(problem, xmin = xmin)
        end
    else
        solver = DirectDisplacementSolver(problem, xmin = xmin)
    end

    solver(Val{true})
    comp = dot(s.u, s.globalinfo.f)

    return u, comp
end
