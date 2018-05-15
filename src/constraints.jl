abstract type AbstractConstraint <: Function end

struct VolConstr{T, dim, TI, TP<:StiffnessTopOptProblem{dim, T}, TS<:AbstractFEASolver} <: AbstractConstraint
    problem::TP
    solver::TS
    volume_fraction::T
    cell_volumes::Vector{T}
    grad::Vector{T}
    total_volume::T
    design_volume::T
    fixed_volume::T
    tracing::Bool
	topopt_trace::TopOptTrace{T,TI}
end
function VolConstr(problem::StiffnessTopOptProblem{dim, T}, solver::AbstractFEASolver, volume_fraction::T, ::Type{TI} = Int; tracing = true) where {dim, T, TI}
    cellvalues = solver.elementinfo.cellvalues
    dh = problem.ch.dh
    vars = solver.vars
    xmin = solver.xmin
    varind = problem.varind
    black = problem.black
    white = problem.white

    cell_volumes = solver.elementinfo.cellvolumes
    grad = zeros(T, length(vars))
    _density = (x)->density(x, xmin)
    for (i, cell) in enumerate(CellIterator(dh))
        if !(black[i]) && !(white[i])
            g = ForwardDiff.derivative(_density, vars[varind[i]])
            grad[varind[i]] = cell_volumes[i]*g
        end
    end
    total_volume = sum(cell_volumes)
    design_volume = total_volume * volume_fraction
    fixed_volume = dot(black, cell_volumes) + dot(white, cell_volumes)*xmin

    return VolConstr{T, dim, TI, typeof(problem), typeof(solver)}(problem, solver, volume_fraction, cell_volumes, grad, total_volume, design_volume, fixed_volume, tracing, TopOptTrace{T, TI}())
end
function (v::VolConstr{T})(x, grad) where {T}
    varind = v.problem.varind
    black = v.problem.black
    white = v.problem.white
    cell_volumes = v.cell_volumes
    total_volume = v.total_volume
    fixed_volume = v.fixed_volume
    design_volume = v.design_volume
    volume_fraction = v.volume_fraction

    tracing = v.tracing
    topopt_trace = v.topopt_trace
    dh = v.problem.ch.dh
    xmin = v.solver.xmin

    vol = fixed_volume
    for (i, cell) in enumerate(CellIterator(dh))
        if !(black[i]) && !(white[i])
            vol += density(x[varind[i]], xmin)*cell_volumes[i]
        end
    end

    constrval = vol / total_volume - volume_fraction
    grad .= v.grad ./ total_volume

    if tracing
        push!(topopt_trace.v_hist, vol/total_volume)
    end
    return constrval
end
