abstract type AbstractConstraint <: Function end

struct VolConstr{T, dim, TI, TV, TP<:StiffnessTopOptProblem{dim, T}, TS<:AbstractFEASolver} <: AbstractConstraint
    problem::TP
    solver::TS
    volume_fraction::T
    cellvolumes::TV
    grad::TV
    total_volume::T
    design_volume::T
    fixed_volume::T
    tracing::Bool
	topopt_trace::TopOptTrace{T,TI}
end
whichdevice(v::VolConstr) = whichdevice(v.cellvolumes)
@define_cu(VolConstr, :cellvolumes, :grad, :problem) # should be optimized to avoid replicating problem

function VolConstr(problem::StiffnessTopOptProblem{dim, T}, solver::AbstractFEASolver, volume_fraction::T, ::Type{TI} = Int; tracing = true) where {dim, T, TI}
    cellvalues = solver.elementinfo.cellvalues
    dh = problem.ch.dh
    vars = solver.vars
    xmin = solver.xmin
    varind = problem.varind
    black = problem.black
    white = problem.white

    cellvolumes = solver.elementinfo.cellvolumes
    grad = zeros(T, length(vars))
    #_density = (x)->density(x, xmin)
    for (i, cell) in enumerate(CellIterator(dh))
        if !(black[i]) && !(white[i])
            #g = ForwardDiff.derivative(_density, vars[varind[i]])
            grad[varind[i]] = cellvolumes[i]#*g
        end
    end
    total_volume = sum(cellvolumes)
    design_volume = total_volume * volume_fraction
    fixed_volume = dot(black, cellvolumes) #+ dot(white, cellvolumes)*xmin

    return VolConstr(problem, solver, volume_fraction, cellvolumes, grad, total_volume, design_volume, fixed_volume, tracing, TopOptTrace{T, TI}())
end
function (v::VolConstr{T})(x, grad) where {T}
    varind = v.problem.varind
    black = v.problem.black
    white = v.problem.white
    cellvolumes = v.cellvolumes
    total_volume = v.total_volume
    fixed_volume = v.fixed_volume
    design_volume = v.design_volume
    volume_fraction = v.volume_fraction

    tracing = v.tracing
    topopt_trace = v.topopt_trace
    dh = v.problem.ch.dh
    xmin = v.solver.xmin

    vol = compute_volume(cellvolumes, x, fixed_volume, varind, black, white)
    
    constrval = vol / total_volume - volume_fraction
    grad .= v.grad ./ total_volume

    if tracing
        push!(topopt_trace.v_hist, vol/total_volume)
    end
    return constrval
end

function compute_volume(cellvolumes::Vector, x, fixed_volume, varind, black, white)
    vol = fixed_volume
    for i in 1:length(cellvolumes)
        if !(black[i]) && !(white[i])
            #vol += density(x[varind[i]], xmin)*cellvolumes[i]
            vol += x[varind[i]]*cellvolumes[i]
        end
    end
    return vol
end

function compute_volume(cellvolumes::CuVector{T}, x, fixed_volume, varind, black, white, ::Val{blocksize} = Val(80), ::Val{threads} = Val(256)) where {T, blocksize, threads}
    result = similar(cellvolumes, T, (blocksize,))
    args = (result, cellvolumes, x, varind, black, white, Val(threads))
    @cuda blocks = blocksize threads = threads volume_kernel(args...)
    CUDAnative.synchronize()
    vol = reduce(+, Array(result)) + fixed_volume
    return vol
end

function volume_kernel(result, cellvolumes::AbstractVector{T}, x, varind, black, white, ::Val{LMEM}) where {T, LMEM}
    @mapreduce_block(i, length(cellvolumes), +, T, LMEM, result, begin
        if !(black[i]) && !(white[i])
            x[varind[i]]*cellvolumes[i]
        else
            zero(T)
        end
    end)

    return
end
