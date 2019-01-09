mutable struct VolumeFunction{T, dim, TI, TV, TP<:StiffnessTopOptProblem{dim, T}, TS<:AbstractFEASolver} <: AbstractFunction{T}
    problem::TP
    solver::TS
    cellvolumes::TV
    grad::TV
    total_volume::T
    fixed_volume::T
    tracing::Bool
	topopt_trace::TopOptTrace{T,TI}
    fevals::Int
    maxfevals::Int
end

function Base.getproperty(vf::VolumeFunction, f::Symbol)
    f === :reuse && return false
    return getfield(vf, f)
end
function Base.setproperty!(vf::VolumeFunction, f::Symbol, v)
    f === :reuse && return false
    return setfield!(vf, f, v)
end
GPUUtils.whichdevice(v::VolumeFunction) = whichdevice(v.cellvolumes)
@define_cu(VolumeFunction, :cellvolumes, :grad, :problem) # should be optimized to avoid replicating problem
Utilities.getsolver(v::VolumeFunction) = v.solver

function VolumeFunction(problem::StiffnessTopOptProblem{dim, T}, solver::AbstractFEASolver, ::Type{TI} = Int; tracing = true, maxfevals = 10^8) where {dim, T, TI}
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
    fixed_volume = dot(black, cellvolumes) #+ dot(white, cellvolumes)*xmin

    return VolumeFunction(problem, solver, cellvolumes, grad, total_volume, fixed_volume, tracing, TopOptTrace{T, TI}(), 0, maxfevals)
end
function (v::VolumeFunction{T})(x, grad) where {T}
    varind = v.problem.varind
    black = v.problem.black
    white = v.problem.white
    cellvolumes = v.cellvolumes
    total_volume = v.total_volume
    fixed_volume = v.fixed_volume
    v.fevals += 1

    tracing = v.tracing
    topopt_trace = v.topopt_trace
    dh = v.problem.ch.dh
    xmin = v.solver.xmin

    vol = compute_volume(cellvolumes, x, fixed_volume, varind, black, white)
    
    constrval = vol / total_volume
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
