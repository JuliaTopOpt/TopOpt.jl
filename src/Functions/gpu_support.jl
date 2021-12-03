using ..CUDASupport
using ..TopOpt: @init_cuda
@init_cuda()
using ..GPUUtils
import ..TopOpt: whichdevice
using ..TopOpt: GPU

### Compliance

whichdevice(c::Compliance) = whichdevice(c.cell_comp)
whichdevice(c::MeanCompliance) = whichdevice(c.compliance)

hutch_rand!(v::CuArray) = v .= round.(CuArrays.CURAND.curand(size(v))) .* 2 .- 1

function Compliance(
    ::GPU,
    problem::StiffnessTopOptProblem{dim,T},
    solver::AbstractDisplacementSolver,
    ::Type{TI}=Int;
    tracing=false,
    logarithm=false,
    maxfevals=10^8,
) where {dim,T,TI}
    comp = T(0)
    cell_comp = zeros(CuVector{T}, getncells(problem.ch.dh.grid))
    grad = CuVector(
        fill(T(NaN), length(cell_comp) - sum(problem.black) - sum(problem.white))
    )
    topopt_trace = TopOptTrace{T,TI}()
    reuse = false
    fevals = TI(0)
    return Compliance(
        problem,
        solver,
        comp,
        cell_comp,
        grad,
        tracing,
        topopt_trace,
        reuse,
        fevals,
        logarithm,
        maxfevals,
    )
end
@define_cu(Compliance, :solver, :cell_comp, :grad, :cheqfilter)

function compute_compliance(
    cell_comp::CuVector{T}, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin
) where {T}
    args = (cell_comp, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin)
    callkernel(dev, comp_kernel1, args)
    CUDAdrv.synchronize(ctx)
    obj = compute_obj(cell_comp, x, varind, black, white, penalty, xmin)

    return obj
end

# CUDA kernels
function comp_kernel1(
    cell_comp::AbstractVector{T},
    grad,
    cell_dofs,
    Kes,
    u,
    black,
    white,
    varind,
    x,
    penalty,
    xmin,
) where {T}
    i = @thread_global_index()
    offset = @total_threads()
    @inbounds while i <= length(cell_comp)
        cell_comp[i] = zero(T)
        Ke = rawmatrix(Kes[i])
        for w in 1:size(Ke, 2)
            for v in 1:size(Ke, 1)
                if Ke isa Symmetric
                    cell_comp[i] += u[cell_dofs[v, i]] * Ke.data[v, w] * u[cell_dofs[w, i]]
                else
                    cell_comp[i] += u[cell_dofs[v, i]] * Ke[v, w] * u[cell_dofs[w, i]]
                end
            end
        end
        if !(black[i] || white[i])
            d = ForwardDiff.Dual{T}(x[varind[i]], one(T))
            if PENALTY_BEFORE_INTERPOLATION
                p = density(penalty(d), xmin)
            else
                p = penalty(density(d, xmin))
            end
            grad[varind[i]] = -p.partials[1] * cell_comp[i]
        end

        i += offset
    end
    return nothing
end

function compute_obj(
    cell_comp::AbstractVector{T},
    x,
    varind,
    black,
    white,
    penalty,
    xmin,
    ::Val{blocksize}=Val(80),
    ::Val{threads}=Val(256),
) where {T,blocksize,threads}
    result = similar(cell_comp, T, (blocksize,))
    args = (result, cell_comp, x, varind, black, white, penalty, xmin, Val(threads))
    @cuda blocks = blocksize threads = threads comp_kernel2(args...)
    CUDAnative.synchronize()
    obj = reduce(+, Array(result))
    return obj
end

function comp_kernel2(
    result,
    cell_comp::AbstractVector{T},
    x,
    varind,
    black,
    white,
    penalty,
    xmin,
    ::Val{LMEM},
) where {T,LMEM}
    @mapreduce_block(
        i,
        length(cell_comp),
        +,
        T,
        LMEM,
        result,
        begin
            w_comp(cell_comp[i], x[varind[i]], black[i], white[i], penalty, xmin)
        end
    )

    return nothing
end
@inline function w_comp(comp::T, x, black, white, penalty, xmin) where {T}
    if PENALTY_BEFORE_INTERPOLATION
        return ifelse(
            black, comp, ifelse(white, xmin * comp, (d = ForwardDiff.Dual{T}(x, one(T));
            p = density(penalty(d), xmin);
            p.value * comp))
        )
    else
        return ifelse(
            black, comp, ifelse(white, xmin * comp, (d = ForwardDiff.Dual{T}(x, one(T));
            p = penalty(density(d, xmin));
            p.value * comp))
        )
    end
end

### Volume

whichdevice(v::Volume) = whichdevice(v.cellvolumes)
@define_cu(Volume, :cellvolumes, :grad, :problem) # should be optimized to avoid replicating problem

function compute_volume(
    cellvolumes::CuVector{T},
    x,
    fixed_volume,
    varind,
    black,
    white,
    ::Val{blocksize}=Val(80),
    ::Val{threads}=Val(256),
) where {T,blocksize,threads}
    result = similar(cellvolumes, T, (blocksize,))
    args = (result, cellvolumes, x, varind, black, white, Val(threads))
    @cuda blocks = blocksize threads = threads volume_kernel(args...)
    CUDAnative.synchronize()
    vol = reduce(+, Array(result)) + fixed_volume
    return vol
end

function volume_kernel(
    result, cellvolumes::AbstractVector{T}, x, varind, black, white, ::Val{LMEM}
) where {T,LMEM}
    @mapreduce_block(
        i,
        length(cellvolumes),
        +,
        T,
        LMEM,
        result,
        begin
            if !(black[i]) && !(white[i])
                x[varind[i]] * cellvolumes[i]
            else
                zero(T)
            end
        end
    )

    return nothing
end
