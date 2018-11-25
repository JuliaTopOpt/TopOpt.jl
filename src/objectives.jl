abstract type AbstractObjective{T} <: Function end

mutable struct ComplianceObj{T, TI<:Integer, TSP<:StiffnessTopOptProblem, FS<:AbstractDisplacementSolver, CF<:AbstractCheqFilter} <: AbstractObjective{T}
	problem::TSP
    solver::FS
    cheqfilter::CF
    comp::T
    cell_comp::Vector{T}
    grad::Vector{T}
    tracing::Bool
    topopt_trace::TopOptTrace{T,TI}
    reuse::Bool
    fevals::TI
    logarithm::Bool
end
function ComplianceObj(problem::StiffnessTopOptProblem{dim, T}, solver::AbstractDisplacementSolver, ::Type{TI}=Int; rmin = T(0), filtering = true, tracing = false, logarithm = false) where {dim, T, TI}
    cheqfilter = CheqFilter{filtering}(solver, rmin)
    comp = T(0)
    cell_comp = zeros(T, getncells(problem.ch.dh.grid))
    grad = fill(T(NaN), length(cell_comp) - sum(problem.black) - sum(problem.white))
    topopt_trace = TopOptTrace{T,TI}()
    reuse = false
    fevals = TI(0)
    return ComplianceObj(problem, solver, cheqfilter, comp, cell_comp, grad, tracing, topopt_trace, reuse, fevals, logarithm)
end

getsolver(obj::AbstractObjective) = obj.solver
getpenalty(obj::AbstractObjective) = getpenalty(getsolver(obj))
setpenalty!(obj::AbstractObjective, p) = setpenalty!(getsolver(obj), p)
getprevpenalty(obj::AbstractObjective) = getprevpenalty(getsolver(obj))

function (o::ComplianceObj{T})(x, grad) where {T}
    @timeit to "Eval obj and grad" begin
        #if o.solver.vars ≈ x && getpenalty(o).p ≈ getprevpenalty(o).p
        #    grad .= o.grad
        #    return o.comp
        #end

        penalty = getpenalty(o)
        cell_dofs = o.problem.metadata.cell_dofs
        u = o.solver.u
        cell_comp = o.cell_comp
        Kes = o.solver.elementinfo.Kes
        black = o.problem.black
        white = o.problem.white
        xmin = o.solver.xmin
        varind = o.problem.varind

        copyto!(o.solver.vars, x)
        if o.reuse
            if !o.tracing
                o.reuse = false
            end
        else
            o.fevals += 1
            o.solver()
        end

        obj = compute_compliance(cell_comp, grad, cell_dofs, cell_comp, Kes, u, 
                            black, white, varind, x, penalty, xmin)

        if o.logarithm
            o.comp = log(obj)
            grad ./= obj
        else
            o.comp = obj# / length(cell_comp)
            #scale!(grad, 1/length(cell_comp))
            #o.comp = obj
        end
        o.cheqfilter(grad)
        o.grad .= grad
        
        if o.tracing
            if o.reuse
                o.reuse = false
            else
                push!(o.topopt_trace.c_hist, obj)
                push!(o.topopt_trace.x_hist, copy(x))
                if length(o.topopt_trace.x_hist) == 1
                    push!(o.topopt_trace.add_hist, 0)
                    push!(o.topopt_trace.rem_hist, 0)
                else
                    push!(o.topopt_trace.add_hist, sum(o.topopt_trace.x_hist[end] .> o.topopt_trace.x_hist[end-1]))
                    push!(o.topopt_trace.rem_hist, sum(o.topopt_trace.x_hist[end] .< o.topopt_trace.x_hist[end-1]))
                end
            end
        end
    end
    return o.comp::T
end

function compute_compliance(cell_comp::Vector{T}, grad, cell_dofs, cell_comp, Kes, u, 
                            black, white, varind, x, penalty, xmin) where {T}
    obj = zero(T)
    @inbounds for i in 1:size(cell_dofs, 2)
        cell_comp[i] = zero(T)
        Ke = Kes[i]
        for w in 1:size(Ke,2)
            for v in 1:size(Ke, 1)
                cell_comp[i] += u[cell_dofs[v,i]]*Ke[v,w]*u[cell_dofs[w,i]]
            end
        end

        if black[i]
            obj += cell_comp[i]
        elseif white[i]
            obj += xmin * cell_comp[i] 
        else
            d = ForwardDiff.Dual{T}(x[varind[i]], one(T))
            p = density(penalty(d), xmin)
            grad[varind[i]] = -p.partials[1] * cell_comp[i]
            obj += p.value * cell_comp[i]
        end
    end

    return obj    
end

# CUDA kernels
function comp_kernel1(cell_comp::CuVector{T}, grad, cell_dofs, cell_comp, Kes, u, 
    black, white, varind, x, penalty, xmin) where {N, T, TV<:SVector{N, T}}
    
    blockid = blockIdx().x + blockIdx().y * gridDim().x
    i = blockid * (blockDim().x * blockDim().y) + (threadIdx().y * blockDim().x) + threadIdx().x
    if i <= length(cell_comp)
        cell_comp[i] = zero(T)
        Ke = Kes[i]
        for w in 1:size(Ke,2)
            for v in 1:size(Ke, 1)
                cell_comp[i] += u[cell_dofs[v,i]]*Ke[v,w]*u[cell_dofs[w,i]]
            end
        end

        if !(black[i] || white[i])
            d = ForwardDiff.Dual{T}(x[varind[i]], one(T))
            p = density(penalty(d), xmin)
            grad[varind[i]] = -p.partials[1] * cell_comp[i]
        end
    end

    return
end

function mapreducedim_kernel_parallel(f, op, R::CuDeviceArray{T}, A::CuDeviceArray{T},
                             CIS, Rlength, Slength) where {T}
    
    for Ri_base in 0:(gridDim().x * blockDim().y):(Rlength-1)
        Ri = Ri_base + (blockIdx().x - 1) * blockDim().y + threadIdx().y
        Ri > Rlength && return
        RI = Tuple(CartesianIndices(R)[Ri])
        S = @cuStaticSharedMem(T, 512)
        Si_folded_base = (threadIdx().y - 1) * blockDim().x
        Si_folded = Si_folded_base + threadIdx().x
        # serial reduction of A into S by Slength ÷ xthreads
        for Si_base in 0:blockDim().x:(Slength-1)
            Si = Si_base + threadIdx().x
            Si > Slength && break
            SI = Tuple(CIS[Si])
            AI = ifelse.(size(R) .== 1, SI, RI)
            if Si_base == 0
                S[Si_folded] = f(A[AI...])
            else
                S[Si_folded] = op(S[Si_folded], f(A[AI...]))
            end
        end
        # block-parallel reduction of S to S[1] by xthreads
        reduce_block(view(S, (Si_folded_base + 1):512), op)
        # reduce S[1] into R
        threadIdx().x == 1 && (R[Ri] = op(R[Ri], S[Si_folded]))
    end
    return
end

@inline function reduce_block(arr, op)
    sync_threads()
    len = blockDim().x
    while len != 1
        sync_threads()
        skip = (len + 1) >> 1
        reduce_to = threadIdx().x - skip
        if 0 < reduce_to <= (len >> 1)
            arr[reduce_to] = op(arr[reduce_to], arr[threadIdx().x])
        end
        len = skip
    end
    sync_threads()
    
    return 
end

function compute_compliance(cell_comp::CuVector{T}, grad, cell_dofs, cell_comp, Kes, u, 
                            black, white, varind, x, penalty, xmin) where {T}

    args1 = (grad, cell_dofs, cell_comp, Kes, u, black, white, varind, x, penalty, xmin)
    callkernel(dev, comp_kernel1, args1)
    CUDAdrv.synchronize(ctx)

    # Use mapreduce for obj
    if black[i]
        obj += cell_comp[i]
    elseif white[i]
        obj += xmin * cell_comp[i] 
    else
        d = ForwardDiff.Dual{T}(x[varind[i]], one(T))
        p = density(penalty(d), xmin)
        grad[varind[i]] = -p.partials[1] * cell_comp[i]
        obj += p.value * cell_comp[i]
    end

    return obj    
end

function (o::ComplianceObj{T})(to, x, grad) where {T}
    penalty = getpenalty(o)
    prev_penalty = getprevpenalty(o)
    if o.solver.vars ≈ x && penalty.p ≈ prev_penalty.p
        grad .= o.grad
        return o.comp
    end

    cell_dofs = o.problem.metadata.cell_dofs
    u = o.solver.u
    cell_comp = o.cell_comp
    Kes = o.solver.elementinfo.Kes
    black = o.problem.black
    white = o.problem.white
    xmin = o.solver.xmin
    varind = o.problem.varind

    copyto!(o.solver.vars, x)
    if o.reuse
        if !o.tracing
            o.reuse = false
        end
    else
        o.fevals += 1
        o.solver()
    end

    obj = zero(T)
    @timeit to "Compute gradient" @inbounds for i in 1:size(cell_dofs, 2)
        cell_comp[i] = zero(T)
        Ke = Kes[i]
        for w in 1:size(Ke,2)
            for v in 1:size(Ke, 1)
                cell_comp[i] += u[cell_dofs[v,i]]*Ke[v,w]*u[cell_dofs[w,i]]
            end
        end

        if black[i]
            obj += cell_comp[i]
        elseif white[i]
            obj += xmin * cell_comp[i] 
        else
            d = ForwardDiff.Dual{T}(x[varind[i]], one(T))
            p = density(penalty(d), xmin)
            grad[varind[i]] = -p.partials[1] * cell_comp[i]
            obj += p.value * cell_comp[i]
        end
    end
    if o.logarithm
        o.comp = log(obj)
        grad ./= obj
    else
        o.comp = obj# / length(cell_comp)
        #scale!(grad, 1/length(cell_comp))
        #o.comp = obj
    end
    @timeit to "Chequerboard filtering" o.cheqfilter(grad)
    o.grad .= grad

    @timeit to "Tracing" if o.tracing
        if o.reuse
            o.reuse = false
        else
            push!(o.topopt_trace.c_hist, obj)
            push!(o.topopt_trace.x_hist, copy(x))
            if length(o.topopt_trace.x_hist) == 1
                push!(o.topopt_trace.add_hist, 0)
                push!(o.topopt_trace.rem_hist, 0)
            else
                push!(o.topopt_trace.add_hist, sum(o.topopt_trace.x_hist[end] .> o.topopt_trace.x_hist[end-1]))
                push!(o.topopt_trace.rem_hist, sum(o.topopt_trace.x_hist[end] .< o.topopt_trace.x_hist[end-1]))
            end
        end
    end
    return o.comp::T
end
