abstract type AbstractObjective{T} <: Function end

mutable struct ComplianceObj{T, TI<:Integer, TV<:AbstractArray, TSP<:StiffnessTopOptProblem, FS<:AbstractDisplacementSolver, CF<:AbstractCheqFilter} <: AbstractObjective{T}
	problem::TSP
    solver::FS
    cheqfilter::CF
    comp::T
    cell_comp::TV
    grad::TV
    tracing::Bool
    topopt_trace::TopOptTrace{T,TI}
    reuse::Bool
    fevals::TI
    logarithm::Bool
end
whichdevice(c::ComplianceObj) = whichdevice(c.cell_comp)

function ComplianceObj(problem, solver::AbstractDisplacementSolver, args...; kwargs...)
    ComplianceObj(whichdevice(solver), problem, solver, args...; kwargs...)
end
function ComplianceObj(::CPU, problem::StiffnessTopOptProblem{dim, T}, solver::AbstractDisplacementSolver, ::Type{TI}=Int; rmin = T(0), filtering = true, tracing = false, logarithm = false) where {dim, T, TI}
    cheqfilter = CheqFilter(Val(filtering), solver, rmin)
    comp = T(0)
    cell_comp = zeros(T, getncells(problem.ch.dh.grid))
    grad = fill(T(NaN), length(cell_comp) - sum(problem.black) - sum(problem.white))
    topopt_trace = TopOptTrace{T,TI}()
    reuse = false
    fevals = TI(0)
    return ComplianceObj(problem, solver, cheqfilter, comp, cell_comp, grad, tracing, topopt_trace, reuse, fevals, logarithm)
end
function ComplianceObj(::GPU, problem::StiffnessTopOptProblem{dim, T}, solver::AbstractDisplacementSolver, ::Type{TI}=Int; rmin = T(0), filtering = true, tracing = false, logarithm = false) where {dim, T, TI}
    cheqfilter = cu(CheqFilter(Val(filtering), solver, rmin))
    comp = T(0)
    cell_comp = zeros(CuVector{T}, getncells(problem.ch.dh.grid))
    grad = CuVector(fill(T(NaN), length(cell_comp) - sum(problem.black) - sum(problem.white)))
    topopt_trace = TopOptTrace{T,TI}()
    reuse = false
    fevals = TI(0)
    return ComplianceObj(problem, solver, cheqfilter, comp, cell_comp, grad, tracing, topopt_trace, reuse, fevals, logarithm)
end

@define_cu(ComplianceObj, :solver, :cell_comp, :grad, :cheqfilter)

getsolver(obj::AbstractObjective) = obj.solver
getpenalty(obj::AbstractObjective) = getpenalty(getsolver(obj))
setpenalty!(obj::AbstractObjective, p) = setpenalty!(getsolver(obj), p)
getprevpenalty(obj::AbstractObjective) = getprevpenalty(getsolver(obj))

function (o::ComplianceObj{T})(x, grad) where {T}
    @unpack cell_comp, solver, tracing, cheqfilter, topopt_trace = o
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes, black, white, varind = elementinfo
    @unpack cell_dofs = metadata

    @timeit to "Eval obj and grad" begin
        #if o.solver.vars ≈ x && getpenalty(o).p ≈ getprevpenalty(o).p
        #    grad .= o.grad
        #    return o.comp
        #end

        penalty = getpenalty(o)    
        copyto!(solver.vars, x)
        if o.reuse
            if !tracing
                o.reuse = false
            end
        else
            o.fevals += 1
            solver()
        end
        obj = compute_compliance(cell_comp, grad, cell_dofs, Kes, u, 
                            black, white, varind, x, penalty, xmin)

        if o.logarithm
            o.comp = log(obj)
            grad ./= obj
        else
            o.comp = obj
            #scale!(grad, 1/length(cell_comp))
            #o.comp = obj
        end
        cheqfilter(grad, x, elementinfo)
        copyto!(o.grad, grad)
        
        if o.tracing
            if o.reuse
                o.reuse = false
            else
                push!(topopt_trace.c_hist, obj)
                if x isa GPUArray
                    push!(topopt_trace.x_hist, Array(x))
                else
                    push!(topopt_trace.x_hist, copy(x))
                end
                if length(topopt_trace.x_hist) == 1
                    push!(topopt_trace.add_hist, 0)
                    push!(topopt_trace.rem_hist, 0)
                else
                    push!(topopt_trace.add_hist, sum(topopt_trace.x_hist[end] .> topopt_trace.x_hist[end-1]))
                    push!(topopt_trace.rem_hist, sum(topopt_trace.x_hist[end] .< topopt_trace.x_hist[end-1]))
                end
            end
        end
    end
    return o.comp::T
end

function compute_compliance(cell_comp::Vector{T}, grad, cell_dofs, Kes, u, 
                            black, white, varind, x, penalty, xmin) where {T}
    obj = zero(T)
    @inbounds for i in 1:size(cell_dofs, 2)
        cell_comp[i] = zero(T)
        Ke = rawmatrix(Kes[i])
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

function compute_compliance(cell_comp::CuVector{T}, grad, cell_dofs, Kes, u, 
                            black, white, varind, x, penalty, xmin) where {T}

    args = (cell_comp, grad, cell_dofs, Kes, u, black, white, varind, x, penalty, xmin)
    callkernel(dev, comp_kernel1, args)
    CUDAdrv.synchronize(ctx)
    obj = compute_obj(cell_comp, x, varind, black, white, penalty, xmin)

    return obj
end

# CUDA kernels
function comp_kernel1(cell_comp::AbstractVector{T}, grad, cell_dofs, Kes, u, 
                        black, white, varind, x, penalty, xmin) where {T}

    i = @thread_global_index()
    offset = @total_threads()
    @inbounds while i <= length(cell_comp)
        cell_comp[i] = zero(T)
        Ke = rawmatrix(Kes[i])
        for w in 1:size(Ke, 2)
            for v in 1:size(Ke, 1)
                if Ke isa Symmetric
                    cell_comp[i] += u[cell_dofs[v,i]]*Ke.data[v,w]*u[cell_dofs[w,i]]
                else
                    cell_comp[i] += u[cell_dofs[v,i]]*Ke[v,w]*u[cell_dofs[w,i]]
                end
            end
        end
        if !(black[i] || white[i])
            d = ForwardDiff.Dual{T}(x[varind[i]], one(T))
            p = density(penalty(d), xmin)
            grad[varind[i]] = -p.partials[1] * cell_comp[i]
        end

        i += offset
	end
    return
end

function compute_obj(cell_comp::AbstractVector{T}, x, varind, black, white, penalty, xmin, ::Val{blocksize} = Val(80), ::Val{threads} = Val(256)) where {T, blocksize, threads}
    result = similar(cell_comp, T, (blocksize,))
    args = (result, cell_comp, x, varind, black, white, penalty, xmin, Val(threads))
    @cuda blocks = blocksize threads = threads comp_kernel2(args...)
    CUDAnative.synchronize()
    obj = reduce(+, Array(result))
    return obj
end

function comp_kernel2(result, cell_comp::AbstractVector{T}, x, varind, black, white, penalty, xmin, ::Val{LMEM}) where {T, LMEM}
    i = @thread_global_index()
    obj = zero(T)
    # # Loop sequentially over chunks of input vector
	offset = @total_threads()
    @inbounds while i <= length(cell_comp)
        obj += w_comp(cell_comp[i], x[varind[i]], black[i], white[i], penalty, xmin)
        i += offset
    end

    # Perform parallel reduction
	tmp_local = @cuStaticSharedMem(T, LMEM)
    local_index = @thread_local_index()
    @inbounds tmp_local[local_index] = obj
    sync_threads()

    offset = @total_threads_per_block() ÷ 2
    @inbounds while offset > 0
        if (local_index <= offset)
            tmp_local[local_index] += tmp_local[local_index + offset]
        end
		sync_threads()
        offset = offset ÷ 2
    end
    if local_index == 1
        @inbounds result[@block_index()] = tmp_local[1]
    end

    return
end

@inline function w_comp(comp::T, x, black, white, penalty, xmin) where {T}
    return ifelse(black, comp,
		   ifelse(white, xmin * comp, 
			       	     (d = ForwardDiff.Dual{T}(x, one(T));
            	            p = density(penalty(d), xmin); p.value * comp)))
end
