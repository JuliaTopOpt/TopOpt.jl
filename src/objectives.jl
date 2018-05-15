abstract type AbstractObjective <: Function end

mutable struct ComplianceObj{T, TI<:Integer, TSP<:StiffnessTopOptProblem, FS<:AbstractDisplacementSolver, CF<:AbstractCheqFilter} <: AbstractObjective
	problem::TSP
    solver::FS
    cheqfilter::CF
    comp::T
    cell_comp::Vector{T}
    grad::Vector{T}
    tracing::Bool
    topopt_trace::TopOptTrace{T,TI}
    reuse::Bool
	nf::TI
end
function ComplianceObj(problem::StiffnessTopOptProblem{dim, T}, solver::AbstractDisplacementSolver, ::Type{TI}=Int; rmin = T(0), filtering = true, tracing = false) where {dim, T, TI}
    cheqfilter = CheqFilter{filtering}(solver, rmin)
    comp = T(0)
    cell_comp = zeros(T, getncells(problem.ch.dh.grid))
    grad = fill(T(NaN), length(cell_comp) - sum(problem.black) - sum(problem.white))
    topopt_trace = TopOptTrace{T,TI}()
    reuse = false
    nf = TI(0)
    return ComplianceObj(problem, solver, cheqfilter, comp, cell_comp, grad, tracing, topopt_trace, reuse, nf)
end

function (o::ComplianceObj{T})(x, grad) where {T}
    if o.solver.vars ≈ x && o.solver.penalty.p ≈ o.solver.prev_penalty.p
        grad .= o.grad
        return o.comp
    end

    penalty = o.solver.penalty
    cell_dofs = o.problem.metadata.cell_dofs
    u = o.solver.u
    cell_comp = o.cell_comp
    Kes = o.solver.elementinfo.Kes
    black = o.problem.black
    white = o.problem.white
    xmin = o.solver.xmin
    varind = o.problem.varind

    o.solver.vars .= x
    if o.reuse
        if !o.tracing
            o.reuse = false
        end
    else
        o.nf += 1
        o.solver()
    end
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
            obj += penalty(xmin) * cell_comp[i] 
        else
            d = ForwardDiff.Dual{T}(x[varind[i]], one(T))
            p = penalty(density(d, xmin))
            grad[varind[i]] = -p.partials[1] * cell_comp[i]
            obj += p.value * cell_comp[i]
        end
    end

    o.comp = log(obj)
    grad ./= obj
    #o.comp = obj / length(cell_comp)
    #grad ./= length(cell_comp)
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
    return o.comp::T
end
function (o::ComplianceObj{T})(to, x, grad) where {T}
    if o.solver.vars ≈ x && o.solver.penalty.p ≈ o.solver.prev_penalty.p
        grad .= o.grad
        return o.comp
    end

    penalty = o.solver.penalty
    cell_dofs = o.problem.metadata.cell_dofs
    u = o.solver.u
    cell_comp = o.cell_comp
    Kes = o.solver.elementinfo.Kes
    black = o.problem.black
    white = o.problem.white
    xmin = o.solver.xmin
    varind = o.problem.varind

    o.solver.vars .= x
    if o.reuse
        if !o.tracing
            o.reuse = false
        end
    else
        o.nf += 1
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
            obj += penalty(xmin) * cell_comp[i] 
        else
            d = ForwardDiff.Dual{T}(x[varind[i]], one(T))
            p = penalty(density(d, xmin))
            grad[varind[i]] = -p.partials[1] * cell_comp[i]
            obj += p.value * cell_comp[i]
        end
    end
    o.comp = log(obj)
    grad ./= obj
    #o.comp = obj / length(cell_comp)
    #grad .= grad ./ length(cell_comp)
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
