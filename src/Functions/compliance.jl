@params mutable struct ComplianceFunction{T} <: AbstractFunction{T}
	problem::StiffnessTopOptProblem
    solver::AbstractDisplacementSolver
    cheqfilter::AbstractCheqFilter
    comp::T
    cell_comp::AbstractVector
    grad::AbstractVector
    tracing::Bool
    topopt_trace::TopOptTrace{T}
    reuse::Bool
    fevals::Integer
    logarithm::Bool
    maxfevals::Int
end
Utilities.getpenalty(c::ComplianceFunction) = c |> getsolver |> getpenalty
Utilities.setpenalty!(c::ComplianceFunction, p) = setpenalty!(getsolver(c), p)

function ComplianceFunction(problem, solver::AbstractDisplacementSolver, args...; kwargs...)
    ComplianceFunction(whichdevice(solver), problem, solver, args...; kwargs...)
end
function ComplianceFunction(::CPU, problem::StiffnessTopOptProblem{dim, T}, solver::AbstractDisplacementSolver, ::Type{TI}=Int; rmin = T(0), filtering = true, tracing = false, logarithm = false, maxfevals = 10^8) where {dim, T, TI}
    cheqfilter = CheqFilter(Val(filtering), solver, rmin)
    comp = T(0)
    cell_comp = zeros(T, getncells(problem.ch.dh.grid))
    grad = fill(T(NaN), length(cell_comp) - sum(problem.black) - sum(problem.white))
    topopt_trace = TopOptTrace{T,TI}()
    reuse = false
    fevals = TI(0)
    return ComplianceFunction(problem, solver, cheqfilter, comp, cell_comp, grad, tracing, topopt_trace, reuse, fevals, logarithm, maxfevals)
end

function (o::ComplianceFunction{T})(x, grad = o.grad) where {T}
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
            setpenalty!(solver, penalty.p)
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
        if o.grad !== grad
            copyto!(o.grad, grad)
        end
        
        if o.tracing
            if o.reuse
                o.reuse = false
            else
                push!(topopt_trace.c_hist, obj)
                push!(topopt_trace.x_hist, copy(x))
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
            p = penalty(density(d, xmin))
            grad[varind[i]] = -p.partials[1] * cell_comp[i]
            obj += p.value * cell_comp[i]
        end
    end

    return obj    
end
