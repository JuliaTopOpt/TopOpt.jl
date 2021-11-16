@params mutable struct Compliance{T} <: AbstractFunction{T}
	problem::StiffnessTopOptProblem
    solver::AbstractDisplacementSolver
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
Utilities.getpenalty(c::Compliance) = c |> getsolver |> getpenalty
Utilities.setpenalty!(c::Compliance, p) = setpenalty!(getsolver(c), p)
Nonconvex.NonconvexCore.getdim(::Compliance) = 1

function Compliance(problem, solver::AbstractDisplacementSolver, args...; kwargs...)
    Compliance(whichdevice(solver), problem, solver, args...; kwargs...)
end
function Compliance(
    ::CPU,
    problem::StiffnessTopOptProblem{dim, T},
    solver::AbstractDisplacementSolver,
    ::Type{TI} = Int;
    tracing = false,
    logarithm = false,
    maxfevals = 10^8,
) where {dim, T, TI}
    comp = T(0)
    cell_comp = zeros(T, getncells(problem.ch.dh.grid))
    grad = fill(T(NaN), length(cell_comp) - sum(problem.black) - sum(problem.white))
    topopt_trace = TopOptTrace{T,TI}()
    reuse = false
    fevals = TI(0)
    return Compliance(problem, solver, comp, cell_comp, grad, tracing, topopt_trace, reuse, fevals, logarithm, maxfevals)
end

function (o::Compliance{T})(x, grad = o.grad) where {T}
    @unpack cell_comp, solver, tracing, topopt_trace = o
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes, black, white, varind = elementinfo
    @unpack cell_dofs = metadata

    @timeit to "Eval obj and grad" begin
        #if o.solver.vars ≈ x && getpenalty(o).p ≈ getprevpenalty(o).p
        #    grad .= o.grad
        #    return o.comp
        #end
        penalty = getpenalty(o)
        if o.reuse
            if !tracing
                o.reuse = false
            end
        else
            o.fevals += 1
            setpenalty!(solver, penalty.p)
            solver.vars .= x
            solver()
        end
        obj = compute_compliance(cell_comp, grad, cell_dofs, Kes, u, 
                                    black, white, varind, solver.vars, penalty, xmin)

        if o.logarithm
            o.comp = log(obj)
            grad ./= obj
        else
            o.comp = obj
            #scale!(grad, 1/length(cell_comp))
            #o.comp = obj
        end
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

function ChainRulesCore.rrule(comp::Compliance, x)
    out = comp(x, comp.grad)
    out_grad = copy(comp.grad)
    return out, Δ -> (nothing, out_grad * Δ)
end

"""
cell_compliance = f_e^T * u_e = u_e^T * (ρ_e * Ke) * u_e.
d(cell compliance)/d(x_e) = f_e^T * d(u_e)/d(x_e) = f_e^T * (- K_e^-1 * d(K_e)/d(x_e) * u_e)
                          = - (K_e^(-1) * f_e)^T * d(K_e)/d(x_e) * u_e
                          = - u_e^T * d(ρ_e)/d(x_e) * K_e * u_e
                          = - d(ρ_e)/d(x_e) * cell_compliance
"""
function compute_compliance(cell_comp::Vector{T}, grad, cell_dofs, Kes, u, 
                            black, white, varind, x, penalty, xmin) where {T}
    obj = zero(T)
    grad .= 0
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
            if PENALTY_BEFORE_INTERPOLATION
                obj += xmin * cell_comp[i] 
            else
                p = penalty(xmin) * cell_comp[i]
            end
        else
            ρe, dρe = get_ρ_dρ(x[varind[i]], penalty, xmin)
            grad[varind[i]] = - dρe * cell_comp[i]
            obj += ρe * cell_comp[i]
        end
    end

    return obj    
end

function compute_inner(inner, u1, u2, solver)
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes, black, white, varind = elementinfo
    @unpack cell_dofs = metadata
    penalty = getpenalty(solver)
    return compute_inner(inner, u1, u2, cell_dofs, Kes,
        black, white, varind, solver.vars, penalty, xmin)
end
function compute_inner(inner::AbstractVector{T}, u1, u2, cell_dofs, Kes,
                            black, white, varind, x, penalty, xmin) where {T}
    obj = zero(T)
    @inbounds for i in 1:size(cell_dofs, 2)
        inner[i] = zero(T)
        cell_comp = zero(T)
        Ke = rawmatrix(Kes[i])
        if !black[i] && !white[i]
            for w in 1:size(Ke,2)
                for v in 1:size(Ke, 1)
                    cell_comp += u1[cell_dofs[v,i]]*Ke[v,w]*u2[cell_dofs[w,i]]
                end
            end
            ρe, dρe = get_ρ_dρ(x[varind[i]], penalty, xmin)
            inner[varind[i]] = - dρe * cell_comp
        end
    end

    return inner
end
