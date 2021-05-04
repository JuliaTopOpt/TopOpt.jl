### Experimental ###

function backsolve!(solver, Mu, global_dofs)
    dh = getdh(solver.problem)
    solver.rhs .= 0
    for i in 1:length(Mu)
        celldofs!(global_dofs, dh, i)
        solver.rhs[global_dofs] .+= Mu[i]
    end
    solver(assemble_f = false)
    return solver.lhs
end

function get_ρ(x_e::T, penalty, xmin) where T
    if PENALTY_BEFORE_INTERPOLATION
        return density(penalty(x_e), xmin)
    else
        return penalty(density(x_e, xmin))
    end
end

function get_ρ_dρ(x_e::T, penalty, xmin) where T
    d = ForwardDiff.Dual{T}(x_e, one(T))
    if PENALTY_BEFORE_INTERPOLATION
        p = density(penalty(d), xmin)
    else
        p = penalty(density(d, xmin))
    end
    g = p.partials[1]
    return p.value, g
end

@inline function get_ϵ(u, ∇ϕ, i, j)
	return 1/2*(u[i]*∇ϕ[j] + u[j]*∇ϕ[i])
end

@params mutable struct TrussBucklingLogDet{T} <: AbstractFunction{T}
    utMu::AbstractVector{T}
    Mu::AbstractArray
    sigma_vm::AbstractVector{T}
    solver
    global_dofs::AbstractVector{<:Integer}
    stress_temp::StressTemp
    fevals::Int
    reuse::Bool
end

function TrussBucklingLogDet(solver; reuse = false)
    T = eltype(solver.u)
    dh = solver.problem.ch.dh
    k = ndofs_per_cell(dh)
    N = getncells(dh.grid)
    global_dofs = zeros(Int, k)
    Mu = zeros(SVector{k, T}, N)
    utMu = zeros(T, N)
    stress_temp = StressTemp(solver)
    sigma_vm = similar(utMu)

    return TrussBucklingLogDet(utMu, Mu, sigma_vm, solver, global_dofs, stress_temp, 0, reuse)
end

```
Compute `logdet(K(x) + Kσ(x))`
```
function (ls::TrussBucklingLogDet{T})(x) where {T}
    @unpack sigma_vm, Mu, utMu, stress_temp = ls
    @unpack reuse, solver, global_dofs = ls
    @unpack penalty, problem, xmin = solver
    E0 = problem.E
    ls.fevals += 1

    @assert length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh)
    solver.vars .= x
    if !reuse
        solver()
        fill_Mu_utMu!(Mu, utMu, solver, stress_temp)
    end
    sigma_vm .= get_sigma_vm.(get_ρ.(x, Ref(penalty), xmin), utMu)
    return copy(sigma_vm)
end

function ChainRulesCore.rrule(vonmises::TrussBucklingLogDet, x)
    @unpack Mu, utMu, stress_temp = vonmises
    @unpack reuse, solver, global_dofs = vonmises
    @unpack penalty, problem, u, xmin = solver
    dh = getdh(problem)
    @unpack Kes = solver.elementinfo
    E0 = problem.E
    vonmises.fevals += 1

    sigma_vm = vonmises(x)
    return sigma_vm, Δ -> (nothing, begin
        for e in 1:length(Mu)
            ρe = get_ρ(x[e], penalty, xmin)
            Mu[e] *= Δ[e] * ρe^2 / sigma_vm[e]
        end
        lhs = backsolve!(solver, Mu, global_dofs)
        map(1:length(Δ)) do e
            ρe, dρe = get_ρ_dρ(x[e], penalty, xmin)
            celldofs!(global_dofs, dh, e)
            t1 = Δ[e] * sigma_vm[e] / ρe * dρe
            @views t2 = -dot(lhs[global_dofs], bcmatrix(Kes[e]) * u[global_dofs]) * dρe
            return t1  + t2
        end
    end)
end