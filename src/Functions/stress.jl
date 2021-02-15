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

get_sigma_vm(ρ_e, utMu_e) = ρ_e * sqrt(utMu_e)

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
@inline function apply_T!(Tu, u, dh, cellidx, global_dofs, cellvalues, ν, ::Val{2})
    # assumes cellvalues is initialized before passing
    Tu[:] .= 0
    # Generalize to higher order field basis functions
    q_point = 1
    n_basefuncs = getnbasefunctions(cellvalues)
    dim = 2
    for a in 1:n_basefuncs
        ∇ϕ = shape_gradient(cellvalues, q_point, a)
        _u = @view u[(@view global_dofs[dim*(a-1) + 1 : a*dim])]
        ϵ_11 = get_ϵ(_u, ∇ϕ, 1, 1)
        ϵ_22 = get_ϵ(_u, ∇ϕ, 2, 2)
        ϵ_12 = get_ϵ(_u, ∇ϕ, 1, 2)

        ϵ_sum = ϵ_11 + ϵ_22

        temp1 = ν/(1-ν^2)
        temp2 = ν*(1+ν)

        Tu[1] += temp1*ϵ_sum + temp2*ϵ_11 # σ[1,1] / E
        Tu[2] += temp1*ϵ_sum + temp2*ϵ_22 # σ[2,2] / E
        Tu[3] += temp2*ϵ_12 # σ[1,2] / E
    end
    return Tu
end
@inline function apply_T!(Tu, u, dh, global_dofs, cellvalues, ν, ::Val{3})
    # assumes cellvalues is initialized before passing
    Tu[:] .= 0
    q_point = 1
    n_basefuncs = getnbasefunctions(cellvalues)
    dim = 3
    for a in 1:n_basefuncs
        ∇ϕ = shape_gradient(cellvalues, q_point, a)
        _u = @view u[(@view global_dofs[dim*(a-1) + 1 : a*dim])]
        ϵ_11 = get_ϵ(_u, ∇ϕ, 1, 1)
        ϵ_22 = get_ϵ(_u, ∇ϕ, 2, 2)
        ϵ_33 = get_ϵ(_u, ∇ϕ, 3, 3)
        ϵ_12 = get_ϵ(_u, ∇ϕ, 1, 2)
        ϵ_23 = get_ϵ(_u, ∇ϕ, 2, 3)
        ϵ_31 = get_ϵ(_u, ∇ϕ, 3, 1)

        ϵ_sum = ϵ_11 + ϵ_22 + ϵ_33

        temp1 = ν/(1-ν^2)
        temp2 = ν*(1+ν)

        Tu[1] += temp1*ϵ_sum + temp2*ϵ_11 # σ[1,1] / E
        Tu[2] += temp1*ϵ_sum + temp2*ϵ_22 # σ[2,2] / E
        Tu[3] += temp1*ϵ_sum + temp2*ϵ_33 # σ[3,3] / E
        Tu[4] += temp2*ϵ_12 # σ[1,2] / E
        Tu[5] += temp2*ϵ_23 # σ[2,3] / E
        Tu[6] += temp2*ϵ_31 # σ[3,1] / E
    end
    return Tu
end

@inline function fill_T!(T, ::Val{3}, cellvalues, E0, ν)
    # assumes cellvalues is initialized before passing
    dim = 3
    temp1 = E0*ν/(1-ν^2)
    temp2 = E0*ν*(1+ν)
    q_point = 1
    n_basefuncs = size(T, 2) ÷ dim
    @assert size(T, 1) == 6
    for a in 1:n_basefuncs
        ∇ϕ = shape_gradient(cellvalues, q_point, a)
        cols = dim * (a - 1) + 1 : dim * a
        T[1, cols[1]] = (temp1 + temp2) * ∇ϕ[1]
        T[2, cols[1]] = temp1 * ∇ϕ[1]
        T[3, cols[1]] = temp1 * ∇ϕ[1]
        T[4, cols[1]] = temp2 * ∇ϕ[2] / 2
        T[5, cols[1]] = 0
        T[6, cols[1]] = temp2 * ∇ϕ[3] / 2
    
        T[1, cols[2]] = temp1 * ∇ϕ[2]
        T[2, cols[2]] = (temp1 + temp2) * ∇ϕ[2]
        T[3, cols[2]] = temp1 * ∇ϕ[2]
        T[4, cols[2]] = temp2 * ∇ϕ[1] / 2
        T[5, cols[2]] = temp2 * ∇ϕ[3] / 2
        T[6, cols[2]] = 0
    
        T[1, cols[3]] = temp1 * ∇ϕ[3]
        T[2, cols[3]] = temp1 * ∇ϕ[3]
        T[3, cols[3]] = (temp1 + temp2) * ∇ϕ[3]
        T[4, cols[3]] = 0
        T[5, cols[3]] = temp2 * ∇ϕ[2] / 2
        T[6, cols[3]] = temp2 * ∇ϕ[1] / 2
    end
    return T
end

@inline function fill_T!(T, ::Val{2}, cellvalues, E0, ν)
    # assumes cellvalues is initialized before passing
    dim = 2
    temp1 = E0*ν/(1-ν^2)
    temp2 = E0*ν*(1+ν)
    q_point = 1
    n_basefuncs = size(T, 2) ÷ dim
    @assert size(T, 1) == 3
    for a in 1:n_basefuncs
        ∇ϕ = shape_gradient(cellvalues, q_point, a)
        cols = dim * (a - 1) + 1 : dim * a
        T[1, cols[1]] = (temp1 + temp2) * ∇ϕ[1]
        T[2, cols[1]] = temp1 * ∇ϕ[1]
        T[3, cols[1]] = temp2 * ∇ϕ[2] / 2
    
        T[1, cols[2]] = temp1 * ∇ϕ[2]
        T[2, cols[2]] = (temp1 + temp2) * ∇ϕ[2]
        T[3, cols[2]] = temp2 * ∇ϕ[1] / 2    
    end
    return T
end

@params struct StressTemp{T}
    VTu::AbstractVector{T}
    Tu::AbstractVector{T}
    Te::AbstractMatrix{T}
    global_dofs::AbstractVector{Int}
end
function StressTemp(solver)
    @unpack u, problem = solver
    @unpack dh = problem.ch
    T = eltype(u)
    dim = TopOptProblems.getdim(problem)
    k = dim == 2 ? 3 : 6
    VTu = zero(MVector{k, T})
    Tu = similar(VTu)
    n_basefuncs = getnbasefunctions(solver.elementinfo.cellvalues)
    Te = zero(MMatrix{k, dim*n_basefuncs, T})
    global_dofs = zeros(Int, ndofs_per_cell(dh))

    return StressTemp(VTu, Tu, Te, global_dofs)
end
Zygote.@nograd StressTemp

function fill_Mu_utMu!(Mu, utMu, solver, stress_temp::StressTemp)
    @unpack problem, elementinfo, u = solver
    @unpack ch, ν, E = problem
    @unpack dh = ch
    @unpack VTu, Tu, Te, global_dofs = stress_temp

    _fill_Mu_utMu!(Mu, utMu, dh, elementinfo, u, E, ν, global_dofs, Tu, VTu, Te)
    return Mu, utMu
end

@inline function _fill_Mu_utMu!(
    Mu::AbstractVector, 
    utMu::AbstractVector{T}, 
    dh::DofHandler{3}, 
    elementinfo, 
    u,
    E0,
    ν, 
    global_dofs = zeros(Int, ndofs_per_cell(dh)), 
    Tu = zeros(T, 6),
    VTu = zeros(T, 6), 
    Te = zeros(T, 6, ndofs_per_cell(dh))
) where {T}
    dim = 3
    @unpack cellvalues = elementinfo
    for (cellidx, cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues, cell)
        # Same for all elements
        fill_T!(Te, Val(3), cellvalues, E0, ν)
        celldofs!(global_dofs, dh, cellidx)
        @views mul!(Tu, Te, u[global_dofs])

		VTu[1] = Tu[1] - Tu[2]/2 - Tu[3]/2
		VTu[2] = -Tu[1]/2 + Tu[2] - Tu[3]/2
		VTu[3] = -Tu[1]/2 - Tu[2]/2 + Tu[3]
		VTu[4] = 3*Tu[4]
		VTu[5] = 3*Tu[5]
		VTu[6] = 3*Tu[6]

        utMu_e = dot(Tu, VTu)
        @assert utMu_e >= 0
        utMu[cellidx] = utMu_e
        Mu[cellidx] = Te' * VTu
	end
	return Mu, utMu
end

@inline function _fill_Mu_utMu!(
    Mu::AbstractVector, 
    utMu::AbstractVector{T}, 
    dh::DofHandler{2}, 
    elementinfo, 
    u, 
    E0,
    ν, 
    global_dofs = zeros(Int, ndofs_per_cell(dh)), 
    Tu = zeros(T, 3),
    VTu = zeros(T, 3), 
    Te = zeros(T, 3, ndofs_per_cell(dh))
) where {T}
    dim = 2
    @unpack cellvalues = elementinfo
    for (cellidx, cell) in enumerate(CellIterator(dh))
        reinit!(cellvalues, cell)
        fill_T!(Te, Val(2), cellvalues, E0, ν)
        celldofs!(global_dofs, dh, cellidx)
        @views mul!(Tu, Te, u[global_dofs])

        VTu[1] = Tu[1] - Tu[2]/2
		VTu[2] = -Tu[1]/2 + Tu[2]
		VTu[3] = 3*Tu[3]

        utMu_e = dot(Tu, VTu)
        @assert utMu_e >= 0
        utMu[cellidx] = utMu_e
        Mu[cellidx] = Te' * VTu
	end
	return Mu, utMu
end

@params mutable struct MacroVonMisesStress{T} <: AbstractFunction{T}
    utMu::AbstractVector{T}
    Mu::AbstractArray
    sigma_vm::AbstractVector{T}
    solver
    global_dofs::AbstractVector{<:Integer}
    stress_temp::StressTemp
    fevals::Int
    reuse::Bool
    maxfevals::Int
end
function MacroVonMisesStress(solver; reuse = false, maxfevals = 10^8)
    T = eltype(solver.u)
    dh = solver.problem.ch.dh
    k = ndofs_per_cell(dh)
    N = getncells(dh.grid)
    global_dofs = zeros(Int, k)
    Mu = zeros(SVector{k, T}, N)
    utMu = zeros(T, N)
    stress_temp = StressTemp(solver)
    sigma_vm = similar(utMu)

    return MacroVonMisesStress(utMu, Mu, sigma_vm, solver, global_dofs, stress_temp, 0, reuse, maxfevals)
end
function (ls::MacroVonMisesStress{T})(x) where {T}
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
function ChainRulesCore.rrule(vonmises::MacroVonMisesStress, x)
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
#getdim(f::MacroVonMisesStress) = length(f.sigma_vm)

@params struct MicroVonMisesStress{T} <: AbstractFunction{T}
    vonmises::MacroVonMisesStress{T}
end
MicroVonMisesStress(args...; kwargs...) = MicroVonMisesStress(MacroVonMisesStress(args...; kwargs...))
function (f::MicroVonMisesStress)(x)
    @unpack vonmises = f
    @unpack sigma_vm, Mu, utMu, stress_temp = vonmises
    @unpack reuse, solver, global_dofs = vonmises
    @unpack penalty, problem, xmin = solver
    vonmises.fevals += 1
    @assert length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh)
    solver.vars .= x
    if !reuse
        solver()
        fill_Mu_utMu!(Mu, utMu, solver, stress_temp)
    end
    out = sqrt.(utMu)
    sigma_vm .= get_ρ.(x, Ref(penalty), xmin) .* out
    return out
end
function ChainRulesCore.rrule(f::MicroVonMisesStress, x)
    @unpack vonmises = f
    @unpack Mu, utMu, stress_temp, sigma_vm = vonmises
    @unpack reuse, solver, global_dofs = vonmises
    @unpack penalty, problem, u, xmin = solver
    dh = getdh(problem)
    @unpack Kes = solver.elementinfo
    E0 = problem.E
    vonmises.fevals += 1

    out = f(x)
    return out, Δ -> (nothing, begin
        for e in 1:length(Mu)
            ρe = get_ρ(x[e], penalty, xmin)
            Mu[e] *= Δ[e] * ρe / sigma_vm[e]
        end
        lhs = backsolve!(solver, Mu, global_dofs)
        map(1:length(Δ)) do e
            ρe, dρe = get_ρ_dρ(x[e], penalty, xmin)
            celldofs!(global_dofs, dh, e)
            return -dot(lhs[global_dofs], bcmatrix(Kes[e]) * u[global_dofs]) * dρe
        end
    end)
end
