function get_dudxe!(solver, u, Ke, xe, penalty, E0, xmin, global_dofs)
    E, dE = get_E_dE(xe, penalty, E0, xmin)
    solver.rhs .= 0
    for i in 1:size(Ke, 2)
        solver.rhs[global_dofs] .-= dE .* Ke[:, i] .* u[global_dofs[i]]
    end
    solver(assemble_f = false)
    return solver.lhs
end

get_sigma_vm(E_e, utMu_e) = E_e * sqrt(utMu_e)

function get_E(x_e::T, penalty, E0, xmin) where T
    return E0 * density(penalty(x_e), xmin)
end

function get_E_dE(x_e::T, penalty, E0, xmin) where T
    d = ForwardDiff.Dual{T}(x_e, one(T))
    p = density(penalty(d), xmin)
    g = p.partials[1] * E0
    return p.value * E0, g
end

struct CorrectedStress <: Function end
function (s::CorrectedStress)(x_e, sigma_vm_e, sigma_bar)
    return x_e * (sigma_vm_e - sigma_bar)
end

struct OffsetCorrectedStress <: Function end
function (s::OffsetCorrectedStress)(x_e, sigma_vm_e, sigma_bar)
    return x_e * sigma_vm_e + (1 - x_e) * sigma_bar
end

function get_s(x_e, sigma_vm_e, sigma_bar)
    return x_e * sigma_vm_e + (1 - x_e) * sigma_bar
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

function fill_T!(T, ::Val{3}, cellvalues, ν)
    # assumes cellvalues is initialized before passing
    dim = 3
    temp1 = ν/(1-ν^2)
    temp2 = ν*(1+ν)
    q_point = 1
    n_basefuncs = getnbasefunctions(cellvalues)
    @assert size(T) == (6, dim*n_basefuncs)
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

function fill_T!(T, ::Val{2}, cellvalues, ν)
    # assumes cellvalues is initialized before passing
    dim = 2
    temp1 = ν/(1-ν^2)
    temp2 = ν*(1+ν)
    q_point = 1
    n_basefuncs = getnbasefunctions(cellvalues)
    @assert size(T) == (3, dim*n_basefuncs)
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
    VTu = zeros(T, k)
    Tu = similar(VTu)
    Te = zeros(T, k, dim * getnbasefunctions(solver.elementinfo.cellvalues))
    global_dofs = zeros(Int, ndofs_per_cell(dh))

    return StressTemp(VTu, Tu, Te, global_dofs)
end

function fill_Mu_utMu!(Mu, utMu, solver, stress_temp)
    @unpack problem, elementinfo, u = solver
    @unpack ch, ν = problem
    @unpack dh = ch
    @unpack VTu, Tu, Te, global_dofs = stress_temp

    _fill_Mu_utMu!(Mu, utMu, dh, elementinfo, u, ν, global_dofs, Tu, VTu, Te)
    return Mu, utMu
end

function _fill_Mu_utMu!( Mu::AbstractMatrix{T}, 
                        utMu::AbstractVector{T}, 
                        dh::DofHandler{3}, 
                        elementinfo, 
                        u, 
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
        Mu_e = @view Mu[:, cellidx]
        fill_T!(Te, Val(3), cellvalues, ν)
        celldofs!(global_dofs, dh, cellidx)
        @views mul!(Tu, Te, u[global_dofs])

		VTu[1] = Tu[1] - Tu[2]/2 - Tu[3]/2
		VTu[2] = -Tu[1]/2 + Tu[2] - Tu[3]/2
		VTu[3] = -Tu[1]/2 - Tu[2]/2 + Tu[3]
		VTu[4] = 3*Tu[4]
		VTu[5] = 3*Tu[5]
		VTu[6] = 3*Tu[6]

        utMu_e = dot(Tu, VTu)
        @assert utMu_e > 0
        utMu[cellidx] = utMu_e
        mul!(Mu_e, Te', VTu)
	end
	return Mu, utMu
end

function _fill_Mu_utMu!( Mu::AbstractMatrix{T}, 
                        utMu::AbstractVector{T}, 
                        dh::DofHandler{2}, 
                        elementinfo, 
                        u, 
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
        Mu_e = @view Mu[:, cellidx]
        fill_T!(Te, Val(2), cellvalues, ν)
        celldofs!(global_dofs, dh, cellidx)
        @views mul!(Tu, Te, u[global_dofs])

        VTu[1] = Tu[1] - Tu[2]/2
		VTu[2] = -Tu[1]/2 + Tu[2]
		VTu[3] = 3*Tu[3]

        utMu_e = dot(Tu, VTu)
        @assert utMu_e > 0
        utMu[cellidx] = utMu_e
        mul!(Mu_e, Te', VTu)
	end
	return Mu, utMu
end

@params mutable struct GlobalStress{T} <: AbstractFunction{T}
    reducer
    reducer_g::AbstractVector{T}
    corrected_stress
    utMu::AbstractVector{T}
    Mu::AbstractArray{T}
    s::AbstractVector{T}
    sigma_bar::T
    solver
    global_dofs::AbstractVector{<:Integer}
    buffer::AbstractVector{T}
    stress_temp::StressTemp
    fevals::Int
    reuse::Bool
    maxfevals::Int
end
function GlobalStress(solver, sigma_bar, reducer = WeightedKNorm(4, 1/length(solver.vars)), stress = OffsetCorrectedStress(); reuse = false, maxfevals = 10^8)
    T = eltype(solver.u)
    dh = solver.problem.ch.dh
    k = ndofs_per_cell(dh)
    N = getncells(dh.grid)
    global_dofs = zeros(Int, k)
    Mu = zeros(T, k, N)
    utMu = zeros(T, N)
    stress_temp = StressTemp(solver)
    s = similar(utMu)
    buffer = zeros(T, ndofs_per_cell(dh))
    reducer_g = similar(utMu)
    
    return GlobalStress(reducer, reducer_g, stress, utMu, Mu, s, sigma_bar, solver, global_dofs, buffer, stress_temp, 0, reuse, maxfevals)
end

function (gs::GlobalStress)(x, g)
    @unpack s, sigma_bar, Mu, utMu, buffer, stress_temp, corrected_stress, reuse = gs
    @unpack solver, global_dofs, buffer, reducer, reducer_g = gs
    @unpack elementinfo, u, penalty, problem, xmin = solver
    @unpack Kes = elementinfo
    @unpack dh = problem.ch
    E0 = problem.E
    gs.fevals += 1

    @assert length(global_dofs) == ndofs_per_cell(solver.problem.ch.dh)
    if !reuse
        solver()
        fill_Mu_utMu!(Mu, utMu, solver, stress_temp)
    end
    s .= corrected_stress.(x, get_sigma_vm.(get_E.(x, Ref(penalty), E0, xmin), utMu), sigma_bar)
    reduced = reducer(s, reducer_g)
    g .= 0
    for e1 in 1:length(g)
        if !reuse
            celldofs!(global_dofs, dh, e1)
            dudxe = get_dudxe!(solver, u, Kes[e1], x[e1], penalty, E0, xmin, global_dofs)
            @views buffer .= dudxe[global_dofs]
        end
        for e2 in 1:length(g)
            utMu_e2 = utMu[e2]
            Ee2, dEe2 = get_E_dE(x[e2], penalty, E0, xmin)
            t1 = (e1 == e2) * (dEe2 * sqrt(utMu_e2))
            if reuse
                t2 = zero(T)
            else
                @views t2 = (Ee2 / sqrt(utMu_e2)) * dot(buffer, Mu[:, e2])
            end
            dsigmae2_dxe1 = t1 + t2
            dse2_dxe1 = (e1 == e2) * (Ee2 * sqrt(utMu_e2) - sigma_bar) + x[e2] * dsigmae2_dxe1
            g[e1] += dse2_dxe1 * reducer_g[e2]
        end
    end
    return reduced
end

struct LogSumExp <: Function end
function (lse::LogSumExp)(s, g_s)
    out = logsumexp(s)
    g_s .= exp.(s .- out)
    return out
end

struct OffsetLogSumExp{T} <: Function
    k::T
end
function (olse::OffsetLogSumExp)(s, g_s)
    lse = logsumexp(s)
    g_s .= exp.(s .- lse)
    return out - olse.k * log(length(s))
end

struct KNorm <: Function
    k::Int
end
function (knorm::KNorm)(s, g_s)
    out = norm(s, knorm.k)
    g_s .= (s ./ out).^(k-1)
    return out
end

struct WeightedKNorm{T} <: Function
    k::Int
    w::T
end
function (wknorm::WeightedKNorm{T})(s, g_s) where {T}
    @unpack k, w = wknorm
    if T <: AbstractVector
        mw = MappedArray(w -> w^(1/k), w)
        out = norm(MappedArray(*, s, mw), k)
    else
        mw = w^(1/k)
        out = norm(BroadcastArray(*, s, mw), k)
    end
    g_s .= (s ./ out).^(k-1) .* mw
    return out
end
