@params mutable struct TrussElementKσ{T} <: AbstractFunction{T}
    problem::TrussProblem
    Kσ_e::AbstractMatrix{T}
    cellidx
    cellvalues::CellValues
    EALγ::AbstractVector
    δmat::AbstractMatrix
end

Base.show(::IO, ::MIME{Symbol("text/plain")}, ::TrussElementKσ) = println("TopOpt element stress stiffness matrix (Kσ_e) construction function")

function TrussElementKσ(problem::TrussProblem{xdim, T}, cellidx, cellvalues) where {xdim, T}
    Es = getE(problem)
    As = getA(problem)
    dh = problem.ch.dh

    # usually ndof_pc = xdim * n_basefuncs
    ndof_pc = ndofs_per_cell(dh)
    n_basefuncs = getnbasefunctions(cellvalues)
    @assert ndof_pc == xdim*n_basefuncs "$ndof_pc, $n_basefuncs"
    @assert n_basefuncs == 2

    global_dofs = zeros(Int, ndof_pc)
    Kσ_e = zeros(T, ndof_pc, ndof_pc)
    EALγ = zeros(T, ndof_pc)
    δmat = zeros(T, ndof_pc, ndof_pc)

    for (ci, cell) in enumerate(CellIterator(dh))
        # TODO directly fetch cell
        if cellidx != ci
            continue
        end
        truss_reinit!(cellvalues, cell, As[cellidx])
        celldofs!(global_dofs, dh, cellidx)
        E = Es[cellidx]
        A = As[cellidx]
        L = norm(cell.coords[1] - cell.coords[2])
        R = compute_local_axes(cell.coords[1], cell.coords[2])
        # local axial projection operator (local axis transformation)
        EALγ = E*A/L*γ
        γ = vcat(-R[:,1], R[:,1])
        for i=2:size(R,2)
            δ = vcat(-R[:,i], R[:,i])
            # @assert δ' * γ ≈ 0
            δmat .+= δ * δ'
        end
        break
    end
    return TrussElementKσ(problem, Kσ_e, cellidx, cellvalues, EALγ, δmat)
end

function (eksig::TrussElementKσ{T})(u_e, x_e) where {T}
    # compute axial force: first-order approx of bar force
    # better approx would be: EA/L * (u3-u1 + 1/(2*L0)*(u4-u2)^2) = EA/L * (γ'*u + 1/2*(δ'*u)^2)
    # see: https://people.duke.edu/~hpgavin/cee421/truss-finite-def.pdf
    @unpack Kσ_e, EALγ, δmat = eksig
    Kσ_e .= 0.0
    Kσ_e .+= δmat
    # ? do we need to do black, white here?
    # x_e scales the cross section
    q_cell = x_e*EALγ'*u_e
    Kσ_e .*= q_cell / L
    return Kσ_e
end

function ChainRulesCore.rrule(eksig::TrussElementKσ, u_e, x_e)
    val = eksig([u_e; x_e])
    function vec_eksig(ux_vec)
        u_e = ux_vec[1:end-1]
        x_e = ux_vec[end]
        return vec(eksig([u_e; x_e]))
    end
    jac = ForwardDiff.jacobian(vec_eksig, [u_e; x_e])
    val, Δ -> begin 
        jtv = jac' * vec(Δ)
        return (NoTangent(), jtv[1:end-1], jtv[end])
    end
end

#####################################

# TODO ElementKσ for volumetic cells