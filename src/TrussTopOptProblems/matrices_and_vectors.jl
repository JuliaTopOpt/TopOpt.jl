using Base: @propagate_inbounds
using ..TopOpt.TopOptProblems: getdh, getE, getν, getdensity, gettypes
using LinearAlgebra: norm
using Ferrite: getdim, getrefshape, value, getlowerdim
using ChainRulesCore

"""
Generate element stiffness matrices
"""
function make_Kes_and_fes(problem::TrussProblem, quad_order=1)
    return make_Kes_and_fes(problem, quad_order, Val{:Static})
end

function make_Kes_and_fes(problem::TrussProblem, ::Type{Val{mat_type}}) where {mat_type}
    return make_Kes_and_fes(problem, 1, Val{mat_type})
end

function make_Kes_and_fes(
    problem::TrussProblem{xdim,T}, quad_order, ::Type{Val{mat_type}}
) where {xdim,T,mat_type}
    dh = getdh(problem)
    Es = getE(problem)
    # ν = getν(problem)
    # ρ = getdensity(problem)
    As = getA(problem)

    # * Shape functions and quadrature rule
    interpolation_space = Ferrite.default_interpolation(
        getcelltype(problem.truss_grid.grid)
    )
    # Lagrange{ξdim, refshape, geom_order}()
    ξdim = getdim(interpolation_space)
    refshape = getrefshape(dh.field_interpolations[1])
    quadrature_rule = QuadratureRule{ξdim,refshape}(quad_order)
    cellvalues = GenericCellScalarValues(T, quadrature_rule, interpolation_space; xdim=xdim)

    # * A Line element's faces are not meaningful in truss problems
    # placeholder to make type right
    facevalues = FaceScalarValues(
        QuadratureRule{ξdim - 1,refshape}(quad_order), interpolation_space
    )

    # * Calculate element stiffness matrices
    n_basefuncs = getnbasefunctions(cellvalues)
    Kesize = xdim * n_basefuncs
    MatrixType, VectorType = gettypes(T, Val{mat_type}, Val{Kesize})
    Kes, weights = _make_Kes_and_weights(
        dh,
        Tuple{MatrixType,VectorType},
        Val{n_basefuncs},
        Val{xdim * n_basefuncs},
        Es,
        As,
        quadrature_rule,
        cellvalues,
    )

    # ! distributed load, not used in a truss problem
    # dloads = _make_dloads(weights, problem, facevalues)

    return Kes, weights, cellvalues, facevalues #dloads, 
end

############################

"""
    GenericCellScalarValues{ξdim,xdim,T<:Real,refshape<:AbstractRefShape} <: CellScalarValues{xdim,T,refshape}

`GenericCellScalarValues` is a generalization of the `Ferrite.CellScalarValues` to separate the reference domain
dimension `ξdim` and the node coordinate dimension `xdim`. While in most of the solid mechanics cases `ξdim = xdim`,
in a truss element (line element with nodes in 2D or 3D), `xdim = 2` or `3` and `ξdim = 1`. 

**Arguments:**
* `ξdim` : reference domain dimension
* `xdim` : node coordinate dimension
"""
struct GenericCellScalarValues{ξdim,xdim,T,refshape} <: CellValues{xdim,T,refshape}
    N::Matrix{T}
    dNdx::Matrix{Vec{xdim,T}}
    dNdξ::Matrix{Vec{ξdim,T}}
    detJdV::Vector{T}
    M::Matrix{T}
    dMdξ::Matrix{Vec{ξdim,T}}
    qr_weights::Vector{T}
end

function GenericCellScalarValues(
    quad_rule::QuadratureRule,
    func_interpol::Interpolation,
    geom_interpol::Interpolation=func_interpol,
)
    return TrussCellScalarValues(Float64, quad_rule, func_interpol, geom_interpol)
end

function GenericCellScalarValues(
    ::Type{T},
    quad_rule::QuadratureRule{ξdim,shape},
    func_interpol::Interpolation,
    geom_interpol::Interpolation=func_interpol;
    xdim=ξdim,
) where {ξdim,T,shape<:Ferrite.AbstractRefShape}
    @assert getdim(func_interpol) == getdim(geom_interpol)
    @assert getrefshape(func_interpol) == getrefshape(geom_interpol) == shape
    n_qpoints = length(getweights(quad_rule))
    # * Function interpolation
    n_func_basefuncs = getnbasefunctions(func_interpol)
    N = fill(zero(T) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdx = fill(zero(Vec{xdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    dNdξ = fill(zero(Vec{ξdim,T}) * T(NaN), n_func_basefuncs, n_qpoints)
    # * Geometry interpolation
    n_geom_basefuncs = getnbasefunctions(geom_interpol)
    M = fill(zero(T) * T(NaN), n_geom_basefuncs, n_qpoints)
    dMdξ = fill(zero(Vec{ξdim,T}) * T(NaN), n_geom_basefuncs, n_qpoints)
    for (qp, ξ) in enumerate(quad_rule.points)
        for i in 1:n_func_basefuncs
            dNdξ[i, qp], N[i, qp] = Ferrite.gradient(
                ξ -> value(func_interpol, i, ξ), ξ, :all
            )
        end
        for i in 1:n_geom_basefuncs
            dMdξ[i, qp], M[i, qp] = Ferrite.gradient(
                ξ -> value(geom_interpol, i, ξ), ξ, :all
            )
        end
    end
    detJdV = fill(T(NaN), n_qpoints)
    return GenericCellScalarValues{ξdim,xdim,T,shape}(
        N, dNdx, dNdξ, detJdV, M, dMdξ, quad_rule.weights
    )
end

# common values
using Ferrite: getnbasefunctions, getngeobasefunctions, getnquadpoints
getn_scalarbasefunctions(cv::GenericCellScalarValues) = size(cv.N, 1)
@propagate_inbounds function shape_gradient(
    cv::GenericCellScalarValues, q_point::Int, base_func::Int
)
    return cv.dNdx[base_func, q_point]
end
@propagate_inbounds function getdetJdV(cv::GenericCellScalarValues, q_point::Int)
    return cv.detJdV[q_point]
end

############################

"""
    pinv(::Tensor{order, dim})

Compute the pseudo-inverse of a Vector tensor.
"""
function pinv(t::Vec{dim,T}) where {dim,T}
    return LinearAlgebra.Transpose{T,Vec{dim,T}}(t / sum(t .^ 2))
end
function ChainRulesCore.rrule(::typeof(pinv), t::Vec{dim,T}) where {dim,T}
    s = sum(t .^ 2)
    TT = LinearAlgebra.Transpose{T,Vec{dim,T}}
    return TT(t / s), Δ -> begin
        nothing, TT(Δ)
    end
end

############################

"""
    Kes, weights = _make_Kes_and_weights(dof_handler, Tuple{MatrixType, VectorType}, Val{n_basefuncs}, Val{dim*n_basefuncs}, Es, As, quadrature_rule, cellvalues)

`weights` : a vector of `xdim*n_basefuncs` vectors, element_id => self-weight load vector, in truss elements, they are all zeros.
"""
function _make_Kes_and_weights(
    dh::DofHandler{xdim,N,T},
    ::Type{Tuple{MatrixType,VectorType}},
    ::Type{Val{n_basefuncs}},
    ::Type{Val{Kesize}},
    Es::Vector{T},
    As::Vector{T},
    quadrature_rule,
    cellvalues::GenericCellScalarValues,
) where {xdim,N,T,MatrixType<:StaticArray,VectorType,n_basefuncs,Kesize}
    nel = getncells(dh.grid)
    Kes = Symmetric{T,MatrixType}[]
    sizehint!(Kes, nel)
    # body_force = ρ .* g # Force per unit volume
    weights = [zeros(VectorType) for i in 1:nel]
    Ke_e = zeros(T, xdim, xdim)
    fe = zeros(T, Kesize)
    Ke_0 = Matrix{T}(undef, Kesize, Kesize)

    celliterator = CellIterator(dh)
    for (k, cell) in enumerate(celliterator)
        Ke_0 .= 0
        truss_reinit!(cellvalues, cell, As[k])
        # fe = weights[k]
        for q_point in 1:getnquadpoints(cellvalues)
            dΩ = getdetJdV(cellvalues, q_point)
            for b in 1:n_basefuncs
                ∇ϕb = shape_gradient(cellvalues, q_point, b)
                # ϕb = shape_value(cellvalues, q_point, b)
                for d2 in 1:xdim
                    # self weight force calculation
                    # fe = @set fe[(b-1)*dim + d2] += ϕb * body_force[d2] * dΩ
                    for a in 1:n_basefuncs
                        ∇ϕa = shape_gradient(cellvalues, q_point, a)
                        # TODO specialized KroneckerDelta struct to make dotdot more efficient
                        Ke_e .= Es[k] * ∇ϕa ⊗ ∇ϕb * dΩ
                        for d1 in 1:xdim
                            #if dim*(b-1) + d2 >= dim*(a-1) + d1
                            Ke_0[xdim * (a - 1) + d1, xdim * (b - 1) + d2] += Ke_e[d1, d2]
                            #end
                        end
                    end
                end
            end
        end
        # weights[k] = fe
        if MatrixType <: SizedMatrix # Work around because full constructor errors
            push!(Kes, Symmetric(SizedMatrix{Kesize,Kesize,T}(Ke_0)))
        else
            push!(Kes, Symmetric(MatrixType(Ke_0)))
        end
    end
    return Kes, weights
end

@inline function truss_reinit!(
    cv::GenericCellScalarValues{ξdim,xdim,T}, ci::CellIterator{xdim,N,T}, crossec::T
) where {ξdim,xdim,N,T}
    Ferrite.check_compatible_geointerpolation(cv, ci)
    return truss_reinit!(cv, ci.coords, crossec)
end

"""
Reinit a cell for a truss element, using the nodal coordinates `x`, cross section `crossec`
"""
function truss_reinit!(
    cv::GenericCellScalarValues{ξdim,xdim,T}, x::AbstractVector{Vec{xdim,T}}, crossec::T
) where {ξdim,xdim,T}
    n_geom_basefuncs = getngeobasefunctions(cv)
    n_func_basefuncs = getn_scalarbasefunctions(cv)
    @assert length(x) == n_geom_basefuncs
    isa(cv, CellVectorValues) && (n_func_basefuncs *= xdim)

    @inbounds for i in 1:length(cv.qr_weights)
        w = cv.qr_weights[i]
        dxdξ = zero(Tensor{1,xdim})
        for j in 1:n_geom_basefuncs
            # in a truss element, x_j ∈ R, dMdξ_j ∈ R, ξ ∈ R
            # cv.dMdξ[j, i] is a 1-1 tensor here
            dxdξ += x[j] * cv.dMdξ[j, i][1]
        end
        # detJ = √(J' J), J = dxdξ ∈ R(n x 1)
        detJ = norm(dxdξ)
        detJ > 0.0 || throw(ArgumentError("det(J) is not positive: det(J) = $(detJ)"))
        cv.detJdV[i] = detJ * w * crossec
        Jinv = pinv(dxdξ)
        for j in 1:n_func_basefuncs
            # cv.dNdξ[j, i] is a 1-1 tensor here
            cv.dNdx[j, i] = cv.dNdξ[j, i][1] * Jinv'
        end
    end
end
