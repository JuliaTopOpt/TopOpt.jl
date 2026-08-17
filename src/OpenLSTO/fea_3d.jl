# 3D hexahedral finite element analysis for the 3D level-set compliance
# driver (a port of the spacedim = 3 branch of M2DO_FEA). Standard trilinear
# Q8 elements with 2x2x2 Gauss integration. Nodes and elements are ordered
# x-fastest (then y, then z) to match the level-set volume-fraction vector.

# Node index (1-based) of grid point (x, y, z).
_hex_node(nelx, nely, x, y, z) = x + (nelx + 1) * y + (nelx + 1) * (nely + 1) * z + 1

# The 8 corner nodes of element (i, j, k), in standard trilinear ordering.
function _hex_element_nodes(nelx, nely, i, j, k)
    return [
        _hex_node(nelx, nely, i, j, k),
        _hex_node(nelx, nely, i + 1, j, k),
        _hex_node(nelx, nely, i + 1, j + 1, k),
        _hex_node(nelx, nely, i, j + 1, k),
        _hex_node(nelx, nely, i, j, k + 1),
        _hex_node(nelx, nely, i + 1, j, k + 1),
        _hex_node(nelx, nely, i + 1, j + 1, k + 1),
        _hex_node(nelx, nely, i, j + 1, k + 1),
    ]
end

# Natural coordinates of the 8 trilinear nodes.
const _HEX_NATURAL = (
    (-1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0),
    (1.0, 1.0, -1.0),
    (-1.0, 1.0, -1.0),
    (-1.0, -1.0, 1.0),
    (1.0, -1.0, 1.0),
    (1.0, 1.0, 1.0),
    (-1.0, 1.0, 1.0),
)

# 2x2x2 Gauss points and (unit) weights.
const _HEX_GAUSS = (
    (-1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)),
    (1 / sqrt(3), -1 / sqrt(3), -1 / sqrt(3)),
    (-1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)),
    (1 / sqrt(3), 1 / sqrt(3), -1 / sqrt(3)),
    (-1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)),
    (1 / sqrt(3), -1 / sqrt(3), 1 / sqrt(3)),
    (-1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)),
    (1 / sqrt(3), 1 / sqrt(3), 1 / sqrt(3)),
)

# Shape function gradients (∂N/∂x, ∂N/∂y, ∂N/∂z) at a natural point. The unit
# cube has Jacobian diag(1/2, 1/2, 1/2), so ∂N/∂x = 2 ∂N/∂ξ.
function _hex_shape_gradients(xi, eta, zeta)
    grads = Vector{NTuple{3,Float64}}(undef, 8)
    for a in 1:8
        xia, etaa, zetaa = _HEX_NATURAL[a]
        dndx = 2.0 * 0.125 * xia * (1 + etaa * eta) * (1 + zetaa * zeta)
        dndy = 2.0 * 0.125 * etaa * (1 + xia * xi) * (1 + zetaa * zeta)
        dndz = 2.0 * 0.125 * zetaa * (1 + xia * xi) * (1 + etaa * eta)
        grads[a] = (dndx, dndy, dndz)
    end
    return grads
end

# Strain-displacement matrix (6 x 24) at a natural point, standard Voigt
# ordering [εxx, εyy, εzz, γxy, γyz, γzx].
function _hex_B_matrix(xi, eta, zeta)
    grads = _hex_shape_gradients(xi, eta, zeta)
    B = zeros(6, 24)
    for a in 1:8
        gx, gy, gz = grads[a]
        c = 3a - 2
        B[1, c] = gx
        B[2, c + 1] = gy
        B[3, c + 2] = gz
        B[4, c] = gy
        B[4, c + 1] = gx
        B[5, c + 1] = gz
        B[5, c + 2] = gy
        B[6, c] = gz
        B[6, c + 2] = gx
    end
    return B
end

"""
    HexMaterial(E, ν, ρ)

Isotropic 3D material with Young's modulus `E`, Poisson's ratio `ν`, and
density `ρ`. Stores the 6x6 Voigt constitutive matrix `C`.
"""
struct HexMaterial
    E::Float64
    nu::Float64
    rho::Float64
    C::Matrix{Float64}
end

function HexMaterial(E::Real, nu::Real, rho::Real=1.0)
    E = Float64(E)
    nu = Float64(nu)
    lambda = E * nu / ((1 + nu) * (1 - 2nu))
    mu = E / (2 * (1 + nu))
    C = [
        lambda+2mu lambda lambda 0.0 0.0 0.0
        lambda lambda+2mu lambda 0.0 0.0 0.0
        lambda lambda lambda+2mu 0.0 0.0 0.0
        0.0 0.0 0.0 mu 0.0 0.0
        0.0 0.0 0.0 0.0 mu 0.0
        0.0 0.0 0.0 0.0 0.0 mu
    ]
    return HexMaterial(E, nu, Float64(rho), C)
end

"""
    HexStudy(nelx, nely, nz, material, fixed_dofs)

A structured `nelx x nely x nelz` hex mesh with a `K u = f` stationary study,
assembled with element area fractions and solved by conjugate gradient.
"""
mutable struct HexStudy
    nelx::Int
    nely::Int
    nelz::Int
    material::HexMaterial
    fixed_dofs::Vector{Int}
    n_dof::Int
    area_fractions::Vector{Float64}
    K::SparseMatrixCSC{Float64,Int}
    f::Vector{Float64}
    u::Vector{Float64}
    gauss_coords::Vector{Vector{Vector{Float64}}}   # per element, per Gauss point
    sensitivities::Vector{Vector{Float64}}           # per element, per Gauss point
end

function HexStudy(
    nelx::Integer,
    nely::Integer,
    nelz::Integer,
    material::HexMaterial,
    fixed_dofs::Vector{Int},
)
    nelx = Int(nelx)
    nely = Int(nely)
    nelz = Int(nelz)
    n_dof = 3 * (nelx + 1) * (nely + 1) * (nelz + 1)
    n_elements = nelx * nely * nelz
    gauss_coords = [Vector{Vector{Float64}}(undef, 8) for _ in 1:n_elements]
    for k in 0:(nelz - 1)
        for j in 0:(nely - 1)
            for i in 0:(nelx - 1)
                e = i + nelx * j + nelx * nely * k + 1
                for g in 1:8
                    xi, eta, zeta = _HEX_GAUSS[g]
                    lx = (xi + 1) / 2
                    ly = (eta + 1) / 2
                    lz = (zeta + 1) / 2
                    gauss_coords[e][g] = [i + lx, j + ly, k + lz]
                end
            end
        end
    end
    sensitivities = [zeros(8) for _ in 1:n_elements]
    return HexStudy(
        nelx,
        nely,
        nelz,
        material,
        fixed_dofs,
        n_dof,
        ones(n_elements),
        spzeros(n_dof, n_dof),
        zeros(n_dof),
        zeros(n_dof),
        gauss_coords,
        sensitivities,
    )
end

function _hex_element_stiffness(material::HexMaterial, Bs::Vector{Matrix{Float64}})
    K = zeros(24, 24)
    for B in Bs
        K .+= (1 / 8) .* (B' * material.C * B)
    end
    return K
end

# Assemble the area-fraction-weighted stiffness matrix, zeroing the rows and
# columns of the fixed dofs and setting their diagonal to one.
function assemble_hex_K!(study::HexStudy)
    nelx, nely, nelz = study.nelx, study.nely, study.nelz
    is_fixed = falses(study.n_dof)
    for d in study.fixed_dofs
        is_fixed[d] = true
    end
    Bs = Matrix{Float64}[]
    for g in 1:8
        xi, eta, zeta = _HEX_GAUSS[g]
        push!(Bs, _hex_B_matrix(xi, eta, zeta))
    end
    Ke = _hex_element_stiffness(study.material, Bs)
    I = Int[]
    J = Int[]
    V = Float64[]
    for k in 0:(nelz - 1)
        for j in 0:(nely - 1)
            for i in 0:(nelx - 1)
                e = i + nelx * j + nelx * nely * k + 1
                af = study.area_fractions[e]
                nodes = _hex_element_nodes(nelx, nely, i, j, k)
                dofs = Int[]
                for n in nodes
                    push!(dofs, 3n - 2, 3n - 1, 3n)
                end
                for ii in 1:24
                    di = dofs[ii]
                    is_fixed[di] && continue
                    for jj in 1:24
                        dj = dofs[jj]
                        is_fixed[dj] && continue
                        push!(I, di)
                        push!(J, dj)
                        push!(V, af * Ke[ii, jj])
                    end
                end
            end
        end
    end
    for d in study.fixed_dofs
        push!(I, d)
        push!(J, d)
        push!(V, 1.0)
    end
    study.K = sparse(I, J, V, study.n_dof, study.n_dof)
    return study.K
end

function assemble_hex_f!(
    study::HexStudy, load_dofs::Vector{Int}, load_values::Vector{Float64}
)
    f = zeros(study.n_dof)
    for (d, v) in zip(load_dofs, load_values)
        f[d] += v
    end
    for d in study.fixed_dofs
        f[d] = 0.0
    end
    study.f = f
    return f
end

function solve_hex!(study::HexStudy)
    study.u = cg_solve(study.K, study.f)
    return study.u
end

# Compliance sensitivity at each Gauss point (strain-energy density times the
# area fraction). Returns the compliance.
function compute_hex_compliance_sensitivities!(study::HexStudy)
    nelx, nely, nelz = study.nelx, study.nely, study.nelz
    C = study.material.C
    objective = dot(study.f, study.u)
    for k in 0:(nelz - 1)
        for j in 0:(nely - 1)
            for i in 0:(nelx - 1)
                e = i + nelx * j + nelx * nely * k + 1
                fill!(study.sensitivities[e], 0.0)
                study.area_fractions[e] <= 0.1 && continue
                nodes = _hex_element_nodes(nelx, nely, i, j, k)
                dofs = Int[]
                for n in nodes
                    push!(dofs, 3n - 2, 3n - 1, 3n)
                end
                ue = study.u[dofs]
                for g in 1:8
                    xi, eta, zeta = _HEX_GAUSS[g]
                    B = _hex_B_matrix(xi, eta, zeta)
                    Bu = B * ue
                    stress_strain = dot(Bu, C * Bu)
                    study.sensitivities[e][g] = -stress_strain * study.area_fractions[e]
                end
            end
        end
    end
    return objective
end

# Interpolate Gauss-point compliance sensitivities to a boundary point by
# weighted least squares (3D, ten basis functions).
function hex_boundary_sensitivity(
    study::HexStudy, boundary_point::Vector{Float64}; radius::Float64=2.0
)
    r2 = radius * radius
    rows = Vector{Vector{Float64}}()
    b = Float64[]
    for e in eachindex(study.area_fractions)
        for g in 1:8
            gp = study.gauss_coords[e][g]
            far = false
            for k in 1:3
                if abs(gp[k] - boundary_point[k]) > 1.5 * radius
                    far = true
                    break
                end
            end
            far && continue
            xb = gp[1] - boundary_point[1]
            yb = gp[2] - boundary_point[2]
            zb = gp[3] - boundary_point[3]
            d2 = xb * xb + yb * yb + zb * zb
            if d2 < r2
                distance = sqrt(d2)
                w = study.area_fractions[e] / distance
                push!(
                    rows,
                    [
                        w,
                        xb * w,
                        yb * w,
                        zb * w,
                        xb * yb * w,
                        xb * zb * w,
                        yb * zb * w,
                        xb * xb * w,
                        yb * yb * w,
                        zb * zb * w,
                    ],
                )
                push!(b, study.sensitivities[e][g] * w)
            end
        end
    end
    length(b) < 10 && return 0.0
    A = permutedims(hcat(rows...))
    x = A \ b
    return x[1]
end
