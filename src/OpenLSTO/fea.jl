# Area-fraction finite element analysis for the level-set compliance problem
# (a port of the `M2DO_FEA` classes used by `projects/compliance/main.cpp`).
#
# A structured grid of 2D plane-stress Q4 elements is solved with a conjugate
# gradient method, then the compliance shape sensitivity is computed at each
# Gauss point and interpolated to the boundary points by weighted least
# squares.
#
# Equivalent machinery already exists elsewhere in TopOpt.jl: the FEA solve is
# `GenericFEASolver`/`DirectSolver`/`CGAssemblySolver`/`CGMatrixFreeSolver`
# (src/FEA/solvers_api.jl), global stiffness assembly is `AssembleKFun`
# (src/Functions/assemble_K.jl), and the compliance objective is
# `ComplianceFun` (src/Functions/compliance.jl). The self-contained solver
# here reproduces OpenLSTO's linear (ersatz) area-fraction weighting, which
# those SIMP-based paths do not.

mutable struct SolidElement
    node_ids::Vector{Int}
    dof::Vector{Int}
    area_fraction::Float64
    centroid::Vector{Float64}
end

mutable struct FEMesh
    nelx::Int
    nely::Int
    n_nodes::Int
    n_elements::Int
    n_dof::Int
    coords::Matrix{Float64}      # n_nodes × 2
    elements::Vector{SolidElement}
end

mutable struct SolidMaterial
    E::Float64
    nu::Float64
    rho::Float64
    h::Float64
    C::Matrix{Float64}
end

function FEMesh(nelx::Integer, nely::Integer)
    nelx = Int(nelx)
    nely = Int(nely)
    n_nodes = (nelx + 1) * (nely + 1)
    n_elements = nelx * nely
    coords = zeros(n_nodes, 2)
    for i in 1:n_nodes
        coords[i, 1] = (i - 1) % (nelx + 1)
        coords[i, 2] = (i - 1) ÷ (nelx + 1)
    end
    elements = SolidElement[]
    w = nelx + 1
    for e in 1:n_elements
        ex = (e - 1) % nelx
        ey = (e - 1) ÷ nelx
        n1 = ex + ey * w + 1
        node_ids = [n1, n1 + 1, n1 + w + 1, n1 + w]
        dof = Int[]
        for n in node_ids
            push!(dof, 2n - 1, 2n)
        end
        cx =
            (
                coords[node_ids[1], 1] +
                coords[node_ids[2], 1] +
                coords[node_ids[3], 1] +
                coords[node_ids[4], 1]
            ) / 4
        cy =
            (
                coords[node_ids[1], 2] +
                coords[node_ids[2], 2] +
                coords[node_ids[3], 2] +
                coords[node_ids[4], 2]
            ) / 4
        push!(elements, SolidElement(node_ids, dof, 1.0, [cx, cy]))
    end
    return FEMesh(nelx, nely, n_nodes, n_elements, 2 * n_nodes, coords, elements)
end

# Plane-stress 4×4 constitutive matrix mapping the strain vector
# [εxx, ∂u_x/∂y, ∂u_y/∂x, εyy] to stress (see `SolidMaterial::SolidMaterial`).
function SolidMaterial(E::Real, nu::Real, rho::Real=1.0; h::Real=1.0)
    E = Float64(E)
    nu = Float64(nu)
    A = [
        1.0 0.0 0.0 0.0
        0.0 0.5 0.5 0.0
        0.0 0.5 0.5 0.0
        0.0 0.0 0.0 1.0
    ]
    D = [
        1.0 0.0 0.0 nu
        0.0 (1 - nu)/2 (1 - nu)/2 0.0
        0.0 (1 - nu)/2 (1 - nu)/2 0.0
        nu 0.0 0.0 1.0
    ]
    D .*= E / (1 - nu^2)
    C = Float64(h) * D * A
    return SolidMaterial(E, nu, Float64(rho), Float64(h), C)
end

# 2×2 Gauss points for the Q4 element (order 2 in each direction).
const GAUSS_POINTS = (
    (-1 / sqrt(3), -1 / sqrt(3)),
    (1 / sqrt(3), -1 / sqrt(3)),
    (-1 / sqrt(3), 1 / sqrt(3)),
    (1 / sqrt(3), 1 / sqrt(3)),
)

# Bilinear shape function values and natural-coordinate gradients.
function shape_values(xi::Float64, eta::Float64)
    return (
        0.25 * (1 - xi) * (1 - eta),
        0.25 * (1 + xi) * (1 - eta),
        0.25 * (1 + xi) * (1 + eta),
        0.25 * (1 - xi) * (1 + eta),
    )
end

function shape_gradients(xi::Float64, eta::Float64)
    return (
        (-0.25 * (1 - eta), -0.25 * (1 - xi)),
        (0.25 * (1 - eta), -0.25 * (1 + xi)),
        (0.25 * (1 + eta), 0.25 * (1 + xi)),
        (-0.25 * (1 + eta), 0.25 * (1 - xi)),
    )
end

# Strain-displacement matrix (4 × 8) and Jacobian determinant at a natural
# point. Column ordering follows the element dof ordering (u_x, u_y per node).
function B_matrix(xs::Vector{Float64}, ys::Vector{Float64}, xi::Float64, eta::Float64)
    dN = shape_gradients(xi, eta)
    J11 = sum(dN[a][1] * xs[a] for a in 1:4)
    J12 = sum(dN[a][1] * ys[a] for a in 1:4)
    J21 = sum(dN[a][2] * xs[a] for a in 1:4)
    J22 = sum(dN[a][2] * ys[a] for a in 1:4)
    detJ = J11 * J22 - J12 * J21
    B = zeros(4, 8)
    for a in 1:4
        gx = (J22 * dN[a][1] - J12 * dN[a][2]) / detJ
        gy = (-J21 * dN[a][1] + J11 * dN[a][2]) / detJ
        B[1, 2a - 1] = gx
        B[2, 2a - 1] = gy
        B[3, 2a] = gx
        B[4, 2a] = gy
    end
    return B, detJ
end

function element_stiffness(mesh::FEMesh, element::SolidElement, C::Matrix{Float64})
    xs = [mesh.coords[n, 1] for n in element.node_ids]
    ys = [mesh.coords[n, 2] for n in element.node_ids]
    K = zeros(8, 8)
    for (xi, eta) in GAUSS_POINTS
        B, detJ = B_matrix(xs, ys, xi, eta)
        K .+= (detJ .* (B' * C * B))  # unit Gauss weights
    end
    return K
end

# Physical coordinates of the Gauss points in an element.
function gauss_point_coords(mesh::FEMesh, element::SolidElement)
    coords = Vector{Vector{Float64}}(undef, 4)
    xs = [mesh.coords[n, 1] for n in element.node_ids]
    ys = [mesh.coords[n, 2] for n in element.node_ids]
    for (j, (xi, eta)) in enumerate(GAUSS_POINTS)
        N = shape_values(xi, eta)
        coords[j] = [sum(N[a] * xs[a] for a in 1:4), sum(N[a] * ys[a] for a in 1:4)]
    end
    return coords
end

mutable struct StationaryStudy
    mesh::FEMesh
    material::SolidMaterial
    fixed_dofs::Vector{Int}
    K::SparseMatrixCSC{Float64,Int}
    f::Vector{Float64}
    u::Vector{Float64}
end

function StationaryStudy(mesh::FEMesh, material::SolidMaterial, fixed_dofs::Vector{Int})
    return StationaryStudy(
        mesh,
        material,
        fixed_dofs,
        spzeros(mesh.n_dof, mesh.n_dof),
        zeros(mesh.n_dof),
        zeros(mesh.n_dof),
    )
end

# Store the material area fraction on each element. Both the stiffness
# assembly and the compliance sensitivity read it from there, mirroring
# `fea_mesh.solid_elements[i].area_fraction` in OpenLSTO.
function set_area_fractions!(study::StationaryStudy, area_fractions::Vector{Float64})
    for (e, element) in enumerate(study.mesh.elements)
        element.area_fraction = area_fractions[e]
    end
    return study
end

# Assemble the area-fraction-weighted stiffness matrix, eliminating the
# constrained (fixed) dofs by zeroing their rows/columns and setting the
# diagonal to one.
function assemble_K_with_area_fractions!(study::StationaryStudy)
    mesh = study.mesh
    Ke = element_stiffness(mesh, mesh.elements[1], study.material.C)
    is_fixed = falses(mesh.n_dof)
    for d in study.fixed_dofs
        is_fixed[d] = true
    end
    I = Int[]
    J = Int[]
    V = Float64[]
    for element in mesh.elements
        af = element.area_fraction
        dofs = element.dof
        for i in 1:8
            di = dofs[i]
            is_fixed[di] && continue
            for j in 1:8
                dj = dofs[j]
                is_fixed[dj] && continue
                push!(I, di)
                push!(J, dj)
                push!(V, af * Ke[i, j])
            end
        end
    end
    for d in study.fixed_dofs
        push!(I, d)
        push!(J, d)
        push!(V, 1.0)
    end
    study.K = sparse(I, J, V, mesh.n_dof, mesh.n_dof)
    return study.K
end

function assemble_f!(
    study::StationaryStudy, load_dofs::Vector{Int}, load_values::Vector{Float64}
)
    mesh = study.mesh
    f = zeros(mesh.n_dof)
    for (d, v) in zip(load_dofs, load_values)
        f[d] += v
    end
    for d in study.fixed_dofs
        f[d] = 0.0
    end
    study.f = f
    return f
end

# Conjugate gradient solve (OpenLSTO's `SolveWithCG` uses the same tolerance).
function cg_solve(
    K::SparseMatrixCSC{Float64,Int},
    b::Vector{Float64};
    tol::Float64=1e-6,
    maxiter::Int=10000,
)
    x = zeros(length(b))
    r = copy(b)
    p = copy(r)
    rsold = dot(r, r)
    for _ in 1:maxiter
        Ap = K * p
        alpha = rsold / dot(p, Ap)
        x .+= alpha .* p
        r .-= alpha .* Ap
        rsnew = dot(r, r)
        sqrt(rsnew) < tol && break
        p .= r .+ (rsnew / rsold) .* p
        rsold = rsnew
    end
    return x
end

function solve!(study::StationaryStudy)
    study.u = cg_solve(study.K, study.f)
    return study.u
end

mutable struct SensitivityAnalysis
    study::StationaryStudy
    Bs::Vector{Matrix{Float64}}                       # one B per Gauss point
    gauss_coords::Vector{Vector{Vector{Float64}}}     # per element, per Gauss point
    sensitivities::Vector{Vector{Float64}}            # per element, per Gauss point
    objective::Float64
end

function SensitivityAnalysis(study::StationaryStudy)
    mesh = study.mesh
    xs = [mesh.coords[n, 1] for n in mesh.elements[1].node_ids]
    ys = [mesh.coords[n, 2] for n in mesh.elements[1].node_ids]
    Bs = Matrix{Float64}[]
    for (xi, eta) in GAUSS_POINTS
        B, _ = B_matrix(xs, ys, xi, eta)
        push!(Bs, B)
    end
    gauss_coords = [gauss_point_coords(mesh, element) for element in mesh.elements]
    sensitivities = [zeros(4) for _ in mesh.elements]
    return SensitivityAnalysis(study, Bs, gauss_coords, sensitivities, 0.0)
end

# Compliance shape sensitivity at each Gauss point (OpenLSTO's
# `ComputeComplianceSensitivities`). The value is the strain-energy density
# times the area fraction; it is normalised downstream in the optimiser, so
# the quadrature weight and Jacobian are not included.
function compute_compliance_sensitivities!(sens::SensitivityAnalysis)
    study = sens.study
    mesh = study.mesh
    C = study.material.C
    for e in eachindex(mesh.elements)
        element = mesh.elements[e]
        fill!(sens.sensitivities[e], 0.0)
        element.area_fraction <= 0.1 && continue
        ue = study.u[element.dof]
        for j in 1:4
            Bu = sens.Bs[j] * ue
            stress_strain = dot(Bu, C * Bu)
            sens.sensitivities[e][j] = -stress_strain * element.area_fraction
        end
    end
    sens.objective = dot(study.f, study.u)
    return sens.objective
end

# Interpolate Gauss-point sensitivities to a boundary point by weighted least
# squares (OpenLSTO's `ComputeBoundarySensitivities` / `SolveLeastSquares`).
function compute_boundary_sensitivity(
    sens::SensitivityAnalysis, boundary_point::Vector{Float64}; radius::Float64=2.0
)
    mesh = sens.study.mesh
    r2 = radius * radius
    rows = Vector{Vector{Float64}}()
    b = Float64[]
    for e in eachindex(mesh.elements)
        element = mesh.elements[e]
        centroid = element.centroid
        far = false
        for k in 1:2
            if abs(centroid[k] - boundary_point[k]) > 1.5 * radius
                far = true
                break
            end
        end
        far && continue
        for j in 1:4
            gp = sens.gauss_coords[e][j]
            xb = gp[1] - boundary_point[1]
            yb = gp[2] - boundary_point[2]
            d2 = xb * xb + yb * yb
            if d2 < r2
                distance = sqrt(d2)
                w = element.area_fraction / distance
                push!(rows, [w, xb * w, yb * w, xb * yb * w, xb * xb * w, yb * yb * w])
                push!(b, sens.sensitivities[e][j] * w)
            end
        end
    end
    length(b) < 10 && return 0.0
    A = permutedims(hcat(rows...))
    x = A \ b
    return x[1]
end
