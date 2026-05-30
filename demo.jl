using StaticArrays
using LinearAlgebra

# 1. Define the Gauss Quadrature points and weights for a 2x2 rule
# The points are at ±1/√3
const GAUSS_PTS = SVector(-1/sqrt(3), 1/sqrt(3))
const GAUSS_WTS = SVector(1.0, 1.0)

"""
    quad4_thermal_matrix(nodes, k)

Calculates the 4x4 local thermal conductivity matrix for a Quad4 element.
`nodes` is a 4x2 matrix (or vector of vectors) of the [x, y] coordinates of the 4 nodes.
`k` is the thermal conductivity of the element.
"""
function quad4_thermal_matrix(nodes::SMatrix{4, 2, Float64}, k::Float64)
    # Initialize an empty 4x4 mutable static matrix for performance
    Ke = @MMatrix zeros(4, 4)
    
    # Loop over the 2x2 Gauss integration points
    for i in 1:2
        for j in 1:2
            xi = GAUSS_PTS[i]
            eta = GAUSS_PTS[j]
            weight = GAUSS_WTS[i] * GAUSS_WTS[j]
            
            # 2. Derivatives of shape functions with respect to local coords (ξ, η)
            # dN/dξ
            dN_dxi = SVector(
                -0.25 * (1 - eta),
                 0.25 * (1 - eta),
                 0.25 * (1 + eta),
                -0.25 * (1 + eta)
            )
            # dN/dη
            dN_deta = SVector(
                -0.25 * (1 - xi),
                -0.25 * (1 + xi),
                 0.25 * (1 + xi),
                 0.25 * (1 - xi)
            )
            
            # Combine into a 2x4 matrix: [dN/dξ ; dN/dη]
            dN_dxi_eta = SMatrix{2, 4}(
                dN_dxi[1], dN_deta[1],
                dN_dxi[2], dN_deta[2],
                dN_dxi[3], dN_deta[3],
                dN_dxi[4], dN_deta[4]
            )
            
            # 3. Calculate the Jacobian matrix (J = dN_dxi_eta * coordinates)
            J = dN_dxi_eta * nodes
            detJ = det(J)
            
            # 4. Calculate the B matrix (derivatives w.r.t real coords x, y)
            # B = J \ dN_dxi_eta (using left division for inverse multiplication)
            B = J \ dN_dxi_eta
            
            # 5. Add the contribution of this Gauss point to the element matrix
            # Integral evaluation: B^T * k * B * det(J) * weight
            Ke += B' * k * B * detJ * weight
        end
    end
    
    return SMatrix(Ke) # Return as immutable for safety and speed
end

"""
    penalized_thermal_matrix(nodes, x_e, p, k0, kmin)

Returns the local conductivity matrix for an element, scaled by its SIMP density.
"""
function penalized_thermal_matrix(nodes::SMatrix{4, 2, Float64}, x_e::Float64, p::Float64, k0::Float64, kmin::Float64)
    # 1. Calculate the penalized thermal conductivity
    k_penalized = kmin + (x_e^p) * (k0 - kmin)
    
    # 2. Generate the standard local matrix using that conductivity
    return quad4_thermal_matrix(nodes, k_penalized)
end