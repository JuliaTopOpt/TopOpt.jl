using StaticArrays
using LinearAlgebra
using Ferrite 

function thermal_conductivity_matrix(cellvalues::CellScalarValues{dim, T}, k_prop::Float64) where {dim, T}

    n_basefuncs = getnbasefunctions(cellvalues)
    
    # Initialize an empty mutable static matrix for speed
    Ke = @MMatrix zeros(T, n_basefuncs, n_basefuncs)
    
    for q_point in 1:getnquadpoints(cellvalues)
        # dΩ is the determinant of the Jacobian multiplied by the quadrature weight
        dΩ = getdetJdV(cellvalues, q_point)
        
        # Integrate B^T * k * B
        for i in 1:n_basefuncs
            ∇N_i = shape_gradient(cellvalues, q_point, i)
            for j in 1:n_basefuncs
                ∇N_j = shape_gradient(cellvalues, q_point, j)
                
                # The dot product of the shape gradients scales the conductivity
                Ke[i, j] += (∇N_i ⋅ ∇N_j) * k_prop * dΩ
            end
        end
    end
    
    # Return as an immutable SMatrix for thread safety
    return SMatrix(Ke)
end
