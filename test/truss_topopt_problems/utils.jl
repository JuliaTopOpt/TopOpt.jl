using LinearAlgebra
using SparseArrays

# TODO replace this with TrussTopOptProblems.compute_local_axes
function global2local_transf_matrix(end_vert_u, end_vert_v)
    @assert length(end_vert_u) == length(end_vert_v)
    @assert length(end_vert_u) == 2 || length(end_vert_u) == 3
    xdim = length(end_vert_u)
    L = norm(end_vert_u-end_vert_v)
    @assert L > 1e-6
    # by convention, the new x axis is along the element's direction
    # directional cosine of the new x axis in the global world frame
    c_x = (end_vert_v[1] - end_vert_u[1])/L
    c_y = (end_vert_v[2] - end_vert_u[2])/L
    R = zeros(3,3)
    Γ = zeros(2,xdim*2)
    if 3 == xdim
        # error("Not Implemented")
        c_z = (end_vert_v[3] - end_vert_u[3]) / L
        if abs(abs(c_z) - 1.0) < eps()
            R[1, 3] = -c_z
            R[2, 2] = 1
            R[3, 1] = c_z
        else
            # local x_axis = element's vector
            new_x = [c_x, c_y, c_z]
            # local y axis = cross product with global z axis
            new_y = -cross(new_x, [0,0,1.0])
            new_y /= norm(new_y)
            new_z = cross(new_x, new_y)
            R[1, :] = new_x
            R[2, :] = new_y
            R[3, :] = new_z
        end
        Γ[1, 1:3] = R[1,:]
        Γ[2, 4:6] = R[1,:]
    elseif 2 == xdim
        R[1,:] = [c_x, c_y, 0]
        R[2,:] = [-c_y, c_x, 0]
        R[3,3] = 1.0
        Γ[1, 1:2] = R[1,1:2]
        Γ[2, 3:4] = R[1,1:2]
        # Γ = [c_x c_y 0 0
        #      0 0 c_x c_y]
    end
    return Γ
end