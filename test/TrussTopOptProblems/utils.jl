using LinearAlgebra
using SparseArrays

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

"""assemble element stiffness matrix into a sparse system stiffness matrix

Parameters
----------
k_list : list of element stiffness matrix
    a list of equal-sized global element stiffness matrix
nV : int
    number of vertices, used to compute total dof for sizing 
    the output sparse matrix
id_map : Matrix
    (nE x (2xnode_dof)) index matrix:
        M[element index] = [nodal dof index]
    e.g. for a single beam
        M[1, :] = [1, 2, 3, 4, 5, 6]
    e.g. for a single truss
        M[1, :] = [1, 2, 3, 4]
"""
function assemble_global_stiffness_matrix(k_list, nV, id_map; exist_e_ids=undef)
    node_dof = Int(size(k_list[1].data,1)/2)
    total_dof = node_dof * Int(nV)

    if exist_e_ids==undef
        exist_e_ids = 1:length(k_list)
    else
        @assert Set(exist_e_ids) <= Set(1:length(k_list))
    end

    iT = Int
    T = Float64
    row = iT[]
    col = iT[]
    data = T[]
    for e_id in exist_e_ids
        Ke = k_list[e_id].data
        @assert size(Ke) == (node_dof*2, node_dof*2)
        for i in 1:node_dof*2
            for j in 1:node_dof*2
                if abs(Ke[i,j]) > eps()
                    push!(row,id_map[e_id, i])
                    push!(col,id_map[e_id, j])
                    push!(data, T(Ke[i,j]))
                end
            end
        end
    end
    K = sparse(row, col, data, total_dof, total_dof)
    return K
end

"""
compute a sparse permutation matrix that performs row permuation if multiply on the left
    Perm[i,j] = 1 to place j-th row of the target matrix to i-th row

Inputs
------
dof_stat: dof status vector
    dof_stat[i]  = 1 if i-th dof is fixed, 0 if free, -1 if not exists
"""
function compute_permutation(dof_stat)
    n_fixed_dof = sum(dof_stat)
    total_dof = length(dof_stat)
    n_free_dof = total_dof - n_fixed_dof + 1

    free_tail = 1
    fix_tail = n_free_dof
    id_map_RO = collect(1:total_dof)
    for i in 1:total_dof
        if 0 == dof_stat[i]
            id_map_RO[free_tail] = i
            free_tail += 1
        elseif 1 == dof_stat[i]
            id_map_RO[fix_tail] = i
            fix_tail += 1
        else
            error("Wrong value: $(dof_stat[i])")
        end
    end
    # @show id_map_RO

    iT = Int
    T = Float64
    # a row permuatation matrix (multiply left)
    perm_row = iT[]
    perm_col= iT[]
    perm_data = T[]
    for i in 1:total_dof
        push!(perm_row,i)
        push!(perm_col,id_map_RO[i])
        push!(perm_data,T(1))
    end
    Perm = sparse(perm_row, perm_col, perm_data, total_dof, total_dof)
    return Perm
end