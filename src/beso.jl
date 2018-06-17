module BESO
using JuAFEM
using Tensors
using TopOpt

struct BESOResult{T}
    topology::Vector{T}
    objval::T
end

struct BESO
    penalty::TP
    result::BESOResult{T}
    topologies::Vector{Vector{T}}
end

function compliance_beso(nels::NTuple{dim,Int}, sizes::NTuple{dim,T}, problem, V::T, p=1, er=0.02, E=1., force = -1., filtering = false, rmin = 3, norm_type = 2, xmin = 0.001, unpenalized_obj = true, tol = 1e-3, tracing = false, x0 = ones(T,prod(nels))) where {dim, T}

    # Define problem
    rectgrid = make_rect_grid(nels,sizes)
    sp = StiffnessProblem{dim,T,typeof(rectgrid)}(rectgrid, problem, E)
    dh, dbc, K, Kes, f = get_fe(sp, T(force))
    nel = prod(nels)
    black_cells = getcellset(sp.grid.grid, "black")
    white_cells = getcellset(sp.grid.grid, "white")
    fixedcells = black_cells ∪ white_cells
    nfc = length(fixedcells)

    # Initialize vectors
    if length(x0) == nel
        x = zeros(Float64, nel-nfc)
        k = 1
        for i in 1:nel
            if i ∉ fixedcells
                x[k] = x0[i]
                k += 1
            end
        end
    elseif length(x0) == nel - nfc
        x = x0
    else
        throw("Initial solution is not of an appropriate size.")
    end
    c_hist = T[]
    v_hist = T[]
    x_hist = Vector{T}[]
    add_hist = Int[]
    rem_hist = Int[]

    u = zeros(T, size(K,1)) # Nodal dofs
    mean_comp = zeros(T, nel) # Mean compliance of cell / density^p
    
    function solve_fea!(u, ρ)
        # Assemble global stiffness matrix and force vector
        TopOpt.assemble!(K, f, ρ, xmin, Kes, dh, dbc, sp)
        # Solve system of equations
        u .= Symmetric(K) \ f
        return
    end

    # Prepare buffers for chequerboard filter
    if filtering
        nnodes = getnnodes(dh.grid)
        nodal_grad = zeros(Float64, nnodes)
        nodal_connect = zeros(Int, nnodes)
        excl_nodes = zeros(Bool, nnodes)
        iter_count = 1
        last_grad = zeros(nel-nfc)

        cell_neighbours = Vector{Int}[]
        sizehint!(cell_neighbours, getncells(dh.grid))
        for (i,cell) in enumerate(CellIterator(dh))
            push!(cell_neighbours, Int[])
            center = mean(cell.coords)
            for n in cell.nodes
                if norm(getnodes(dh.grid, n).x - center, norm_type) < rmin
                    push!(cell_neighbours[i], n)
                end
            end
        end
    end

    function cheqfilter!(grad)
        nodal_grad .= 0.
        nodal_connect .= 0
        excl_nodes .= false
        k = 1
        for (i,cell) in enumerate(CellIterator(dh))
            if i ∉ fixedcells
                for n in cell.nodes
                    nodal_grad[n] += grad[k]
                    nodal_connect[n] += 1
                end
                k += 1
            end
        end
        excl_nodes .= nodal_connect .== 0
        nodal_connect .= max.(nodal_connect, 1)
        nodal_grad ./= nodal_connect

        grad .= 0.
        k = 1
        for (i,cell) in enumerate(CellIterator(dh))
            if i ∉ fixedcells
                total_w = 0.
                center = mean(cell.coords)
                for n in cell_neighbours[i]
                    if !excl_nodes[n]
                        wij = max(rmin - norm(getnodes(dh.grid, n).x - center), 0)
                        grad[k] += wij * nodal_grad[n]
                        total_w += wij
                    end
                end
                grad[k] /= total_w
                k += 1
            end
        end
        if iter_count > 1
            grad .= 1/2 .* (grad .+ last_grad)
        end
        last_grad .= grad
        iter_count += 1
        return 
    end

    # Compute objective and constraint with derivatives
    nf = 0
    function compliance_objective(x, sens)
        nf += 1
        solve_fea!(u, x)
        obj = zero(T)
        k = 1
        for (i,cell) in enumerate(CellIterator(dh))
            mean_comp[i] = 1/2*compliance(u[celldofs(cell)], Kes[i])
            if i ∈ black_cells
                obj += mean_comp[i]                
            elseif i ∈ white_cells
                obj += xmin^p * mean_comp[i]                
            else
                sens[k] = x[k]^(p-1) * mean_comp[i]
                obj += x[k]^p * mean_comp[i]
                k += 1
            end
        end
        if filtering
            cheqfilter!(sens)
        end
        if tracing
            push!(c_hist, obj)
            push!(x_hist, copy(x))
            if length(x_hist) == 1
                push!(add_hist, 0)
                push!(rem_hist, 0)
            else
                push!(add_hist, sum(x_hist[end] .> x_hist[end-1]))
                push!(rem_hist, sum(x_hist[end] .< x_hist[end-1]))
            end 
            #push!(v_hist, ((sum(x) + length(black_cells) + xmin*length(white_cells)) - nel*xmin) / (1. - xmin) / prod(nels))
            push!(v_hist, (sum(x) + length(black_cells) + xmin*length(white_cells)) / prod(nels))
        end
        return obj
    end
    
    # Soft-kill BESO

    ndim = nel-nfc
    change = 1.
    vol = 1.
    sens = zeros(T, ndim)
    oldsens = similar(sens)
    senstol = tol / 100

    function add_del!(x, sens, targetV)
        l1, l2 = min(sens...), max(sens...)
        while (l2 - l1) / l2 > senstol
            th = (l1 + l2) / 2
            x .= max.(sign.(sens .- th), xmin)
            if sum(x) - targetV * nel > 0
                l1 = th
            else
                l2 = th
            end
        end
        return
    end

    obj_trace = zeros(T, 10)
    i = 0
    while change > tol
        i += 1
        vol = max(vol*(1-er), V)
        i > 1 && (oldsens .= sens)
        for j in 2:10; obj_trace[j-1] = obj_trace[j]; end
        obj_trace[10] = compliance_objective(x, sens)
        add_del!(x, sens, vol)
        if i >= 10
            a = sum(obj_trace[1:5])
            b = sum(obj_trace[6:10])
            change = abs(a-b)/b
        end
    end

    # Postprocessing
    for i in 1:length(x_hist)
        topology = zeros(T, nel)
        k = 1
        for j in 1:nel
            if j ∈ black_cells
                topology[j] = 1.
            elseif j ∈ white_cells
                topology[j] = xmin
            else
                topology[j] = x_hist[i][k]
                k += 1
            end
        end
        x_hist[i] = topology
    end

    if !tracing
        topology = zeros(T, nel)
        k = 1
        for i in 1:nel
            if i ∈ black_cells
                topology[i] = 1.            
            elseif i ∈ white_cells
                topology[i] = xmin
            else
                topology[i] = x[k]
                k += 1
            end
        end
    end

    if unpenalized_obj
        obj = zero(T)
        for (i,cell) in enumerate(CellIterator(dh))
            obj += 1/2*topology[i]*compliance(u[celldofs(cell)], Kes[i])
        end
    end

    beso_results = History(c_hist, v_hist, x_hist, add_hist, rem_hist)

    return topology, obj, nf, K, Kes, dh, beso_results
end
end