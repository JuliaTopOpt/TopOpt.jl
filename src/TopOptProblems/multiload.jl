"""
    MultiLoad(problem, scenarios)

Multi-load-case wrapper that generates stochastic load scenarios for robust
topology optimization. `scenarios` is the number of random load cases to draw.
Use with `MeanComplianceFun` or `BlockComplianceFun`.

Usage example:

```
using Distributions, LinearAlgebra, TopOpt

f1 = RandomMagnitudeFun([0, -1], Uniform(0.5, 1.5))
f2 = RandomMagnitudeFun(normalize([1, -1]), Uniform(0.5, 1.5))
f3 = RandomMagnitudeFun(normalize([-1, -1]), Uniform(0.5, 1.5))

base_problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), 1.0, 0.3, 1.0)
problem = MultiLoad(base_problem, [(160, 20) => f1, (80, 40) => f2, (120, 0) => f3], 10000)
```
"""
struct MultiLoad{dim,T,TP<:StiffnessTopOptProblem{dim,T},TF} <:
       StiffnessTopOptProblem{dim,T}
    problem::TP
    F::TF
end
@forward_property MultiLoad problem
for F in
    (:getE, :getν, :nnodespercell, :getcloaddict, :getdim, :getpressuredict, :getfacesets)
    @eval $F(p::MultiLoad) = $F(p.problem)
end
function MultiLoad(problem::StiffnessTopOptProblem, N::Int, load_rules::Vector{<:Pair})
    I = Int[]
    J = Int[]
    V = Float64[]
    for (pos, f) in load_rules
        dofs = find_nearest_dofs(problem, pos)
        for i in 1:N
            load = f()
            append!(I, dofs)
            push!(J, fill(i, length(dofs))...)
            append!(V, load)
        end
    end
    F = sparse(I, J, V, ndofs(problem.ch.dh), N)
    return MultiLoad(problem, F)
end
function MultiLoad(
    problem::StiffnessTopOptProblem, N::Int, dist::Distributions.Distribution=Uniform(-2, 2)
)
    F = generate_random_loads(problem, N, dist, random_direction)
    return MultiLoad(problem, F)
end

function find_nearest_dofs(problem, p)
    grid = problem.ch.dh.grid
    shortest = Inf
    closest = 0
    for (i, n) in enumerate(grid.nodes)
        dist = norm(n.x .- p)
        if dist < shortest
            shortest = dist
            closest = i
        end
    end
    closest != 0 ||
        throw(ArgumentError("MultiLoad: no node found near the specified coordinates"))
    return problem.metadata.node_dofs[:, closest]
end

"""
    RandomMagnitudeFun

Random load-magnitude sampler used inside `MultiLoad`.
"""
struct RandomMagnitudeFun{Tf,Tdist} <: Function
    f::Tf
    dist::Tdist
end
(rm::RandomMagnitudeFun)() = rm.f .* rand(rm.dist)

function random_direction()
    theta = rand() * 2 * π
    return [cos(theta), sin(theta)]
end

function get_surface_dofs(problem::StiffnessTopOptProblem)
    dh = problem.ch.dh
    grid = dh.grid
    node_dofs = problem.metadata.node_dofs

    surface_node_inds = Int[]
    for (setname, facets) in grid.facetsets
        for (cellind, faceind) in facets
            cell = getcells(grid, cellind)
            face = Ferrite.facets(cell)[faceind]
            append!(surface_node_inds, collect(face))
        end
    end
    unique!(surface_node_inds)
    return setdiff(node_dofs[:, surface_node_inds], problem.ch.prescribed_dofs)
end

function generate_random_loads(
    problem::StiffnessTopOptProblem,
    N::Int,
    scalar::Distributions.Distribution=Distributions.Uniform(-2, 2),
    direction::Function=random_direction,
)
    loadrule = () -> direction() .* rand(scalar)
    surface_dofs = get_surface_dofs(problem)

    # surface_dofs is a flat vector of DOFs after setdiff
    # Group them by node: n_dofs_per_node DOFs per node
    n_dofs_per_node = size(problem.metadata.node_dofs, 1)
    n_surface_nodes = length(surface_dofs) ÷ n_dofs_per_node

    # Create node start indices in the flat vector
    node_indices = [
        (n_dofs_per_node * (i - 1) + 1):(n_dofs_per_node * i) for i in 1:n_surface_nodes
    ]

    FI = Int[]
    FJ = Int[]
    FV = Float64[]
    selected_nodes = rand(1:n_surface_nodes, N)
    for i in 1:N
        load = loadrule()
        idx_range = node_indices[selected_nodes[i]]
        dofs = surface_dofs[idx_range]
        append!(FI, dofs)
        append!(FJ, fill(i, length(dofs)))
        append!(FV, load)
    end
    return sparse(FI, FJ, FV)
end
