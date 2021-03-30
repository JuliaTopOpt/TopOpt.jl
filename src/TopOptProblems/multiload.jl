"""
Usage example:

```
using Distributions, LinearAlgebra, TopOpt

f1 = RandomMagnitude([0, -1], Uniform(0.5, 1.5))
f2 = RandomMagnitude(normalize([1, -1]), Uniform(0.5, 1.5))
f3 = RandomMagnitude(normalize([-1, -1]), Uniform(0.5, 1.5))

base_problem = PointLoadCantilever(Val{:Linear}, (160, 40), (1.0, 1.0), 1.0, 0.3, 1.0)
problem = MultiLoad(base_problem, [(160, 20) => f1, (80, 40) => f2, (120, 0) => f3], 10000)
"""
struct MultiLoad{dim, T, TP <: StiffnessTopOptProblem{dim, T}, TF} <: StiffnessTopOptProblem{dim, T}
    problem::TP
    F::TF
end
@forward_property MultiLoad problem
for F in (:getE, :getν, :nnodespercell, :getcloaddict, :getdim, :getpressuredict, :getfacesets)
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
    problem::StiffnessTopOptProblem,
    N::Int,
    dist::Distributions.Distribution = Uniform(-2, 2),
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
    @assert closest != 0
    return problem.metadata.node_dofs[:, closest]
end

struct RandomMagnitude{Tf, Tdist} <: Function
    f::Tf
    dist::Tdist
end
(rm::RandomMagnitude)() = rm.f .* rand(rm.dist)

function random_direction()
    theta = rand() * 2 * π
    return [cos(theta), sin(theta)]
end

function get_surface_dofs(problem::StiffnessTopOptProblem)
	dh = problem.ch.dh
	boundary_matrix = dh.grid.boundary_matrix
	interpolation = dh.field_interpolations[1]
	celliterator = Ferrite.CellIterator(dh)
	node_dofs = problem.metadata.node_dofs

	faces, cells, _ = findnz(boundary_matrix)
    surface_node_inds = Int[]
    for i in 1:length(cells)
	    cellind = cells[i]
	    faceind = faces[i]
	    face = [Ferrite.faces(interpolation)[faceind]...]
	    Ferrite.reinit!(celliterator, cellind)
        nodeinds = celliterator.nodes[face]
        append!(surface_node_inds, nodeinds)
    end
    unique!(surface_node_inds)
    return setdiff(node_dofs[:, surface_node_inds], problem.ch.prescribed_dofs)
end

function generate_random_loads(
    problem::StiffnessTopOptProblem,
    N::Int,
    scalar::Distributions.Distribution = Distributions.Uniform(-2, 2),
    direction::Function = random_direction,
)
    loadrule = () -> direction() .* rand(scalar)
    surface_dofs = get_surface_dofs(problem)

    FI = Int[]
	FJ = Int[]
	FV = Float64[]
    nodeinds = rand(1:size(surface_dofs, 2), N)
	for i in 1:N
        load = loadrule()
        dofs = surface_dofs[:, nodeinds[i]]
	    append!(FI, dofs)
	    push!(FJ, i, i)
	    append!(FV, load)
	end
	return sparse(FI, FJ, FV)
end
