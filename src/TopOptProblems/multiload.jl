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
function MultiLoad(problem::StiffnessTopOptProblem, load_rules, N::Int)
    I = Int[]
    J = Int[]
    V = Float64[]
    for (pos, f) in load_rules
        dofs = find_nearest_dofs(problem, pos)
        for i in 1:N
            load = f()
            append!(I, dofs)
            push!(J, i, i)
            append!(V, load)
        end
    end
    F = sparse(I, J, V, ndofs(problem.ch.dh), N)
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
