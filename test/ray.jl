# Run this block only once. #= and =# are used to make multi-line comments in Julia.
#=
using Pkg
pkg"add https://github.com/KristofferC/JuAFEM.jl.git"
pkg"add https://github.com/mohamed82008/VTKDataTypes.jl#master"
pkg"add https://github.com/mohamed82008/KissThreading.jl#master"
pkg"add https://github.com/mohamed82008/TopOpt.jl#mt/ray"
=#

using TopOpt

function get_obj_constr(;
    # Number of elements, see example below
    nels,
    # Pin locations, see example below
    pins,
    # Load dictionary, see example below
    loads,
    # Volume fraction
    V = 0.5,
    xmin = 0.0,
    # Penalty value, power penalty is used. This is useless if the input is 0s and 1s only and xmin == 0.
    p = 1.0,
    # Set to DensityFilter if you want a density filter
    filter = nothing,
    # Filter radious
    rmin = 0.0,
)
    problem = RayProblem(nels, pins, loads)
    penalty = TopOpt.PowerPenalty(p)
    # Define a finite element solver
    solver = FEASolver(Displacement, Direct, problem, xmin = xmin, penalty = penalty, qr = true)
    # Define compliance objective
    obj = TopOpt.Objective(Compliance(problem, solver, filterT = filter, rmin = rmin, tracing = false, logarithm = false))
    # Define volume constraint
    constr = TopOpt.Constraint(Volume(problem, solver, filterT = filter, rmin = rmin), V)

    return obj, constr
end

number_of_elements = (50, 20)
pin_locations = [
    [1, 20],
]
loads_dictionary = Dict(
    [5, 5] => [1.0, -1.0],
    [7, 18] => [1.0, -1.0],
)

obj, constr = get_obj_constr(nels = number_of_elements, pins = pin_locations, loads = loads_dictionary)
x0 = ones(prod(number_of_elements))
