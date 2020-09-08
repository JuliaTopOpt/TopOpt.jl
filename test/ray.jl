# Run this block only once. #= and =# are used to make multi-line comments in Julia.
#=
using Pkg
pkg"add https://github.com/KristofferC/JuAFEM.jl.git"
pkg"add https://github.com/mohamed82008/VTKDataTypes.jl#master"
pkg"add https://github.com/mohamed82008/KissThreading.jl#master"
pkg"add https://github.com/mohamed82008/TopOpt.jl#mt/ray"
pkg"add PyPlot"
=#

using TopOpt, PyPlot

# Gets the objective and constraint functions
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

# Shows the topology. Setting the `file` keyword argument to some file name, e.g. "file.png".
function show_topology(x, obj; file = nothing)
    problem = obj.f.problem
    cheqfilter = obj.f.cheqfilter
    # Close the plot
    close()
    # Create an image
    x = cheqfilter isa DensityFilter ? cheqfilter(x) : x
    image = TopOptProblems.RectilinearTopology(problem, x)
    # Show the image
    PyPlot.imshow(1 .- image, cmap="gray", origin="lower")
    # Save the image
    if file != nothing
        savefig(file)
    end
end

# Example

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

obj(x0), constr(x0)

# Shows the design
show_topology(x0, obj)

# Shows the design and saves it to a file
show_topology(x0, obj, file = "file.png")

# Closes the plot
# close()
