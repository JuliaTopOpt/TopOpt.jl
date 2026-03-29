using TopOpt

E = 1.0 # Young’s modulus in MPa
ν = 0.3 # Poisson’s ratio
f = 1.0 # downward force in N - negative is upward
nels = (160, 40) # number of elements
elsizes = (1.0, 1.0) # the size of each element in mm
order = :Linear # shape function order
problem = PointLoadCantilever(Val{order}, nels, elsizes, E, ν, f)

E = 1.0 # Young’s modulus in MPa
ν = 0.3 # Poisson’s ratio
f = 1.0 # downward force in N - negative is upward
nels = (60, 20) # number of elements
elsizes = (1.0, 1.0) # the size of each element in mm
order = :Quadratic # shape function order
problem = HalfMBB(Val{order}, nels, elsizes, E, ν, f)

E = 1.0 # Young’s modulus in MPa
ν = 0.3 # Poisson’s ratio
f = 1.0 # downward force in N - negative is upward
order = :Quadratic # shape function order
problem = LBeam(
    Val{order}; length=100, height=100, upperslab=50, lowerslab=50, E=1.0, ν=0.3, force=1.0
)

order = :Quadratic # shape function order
problem = TieBeam(Val{order})

filename = "../data/problem.inp" # path to inp file
problem = InpStiffness(filename);

path_to_file = "../data/tim_2d.json" # path to json file
mats = TrussFEAMaterial(10.0, 0.3) # Young’s modulus and Poisson’s ratio
crossecs = TrussFEACrossSec(800.0) # Cross-sectional area
node_points, elements, _, _, fixities, load_cases = load_truss_json(path_to_file)
loads = load_cases["0"]
problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crossecs);

using JSON
f = JSON.parsefile(path_to_file)

E = 1.0 # Young’s modulus in MPa
ν = 0.3 # Poisson’s ratio
nels = (60, 20) # number of boundary trusses
elsizes = (1.0, 1.0) # the length of each boundary truss in mm
force = 1.0 # upward force in N - negative is downward
problem = PointLoadCantileverTruss(nels, elsizes, E, ν, force; k_connect=1);

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl
