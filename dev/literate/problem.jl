# # Problem domain definition

# In this section, the syntax to construct multiple problem types will be shown. In `TopOpt.jl`, there are a number of standard topology optimization problem domains that can be defined using a few lines of code. This is for the convenience of testing and comparing algorithms and formulations on standard testing problems.

# ## Continuum problems
# ### 2D and 3D point load cantilever beam

# ![cantilever](https://user-images.githubusercontent.com/19524993/165186251-f26e9dbc-a224-4fa0-b0c6-d0210a00d426.jpg)

# In this problem, the domain is divided into equally sized rectangular, quadrilateral elements. The number of elements (`nels`) and element sizes (`elsizes`) can be used to control the resolution of the problem's domain as well as its dimension. For instance, using `nels = (160, 40)` and `elsizes = (1.0, 1.0)` constructs a 2D problem domain of size 160 mm x 40 mm problem domain where each element is 1 mm x 1 mm. While `nels = (160, 40, 40)` and `elsizes = (1.0, 1.0, 2.0)` constructs a 3D problem domain of size 160 mm x 40 mm x 40 mm where each element is 1 mm x 1 mm x 2 mm.

# Additionally, the Young’s modulus, Poisson’s ratio and downward force magnitude. Finally, the order of the geometric and field shape functions can be specified using either `:Linear` or `:Quadratic` as shown below.

using TopOpt

E = 1.0 # Young’s modulus in MPa
ν = 0.3 # Poisson’s ratio
f = 1.0 # downward force in N - negative is upward
nels = (160, 40) # number of elements
elsizes = (1.0, 1.0) # the size of each element in mm
order = :Linear # shape function order
problem = PointLoadCantilever(Val{order}, nels, elsizes, E, ν, f)

# ### 2D and 3D half Messerschmitt–Bolkow–Blohm (MBB) beam problem

# ![halfmbb](https://user-images.githubusercontent.com/19524993/165186211-3bfe26d8-82c9-4ae4-a37d-baa03d19b47c.jpg)

# A similar problem type exists for the well known half MBB problem shown above. The constructor and parameters are similar to that of the point load cantilever beam. Also 2D and 3D variants exist by changing the lengths of `nels` and `elsizes`.

E = 1.0 # Young’s modulus in MPa
ν = 0.3 # Poisson’s ratio
f = 1.0 # downward force in N - negative is upward
nels = (60, 20) # number of elements
elsizes = (1.0, 1.0) # the size of each element in mm
order = :Quadratic # shape function order
problem = HalfMBB(Val{order}, nels, elsizes, E, ν, f)

# ### 2D L-beam problem

# ![lbeam](https://user-images.githubusercontent.com/19524993/165194043-e2f1b4f2-940a-478d-ac94-4399e1524a81.jpg)

# The L-beam is another well known testing problem in topology optimization available in `TopOpt.jl`. To construct an L-beam problem, you can use the following constructor. The L-beam problem is only a 2D problem.

E = 1.0 # Young’s modulus in MPa
ν = 0.3 # Poisson’s ratio
f = 1.0 # downward force in N - negative is upward
order = :Quadratic # shape function order
problem = LBeam(
    Val{order}; length=100, height=100, upperslab=50, lowerslab=50, E=1.0, ν=0.3, force=1.0
)

# where `E`, `ν` and `force` are the Young's modulus, Poisson's ratio and downward force respectively. The definition of `length`, `height`, `upperslab` and `lowerslab` are shown below. Each element is assumed to be a 1 mm x 1 mm element. The load is always applied at the midpoint of the "lowerslab" side. A positive value for the force is downward and a negative value is upward.

# ```
#         upperslab   
#        ............
#        .          .
#        .          .
#        .          . 
# height .          .                     
#        .          ......................
#        .                               .
#        .                               . lowerslab
#        .                               .
#        .................................
#                     length

# ```

# ### 2D tie-beam problem

# ![tiebeam](https://user-images.githubusercontent.com/19524993/165222174-927bfb06-ee6a-4eb0-b1df-4a32aa1474d5.png)

# The tie-beam problem shown above is a well-known problem in topology optimization literature. A distributed load of 1 N/mm is applied on the elements specified in the figure. To construct an instance of the tie-beam problem for a certain order of shape functions, you can use:

order = :Quadratic # shape function order
problem = TieBeam(Val{order})

# The tie-beam problem only exists as a 2D problem.

# ### Reading INP files

# Instead of defining a problem type programmatically, one can also use CAD/CAE software to define a 2D/3D problem domain using a graphical user interface and then export a .inp file from the CAD/CAE software. The .inp file can then be read into TopOpt.jl using:

filename = "../data/problem.inp" # path to inp file
problem = InpStiffness(filename);

# For example, the following problem with fixed load, distributed loads and tetrahedral elements was defined usign FreeCAD and imported into TopOpt.jl to perform topology optimization. More information on how to specify the supports and loads in FreeCAD can be found in the webpage of FreeCAD's [FEM Workbench](https://wiki.freecad.org/FEM_Workbench).

# ![inpfile](https://user-images.githubusercontent.com/19524993/165223774-2705347e-369f-463e-80f5-4e30093251c1.PNG)

# ## Truss problems
# ### 2D and 3D truss problem from json file

# Data for constructing a 2D/3D truss problems can be imported from a json file:

path_to_file = "../data/tim_2d.json" # path to json file
mats = TrussFEAMaterial(10.0, 0.3) # Young’s modulus and Poisson’s ratio
crossecs = TrussFEACrossSec(800.0) # Cross-sectional area
node_points, elements, _, _, fixities, load_cases = load_truss_json(path_to_file)
loads = load_cases["0"]
problem = TrussProblem(Val{:Linear}, node_points, elements, loads, fixities, mats, crossecs);

# The structure of the JSON file can be parsed to a dictionary using the code below.
using JSON
f = JSON.parsefile(path_to_file)

# The JSON file contains important information for nodal geometry, element connectivity, supports, materials, and cross sections,
# where every key is a string, and the corresponding value can be a single number, dictionary of values, or a list of dictionary.
#
# ⚠️ Be careful that unlike the normal one-indexed convention used in Julia, **all the index information in the JSON file starts from ZERO**.
# This is just a choice out of convenience, because some of the developers use python in their CAD software to generate these JSONs.

# The MUST-HAVE key-value pairs of the JSON are:
# ```
#   "dimension"     : model dimension (2 or 3)
#   "node_num"      : total number of nodes (::Int)
#   "element_num    : total number of elements (::Int)
#   "nodes" : a list of dictionaries, where each dict is structured as followed
#           {
#               "node_ind" : index of a node, starts from 0.
#               "points" : a list of x,y,z coordinate of the node (Float)
#                   [x , y , z] (or [x, y] if dimension=2)
#           }
#   "elements" : a list of dictionaries, where each dict is structured as followed
#           {
#               "end_node_inds" : [
#                     start node index, end node index in Int (zero-indexed!)
#                  ],
#               "elem_tag" : tag of your element in string to link 
#                   to its corresponding material and cross section, e.g. `steel_set1`
#           }
#   "supports" : a list of dictionaries, where each dict is structured as followed
#          {
#           "node_ind": 0, 
#           "condition": a boolean list marking the fixed dofs, [tx,ty] for 2D, 
#              [tx,ty,tz] for 3D
#           }
#   "loadcases" : a dictionary of load cases that maps a load case index to a dictionary
#           { "0":
#               {
#                 "lc_ind" : 0,  #(same as the load case index 0)
#                 "ploads" : [#a list of dictionaries for each load
#                         {
#                             "node_ind" : node index where the point load is applied
#                             "force" : a list of force magitude along x y z axis
#                                     [x,y,z] (or [x,y] for 2D)
#                             "loadcase" : 0  (same as lc_ind)
#                         }
#               }
#           }      
#   "materials" : a list of dictionaries, where each dict is structured as followed
#         {
#          "name": material name in string,
#          "elem_tags": a list of element tags to which the material is applied, e.g. `steel_set1`, 
#          "E": Young's modulus
#          "G12": in-plane shear modulus, not used for truss models now
#          "density": material density, not used now 
#         }
#   "cross_secs" : a list of dictionaries, where each dict is structured as followed
#         {
#          "name": cross section name in string, 
#          "elem_tags": [],
#          "A": 1.0, # cross section area
#         }
# ```
# When parsed by the `load_truss_json` function, materials and cross sections with `element_tags = []`
# will be treated as the fallback material or cross section for elements with an empty `elem_tag` (`""`)

# The following entries are optional for providing additional information about the model:
# ```
#   "unit"          : "meter" 
#   "model_type"    : "truss"
#   "generate_time" :  generated date-time in string
#   "TO_model_type" : "ground_mesh"
#   "model_name"    : "NameOfYourModel"
#   "_info"         : "AdditionalInfo"
# ```

# ### 2D and 3D truss point load cantilever beam

# ![truss_cantilever](https://user-images.githubusercontent.com/19524993/165228327-def94c04-6505-4f13-afea-4f28a370380e.png)

# Much like the continuum 2D/3D point load cantilever beam, you can also create a 2D/3D truss-based cantilever beam with a point load as shown above using the following syntax.

E = 1.0 # Young’s modulus in MPa
ν = 0.3 # Poisson’s ratio
nels = (60, 20) # number of boundary trusses
elsizes = (1.0, 1.0) # the length of each boundary truss in mm
force = 1.0 # upward force in N - negative is downward
problem = PointLoadCantileverTruss(nels, elsizes, E, ν, force; k_connect=1);

# `nels`, `elsizes`, `E` and `ν` have an analagous intepretation to the continuum cantilever beam. `force` is the upward concentrated force in Newton (downward is negative). `k_connect` is the k-ring of each node defining the connectivity of the nodes in the graph, default is 1. For a 2D domain, a node will be connected to `8` neighboring nodes if `k_connect = 1`, and `8 + 16 = 24` neighboring nodes if `k_connect = 2`.
