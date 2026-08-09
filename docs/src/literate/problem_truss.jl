# # Truss problem domain definition

# In this section, the syntax to construct truss topology optimization problem domains is shown. This is a continuation of the problem domain definition page; see the continuum problems page for the first part.

# ## Truss problems
# ### 2D and 3D truss problem from json file

using TopOpt

# Data for constructing a 2D/3D truss problems can be imported from a json file:

path_to_file = joinpath(@__DIR__, "..", "data", "tim_2d.json") # path to json file
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