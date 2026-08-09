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

filename = joinpath(@__DIR__, "..", "data", "problem.inp") # path to inp file
problem = InpStiffness(filename);

# For example, the following problem with fixed load, distributed loads and tetrahedral elements was defined usign FreeCAD and imported into TopOpt.jl to perform topology optimization. More information on how to specify the supports and loads in FreeCAD can be found in the webpage of FreeCAD's [FEM Workbench](https://wiki.freecad.org/FEM_Workbench).

# ![inpfile](https://user-images.githubusercontent.com/19524993/165223774-2705347e-369f-463e-80f5-4e30093251c1.PNG)
