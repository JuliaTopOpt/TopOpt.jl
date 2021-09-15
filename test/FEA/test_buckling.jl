using Test
using TopOpt
import Ferrite
import GeometryBasics
import Makie

# Define problem, can also be imported from .inp files
# nels = (30,10)
# sizes = (1.0,1.0)
nels = (20, 10, 10) 
sizes = (1.0,1.0,1.0)
E = 1.0;
ν = 0.3;
force = -1.0;
# problem = PointLoadCantilever(nels, sizes, E, ν, force)
problem = PointLoadCantilever(Val{:Linear}, nels, sizes, E, ν, force);
# @show TopOpt.TopOptProblems.getdim(problem)

# Build element stiffness matrices and force vectors
# einfo = ElementFEAInfo(problem);

# Assemble global stiffness matrix and force vector
# ginfo = assemble(problem, einfo);

# Solve for node displacements
# u = ginfo.K \ ginfo.f
# @show u

solver = FEASolver(Direct, problem)
solver()
u = solver.u

# ncells = Ferrite.getncells(problem)
# result_mesh = GeometryBasics.Mesh(problem, ones(ncells));
# GLMakie.mesh(result_mesh);

if get(ENV, "CI", nothing) != "true"
    @eval begin
        using TopOpt.TopOptProblems.Visualization: visualize
        # fig = visualize(problem)
        fig = visualize(problem, u; default_exagg_scale=0.07, scale_range=10.0, vector_linewidth=3, vector_arrowsize=0.5)
        # Makie.display(fig)
    end
end

# using GeometryBasics, Makie
# using Ferrite
# full_topology = ones(Float64, Ferrite.getncells(problem))
# result_mesh = GeometryBasics.Mesh(problem, full_topology);
# mesh(result_mesh);

# # Get the stiffness matrices for buckling analysis
# K, Kσ = TopOpt.TopOptProblems.buckling(problem, ginfo, einfo)

# using IterativeSolvers

# # Find the maximum eigenvalue of system Kσ x = 1/λ K x
# r = lobpcg(Kσ.data, K.data, true, 1)

# # Minimum eigenvalue of the system K x = λ Kσ x
# λ = 1/r.λ[1]

# https://mohamed82008.github.io/ScienceLounge/text/2018/07/11/The-buckle-dance/