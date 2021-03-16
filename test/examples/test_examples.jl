# Test the example scripts

module TestPointLoadCantileverExample
    println("PointLoadCantilever Example")
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../../docs/src/literate/point_load_cantilever.jl"))
        end
    end
end

# module TestBESOExample
#     println("BESO Example")
#     mktempdir() do dir
#         cd(dir) do
#             include(joinpath(@__DIR__, "../../docs/src/literate/beso.jl"))
#         end
#     end
# end

# module TestGESOExample
#     println("GESO Example")
#     mktempdir() do dir
#         cd(dir) do
#             include(joinpath(@__DIR__, "../../docs/src/literate/geso.jl"))
#         end
#     end
# end