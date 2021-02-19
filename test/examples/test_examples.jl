# Test the scripts

module TestPointLoadCantileverExample
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../../docs/src/literate/point_load_cantilever.jl"))
        end
    end
end

module TestBESOExample
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../../docs/src/literate/half_mbb_beso.jl"))
        end
    end
end

module TestBESOExample
    mktempdir() do dir
        cd(dir) do
            include(joinpath(@__DIR__, "../../docs/src/literate/half_mbb_beso.jl"))
        end
    end
end


