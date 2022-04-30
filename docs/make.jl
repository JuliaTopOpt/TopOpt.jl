using Documenter, TopOpt
using DocumenterCitations

# Load packages to avoid precompilation output in the docs
# import ...

# Generate examples
include("generate.jl")

GENERATED_EXAMPLES = [
    joinpath("examples", f) for
    f in ("simp.md", "beso.md", "geso.md", "csimp.md", "global_stress.md")
]

bib = CitationBibliography(joinpath(@__DIR__, "biblio", "ref.bib"))
makedocs(
    bib;
    sitename="TopOpt.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    # doctest = false,
    pages=[
        "Home" => "index.md",
        "Problem types" => "examples/problem.md",
        "Examples" => GENERATED_EXAMPLES,
        "API Reference" => ["reference/TopOptProblems.md", "reference/Algorithms.md"],
        "Bibliography" => "bibliography.md",
    ],
)

# # make sure there are no *.vtu files left around from the build
# cd(joinpath(@__DIR__, "build", "examples")) do
#     foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
# end

if get(ENV, "CI", nothing) == "true"
    deploydocs(; repo="github.com/JuliaTopOpt/TopOpt.jl.git", push_preview=true)
end
