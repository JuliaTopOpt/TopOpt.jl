using Documenter, TopOpt
using DocumenterCitations
using DocumenterCodeBlocks

bib = CitationBibliography(joinpath(@__DIR__, "biblio", "ref.bib"))

makedocs(;
    sitename="TopOpt.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    doctest=true,
    checkdocs=:all,
    plugins=[bib, CodeBlocks()],
    pages=[
        "Home" => "index.md",
        "Tutorials" => "tutorials/index.md",
        "Problem types" => [
            "Continuum" => "reference/TopOptProblems.md",
            "Truss" => "reference/TrussTopOptProblems.md",
        ],
        "Differentiable functions" => "functions.md",
        "API Reference" => [
            "FEA solvers" => "reference/FEA.md",
            "Functions" => "reference/Functions.md",
            "Filters" => "reference/CheqFilters.md",
            "Algorithms" => "reference/Algorithms.md",
            "Utilities" => "reference/Utilities.md",
            "Level-set (OpenLSTO)" => "reference/OpenLSTO.md",
        ],
        "Bibliography" => "bibliography.md",
    ],
)
