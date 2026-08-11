using Documenter, TopOpt
using DocumenterCitations

bib = CitationBibliography(joinpath(@__DIR__, "biblio", "ref.bib"))

makedocs(;
    sitename="TopOpt.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    doctest=true,
    checkdocs=:all,
    warnonly=[:cross_references],
    plugins=[bib],
    pages=[
        "Home" => "index.md",
        "Problem types" => [
            "Continuum" => "reference/TopOptProblems.md",
            "Truss" => "reference/TrussTopOptProblems.md",
        ],
        "Functions" => "functions.md",
        "API Reference" => [
            "FEA" => "reference/FEA.md",
            "Filters" => "reference/CheqFilters.md",
            "Functions" => "reference/Functions.md",
            "Algorithms" => "reference/Algorithms.md",
            "Utilities" => "reference/Utilities.md",
        ],
        "Bibliography" => "bibliography.md",
    ],
)
