using Documenter, TopOpt
using DocumenterCitations

# Load packages to avoid precompilation output in the docs
# import ...

# Generate examples (unless a sharded CI build pre-generated them)
if get(ENV, "DOCS_SKIP_GENERATE", "false") != "true"
    include("generate.jl")
end

GENERATED_EXAMPLES = [
    joinpath("examples", f) for f in (
        "simp.md",
        "beso.md",
        "geso.md",
        "csimp.md",
        "global_stress.md",
        "local_stress.md",
        "TOBS.md",
        "heat_tree.md",
        "heat_sink.md",
        "multimaterial.md",
        "neural.md",
        "neural2.md",
        "mixed_integer_truss.md",
        "buckling.md",
    )
    if isfile(joinpath(@__DIR__, "src", "examples", f))
]

PROBLEM_EXAMPLES = [
    "Continuum problems" => "examples/problem_continuum.md",
    "Truss problems" => "examples/problem_truss.md",
]
PROBLEM_EXAMPLES = [p for p in PROBLEM_EXAMPLES if isfile(joinpath(@__DIR__, "src", p.second))]

bib = CitationBibliography(joinpath(@__DIR__, "biblio", "ref.bib"))
makedocs(;
    sitename="TopOpt.jl",
    format=Documenter.HTML(; prettyurls=get(ENV, "CI", nothing) == "true"),
    plugins=[bib],
    warnonly=true,
    pages=[
        "Home" => "index.md",
        "Problem types" => PROBLEM_EXAMPLES,
        "Functions" => "functions.md",
        "Examples" => GENERATED_EXAMPLES,
        "API Reference" => [
            "reference/TopOptProblems.md",
            "reference/TrussTopOptProblems.md",
            "reference/FEA.md",
            "reference/CheqFilters.md",
            "reference/Functions.md",
            "reference/Utilities.md",
            "reference/Algorithms.md",
        ],
        "Bibliography" => "bibliography.md",
    ],
)

if get(ENV, "CI", nothing) == "true"
    deploydocs(; repo="github.com/JuliaTopOpt/TopOpt.jl.git", push_preview=true)
end