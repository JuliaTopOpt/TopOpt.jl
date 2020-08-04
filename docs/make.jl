using Documenter, TopOpt

makedocs(
    sitename = "TopOpt.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    pages = [
        "index.md",
        "TopOptProblems" => "TopOptProblems.md",
    ],
)
