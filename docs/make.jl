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

deploydocs(
    repo = "github.com/mohamed82008/TopOpt.jl.git",
)