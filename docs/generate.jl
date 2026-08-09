# generate examples
using Literate: Literate

EXAMPLE_DIR = joinpath(@__DIR__, "src", "literate")
GENERATED_DIR = joinpath(@__DIR__, "src", "examples")
mkpath(GENERATED_DIR)

# Optional sharding for parallel CI builds. When DOCS_EXAMPLE_TOTAL > 1, each
# shard (DOCS_EXAMPLE_INDEX, 1-based) generates only its share of the example
# notebooks so the work can be split across concurrent jobs.
const INDEX = parse(Int, get(ENV, "DOCS_EXAMPLE_INDEX", "1"))
const TOTAL = parse(Int, get(ENV, "DOCS_EXAMPLE_TOTAL", "1"))
@assert INDEX >= 1 && TOTAL >= 1
@assert INDEX <= TOTAL "DOCS_EXAMPLE_INDEX ($INDEX) must be <= DOCS_EXAMPLE_TOTAL ($TOTAL)"

jl_files = sort(filter(f -> endswith(f, ".jl"), readdir(EXAMPLE_DIR)))
my_files = [f for (i, f) in enumerate(jl_files) if (i - 1) % TOTAL == INDEX - 1]
@info "Generating examples" INDEX TOTAL total_files = length(jl_files) my_files = my_files

for example in my_files
    input = abspath(joinpath(EXAMPLE_DIR, example))
    script = Literate.script(input, GENERATED_DIR)
    code = strip(read(script, String))
    mdpost(str) = replace(str, "@__CODE__" => code)
    Literate.markdown(input, GENERATED_DIR; postprocess=mdpost)
    Literate.notebook(input, GENERATED_DIR; execute=true)
end

# Copy static images and warn about ignored files. Only the first shard does
# this so artifact uploads from different shards do not collide on image files.
if INDEX == 1
    for example in readdir(EXAMPLE_DIR)
        if any(endswith.(example, [".png", ".jpg", ".gif"]))
            cp(joinpath(EXAMPLE_DIR, example), joinpath(GENERATED_DIR, example); force=true)
        elseif !endswith(example, ".jl")
            @warn "ignoring $example"
        end
    end
end

# remove any .vtu files in the generated dir (should not be deployed)
cd(GENERATED_DIR) do
    foreach(file -> endswith(file, ".vtu") && rm(file; force=true), readdir())
end
