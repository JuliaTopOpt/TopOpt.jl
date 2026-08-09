# generate examples
using Literate: Literate

const EXAMPLE_DIR = joinpath(@__DIR__, "src", "literate")
const GENERATED_DIR = joinpath(@__DIR__, "src", "examples")

# Manual shard assignment so the division is explicit and balanced.
# Each entry is the list of .jl files that shard generates (in order).
const SHARDS = (
    ["beso.jl", "geso.jl"],
    ["problem_continuum.jl"],
    ["problem_truss.jl"],
    ["csimp.jl", "simp.jl"],
    ["TOBS.jl", "heat_tree.jl"],
    ["global_stress.jl"],
    ["local_stress.jl"],
    ["heat_sink.jl", "multimaterial.jl"],
    ["mixed_integer_truss.jl", "neural.jl"],
    ["neural2.jl", "buckling.jl"],
)

"""
    generate_example(example_file, example_dir, output_dir)

Generate Literate markdown + notebook + script for a single example file.
Used by the test harness to produce docs output alongside running the example
as a test, avoiding a separate docs-generation CI stage.
"""
function generate_example(example_file::AbstractString, example_dir::AbstractString, output_dir::AbstractString)
    mkpath(output_dir)
    input = abspath(joinpath(example_dir, example_file))
    isfile(input) || error("Example file not found: $input")
    script = Literate.script(input, output_dir)
    code = strip(read(script, String))
    mdpost(str) = replace(str, "@__CODE__" => code)
    Literate.markdown(input, output_dir; postprocess=mdpost)
    Literate.notebook(input, output_dir; execute=false)
    # Clean up .vtu files
    cd(output_dir) do
        foreach(file -> endswith(file, ".vtu") && rm(file; force=true), readdir())
    end
end

"""
    copy_static_images(example_dir, output_dir)

Copy .png/.jpg/.gif files from the literate directory to the output directory.
"""
function copy_static_images(example_dir::AbstractString, output_dir::AbstractString)
    mkpath(output_dir)
    for example in readdir(example_dir)
        if any(endswith.(example, [".png", ".jpg", ".gif"]))
            cp(joinpath(example_dir, example), joinpath(output_dir, example); force=true)
        end
    end
end

# Run full shard generation only when executed directly (not when included
# by the test harness for its helper functions).
if get(ENV, "DOCS_SKIP_GENERATE", "false") != "true" && abspath(PROGRAM_FILE) == @__FILE__
    mkpath(GENERATED_DIR)

    const INDEX = parse(Int, get(ENV, "DOCS_EXAMPLE_INDEX", "1"))
    const TOTAL = parse(Int, get(ENV, "DOCS_EXAMPLE_TOTAL", "1"))
    @assert INDEX >= 1 && TOTAL >= 1
    @assert INDEX <= TOTAL "DOCS_EXAMPLE_INDEX ($INDEX) must be <= DOCS_EXAMPLE_TOTAL ($TOTAL)"

    if TOTAL == 1
        my_files = reduce(vcat, collect(SHARDS))
    else
        @assert TOTAL == length(SHARDS) "DOCS_EXAMPLE_TOTAL ($TOTAL) must equal 1 or the number of manual shards ($(length(SHARDS)))"
        my_files = SHARDS[INDEX]
    end
    @info "Generating examples" INDEX TOTAL total_files = length(my_files) my_files = my_files

    for example in my_files
        generate_example(example, EXAMPLE_DIR, GENERATED_DIR)
    end

    # Copy static images. Only the first shard does this so artifact uploads
    # from different shards do not collide on image files.
    if INDEX == 1
        for example in readdir(EXAMPLE_DIR)
            if any(endswith.(example, [".png", ".jpg", ".gif"]))
                cp(joinpath(EXAMPLE_DIR, example), joinpath(GENERATED_DIR, example); force=true)
            elseif !endswith(example, ".jl")
                @warn "ignoring $example"
            end
        end
    end
end
