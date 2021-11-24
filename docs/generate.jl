# generate examples
import Literate

EXAMPLE_DIR = joinpath(@__DIR__, "src", "literate")
GENERATED_DIR = joinpath(@__DIR__, "src", "examples")
mkpath(GENERATED_DIR)
for example in readdir(EXAMPLE_DIR)
    if endswith(example, ".jl")
        input = abspath(joinpath(EXAMPLE_DIR, example))
        script = Literate.script(input, GENERATED_DIR)
        code = strip(read(script, String))
        mdpost(str) = replace(str, "@__CODE__" => code)
        Literate.markdown(input, GENERATED_DIR, postprocess = mdpost)
        Literate.notebook(input, GENERATED_DIR, execute = true)
    elseif any(endswith.(example, [".png", ".jpg", ".gif"]))
        cp(joinpath(EXAMPLE_DIR, example), joinpath(GENERATED_DIR, example); force = true)
    else
        @warn "ignoring $example"
    end
end

# remove any .vtu files in the generated dir (should not be deployed)
cd(GENERATED_DIR) do
    foreach(file -> endswith(file, ".vtu") && rm(file), readdir())
end
