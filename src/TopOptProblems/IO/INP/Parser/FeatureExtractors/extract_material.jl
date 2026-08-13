function extract_material(file, (::Type{TF})=Float64) where {TF}
    elastic_heading_pattern = r"\*ELASTIC"
    Emu_pattern = r"(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)"
    line = readline(file)
    m = match(elastic_heading_pattern, line)
    m === nothing &&
        throw(ArgumentError("Expected *ELASTIC section in material definition"))
    line = readline(file)
    m = match(Emu_pattern, line)
    m === nothing && throw(
        ArgumentError(
            "Failed to parse Young's modulus and Poisson ratio from material data"
        ),
    )
    E = parse(TF, something(m[1]))
    mu = parse(TF, something(m[2]))
    line = readline(file)
    return E, mu, line
end
