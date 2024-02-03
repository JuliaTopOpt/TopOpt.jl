function extract_cload!(
    cloads::Dict{TI,Vector{TF}}, file, ::Type{Val{dim}}
) where {TI,TF,dim}
    pattern = r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
    line = readline(file)
    m = match(stopping_pattern, line)
    while m isa Nothing
        m = match(pattern, line)
        if m != nothing
            nodeidx = parse(TI, m[1])
            dof = parse(Int, m[2])
            load = parse(TF, m[3])
            if haskey(cloads, nodeidx)
                cloads[nodeidx][dof] += load
            else
                cloads[nodeidx] = zeros(TF, dim)
                cloads[nodeidx][dof] = load
            end
        end
        line = readline(file)
        m = match(stopping_pattern, line)
    end
    return line
end
