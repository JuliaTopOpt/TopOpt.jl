function extract_cload!(
    cloads::Dict{TI,Vector{TF}}, file, ::Type{Val{dim}}
) where {TI,TF,dim}
    pattern = r"\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
    return consume_section(file) do line
        m = match(pattern, line)
        if m != nothing
            nodeidx = parse(TI, something(m[1]))
            dof = parse(Int, something(m[2]))
            load = parse(TF, something(m[3]))
            dof <= dim || throw(
                ArgumentError(
                    "*CLOAD dof $dof exceeds the problem dimension $dim (rotational and other dofs are not supported).",
                ),
            )
            if haskey(cloads, nodeidx)
                cloads[nodeidx][dof] += load
            else
                cloads[nodeidx] = zeros(TF, dim)
                cloads[nodeidx][dof] = load
            end
        end
    end
end
