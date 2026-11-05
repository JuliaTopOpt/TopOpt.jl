function extract_dload!(
    dloads::Dict{String,TF},
    facesets::Dict{String,Vector{Tuple{TI,TI}}},
    file,
    ::Type{Val{dim}},
    offset::TI,
) where {TI,TF,dim}
    # The magnitude must accept integer values (`-3` as well as `-3.0`).
    pattern = r"(\d+)\s*,\s*P(\d+)\s*,\s*(\-?\d+(\.\d*)?(e[-\+]?\d*)?)"
    dload_heading_pattern = r"\*DLOAD"

    faceset_name = "DLOAD_SET_$(length(dloads)+1)"
    facesets[faceset_name] = Tuple{TI,TI}[]

    first = true
    prevload = zero(TF)
    load = zero(TF)

    nextline = consume_section(file) do line
        m = match(dload_heading_pattern, line)
        if m != nothing
            dloads[faceset_name] = load
            first = true
            faceset_name = "DLOAD_SET_$(length(dloads)+1)"
            facesets[faceset_name] = Tuple{TI,TI}[]
        end
        m = match(pattern, line)
        if m != nothing
            cellidx = parse(TI, something(m[1])) - offset
            faceidx = parse(TI, something(m[2]))
            load = parse(TF, something(m[3]))
            if !first && prevload != load
                throw("Loads in the same DLOAD set are not equal.")
            end
            prevload = load
            if first
                first = false
            end
            push!(facesets[faceset_name], (cellidx, faceidx))
        end
    end
    dloads[faceset_name] = load

    return nextline
end
