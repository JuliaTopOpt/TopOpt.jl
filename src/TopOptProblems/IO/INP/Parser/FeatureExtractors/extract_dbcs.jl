function extract_nodedbcs!(node_dbcs::Dict{String,Vector{Tuple{TI,TF}}}, file) where {TI,TF}
    pattern_zero = r"([^,]+)\s*,\s*(\d)"
    pattern_range = r"([^,]+)\s*,\s*(\d)\s*,\s*(\d)"
    # The value is optional and, when present, must be parsed even without a
    # decimal point (e.g. `0`, `-1`) -- previously integer values were either
    # dropped or misparsed as zero-valued range constraints.
    pattern_other = r"([^,]+)\s*,\s*(\d)\s*,\s*(\d)\s*,\s*(\-?\d+(\.\d*)?(e[-\+]?\d*)?)"
    return consume_section(file) do line
        m = match(pattern_other, line)
        if m != nothing
            nodesetname = something(m[1])
            dof1 = parse(TI, something(m[2]))
            dof2 = parse(TI, something(m[3]))
            val = parse(TF, something(m[4]))
            if haskey(node_dbcs, nodesetname)
                for dof in dof1:dof2
                    push!(node_dbcs[nodesetname], (dof, val))
                end
            else
                node_dbcs[nodesetname] = [(dof1, val)]
                for dof in (dof1 + 1):dof2
                    push!(node_dbcs[nodesetname], (dof, val))
                end
            end
            return
        end
        m = match(pattern_range, line)
        if m != nothing
            nodesetname = something(m[1])
            dof1 = parse(TI, something(m[2]))
            dof2 = parse(TI, something(m[3]))
            if haskey(node_dbcs, nodesetname)
                for dof in dof1:dof2
                    push!(node_dbcs[nodesetname], (dof, zero(TF)))
                end
            else
                node_dbcs[nodesetname] = [(dof1, zero(TF))]
                for dof in (dof1 + 1):dof2
                    push!(node_dbcs[nodesetname], (dof, zero(TF)))
                end
            end
            return
        end
        m = match(pattern_zero, line)
        if m != nothing
            nodesetname = something(m[1])
            dof = parse(TI, something(m[2]))
            if haskey(node_dbcs, nodesetname)
                push!(node_dbcs[nodesetname], (dof, zero(TF)))
            else
                node_dbcs[nodesetname] = [(dof, zero(TF))]
            end
            return
        end
    end
end
