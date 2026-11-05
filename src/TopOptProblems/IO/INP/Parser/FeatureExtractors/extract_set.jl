"""
    extract_set!(sets, setname, file, offset=0; generated=false)

Extract a node/element set definition from an INP file. With `generated=true`
(the `*Nset/*Elset, ..., generate` construct), the single data row
`<first>, <last>[, <increment>]` is expanded into the full (inclusive) range,
with element indices shifted by `offset` (1-based conversion). Without
`generate`, plain indices and named references to previously defined sets are
collected.
"""
function extract_set!(
    sets::Dict{String,TV},
    setname::AbstractString,
    file,
    offset=0;
    generated::Bool=false,
) where {TI,TV<:AbstractVector{TI}}
    sets[setname] = Int[]
    vector = sets[setname]

    if generated
        # A generated set has exactly one data row: <first>, <last>[, <inc>].
        line = eof(file) ? nothing : readline(file)
        if line !== nothing && match(stopping_pattern, line) === nothing
            nums = [parse(Int, m.match) for m in eachmatch(r"(\d+)", line)]
            if length(nums) >= 2
                first, last = nums[1], nums[2]
                inc = length(nums) >= 3 ? nums[3] : 1
                append!(vector, (first:inc:last) .- offset)
            end
        end
        # Return the following section heading (or nothing at EOF).
        return eof(file) ? nothing : readline(file)
    end

    pattern_single = r"^(\d+)"
    pattern_subset = r"^([^,]+)"
    return consume_section(file) do line
        # Ignore comment lines (`**...`) and separator lines (`****...`).
        occursin(r"^\s*\*", line) && return
        if match(pattern_single, line) !== nothing
            m = eachmatch(r"(\d+)", line)
            for _m in m
                push!(vector, parse(TI, _m.match) - offset)
            end
        else
            m = match(pattern_subset, line)
            if m !== nothing
                subsetname = String(something(m[1]))
                if haskey(sets, subsetname)
                    append!(vector, sets[subsetname])
                else
                    throw(
                        ArgumentError(
                            "INP set \"$subsetname\" references a set that has not " *
                            "been defined yet (forward references are not supported).",
                        ),
                    )
                end
            end
        end
    end
end
