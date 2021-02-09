import JSON

function parse_truss_json(file_path::String)
    data = JSON.parsefile(file_path)
    ndim = data["dimension"]
    n = data["node_num"]
    m = data["element_num"]
    iT = Int
    T = Float64

    node_points = Dict{iT, SVector{ndim, T}}()
    for (i, ndata) in enumerate(data["nodes"])
        node_points[i] = convert(SVector{ndim,T}, ndata["point"])
        if "node_ind" in keys(ndata)
            @assert 1 + ndata["node_ind"] == i
        end
    end
    @assert length(node_points) == n

    elements = Dict{iT, Tuple{iT,iT}}()
    element_inds_from_tag = Dict()
    for (i, edata) in enumerate(data["elements"])
        elements[i] = (edata["end_node_inds"]...,) .+ 1
        if "elem_ind" in keys(edata)
            @assert 1+edata["elem_ind"] == i
        end
        elem_tag = edata["elem_tag"]
        if elem_tag ∉ keys(element_inds_from_tag)
            element_inds_from_tag[elem_tag] = []
        end
        push!(element_inds_from_tag[elem_tag], i)
    end
    @assert length(elements) == m

    E_from_tag = Dict()
    ν_from_tag = Dict()
    for mat in data["materials"]
        # if elem_tag list has length 0, use tag `nothing`
        mat["elem_tags"] = length(mat["elem_tags"]) == 0 ? [nothing] : mat["elem_tags"]
        for e_tag in mat["elem_tags"]
            if e_tag in keys(E_from_tag)
                @warn "Multiple materials assigned to the same element tag |$(e_tag)|!"
            end
            E_from_tag[e_tag] = T(mat["E"])
            if "mu" ∈ keys(mat)
                ν_from_tag[e_tag] = T(mat["mu"])
            else
                ν_from_tag[e_tag] = 0.0
            end
        end
    end
    A_from_tag = Dict()
    for cs in data["cross_secs"]
        # if elem_tag list has length 0, use tag `nothing`
        cs["elem_tags"] = length(cs["elem_tags"]) == 0 ? [nothing] : cs["elem_tags"]
        for e_tag in cs["elem_tags"]
            if e_tag in keys(A_from_tag)
                @warn "Multiple cross secs assigned to the same element tag |$(e_tag)|!"
            end
            A_from_tag[e_tag] = T(cs["A"])
        end
    end
    mats = Array{TrussFEAMaterial}(undef, m)
    crosssecs = Array{TrussFEACrossSec}(undef, m)
    for (tag, e_ids) in element_inds_from_tag
        # @show tag, e_ids
        for ei in e_ids
            if !(tag ∈ keys(A_from_tag))
                # use default material (key `nothing`)
                A = A_from_tag[nothing]
            else
                A = A_from_tag[tag]
            end
            crosssecs[ei] = TrussFEACrossSec(A)

            if !(tag ∈ keys(E_from_tag))
                # use default material (key `nothing`)
                E = E_from_tag[nothing]
            else
                E = E_from_tag[tag]
            end
            if !(tag ∈ keys(ν_from_tag))
                # use default material (key `nothing`)
                ν = ν_from_tag[nothing]
            else
                ν = ν_from_tag[tag]
            end
            mats[ei] = TrussFEAMaterial(E, ν)
        end
    end

    # TODO only translation dof for now
    @assert(length(data["supports"]) > 0)
    fixities = Dict{iT, SVector{ndim, Bool}}()
    for sdata in data["supports"]
        supp_v = iT(sdata["node_ind"])+1
        fixities[supp_v] = sdata["condition"][1:ndim]
    end

    load_cases = Dict()
    for (lc_ind, lc_data) in data["loadcases"]
        nploads = length(lc_data["ploads"])
        @assert nploads > 0
        ploads = Dict{iT, SVector{ndim, T}}()
        for pl in lc_data["ploads"]
            load_v = pl["node_ind"]+1
            ploads[load_v] = convert(SVector{ndim,T}, pl["force"])
        end
        load_cases[lc_ind] = ploads
    end

    return node_points, elements, mats, crosssecs, fixities, load_cases
end