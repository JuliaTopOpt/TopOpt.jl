# FileIO.add_format(format"GEO", "TrussGEO", [".geo"])

function load_truss_geo(filepath::AbstractString)
    open(filepath) do io
        load_truss_geo(io)
    end
end

function load_truss_geo(io::IOStream)
    iT = Int
    T = Float64
    ndim = 2

    name = readline(io)
    nnodes, nelements = parse.(iT, split(strip(chomp(readline(io)))))

    node_points = Dict{iT,SVector{ndim,T}}()
    for i = 1:nnodes
        strs = split(strip(chomp(readline(io))))
        node_id = parse(iT, strs[1])
        point = parse.(T, strs[2:end])
        @assert length(point) == ndim && node_id == i
        node_points[node_id] = SVector{ndim,T}(point...)
    end

    elements = Dict{iT,Tuple{iT,iT}}()
    for i = 1:nelements
        strs = split(strip(chomp(readline(io))))
        elem_id = parse(iT, strs[1])
        node_ids = parse.(iT, strs[2:end])
        @assert length(node_ids) == 2 && elem_id == i
        elements[i] = Tuple(node_ids)
    end

    nloadcases = parse(iT, strip(chomp(readline(io))))
    nloaded_nodes = parse(iT, strip(chomp(readline(io))))
    @assert nloadcases == 1
    load_cases = Dict()
    for lc_ind = 1:nloadcases
        ploads = Dict{iT,SVector{ndim,T}}()
        for pl = 1:nloaded_nodes
            strs = split(strip(chomp(readline(io))))
            node_id = parse(iT, strs[1])
            load = parse.(T, strs[2:end])
            @assert length(load) == ndim
            ploads[node_id] = SVector{ndim,T}(load...)
        end
        load_cases[lc_ind] = ploads
    end

    nfixities = parse(iT, strip(chomp(readline(io))))
    fixities = Dict{iT,SVector{ndim,Bool}}()
    for i = 1:nfixities
        strs = split(strip(chomp(readline(io))))
        node_id = parse(iT, strs[1])
        condition = map(!, parse.(Bool, strs[2:end]))
        @assert length(condition) == ndim
        fixities[node_id] = condition
    end

    return node_points, elements, fixities, load_cases
end
