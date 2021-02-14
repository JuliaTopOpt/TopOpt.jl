"""
    @params struct_def

A macro that changes all fields' types to type parameters while respecting the type bounds specificied by the user. For example:
```
@params struct MyType{T}
    f1::T
    f2::AbstractVector{T}
    f3::AbstractVector{<:Real}
    f4
end
```
will define:
```
struct MyType{T, T1 <: T, T2 <: AbstractVector{T}, T3 <: AbstractVector{<:Real}, T4}
    f1::T1
    f2::T1
    f3::T3
    f4::T4
end
```
The default non-parameteric constructor, e.g. `MyType(f1, f2, f3, f4)`, will always work if all the type parameters used in the type header are used in the field types. Using a type parameter in the type header that is not used in the field types such as:
```
struct MyType{T}
    f1
    f2
end
```
is not recommended.
"""
macro params(struct_expr)
    header = struct_expr.args[2]
    fields = @view struct_expr.args[3].args[2:2:end]
    params = []
    for i in 1:length(fields)
        x = fields[i]
        T = gensym()
        if x isa Symbol
            push!(params, T)
            fields[i] = :($x::$T)
        elseif x.head == :(::)
            abstr = x.args[2]
            var = x.args[1]
            push!(params, :($T <: $abstr))
            fields[i] = :($var::$T)
        end
    end
    if header isa Symbol && length(params) > 0
        struct_expr.args[2] = :($header{$(params...)})
    elseif header.head == :curly
        append!(struct_expr.args[2].args, params)
    elseif header.head == :<:
        if struct_expr.args[2].args[1] isa Symbol
            name = struct_expr.args[2].args[1]
            struct_expr.args[2].args[1] = :($name{$(params...)})
        elseif header.head == :<: && struct_expr.args[2].args[1] isa Expr
            append!(struct_expr.args[2].args[1].args, params)
        else
            error("Unidentified type definition.")
        end
    else
        error("Unidentified type definition.")
    end
    esc(struct_expr)
end

struct RaggedArray{TO, TV}
    offsets::TO
    values::TV
end

function RaggedArray(vv::Vector{Vector{T}}) where T
    offsets = [1; 1 .+ accumulate(+, collect(length(v) for v in vv))]
    values = Vector{T}(undef, offsets[end]-1)
    for (i, v) in enumerate(vv)
        r = offsets[i]:offsets[i+1]-1
        values[r] .= v
    end
    RaggedArray(offsets, values)
end

function Base.getindex(ra::RaggedArray, i)
    @assert 1 <= i < length(ra.offsets)
    r = ra.offsets[i]:ra.offsets[i+1]-1
    @assert 1 <= r.start && r.stop <= length(ra.values)
    return @view ra.values[r]
end
function Base.getindex(ra::RaggedArray, i, j)
    @assert 1 <= j < length(ra.offsets)
    r = ra.offsets[j]:ra.offsets[j+1]-1
    @assert 1 <= i <= length(r)
    return ra.values[r[i]]
end
function Base.setindex!(ra::RaggedArray, v, i, j)
    @assert 1 <= j < length(ra.offsets)
    r = ra.offsets[j]:ra.offsets[j+1]-1
    @assert 1 <= i <= length(r)
    ra.values[r[i]] = v
end

function find_varind(black, white, ::Type{TI}=Int) where TI
    nel = length(black)
    nel == length(white) || throw("Black and white vectors should be of the same length")
    varind = zeros(TI, nel)
    k = 1
    for i in 1:nel
        if !black[i] && !white[i]
            varind[i] = k
            k += 1
        end
    end
    return varind
end

function find_black_and_white(dh)
    black = falses(getncells(dh.grid))
    white = falses(getncells(dh.grid))
    if haskey(dh.grid.cellsets, "black")
        for c in grid.cellsets["black"]
            black[c] = true
        end
    end
    if haskey(dh.grid.cellsets, "white")
        for c in grid.cellsets["white"]
            white[c] = true
        end
    end
    
    return black, white
end

YoungsModulus(p) = getE(p)
PoissonRatio(p) = getν(p)

function compliance(Ke, u, dofs)
    comp = zero(eltype(u))
    for i in 1:length(dofs)
        for j in 1:length(dofs)
            comp += u[dofs[i]]*Ke[i,j]*u[dofs[j]]
        end
    end
    comp
end

function meandiag(K::AbstractMatrix)
    z = zero(eltype(K))
    for i in 1:size(K, 1)
        z += abs(K[i, i])
    end
    return z / size(K, 1)
end

density(var, xmin) = var*(1-xmin) + xmin

macro debug(expr)
    return quote
        if DEBUG[]
            $(esc(expr))
        end
    end
end

@generated function _getproperty(c::T, ::Val{fallback}, ::Val{f}) where {T, fallback, f}
    f ∈ fieldnames(T) && return :(getfield(c, $(QuoteNode(f))))
    return :(getproperty(getfield(c, $(QuoteNode(fallback))), $(QuoteNode(f))))
end
@generated function _setproperty!(c::T, ::Val{fallback}, ::Val{f}, val) where {T, fallback, f}
    f ∈ fieldnames(T) && return :(setfield!(c, $(QuoteNode(f)), val))
    return :(setproperty!(getfield(c, $(QuoteNode(fallback))), $(QuoteNode(f)), val))
end
macro forward_property(T, field)
    quote
        Base.getproperty(c::$(esc(T)), f::Symbol) = _getproperty(c, Val($(QuoteNode(field))), Val(f))
        Base.setproperty!(c::$(esc(T)), f::Symbol, val) = _setproperty!(c, Val($(QuoteNode(field))), Val(f), val)
    end
end
