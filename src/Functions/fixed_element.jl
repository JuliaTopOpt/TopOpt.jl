"""
    FixedElementProjector

Callable struct that maps free design variables to a full density vector.
Implements the projection with ChainRulesCore.rrule for automatic differentiation.

# Fields
- `nel::Int`: Total number of elements
- `black::BitVector`: Fixed solid elements (density = 1)
- `white::BitVector`: Fixed void elements (density = 0)
- `free::BitVector`: Free design elements
"""
struct FixedElementProjector
    nel::Int
    black::BitVector
    white::BitVector
    free::BitVector
    
    function FixedElementProjector(nel::Int, black::BitVector, white::BitVector)
        length(black) == nel || throw(ArgumentError("black must have length $nel"))
        length(white) == nel || throw(ArgumentError("white must have length $nel"))
        any(black .& white) && throw(ArgumentError("an element cannot be both black and white"))
        free = .!black .& .!white
        new(nel, black, white, free)
    end
end

"""
    (p::FixedElementProjector)(x_free::AbstractVector{T}) -> Vector{T}

Project free design variables to full density vector by direct copying.

- Fixed black elements → density = 1.0
- Fixed white elements → density = 0.0  
- Free elements → copy x_free values directly

# Arguments
- `x_free::AbstractVector{T}`: Free design variables (typically in [0, 1])

# Returns
- `ρ::Vector{T}`: Full density vector of length p.nel
"""
function (p::FixedElementProjector)(x_free::AbstractVector{T}) where T
    nfree = count(p.free)
    length(x_free) == nfree || throw(ArgumentError(
        "x_free length ($(length(x_free))) must match number of free elements ($nfree)"))
    
    ρ = Vector{T}(undef, p.nel)
    ρ[p.black] .= T(1.0)
    ρ[p.white] .= T(0.0)
    ρ[p.free] .= x_free
    
    return ρ
end

"""
    ChainRulesCore.rrule(p::FixedElementProjector, x_free)

Reverse-mode AD rule for `FixedElementProjector`.

The pullback propagates gradients only through free elements, with zero
gradient through fixed (black/white) elements.
"""
function ChainRulesCore.rrule(p::FixedElementProjector, x_free::AbstractVector{T}) where T
    y = p(x_free)
    
    function projector_pullback(Δy)
        # Only free elements contribute to gradient
        ∂x_free = Δy[p.free]
        return (ChainRulesCore.NoTangent(), ∂x_free)
    end
    
    return y, projector_pullback
end

"""
    get_free_variables(p::FixedElementProjector) -> BitVector

Return a BitVector indicating which elements are free design variables.
"""
get_free_variables(p::FixedElementProjector) = p.free

"""
    get_free_variable_count(p::FixedElementProjector) -> Int

Return the number of free design variables.
"""
get_free_variable_count(p::FixedElementProjector) = count(p.free)

"""
    get_fixed_element_projector(problem, black_cells, white_cells) -> FixedElementProjector

Create a projector that maps free design variables to a full density vector.

# Arguments
- `problem`: A TopOpt problem (e.g., `PointLoadCantilever`, `HalfMBB`, etc.)
- `black_cells`: Indices of elements fixed to solid (density = 1)
- `white_cells`: Indices of elements fixed to void (density = 0)

# Returns
- `FixedElementProjector`: A callable struct that maps `x_free -> ρ_full`

# Example
```julia
problem = PointLoadCantilever((60, 20), (1.0, 1.0), 1.0, 0.3, 1.0)
nel = prod(problem.nels)

# Fix first row solid and last row void
black = 1:problem.nels[1]  # First row
white = (nel - problem.nels[1] + 1):nel  # Last row

# Create projector
projector = get_fixed_element_projector(problem, black, white)

# Use in optimization
x_free = fill(0.5, count(projector.free))  # Initialize free variables
ρ = projector(x_free)  # Full density vector
```
"""
function get_fixed_element_projector(problem, black_cells::AbstractVector{<:Integer}, 
                                      white_cells::AbstractVector{<:Integer})
    nel = getncells(problem.ch.dh.grid)  # Number of elements
    black = falses(nel)
    white = falses(nel)
    black[black_cells] .= true
    white[white_cells] .= true
    return FixedElementProjector(nel, black, white)
end

"""
    get_fixed_element_projector(nel::Int, black_cells, white_cells) -> FixedElementProjector

Create a projector given the total number of elements.

# Arguments
- `nel::Int`: Total number of elements
- `black_cells`: Indices of elements fixed to solid (density = 1)
- `white_cells`: Indices of elements fixed to void (density = 0)

# Returns
- `FixedElementProjector`: A callable struct that maps `x_free -> ρ_full`
"""
function get_fixed_element_projector(nel::Int, black_cells::AbstractVector{<:Integer}, 
                                      white_cells::AbstractVector{<:Integer})
    black = falses(nel)
    white = falses(nel)
    black[black_cells] .= true
    white[white_cells] .= true
    return FixedElementProjector(nel, black, white)
end

export FixedElementProjector, get_fixed_element_projector, 
       get_free_variables, get_free_variable_count