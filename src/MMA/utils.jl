struct Iteration
    i::Int
end

MatrixOf(::Type{Vector{T}}) where T = Matrix{T}
MatrixOf(::Type{CuVector{T}}) where T = CuMatrix{T}
zerosof(::Type{TM}, n...) where TM = (TM(undef, n...) .= 0)
onesof(::Type{TM}, n...) where TM = (TM(undef, n...) .= 1)
infsof(::Type{TM}, n...) where TM = (TM(undef, n...) .= Inf)
ninfsof(::Type{TM}, n...) where TM = (TM(undef, n...) .= -Inf)
nansof(::Type{TM}, n...) where TM = (TM(undef, n...) .= NaN)

@inline minus_plus(a, b) = a - b, a + b

@inline or(a,b) = a || b

macro matdot(v, A, j)
    r = gensym()
    T = gensym()
    esc(quote
        $T = promote_type(eltype($v), eltype($A))
        $r = zero($T)
        for i in 1:length($v)
            $r += $v[i] * $A[$j, i]
        end
        $r
    end)
end

function check_error(m, x0)
    if length(x0) != dim(m)
        throw(ArgumentError("initial variable must have same length as number of design variables"))
    end

    Threads.@threads for j in 1:length(x0)
        # x is not in box
        if !(min(m, j) <= x0[j] <= max(m,j))
            throw(ArgumentError("initial variable at index $j outside box constraint"))
        end
    end
end
