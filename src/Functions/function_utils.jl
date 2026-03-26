function get_ρ(
    x_e::T, penalty::AbstractPenalty{T}, xmin::T
) where {T<:Real}
    if PENALTY_BEFORE_INTERPOLATION
        return density(penalty(x_e), xmin)
    else
        return penalty(density(x_e, xmin))
    end
end

function get_ρ_dρ(
    x_e::T, penalty::AbstractPenalty{T}, xmin::T
) where {T<:Real}
    d = ForwardDiff.Dual{T}(x_e, one(T))
    if PENALTY_BEFORE_INTERPOLATION
        p = density(penalty(d), xmin)
    else
        p = penalty(density(d, xmin))
    end
    g = p.partials[1]
    return p.value, g
end
