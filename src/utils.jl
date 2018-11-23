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

@inline density(var, xmin) = var*(1-xmin) + xmin

macro debug(expr)
    return quote
        if DEBUG[]
            $(esc(expr))
        end
    end
end
