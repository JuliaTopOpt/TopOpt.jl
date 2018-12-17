# Update move limits
function update_limits!(primal_data, m, k, x1, x2, s_init, s_incr, s_decr)
    @unpack L, x, U, = primal_data
    for j in 1:dim(m)
        if k == 1 || k == 2
            # Equation 11 in Svanberg
            L[j] = x[j] - s_init * (max(m,j) - min(m, j))
            U[j] = x[j] + s_init * (max(m,j) - min(m, j))
        else
            # Equation 12 in Svanberg
            if sign(x[j] - x1[j]) != sign(x1[j] - x2[j])
                L[j] = x[j] - (x1[j] - L[j]) * s_decr
                U[j] = x[j] + (U[j] - x1[j]) * s_decr
            # Equation 13 in Svanberg
            else
                L[j] = x[j] - (x1[j] - L[j]) * s_incr
                U[j] = x[j] + (U[j] - x1[j]) * s_incr
            end
        end
    end
end

function compute_mma!(primal_data, m)
    @unpack x, L, U, α, β, f_val, g_val, ∇f, ∇g, p, q, p0, q0, r = primal_data
    # Bound limits
    for j = 1:dim(m)
        μ = 0.1
        α[j] = max(L[j] + μ * (x[j] - L[j]), min(m, j))
        β[j] = min(U[j] - μ * (U[j] - x[j]), max(m, j))
    end

    r0 = 0.0
    for i in 0:length(constraints(m))
        if i == 0
            ri = f_val[]
            ∇fi = @view ∇f[:]
        else
             ri = g_val[i]
             ∇fi = @view ∇g[:,i]
        end
        for j in 1:dim(m)
            Ujxj = U[j] - x[j]
            xjLj = x[j] - L[j]
            if ∇fi[j] > 0
                p_ij = abs2(Ujxj) * ∇fi[j]
                q_ij = 0.0
            else
                p_ij = 0.0
                q_ij = -abs2(xjLj) * ∇fi[j]
            end
            ri -= p_ij / Ujxj + q_ij / xjLj
            if i == 0
                p0[j] = p_ij
                q0[j] = q_ij
            else
                p[j, i] = p_ij
                q[j, i] = q_ij
            end
        end
        if i == 0
            r0 = ri
        else
            r[i] = ri
        end
    end
    primal_data.r0[] = r0
end

function compute_dual!(λ, primal_data)
    @unpack p0, q0, p, q, L, U, x, r0, r = primal_data
    update_x!(primal_data, λ)

    #Optimal value of Lagrangian at λ
    φ = r0[] + dot(λ, r)
    @inbounds for j = 1:length(x)
        φ += (p0[j] + matdot(λ, p, j)) / (U[j] - x[j])
        φ += (q0[j] + matdot(λ, q, j)) / (x[j] - L[j])
    end
    return -φ
end

function compute_dual_grad!(∇φ, λ, primal_data)
    @unpack r, p, q, L, U, x = primal_data
    update_x!(primal_data, λ)
    for i = 1:length(λ)
        ∇φ[i] = r[i]
        for j = 1:length(x)
            ∇φ[i] += p[j,i] / (U[j] - x[j])
            ∇φ[i] += q[j,i] / (x[j] - L[j])
        end
    end
    # Negate since we have a maximization problem
    scale!(∇φ, -1.0)
    return
end

# Updates x to be the analytical optimal point in the dual
# problem for a given λ
function update_x!(primal_data, λ)
    @unpack x, p0, q0, p, q, L, U, α, β = primal_data
    @inbounds for j in 1:length(x)
        fpj = sqrt(p0[j] + matdot(λ, p, j))
        fqj = sqrt(q0[j] + matdot(λ, q, j))
        x[j] = (fpj * L[j] + fqj * U[j]) / (fpj + fqj)
        if x[j] > β[j]
            x[j] = β[j]
        elseif x[j] < α[j]
            x[j] = α[j]
        end
    end
    return 
end
