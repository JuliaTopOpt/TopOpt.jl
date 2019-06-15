@params mutable struct BESOResult{T}
    topology::AbstractVector{T}
    objval::T
    change::T
    converged::Bool
    fevals::Int
end

@params struct BESO{T, TO, TC} <: TopOptAlgorithm 
    obj::TO
    constr::TC
    vars::Vector{T}
    topology::Vector{T}
    er::T
    maxiter::Int
    p::T
    sens::AbstractVector{T}
    old_sens::AbstractVector{T}
    obj_trace::MVector{<:Any, T}
    tol::T
    sens_tol::T
    result::BESOResult{T}
end

function BESO(obj::Objective{<:Any, <:ComplianceFunction}, constr::Constraint{<:Any, <:VolumeFunction}; maxiter = 200, tol = 0.0001, p = 3.0, er = 0.02, sens_tol = tol/100, k = 10)
    T = typeof(obj.comp)
    topology = zeros(T, getncells(obj.problem.ch.dh.grid))
    result = BESOResult(topology, T(NaN), T(NaN), false, 0)
    black = obj.problem.black
    white = obj.problem.white
    nvars = length(topology) - sum(black) - sum(white)
    vars = zeros(T, nvars)
    sens = zeros(T, nvars)
    old_sens = zeros(T, nvars)
    obj_trace = zeros(MVector{k, T})

    return BESO(obj, constr, vars, topology, er, maxiter, p, sens, old_sens, obj_trace, tol, sens_tol, result)
end

update_penalty!(b::BESO, p::Number) = (b.p = p)

function (b::BESO{T, TO, TC})(x0 = copy(b.obj.solver.vars)) where {TO<:Objective{<:Any, <:ComplianceFunction}, TC<:Constraint{<:Any, <:VolumeFunction}, T}
    @unpack sens, old_sens, er, tol, maxiter = b
    @unpack obj_trace, topology, sens_tol, vars = b    
    @unpack varind, black, white = b.obj.f.problem
    @unpack total_volume, cellvolumes = b.constr.f
    V = b.constr.s
    k = length(obj_trace)

    # Initialize the topology
    for i in 1:length(topology)
        if black[i]
            topology[i] = 1
        elseif white[i]
            topology[i] = 0
        else
            topology[i] = round(x0[varind[i]])
            vars[varind[i]] = topology[i]
        end
    end

    # Calculate the current volume fraction
    true_vol = vol = dot(topology, cellvolumes) / total_volume
    # Main loop
    change = T(1)
    iter = 0
    while (change > tol || true_vol > V) && iter < maxiter
        iter += 1
        if iter > 1
            old_sens .= sens
        end
        vol = max(vol*(1-er), V)
        for j in max(2, k-iter+2):k
            obj_trace[j-1] = obj_trace[j]
        end
        setpenalty!(b.obj, b.p)
        obj_trace[k] = b.obj(vars, sens)
        rmul!(sens, -1)
        if iter > 1
            @. sens = (sens + old_sens) / 2
        end
        l1, l2 = minimum(sens), maximum(sens)
        while (l2 - l1) / l2 > sens_tol
            th = (l1 + l2) / 2
            for i in 1:length(topology)
                if !black[i] && !white[i]
                    topology[i] = T(sign(sens[varind[i]] - th) > 0)
                    vars[varind[i]] = topology[i]
                end
            end
            if dot(topology, cellvolumes) - vol * total_volume > 0
                l1 = th
            else
                l2 = th
            end
        end
        true_vol = dot(topology, cellvolumes) / total_volume
        if iter >= k
            l = sum(@view obj_trace[1:k÷2])
            h = sum(@view obj_trace[k÷2+1:k])
            change = abs(l-h)/h
        end
    end

    objval = obj_trace[k]
    converged = change <= tol
    fevals = iter
    @pack! b.result = change, objval, converged, fevals

    return b.result
end
