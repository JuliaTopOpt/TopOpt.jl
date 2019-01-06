mutable struct BESOResult{T}
    topology::Vector{T}
    objval::T
    change::T
    converged::Bool
    fevals::Int
end

struct BESO{TO, TC, T, TP} <: TopOptAlgorithm 
    obj::TO
    constr::TC
    vars::Vector{T}
    topology::Vector{T}
    er::T
    maxiter::Int
    penalty::TP
    sens::Vector{T}
    old_sens::Vector{T}
    obj_trace::MVector{10,T}
    tol::T
    sens_tol::T
    result::BESOResult{T}
end

function BESO(obj::Objective{<:ComplianceFunction}, constr::Constraint{<:VolumeFunction}; maxiter = 1000, tol = 0.001, p = 3., er=0.02, sens_tol = tol/100)
    penalty = obj.solver.penalty
    penalty.p = p
    T = typeof(obj.comp)
    topology = zeros(T, getncells(obj.problem.ch.dh.grid))
    result = BESOResult(topology, T(NaN), T(NaN), false, 0)
    black = obj.problem.black
    white = obj.problem.white
    nvars = length(topology) - sum(black) - sum(white)
    vars = zeros(T, nvars)
    sens = zeros(T, nvars)
    old_sens = zeros(T, nvars)
    obj_trace = zeros(MVector{10, T})

    return BESO{typeof(obj), typeof(constr), T, typeof(penalty)}(obj, constr, vars, topology, er, maxiter, penalty, sens, old_sens, obj_trace, tol, sens_tol, result)
end

update_penalty!(b::BESO, p::Number) = (b.penalty.p = p)

function (b::BESO{TO, TC, T})(x0=copy(b.obj.solver.vars)) where {TO<:Objective{<:ComplianceFunction}, TC<:Constraint{<:VolumeFunction}, T}
    @unpack sens, old_sens, er, tol, maxiter = b
    @unpack obj_trace, topology, sens_tol, vars = b    
    @unpack varind, black, white = b.obj.problem
    @unpack volume_fraction, total_volume, cellvolumes = b.constr
    V = volume_fraction

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
    true_vol = vol = dot(topology, cellvolumes)/total_volume
    # Main loop
    change = T(1)
    iter = 0
    while (change > tol || true_vol > V) && iter < maxiter
        iter += 1
        if iter > 1
            old_sens .= sens
        end
        vol = max(vol*(1-er), V)
        for j in max(2, 10-iter+2):10
            obj_trace[j-1] = obj_trace[j]
        end
        obj_trace[10] = b.obj(vars, sens)
        scale!(sens, -1)
        if iter > 1
            @. sens = (sens + old_sens) / 2
        end
        l1, l2 = minimum(sens), maximum(sens)
        while (l2 - l1) / l2 > sens_tol
            th = (l1 + l2) / 2
            for i in 1:length(topology)
                if !black[i] && !white[i]
                    topology[i] = (sign(sens[varind[i]] - th) > 0)
                    vars[varind[i]] = topology[i]
                end
            end
            if dot(topology, cellvolumes) - vol * total_volume > 0
                l1 = th
            else
                l2 = th
            end
        end
        true_vol = dot(topology, cellvolumes)/total_volume
        if iter >= 10
            l = sum(@view obj_trace[1:5])
            h = sum(@view obj_trace[6:10])
            change = abs(l-h)/h
        end
    end

    b.result.objval = obj_trace[10]
    b.result.change = change
    b.result.converged = change <= tol
    b.result.fevals = iter

    return b.result
end
