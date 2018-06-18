mutable struct BESOResult{T}
    topology::BitVector
    objval::T
    change::T
    converged::Bool
    iters::Int
end

struct BESO{TO, TC, T, TP} <: TopOptAlgorithm 
    obj::TO
    constr::TC
    topology::BitVector
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

function BESO(obj::ComplianceObj, constr::VolConstr; maxiter = 1000, tol = 0.001, p = 3., er=0.02, sens_tol = tol/100)
    penalty = obj.solver.penalty
    penalty.p = p
    T = typeof(obj.comp)
    topology = trues(getncells(obj.problem.ch.dh.grid))
    result = BESOResult(topology, T(NaN), T(NaN), false, 0)
    sens = zeros(T, length(obj.problem.varind))
    old_sens = zeros(T, length(obj.problem.varind))
    obj_trace = zeros(MVector{10, T})
    return BESO{typeof(obj), typeof(constr), T, typeof(penalty)}(obj, constr, topology, er, maxiter, penalty, sens, old_sens, obj_trace, tol, sens_tol, result)
end

update_penalty!(b::BESO, p::Number) = (b.penalty.p = p)

function (b::BESO{TO, TC, T})(x0=b.obj.solver.vars) where {TO<:ComplianceObj, TC<:VolConstr, T}
    sens = b.sens
    old_sens = b.old_sens
    er = b.er
    tol = b.tol
    maxiter = b.maxiter
    obj_trace = b.obj_trace
    topology = b.topology
    sens_tol = b.sens_tol

    # Get black and white
    black = b.obj.problem.black
    white = b.obj.problem.white

    # Initialize the topology
    topology .= round.(x0)
    topology[black] .= 1
    topology[white] .= 0

    # Get the desired volume fraction and total volume
    V = b.constr.volume_fraction
    total_volume = b.constr.total_volume

    # Calculate the current volume fraction
    vol = dot(topology, b.constr.cell_volumes)/total_volume

    # Main loop
    change = T(1)
    i = 0
    while change > tol && i < maxiter
        i += 1
        vol = max(vol*(1-er), V)
        for j in max(2, 10-i+2):10
            obj_trace[j-1] = obj_trace[j]
        end
        obj_trace[10] = b.obj(topology, sens)
        scale!(sens, -1)
        if i > 1
            @. sens = (sens + old_sens) / 2
        end
        l1, l2 = minimum(sens), maximum(sens)
        while (l2 - l1) / l2 > sens_tol
            th = (l1 + l2) / 2
            @. topology = ((sign(sens - th) > 0) | black) & !white
            if dot(topology, b.constr.cell_volumes) - vol * total_volume > 0
                l1 = th
            else
                l2 = th
            end
        end
        if i >= 10
            l = sum(@view obj_trace[1:5])
            h = sum(@view obj_trace[6:10])
            change = abs(l-h)/h
        end
    end
    
    b.result.topology .= topology
    b.result.objval = obj_trace[10]
    b.result.change = change
    b.result.converged = change <= tol
    b.result.iters = i

    return b.result
end
