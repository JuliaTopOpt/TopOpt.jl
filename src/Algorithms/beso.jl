mutable struct BESOResult{T,Tt<:AbstractVector{T}}
    topology::Tt
    objval::T
    change::T
    converged::Bool
    fevals::Int
end

"""
The BESO algorithm, see [HuangXie2010](@cite).
"""
struct BESO{
    T,
    Tc<:Compliance,
    Tv1<:Volume,
    Tv2,
    Tf,
    Ts<:AbstractVector{T},
    To1<:AbstractVector{T},
    To2<:MVector{<:Any,T},
    Tr<:BESOResult{T},
} <: TopOptAlgorithm
    comp::Tc
    vol::Tv1
    vol_limit::Tv2
    filter::Tf
    vars::Vector{T}
    topology::Vector{T}
    er::T
    maxiter::Int
    penalty::AbstractPenalty{T}
    sens::Ts
    old_sens::To1
    obj_trace::To2
    tol::T
    sens_tol::T
    result::Tr
end
Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::BESO) = println(io, "TopOpt BESO algorithm")

function BESO(
    comp::Compliance,
    vol::Volume,
    vol_limit,
    filter;
    maxiter=200,
    tol=0.0001,
    p=3.0,
    er=0.02,
    sens_tol=tol / 100,
    k=10,
)
    penalty = comp.solver.penalty
    setpenalty!(penalty, p)
    solver = comp.solver
    T = eltype(solver.vars)
    nel = Ferrite.getncells(solver.problem.ch.dh.grid)
    topology = zeros(T, nel)
    result = BESOResult(topology, T(NaN), T(NaN), false, 0)
    # BESO now works with full design variables
    nvars = nel
    vars = zeros(T, nvars)
    sens = zeros(T, nvars)
    old_sens = zeros(T, nvars)
    obj_trace = zeros(MVector{k,T})

    return BESO(
        comp,
        vol,
        vol_limit,
        filter,
        vars,
        topology,
        er,
        maxiter,
        penalty,
        sens,
        old_sens,
        obj_trace,
        tol,
        sens_tol,
        result,
    )
end

function Utilities.setpenalty!(b::BESO, p::Number)
    b.penalty.p = p
    return b
end

function (b::BESO)(x0=copy(b.comp.solver.vars); black=BitVector(), white=BitVector())
    T = eltype(x0)
    @unpack sens, old_sens, er, tol, maxiter = b
    @unpack obj_trace, topology, sens_tol, vars = b
    @unpack solver = b.comp
    @unpack total_volume, cellvolumes = b.vol
    V = b.vol_limit
    k = length(obj_trace)

    nel = length(x0)

    # Validate black and white vectors
    if !isempty(black)
        @assert length(black) == nel "black must have length $nel (got $(length(black)))"
    end
    if !isempty(white)
        @assert length(white) == nel "white must have length $nel (got $(length(white)))"
    end
    if !isempty(black) && !isempty(white)
        @assert !any(black .& white) "elements cannot be both black and white"
    end

    # Initialize the topology (work with full design variables)
    for i in 1:length(topology)
        topology[i] = round(x0[i])
        vars[i] = topology[i]
    end

    # Initialize black elements to 1 (solid) and white elements to 0 (void)
    @inbounds for i in 1:length(topology)
        if !isempty(black) && black[i]
            topology[i] = T(1)
            vars[i] = T(1)
        elseif !isempty(white) && white[i]
            topology[i] = T(0)
            vars[i] = T(0)
        end
    end

    # Calculate the current volume fraction
    true_vol = vol = dot(topology, cellvolumes) / total_volume
    # Main loop
    change = T(1)
    iter = 0
    @assert typeof(solver.penalty) === typeof(b.penalty)
    setpenalty!(solver, b.penalty.p)
    f = x -> b.comp(b.filter(PseudoDensities(x)))
    while (change > tol || true_vol > V) && iter < maxiter
        iter += 1
        if iter > 1
            old_sens .= sens
        end
        vol = max(vol * (1 - er), V)
        for j in max(2, k - iter + 2):k
            obj_trace[j - 1] = obj_trace[j]
        end
        obj_trace[k], pb = Zygote.pullback(f, vars)
        sens = pb(1.0)[1]
        rmul!(sens, -1)
        if iter > 1
            @. sens = (sens + old_sens) / 2
        end
        l1, l2 = minimum(sens), maximum(sens)
        while (l2 - l1) / l2 > sens_tol
            th = (l1 + l2) / 2
            for i in 1:length(topology)
                # Skip updating black and white elements
                if !isempty(black) && black[i]
                    continue
                end
                if !isempty(white) && white[i]
                    continue
                end
                topology[i] = T(sign(sens[i] - th) > 0)
                vars[i] = topology[i]
            end
            if dot(topology, cellvolumes) - vol * total_volume > 0
                l1 = th
            else
                l2 = th
            end
        end
        true_vol = dot(topology, cellvolumes) / total_volume
        if iter >= k
            l = sum(@view obj_trace[1:(k ÷ 2)])
            h = sum(@view obj_trace[(k ÷ 2 + 1):k])
            change = abs(l - h) / h
        end
    end

    objval = obj_trace[k]
    converged = change <= tol
    fevals = iter
    @pack! b.result = change, objval, converged, fevals

    return b.result
end
