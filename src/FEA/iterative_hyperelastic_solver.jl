abstract type AbstractHyperelasticSolver <: AbstractFEASolver end

mutable struct HyperelasticCompressibleDisplacementSolver{
    T,
    dim,
    TP1<:AbstractPenalty{T},
    TP2<:StiffnessTopOptProblem{dim,T},
    TG<:GlobalFEAInfo_hyperelastic{T},
    TE<:ElementFEAInfo_hyperelastic{dim,T},
    Tu<:AbstractVector{T},
    TF<:AbstractVector{<:AbstractMatrix{T}},
} <: AbstractHyperelasticSolver
    mp
    problem::TP2
    globalinfo::TG
    elementinfo::TE
    u::Tu # JGB: u --> u0
    F::TF
    vars::Tu
    penalty::TP1
    prev_penalty::TP1
    xmin::T
    tsteps::Int
    ntsteps::Int
end
mutable struct HyperelasticNearlyIncompressibleDisplacementSolver{
    T,
    dim,
    TP1<:AbstractPenalty{T},
    TP2<:StiffnessTopOptProblem{dim,T},
    TG<:GlobalFEAInfo_hyperelastic{T},
    TE<:ElementFEAInfo_hyperelastic{dim,T},
    Tu<:AbstractVector{T},
    TF<:AbstractVector{<:AbstractMatrix{T}},
} <: AbstractHyperelasticSolver
    mp
    problem::TP2
    globalinfo::TG
    elementinfo::TE
    u::Tu # JGB: u --> u0
    F::TF
    vars::Tu
    penalty::TP1
    prev_penalty::TP1
    xmin::T
    tsteps::Int
    ntsteps::Int
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, x::HyperelasticCompressibleDisplacementSolver)
    return println("TopOpt compressible hyperelastic solver")
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, x::HyperelasticNearlyIncompressibleDisplacementSolver)
    return println("TopOpt nearly-incompressible hyperelastic solver")
end
function HyperelasticDisplacementSolver(
    mp, # JGB: add type later
    sp::StiffnessTopOptProblem{dim,T}; # JGB: eventually add ::HyperelaticParam type
    xmin=T(1)/1000,
    penalty=PowerPenalty{T}(1),
    prev_penalty=deepcopy(penalty),
    quad_order=default_quad_order(sp),
    tstep = 1,
    ntsteps = 20,
    nearlyincompressible=false
) where {dim,T}
    u = zeros(T, ndofs(sp.ch.dh))
    ts0 = tstep/ntsteps
    update!(sp.ch,ts0) # set initial time-step (adjusts dirichlet bcs)
    apply!(u,sp.ch) # apply dbc for initial guess
    elementinfo = ElementFEAInfo_hyperelastic(mp, sp, u, quad_order, Val{:Static}, nearlyincompressible; ts = ts0) # JGB: add u
    F = [zeros(typeof(elementinfo.Fes[1])) for _ in 1:getncells(sp.ch.dh.grid)]
    globalinfo = GlobalFEAInfo_hyperelastic(sp) # JGB: small issue this leads to symmetric K initialization
    #u = zeros(T, ndofs(sp.ch.dh)) # JGB
    vars = fill(one(T), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    varind = sp.varind
    return nearlyincompressible ?
        HyperelasticNearlyIncompressibleDisplacementSolver(mp, sp, globalinfo, elementinfo, u, F, vars, penalty, prev_penalty, xmin, tstep, ntsteps) :
        HyperelasticCompressibleDisplacementSolver(mp, sp, globalinfo, elementinfo, u, F, vars, penalty, prev_penalty, xmin, tstep, ntsteps)
    #if nearlyincompressible
    #    return HyperelasticNearlyIncompressibleDisplacementSolver(mp, sp, globalinfo, elementinfo, u, F, vars, penalty, prev_penalty, xmin, tstep, ntsteps) 
    #else 
    #    return HyperelasticCompressibleDisplacementSolver(mp, sp, globalinfo, elementinfo, u, F, vars, penalty, prev_penalty, xmin, tstep, ntsteps)
    #end
end
function (s::HyperelasticCompressibleDisplacementSolver{T})(
    ::Type{Val{safe}}=Val{false},
    ::Type{newT}=T;
    assemble_f=true,
    kwargs...,
) where {T,safe,newT}
    elementinfo = s.elementinfo
    globalinfo = s.globalinfo
    dh = s.problem.ch.dh
    ch = s.problem.ch

    _ndofs = ndofs(dh)
    un = zeros(_ndofs) # previous solution vector

    NEWTON_TOL = 1e-8
    NEWTON_MAXITER = 30
    CG_MAXITER = 1000
    TS_MAXITER_ABS = 200
    TS_MAXITER_REL = 10

    function HyperelasticSolverCore(ts)
        update!(ch,ts)
        apply!(un,ch) 
        #println(maximum(un))
        u  = zeros(_ndofs) 
        Δu = zeros(_ndofs)
        ΔΔu = zeros(_ndofs)

        newton_itr = 0
        normg = zeros(NEWTON_MAXITER)
        while true; newton_itr += 1
            u .= un .+ Δu # current trial solution
            # Compute residual norm for current guess
            elementinfo = ElementFEAInfo_hyperelastic(s.mp, s.problem, u, default_quad_order(s.problem), Val{:Static}; ts=ts)
            assemble_hyperelastic!(globalinfo,s.problem,elementinfo,s.vars,getpenalty(s),s.xmin,assemble_f=assemble_f)
            apply_zero!(globalinfo.K,globalinfo.g,ch)
            normg[newton_itr] = norm(globalinfo.g)
            println("Tstep: $ts / 1]. Iteration: $newton_itr. normg is equal to " * string(normg[newton_itr]))
            # Check for convergence
            if normg[newton_itr] < NEWTON_TOL
                break
            elseif newton_itr > 1 && normg[newton_itr] > normg[newton_itr-1]
                error("Newton iteration resulted in an increase in normg, aborting")
            elseif newton_itr > NEWTON_MAXITER
                error("Reached maximum Newton iterations, aborting")
            end
            # Compute increment using conjugate gradients
            IterativeSolvers.cg!(ΔΔu, globalinfo.K, globalinfo.g; maxiter=CG_MAXITER)
            apply_zero!(ΔΔu, ch)
            Δu .-= ΔΔu
        end
        un = u
    end

    inc_up = 1.1
    inc_down = 0.5
    delay_up = 5
    delay_down = 5
    inc_delay = 10 # integer 
    ntsteps = s.ntsteps
    iter_ts = 0
    ts = 0
    Δts0 = 1/ntsteps
    Δts = Δts0
    conv = zeros(TS_MAXITER_ABS)
    #for tstep ∈ 1:ntsteps
    #    ts = tstep/ntsteps
    #end
    while ts < 1 
        iter_ts += 1
        ts += Δts
        if ts > 1 
            Δts = 1 - ts
            ts = 1
        elseif iter_ts > TS_MAXITER_REL && sum(conv[iter_ts-(TS_MAXITER_REL):iter_ts-1]) == 0
            error("Reached maximum number of successive failed ts iterations ($TS_MAXITER_REL), aborting")
        elseif iter_ts > TS_MAXITER_ABS
            error("Reached maximum number of allowed ts iterations ($TS_MAXITER_ABS), aborting")
        end
        try
            HyperelasticSolverCore(ts)
            conv[iter_ts] = 1
            if sum(conv[1:iter_ts]) == iter_ts || (iter_ts - findlast(x -> x == 0, conv[1:iter_ts-1])) > delay_up + 1 # increase Δts if it's never failed or if it's been 'inc_delay' or more successful iterations since last divergence
                Δts *= inc_up
            end
        catch
            conv[iter_ts] = 0
            ts -= Δts # revert to previous successful ts
            println("REMINDER TO CHECK THIS!!")
            if any(x -> x == 1, conv) && (iter_ts - findlast(x -> x == 1, conv[1:iter_ts-1])) < delay_down # decrease Δts a little if it's been 'down_delay' or less failures in a row
                Δts *= 1/inc_up
            else
                Δts *= inc_down # otherwise make a big decrease in Δts
            end
        end
        println("ts = $ts. Δts/Δts0 = $(Δts/Δts0)")
    end
    s.u .= un
    s.F .= elementinfo.Fes
    return nothing
end

function (s::HyperelasticNearlyIncompressibleDisplacementSolver{T})(
    ::Type{Val{safe}}=Val{false},
    ::Type{newT}=T;
    assemble_f=true,
    reuse_fact=false,
    kwargs...,
) where {T,safe,newT}
    elementinfo = s.elementinfo
    globalinfo = s.globalinfo
    dh = s.problem.ch.dh
    ch = s.problem.ch

    _ndofs = ndofs(dh)
    un = zeros(_ndofs) # previous solution vector

    NEWTON_TOL = 1e-8
    NEWTON_MAXITER = 30
    CG_MAXITER = 1000

    ntsteps = s.ntsteps
    for tstep ∈ 1:ntsteps
        ts = tstep/ntsteps
        update!(ch,ts)
        apply!(un,ch) 
        println(maximum(un))
        u  = zeros(_ndofs) 
        Δu = zeros(_ndofs)
        ΔΔu = zeros(_ndofs)
        
        newton_itr = 0
        normg = zeros(NEWTON_MAXITER)
        while true; newton_itr += 1
            u .= un .+ Δu # current trial solution
            # Compute residual norm for current guess
            elementinfo = ElementFEAInfo_hyperelastic(s.mp, s.problem, u, default_quad_order(s.problem), Val{:Static}; ts=ts)
            assemble_hyperelastic!(globalinfo,s.problem,elementinfo,s.vars,getpenalty(s),s.xmin,assemble_f=assemble_f)
            apply_zero!(globalinfo.K,globalinfo.g,ch)
            normg[newton_itr] = norm(globalinfo.g)
            println("Tstep: $tstep / $ntsteps. Iteration: $newton_itr. normg is equal to " * string(normg[newton_itr]))
            # Check for convergence
            if normg[newton_itr] < NEWTON_TOL
                break
            elseif newton_itr > NEWTON_MAXITER
                error("Reached maximum Newton iterations, aborting")
            end
            # Compute increment using conjugate gradients
            IterativeSolvers.cg!(ΔΔu, globalinfo.K, globalinfo.g; maxiter=CG_MAXITER)
            apply_zero!(ΔΔu, ch)
            Δu .-= ΔΔu
        end
        un = u
    end
    s.u .= un
    s.F .= elementinfo.Fes
    return nothing
end