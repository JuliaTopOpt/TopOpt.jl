abstract type AbstractOptimizer end

mutable struct MMAOptimizer{T, TM<:MMAModel{T}, TO, TSO, TObj, TConstr, TW} <: AbstractOptimizer
    model::TM
    s_init::T
    s_decr::T
    s_incr::T
    optimizer::TO
    suboptimizer::TSO
    dual_caps::Tuple{T,T}
    obj::TObj
    constr::TConstr
    x_abschange::T
    x_converged::Bool
    f_abschange::T
    f_converged::Bool
    g_residual::T
    g_converged::Bool
    workspace::TW
end
GPUUtils.whichdevice(o::MMAOptimizer) = o.model

function Functions.maxedfevals(o::MMAOptimizer)
    maxedfevals(o.obj) || maxedfevals(o.constr)
end
@inline function Functions.maxedfevals(c::Tuple{Vararg{Constraint}})
    maxedfevals(c[1]) || maxedfevals(Base.tail(c))
end
function Functions.maxedfevals(c::Vector{<:Constraint})
    all(c -> maxedfevals(c), c)
end

MMAOptimizer(args...; kwargs...) = MMAOptimizer{CPU}(args...; kwargs...)
MMAOptimizer{T}(args...; kwargs...) where T = MMAOptimizer(T(), args...; kwargs...)
function MMAOptimizer(  device::Tdev, 
                        obj::Objective{<:AbstractFunction{T}}, 
                        constr, 
                        opt = MMA.MMA87(), 
                        subopt = Optim.ConjugateGradient(), 
                        suboptions = Optim.Options(x_tol = sqrt(eps(T)), f_tol = zero(T), g_tol = zero(T));

                        maxiter = 100,
                        xtol = 0.001,
                        ftol = xtol,
                        grtol = NaN,
                        s_init = 0.5,
                        s_incr = 1.2,
                        s_decr = 0.7,
                        dual_caps = MMA.default_dual_caps(opt, T)
                    ) where {T, Tdev}

    solver = getsolver(obj)
    nvars = length(solver.vars)
    xmin = solver.xmin

    model = MMAModel{Tdev}(nvars, obj, maxiter=maxiter, ftol=ftol, grtol=grtol, xtol=xtol, store_trace=false, extended_trace=false)

    box!(model, zero(T), one(T))
    ineq_constraint!(model, constr)

    if Tdev <: CPU && whichdevice(obj) isa GPU
        x0 = Array(solver.vars)
    else
        x0 = solver.vars
    end
    workspace = MMA.MMAWorkspace(model, x0, opt, subopt; s_init=s_init, 
    s_incr=s_incr, s_decr=s_decr, dual_caps=dual_caps)    

    return MMAOptimizer(model, s_init, s_decr, s_incr, opt, subopt, dual_caps, obj, constr, T(NaN), false, T(NaN), false, T(NaN), false, workspace)
end

Utilities.getpenalty(o::MMAOptimizer) = getpenalty(o.obj)
Utilities.setpenalty!(o::MMAOptimizer, p) = setpenalty!(o.obj, p)

function (o::MMAOptimizer)(x0::AbstractVector)
    mma_results = @timeit to "MMA" begin
        workspace = o.workspace
        T = eltype(x0)
        workspace.x .= x0
        n_i = length(MMA.constraints(workspace.model))
        reuse = o.obj.reuse
        workspace.f_x = @timeit to "Eval obj and constr" MMA.eval_objective(workspace.model, workspace.x, workspace.∇f_x)
        #assess_convergence(workspace)
        workspace.f_calls, workspace.g_calls = 1, 1
        #workspace.f_x_previous = T(NaN) 
        @timeit to "Eval obj and constr" map!((i)->MMA.eval_constraint(workspace.model, i, workspace.x, @view(workspace.∇g[:,i])), workspace.g, 1:n_i)
        workspace.primal_data.r0[] = 0
        workspace.primal_data.f_val[] = workspace.f_x
        # Assess multiple types of convergence
        workspace.x_converged, workspace.f_converged, workspace.gr_converged = false, false, false
        f_increased, converged = false, false
        workspace.x_residual = T(Inf)
        workspace.f_residual = T(Inf)
        workspace.gr_residual = T(Inf)
        workspace.outer_iter = 0
        workspace.iter = 0
        r = @timeit to "MMA optimize" MMA.optimize!(#=to, =#workspace)
        @timeit to "Pack MMA result" pack_results!(o, r)
        r
    end
    return mma_results
end

# For adaptive SIMP
function (o::MMAOptimizer)(workspace::MMA.MMAWorkspace)
    mma_results = MMA.optimize!(workspace)
    #@code_warntype mma_results = MMA.optimize!(workspace)
    pack_results!(o, mma_results)
    return mma_results
end

function pack_results!(o, r)
    o.x_abschange = r.x_abschange
    o.x_converged = r.x_converged
    o.f_abschange = r.f_abschange
    o.f_converged = r.f_converged
    o.g_residual = r.g_residual
    o.g_converged = r.g_converged

    return o
end
