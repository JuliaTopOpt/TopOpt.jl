@with_kw struct Tolerances{Txtol, Tftol, Tgrtol}
    xtol::Txtol = 1e-4
    ftol::Tftol = 1e-6
    grtol::Tgrtol = 1e-6
end
function (tol::Tolerances{<:Function, <:Function, <:Function})(i)
    return Tolerances(tol.xtol(i), tol.ftol(i), tol.grtol(i))
end

struct DualData{TSol, Tv}
    λ::TSol 
    l::Tv
    u::Tv
end
function DualData(model::Model{T, TV}) where {T, TV}
    n_i = length(constraints(model))
    λ = DualSolution(model)
    # Initial data and bounds for Optim to solve dual problem
    l = Vector(zerosof(TV, n_i))
    u = Vector(infsof(TV, n_i))
    return DualData(λ, l, u)
end

@with_kw mutable struct Options{T, Ttol <: Tolerances, TSubOptions <: Optim.Options}
    maxiter::Int = 1000
    outer_maxiter::Int = 10^8
    tol::Ttol = Tolerances()
    s_init::T = 0.5
    s_incr::T = 1.2
    s_decr::T = 0.7
    dual_caps::Tuple{T, T} = MMA.default_dual_caps(Float64)
    store_trace::Bool = false
    show_trace::Bool = false
    extended_trace::Bool = false
    subopt_options::TSubOptions = Optim.Options(allow_f_increases = false)
end
@inline function Base.getproperty(o::Options, f::Symbol)
    f === :xtol && return o.tol.xtol
    f === :ftol && return o.tol.ftol
    f === :grtol && return o.tol.grtol
    return getfield(o, f)
end

mutable struct Workspace{T, Tx, Tc, TO, TSO, TTrace <: OptimizationTrace{T}, TModel <: Model{T}, TPD <: PrimalData{T}, TXUpdater <: XUpdater{T, Tx, TPD}, Tdual_data <: DualData, TOptions, TState}
    model::TModel
    optimizer::TO
    suboptimizer::TSO
	primal_data::TPD
	asymptotes_updater::AsymptotesUpdater{T, Tx, TModel}
	variable_bounds_updater::VariableBoundsUpdater{T, Tx, TPD, TModel}
	cvx_grad_updater::ConvexApproxGradUpdater{T, Tx, TPD, TModel}
	lift_updater::LiftUpdater{T, Tc, TPD}
	lift_resetter::LiftResetter{T, Tc}
	x_updater::TXUpdater
    dual_data::Tdual_data
    dual_obj::DualObjVal{TPD, TXUpdater}
	dual_obj_grad::DualObjGrad{TPD, TXUpdater}
    tracing::Bool
    tr::TTrace
    outer_iter::Int
    iter::Int
	f_calls::Int
	g_calls::Int
    options::TOptions
    convstate::TState
end
const Workspace87{T, TV1, TV2, TM, TSO, TModel, TPD} = Workspace{T, TV1, TV2, TM, MMA87, TSO, TModel, TPD}
const Workspace02{T, TV1, TV2, TM, TSO, TModel, TPD} = Workspace{T, TV1, TV2, TM, MMA02, TSO, TModel, TPD}

Base.show(io::IO, w::Workspace87) = print(io, "Workspace for the method of moving asymptotes of 1987.")
Base.show(io::IO, w::Workspace02) = print(io, "Workspace for the method of moving asymptotes of 2002.")

function PrimalData(model::Model{T, TV}, optimizer, x0) where {T, TV}
    n_i = length(constraints(model))
    n_j = dim(model)

    x, x1, x2 = copy(x0), copy(x0), copy(x0)
    α, β, σ = zerosof(TV, n_j), zerosof(TV, n_j), zerosof(TV, n_j)

    # Buffers for p0, pji, q0, qji
    p0, q0 = zerosof(TV, n_j), zerosof(TV, n_j)
    TM = MatrixOf(TV)
    p, q = zerosof(TM, n_j, n_i), zerosof(TM, n_j, n_i)
    if optimizer isa MMA87
        ρ = Vector(zerosof(TV, n_i))
    else
        ρ = Vector(onesof(TV, n_i))
    end
    r = Vector(zerosof(TV, n_i))
    ∇f_x = nansof(TV, length(x))
    f_x = eval_objective(model, x, ∇f_x)
    f_x_previous = T(NaN)
    g = Vector(zerosof(TV, n_i))
    ∇g = zerosof(TM, n_j, n_i)
    
    return PrimalData(  σ, α, β, p0, q0, p, q, ρ, r, 
                        zero(T), x0, x, x1, x2, f_x, 
                        f_x_previous, g, ∇f_x, ∇g
                    )
end

function AsymptotesUpdater(primal_data, model, options)
    @unpack σ, x, x1, x2 = primal_data
    @unpack s_init, s_incr, s_decr = options
    return AsymptotesUpdater(model, σ, x, x1, x2, s_init, s_incr, s_decr)
end

function LiftUpdater(primal_data, model::Model{T, TV}) where {T, TV}
    @unpack ρ, g = primal_data
    n_i = length(constraints(model))
    n_j = dim(model)
    ng_approx = Vector(zerosof(TV, n_i))
    return LiftUpdater(primal_data, ρ, g, ng_approx, n_j)
end

function DualSolution(model::Model{T, TV}) where {T, TV}
    dev = whichdevice(model)
    n_i = length(constraints(model))
    if dev isa CPU
        return DualSolution{:CPU}(Vector(onesof(TV, n_i)))
    else
        return DualSolution{:GPU}(Vector(onesof(TV, n_i)))
    end
end

function Workspace( model::Model{T, TV}, 
                    x0::TV, 
                    optimizer::TO = MMA02(), 
                    suboptimizer = Optim.ConjugateGradient();
                    options = Options()
                ) where {T, TV, TO}

    primal_data = PrimalData(model, optimizer, x0)
    asymptotes_updater = AsymptotesUpdater(primal_data, model, options)
    variable_bounds_updater = VariableBoundsUpdater(primal_data, model, T(μ))
    cvx_grad_updater = ConvexApproxGradUpdater(primal_data, model)
    lift_updater = LiftUpdater(primal_data, model)
    x_updater = XUpdater(primal_data)
    dual_data = DualData(model)
    dual_obj = DualObjVal(primal_data, x_updater, dual_data.λ)
    dual_obj_grad = DualObjGrad(primal_data, x_updater, dual_data.λ)
    lift_resetter = LiftResetter(primal_data.ρ, T(ρmin))

    f_calls, g_calls = 1, 1
    convstate = ConvergenceState()

    # Evaluate the constraints and their gradients
    @unpack g, ∇g, x = primal_data
    update_constraints!(g, ∇g, model, x)
    tr = OptimizationTrace{T, TO}()
    tracing = (options.store_trace || options.extended_trace || options.show_trace)

    # Iteraton counter
    outer_iter = 0
    iter = 0

    return Workspace(
        model, optimizer, suboptimizer, primal_data, asymptotes_updater, 
        variable_bounds_updater, cvx_grad_updater, lift_updater, 
        lift_resetter, x_updater, dual_data, dual_obj, dual_obj_grad, 
        tracing, tr, outer_iter, iter, f_calls, g_calls, options, convstate
    )
end

function update_constraints!(g, ∇g, model, x)
    for i in 1:length(g)
        g[i] = eval_constraint(model, i, x, @view(∇g[:,i]))
    end
    return g
end

@with_kw mutable struct ConvergenceState{T}
    x_converged::Bool = false
    f_converged::Bool = false
    gr_converged::Bool = false
    x_residual::T = Inf
    f_residual::T = Inf
    gr_residual::T = Inf
    f_increased::Bool = false
    converged::Bool = false
end
function ConvergenceState(::Type{T}) where T
    ConvergenceState(false, false, false, T(Inf), T(Inf), T(Inf), false, false)
end

function assess_convergence(workspace::Workspace)
    @unpack options, primal_data = workspace
    @unpack x, x1, f_x, f_x_previous, ∇f_x = primal_data
    return assess_convergence(  x, 
                                x1, 
                                f_x, 
                                f_x_previous, 
                                ∇f_x, 
                                options.xtol, 
                                options.ftol, 
                                options.grtol
                            )
end
