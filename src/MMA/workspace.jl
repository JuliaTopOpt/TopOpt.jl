mutable struct MMAWorkspace{T, TV1, TV2, TM, TO, TSO, TSubOptions, TTrace<:OptimizationTrace{T}, TModel<:MMAModel{T}, TPD<:PrimalData{T}}
    model::TModel
    optimizer::TO
    suboptimizer::TSO
    suboptions::TSubOptions
    x0::TV1
    x::TV1
	x1::TV1
	x2::TV1
	λ::TV2
	l::TV2
	u::TV2
	∇f_x::TV1
	g::TV2
	ng_approx::TV2
	∇g::TM
	f_x::T
	f_calls::Int
	g_calls::Int
	f_x_previous::T
	primal_data::TPD
	tr::TTrace
	tracing::Bool
	converged::Bool
	x_converged::Bool
	f_converged::Bool
	gr_converged::Bool
	f_increased::Bool
	x_residual::T
	f_residual::T
	gr_residual::T
	asymptotes_updater::AsymptotesUpdater{T, TV1, TModel}
	variable_bounds_updater::VariableBoundsUpdater{T, TPD, TModel}
	cvx_grad_updater::ConvexApproxGradUpdater{T, TPD, TModel}
	lift_updater::LiftUpdater{T, TV2, TPD}
	lift_resetter::LiftResetter{T, TV2}
	x_updater::XUpdater{TPD}
	dual_obj::DualObjVal{T, TPD}
	dual_obj_grad::DualObjGrad{TPD}
    dual_caps::Tuple{T, T}
	outer_iter::Int
    iter::Int
end
const MMA87Workspace{T, TV1, TV2, TM, TSO, TModel, TPD} = MMAWorkspace{T, TV1, TV2, TM, MMA87, TSO, TModel, TPD}
const MMA02Workspace{T, TV1, TV2, TM, TSO, TModel, TPD} = MMAWorkspace{T, TV1, TV2, TM, MMA02, TSO, TModel, TPD}

Base.show(io::IO, w::MMA87Workspace) = print(io, "Workspace for the method of moving asymptotes of 1987.")
Base.show(io::IO, w::MMA02Workspace) = print(io, "Workspace for the method of moving asymptotes of 2002.")

function MMAWorkspace(model::MMAModel{T,TV}, x0::TV, optimizer=MMA02(), suboptimizer=Optim.ConjugateGradient(); suboptions=Optim.Options(x_tol=sqrt(eps(T)), f_tol=sqrt(eps(T)), g_tol=sqrt(eps(T))), s_init=T(0.5), s_incr=T(1.2), s_decr=T(0.7), dual_caps=default_dual_caps(optimizer, T)) where {T, TV}

    n_i = length(constraints(model))
    n_j = dim(model)
    x, x1, x2 = copy(x0), copy(x0), copy(x0)
    TM = MatrixOf(TV)

    # Buffers for bounds and move limits
    α, β, σ = zerosof(TV, n_j), zerosof(TV, n_j), zerosof(TV, n_j)

    # Buffers for p0, pji, q0, qji
    p0, q0 = zerosof(TV, n_j), zerosof(TV, n_j)
    p, q = zerosof(TM, n_j, n_i), zerosof(TM, n_j, n_i)
    if optimizer isa MMA87
        ρ = Vector(zerosof(TV, n_i))
    else
        ρ = Vector(onesof(TV, n_i))
    end
    r = Vector(zerosof(TV, n_i))
    
    # Initial data and bounds for Optim to solve dual problem
    λ = Vector(onesof(TV, n_i))
    l = Vector(zerosof(TV, n_i))
    u = Vector(infsof(TV, n_i))
    #u .= 1e50
    # Objective gradient buffer

    ∇f_x = nansof(TV, length(x))
    g = Vector(zerosof(TV, n_i))
    ng_approx = Vector(zerosof(TV, n_i))
    ∇g = zerosof(TM, n_j, n_i)
    
    f_x::T = eval_objective(model, x, ∇f_x)
    f_calls, g_calls = 1, 1
    f_x_previous = T(NaN)

    # Evaluate the constraints and their gradients
    update_constraints!(g, ∇g, model, x)

    # Build a primal data struct storing all primal problem's info
    primal_data = PrimalData(σ, α, β, p0, q0, p, q, ρ, r, Ref(zero(T)), x, x1, Ref(f_x), g, ∇f_x, ∇g)
    
    tr = OptimizationTrace{T, MMA87}()
    tracing = (model.store_trace || model.extended_trace || model.show_trace)

    converged = false
    # Assess multiple types of convergence
    x_converged, f_converged, gr_converged = false, false, false
    f_increased = false
    x_residual = T(Inf)
    f_residual = T(Inf)
    gr_residual = T(Inf)

    asymptotes_updater = AsymptotesUpdater(model, σ, x, x1, x2, s_init, s_incr, s_decr)
    variable_bounds_updater = VariableBoundsUpdater(primal_data, model, T(μ))
    cvx_grad_updater = ConvexApproxGradUpdater(primal_data, model)
    lift_updater = LiftUpdater(primal_data, ρ, g, ng_approx, n_j)
    lift_resetter = LiftResetter(ρ, T(ρmin))

    x_updater = XUpdater(primal_data)
    dual_obj = DualObjVal(primal_data, x_updater)
    dual_obj_grad = DualObjGrad(primal_data, x_updater)

    # Iteraton counter
    outer_iter = 0
    iter = 0

    TO = typeof(optimizer)
    TSO = typeof(suboptimizer)
    TModel = typeof(model)
    TPD = typeof(primal_data)
    TSubOptions = typeof(suboptions)
    return MMAWorkspace(
        model, optimizer, suboptimizer, suboptions, x0, x, x1, x2, λ, l, u, ∇f_x, g, 
        ng_approx, ∇g, f_x, f_calls, g_calls, f_x_previous, primal_data, tr, tracing, 
        converged, x_converged, f_converged, gr_converged, f_increased, x_residual, 
        f_residual, gr_residual, asymptotes_updater, variable_bounds_updater, 
        cvx_grad_updater, lift_updater, lift_resetter, x_updater, dual_obj, 
        dual_obj_grad, dual_caps, outer_iter, iter)
end

function update_constraints!(g, ∇g, model, x)
    for i in 1:length(g)
        g[i] = eval_constraint(model, i, x, @view(∇g[:,i]))
    end
    return g
end

function assess_convergence(workspace::MMAWorkspace)
    @unpack model, x, x1, f_x, f_x_previous, ∇f_x = workspace
    return assess_convergence(x, x1, f_x, f_x_previous, ∇f_x, xtol(model), ftol(model), grtol(model))
end
