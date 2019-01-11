mutable struct AdaptiveSIMP{T,TO,TP,TFC} <: AbstractSIMP
    simp::SIMP{T,TO,TP}
    pstart::T
    pfinish::T
    reuse::Bool
    result::SIMPResult{T}
    ftol_cont::TFC #speed::T
    Δp::T
    maxiter::Int
    innerpolynomial::PolynomialFit{T}
    prev_f::T
    prev_dfdp::T
    prev_ftol::T
    toltrace::Vector{T}
    progresstrace::Vector{T}
end
GPUUtils.whichdevice(a::AdaptiveSIMP) = whichdevice(a.simp)

function AdaptiveSIMP(simp::SIMP{T}, ::Type{Val{filtering}}=Val{false}; 
    pstart = T(1), 
    pfinish = T(5),
    ftol_cont = ExponentialContinuation{T}(T(0.01), T(4log(0.7)), T(0), 100, T(1e-4)),
    Δp = T(0.1),
    reuse = true,
    maxiter = 500,
    ratio = 0.7) where {T, filtering}

    ncells = getncells(simp.optimizer.obj.problem)
    result = NewSIMPResult(T, simp.optimizer, ncells)
    innerpolynomial = PolynomialFit{T}(3, ratio)

    return AdaptiveSIMP(simp, pstart, pfinish, reuse, result, ftol_cont, Δp, maxiter, innerpolynomial, T(NaN), T(NaN), T(0), T[], T[])
end

function (asimp::AdaptiveSIMP)(x0=asimp.simp.optimizer.obj.solver.vars)
    @unpack pstart, pfinish, Δp = asimp
    @unpack obj = asimp.simp.optimizer
    T = eltype(x0)

    p = pstart
    workspace = setup_workspace(asimp, x0, p)    
    @unpack primal_data, options, convstate = workspace
    steps = round(Int, (pfinish - pstart)/Δp)

    maxiter = options.maxiter
    fevals1, fevals2, fevals = obj.fevals, obj.fevals, obj.fevals
    for i in 1:steps
        obj.fevals >= asimp.maxiter && break
        workspace.outer_iter = workspace.iter = 0
        options.maxiter = 1

        f_x_previous = primal_data.f_x_previous[]
        fevals2 = fevals1
        fevals1 = fevals

        workspace.converged = false
        if i == steps
            obj.reuse = false
        else
            obj.reuse = asimp.reuse
        end

        setpenalty!(asimp.simp, p + Δp)
        simp_results = _innersolve!(asimp, workspace)
        
        if workspace.converged
            if obj.reuse
                primal_data.f_x[] = primal_data.f_x_previous[]
                primal_data.f_x_previous = f_x_previous
                primal_data.x .= primal_data.x1
                primal_data.x1 .= primal_data.x2
            end
        else
            obj.reuse = false
            while !convstate.converged && obj.fevals < asimp.maxiter && workspace.iter < maxiter
                options.maxiter += 1
                _innersolve!(asimp, workspace)
            end
        end
        fevals = obj.fevals
        x_hist = asimp.simp.optimizer.obj.topopt_trace.x_hist
        x = x_hist[fevals]
        x1 = x_hist[fevals1]
        x2 = x_hist[fevals2]
        same = true
        for _x in (x1, x2)
            for j in 1:length(x)
                if round(x[j]) != round(_x[j])
                    same = false
                    break
                end
            end
        end
    end
    options.maxiter = maxiter
    asimp.result = asimp.simp.result

    return asimp.result
end

function setup_workspace(asimp::AdaptiveSIMP, x0::AbstractArray{T}, p) where T
    @unpack model, s_init, s_decr, s_incr, optimizer, suboptimizer, dual_caps, obj, constr, x_abschange, x_converged, f_abschange, f_converged, g_residual, g_converged = asimp.simp.optimizer

    @unpack innerpolynomial, simp = asimp
    # Set penalty as starting penalty
    setpenalty!(simp, p)
    simp.optimizer.obj.reuse = false

    # Does the first function evaluation
    # Number of function evaluations is the number of iterations plus 1
    workspace = MMA.Workspace(model, x0, optimizer, suboptimizer, options = MMA.Options(s_init=s_init, s_incr=s_incr, s_decr=s_decr, dual_caps=dual_caps))
    # Record the first value in the polynomial fit struct
    newvalue!(innerpolynomial, obj.fevals, workspace.f_x)

    # Initial workspace.iter is 0
    # Tolerance is pretty low
    MMA.ftol!(workspace.model, asimp.ftol_cont(0))
    MMA.grtol!(workspace.model, zero(T))
    MMA.xtol!(workspace.model, zero(T))

    return workspace
end

function _innersolve!(asimp::AdaptiveSIMP{T}, workspace::MMA.Workspace) where T
    @unpack innerpolynomial, pstart, pfinish, simp = asimp
    @unpack optimizer, penalty = asimp.simp
    @unpack p = penalty
    @unpack obj = optimizer
    @unpack xmin = obj.solver
    @unpack model = workspace

    local simp_results

    while !workspace.converged && obj.fevals < asimp.maxiter && workspace.iter < model.maxiter[]
        if order(innerpolynomial) < maxorder(innerpolynomial)
            ftol = asimp.ftol_cont(0)
        else
            ftol = get_tol(asimp)
        end
        push!(asimp.toltrace, ftol)
        MMA.ftol!(model, ftol)
        simp_results = simp(workspace)
        newvalue!(innerpolynomial, obj.fevals, workspace.f_x)
    end

    return simp_results
end

function get_tol(asimp::AdaptiveSIMP{T}) where {T}
    p = getpenalty(asimp.simp).p
    obj = asimp.simp.optimizer.obj
    poly = asimp.innerpolynomial
    # If not converged yet, time to do tolerance adaptation
    # Fit the quadratic
    innerpolycoeffs = solve!(poly)

    # Set the gradient tolerance as the speed/innerpolynomial''    
    if p >= asimp.pfinish - sqrt(eps(T))
        progress = one(T)
    else
        # Get the second derivative of the quadratic
        zero_derivative = poly(obj.fevals)
        first_derivative = poly'(obj.fevals)
        second_derivative = poly''(obj.fevals)
        third_derivative = poly'''(obj.fevals)

        f = first_derivative / zero_derivative
        s = second_derivative / zero_derivative
        t = third_derivative / zero_derivative
        ds = abs(f) + abs(s) + abs(t)
        #_progress = log(1 + 1/max(ds, sqrt(eps(T))))
        #maxprogress = log(1 + 1/sqrt(eps(T)))
        #progress = clamp(_progress/maxprogress, (p - asimp.pstart)/(asimp.pfinish - asimp.pstart), 1)
        #progress = _progress/maxprogress
        progress = (p - asimp.pstart)/(asimp.pfinish - asimp.pstart)
    end
    push!(asimp.progresstrace, progress)
    ftol = asimp.ftol_cont(progress)

    return ftol
end

#=
Utilities.getpenalty(obj) = obj.solver.penalty

dFdp(obj::Objective{<:ComplianceFunction}) = dFdp(obj, getpenalty(obj))
function dFdp(obj::Objective{<:ComplianceFunction{T}}, penalty::PowerPenalty{T}) where T
    p = penalty.p
    cell_comp = obj.cell_comp
    vars = obj.solver.vars
    xmin = obj.solver.xmin

    return -mapreduce(e -> power_penalty_dFdpterm(vars[e], xmin, cell_comp[e], p), +, zero(T), 1:length(cell_comp))
end

function power_penalty_dFdpterm(var, xmin, cellcomp, p)
    return cellcomp*log(var)*var^p
end
=#