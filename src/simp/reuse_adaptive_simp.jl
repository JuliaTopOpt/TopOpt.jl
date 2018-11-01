mutable struct ReuseAdaptiveSIMP{T,TO,TP,TFC,TC<:CheqFilter} <: AbstractSIMP
    simp::SIMP{T,TO,TP}
    pstart::T
    pfinish::T
    reuse::Bool
    result::SIMPResult{T}
    ftol_cont::TFC #speed::T
    entertol::T
    Δp::T
    maxiter::Int
    innerpolynomial::PolynomialFit{T}
    prev_f::T
    prev_dfdp::T
    prev_ftol::T
    usederivative::Bool
    cheqfilter::TC
    adapt_pstep::Bool
end

function ReuseAdaptiveSIMP(simp::SIMP{T}, ::Type{Val{filtering}}=Val{false}; 
    pstart = T(1), 
    pfinish = T(5),
    ftol_cont = ExponentialContinuation{T}(T(0.01), T(4log(0.7)), T(0), 100, T(1e-4)), #speed = T(0.1),
    entertol = T(1),
    Δp = T(0.1),
    reuse = true,
    maxiter = 100,
    adalength = 3,
    usederivative = true,
    ratio = 0.5,
    rmin = 1.0,
    adapt_pstep = true) where {T, filtering}

    ncells = getncells(simp.optimizer.obj.problem)
    result = NewSIMPResult(T, ncells)

    innerpolynomial = PolynomialFit{T}(2, ratio)
    # `adalength` is the number of steps backward we consider when trying to adapt the penalty steps
    cheqfilter = CheqFilter{filtering}(simp.optimizer.obj.solver, T(rmin))

    return ReuseAdaptiveSIMP(simp, pstart, pfinish, reuse, result, ftol_cont, #=speed,=# entertol, Δp, maxiter, innerpolynomial, T(NaN), T(NaN), T(0), usederivative, cheqfilter, adapt_pstep)
end

function (rasimp::ReuseAdaptiveSIMP)(x0=rasimp.simp.optimizer.obj.solver.vars)
    @unpack pstart, pfinish, Δp = rasimp
    obj = rasimp.simp.optimizer.obj

    # Make an initial inner solve with the pstart creating the MMA workspace in the process
    p = pstart
    workspace = innersolve!(rasimp, x0, p)
    # Record the first value in the polynomial fit objects
    # Make a second data point
    while p < pfinish && obj.fevals < rasimp.maxiter
        #obj.cheqfilter(obj.solver.vars)
        deltap = ceil(abs(rasimp.prev_f) * 4 * MMA.ftol(workspace.model) / dFdp(rasimp.simp.optimizer.obj), 0.1)
        p = p + deltap
        maxiter = workspace.model.maxiter[]
        workspace.converged = false
        innersolve!(rasimp, workspace, p, false)
        # Check that we got past the reuse iteration, if any. If not, keeps going.
        obj.fevals >= rasimp.maxiter && break
    end
    rasimp.result = rasimp.simp.result
    return rasimp.result
end

function innersolve!(asimp::AdaptiveSIMP, x0::AbstractArray{T}, p) where T
    prev_l = length(asimp.simp.topologies)
    prev_fevals = asimp.simp.optimizer.obj.fevals

    innerpolynomial = asimp.innerpolynomial

    # Set penalty as starting penalty
    asimp.simp.penalty.p = p
    # Turn off solution reuse
    asimp.simp.optimizer.obj.reuse = false

    # Solve the SIMP with reuse as false and max 3 function evaluations and eps tolerance, 2 iterations with trace recording the objective values

    @unpack model, s_init, s_decr, s_incr, optimizer, suboptimizer, dual_caps, obj, constr, x_abschange, x_converged, f_abschange, f_converged, g_residual, g_converged = asimp.simp.optimizer

    # Resetting the adaptation polynomial
    #reset!(innerpolynomial)
        
    # Does the first function evaluation
    # Number of function evaluations is the number of iterations plus 1
    workspace = MMA.MMAWorkspace(model, x0, optimizer, suboptimizer, s_init=s_init, s_incr=s_incr, s_decr=s_decr, dual_caps=dual_caps)
    workspace.model.maxiter[] = 0
    # Record the first value in the polynomial fit struct
    newvalue!(innerpolynomial, obj.fevals, workspace.f_x)

    # Initial workspace.iter is 0
    # Tolerance is pretty low
    MMA.ftol!(workspace.model, sqrt(eps(T)))
    MMA.grtol!(workspace.model, sqrt(eps(T)))
    MMA.xtol!(workspace.model, sqrt(eps(T)))
    ftol = sqrt(eps(T))
    if obj.fevals >= asimp.maxiter
        return workspace
    end
    mma_results = _innersolve!(asimp, workspace)
    update_result!(asimp.simp, mma_results, prev_l, prev_fevals)
    return workspace
end

function innersolve!(asimp::AdaptiveSIMP, workspace::MMA.MMAWorkspace, p, reuse = true)
    prev_l = length(asimp.simp.topologies)
    prev_fevals = asimp.simp.optimizer.obj.fevals

    asimp.simp.penalty.p = p
    # Turn on solution reuse
    obj = asimp.simp.optimizer.obj
    obj.reuse = reuse && asimp.reuse

    # Solve the SIMP with reuse as true and max 3 function evaluations with last tolerance, 3 iterations with trace recording the objective values and dCdp

    # Resetting the adaptation polynomial
    #reset!(asimp.innerpolynomial)
    
    # Number of function evaluations is the number of iterations (not plus one as before)
    workspace.model.maxiter[] += 1
    # Use the last tolerance value
    # This runs 1 dummy iteration since reuse is true
    @debug println("Before MMA")
    @debug @show asimp.simp.optimizer.obj.fevals
    @debug @show MMA.xtol(workspace.model)
    @debug @show MMA.ftol(workspace.model)
    @debug @show MMA.grtol(workspace.model)
    @debug @show workspace.iter
    @debug @show workspace.model.maxiter[]
    @debug @show asimp.simp.optimizer.obj.reuse
    mma_results = asimp.simp.optimizer(workspace)
    @debug @show workspace.iter
    @debug @show asimp.simp.optimizer.obj.fevals
    @debug @show workspace.x_converged
    @debug @show workspace.f_converged
    @debug @show workspace.gr_converged
    @debug println("After MMA")
    newvalue!(asimp.innerpolynomial, obj.fevals, workspace.f_x)
    if workspace.converged || obj.fevals >= asimp.maxiter
        asimp.prev_f = workspace.f_x
        asimp.prev_dfdp = dFdp(asimp.simp.optimizer.obj)
    else
        @debug println("_innersolve!")
        mma_results = _innersolve!(asimp, workspace, 1)
    end
    update_result!(asimp.simp, mma_results, prev_l, prev_fevals)
    return workspace
end

function _innersolve!(asimp::AdaptiveSIMP{T}, workspace::MMA.MMAWorkspace, offset=0) where T
    @unpack innerpolynomial, pstart, pfinish = asimp
    p = asimp.simp.penalty.p
    obj = asimp.simp.optimizer.obj
    xmin = obj.solver.xmin

    # This runs 2 iterations
    workspace.model.maxiter[] += 1
    mma_results = asimp.simp.optimizer(workspace)
    ftol = MMA.ftol(workspace.model)
    prev_ftol = T(NaN)
    @debug @show ftol
    #@show workspace.model.maxiter[]
    if !workspace.converged && obj.fevals < asimp.maxiter
        # Record the second value in the polynomial fit struct
        newvalue!(innerpolynomial, obj.fevals, workspace.f_x)
        #@assert order(innerpolynomial) == 1
        workspace.model.maxiter[] += 1
        mma_results = asimp.simp.optimizer(workspace)
        if !workspace.converged && obj.fevals < asimp.maxiter
            # Record the third value in the polynomial fit struct
            newvalue!(innerpolynomial, obj.fevals, workspace.f_x)

            # If not converged yet, time to do tolerance adaptation
            # Fit the quadratic
            innerpolycoeffs = solve!(innerpolynomial)
            @debug @show innerpolycoeffs
            # Get the second derivative of the quadratic
            zero_derivative = innerpolynomial(obj.fevals)
            first_derivative = innerpolynomial'(obj.fevals)
            second_derivative = innerpolynomial''(obj.fevals)
            @debug @show second_derivative
            # Set the gradient tolerance as the speed/innerpolynomial''    
            prev_ftol = ftol
            progress = log(1 + 1/max(abs(second_derivative/zero_derivative) + abs(first_derivative/zero_derivative), sqrt(eps(T))))
            maxprogress = log(1 + 1/sqrt(eps(T)))
            #progress = p - pstart
            #maxprogress = pfinish - pstart
            #tol_cont = ExponentialContinuation{T}(asimp.speed, T(4log(0.7)), T(0), 100, T(1e-4))
            ##Uncomment##grtol = max(asimp.speed*(1 - progress/maxprogress), sqrt(eps(T)))
            ftol = asimp.ftol_cont(progress/maxprogress)
            
            #grtol = speed*progress/maxprogress
            #grtol = max(speed*sqrt(abs(second_derivative))/abs(first_derivative), sqrt(eps(T)))
            @debug @show ftol
            #@show workspace.model.maxiter[]
            #MMA.grtol!(workspace.model, grtol)
            #MMA.xtol!(workspace.model, grtol)
            MMA.ftol!(workspace.model, ftol)
            while ftol != prev_ftol && !MMA.assess_convergence(workspace)[end] && obj.fevals < asimp.maxiter
                workspace.model.maxiter[] += 1
                prev_l = length(asimp.simp.topologies)
                prev_fevals = asimp.simp.optimizer.obj.fevals
                prev_f = workspace.f_x
                mma_results = asimp.simp.optimizer(workspace)
                # Should we check if converged here and break??
                # Record the workspace.model.maxiter value in the polynomial fit struct
                #if (obj.fevals != prev_fevals || workspace.f_x != prev_f)
                    newvalue!(innerpolynomial, obj.fevals, workspace.f_x)
                    # Solve for coefficients
                    innerpolycoeffs = solve!(innerpolynomial)
                    @debug @show innerpolycoeffs
                    # Find second derivative
                    zero_derivative =  innerpolynomial(obj.fevals)
                    first_derivative = innerpolynomial'(obj.fevals)
                    second_derivative = innerpolynomial''(obj.fevals)
                    @debug @show second_derivative
                    # Update the gradient tolerance
                    prev_ftol = ftol
                    progress = log(1 + 1/max(abs(second_derivative/zero_derivative) + abs(first_derivative/zero_derivative), sqrt(eps(T))))
                    maxprogress = log(1 + 1/sqrt(eps(T)))
                    #progress = p - pstart
                    #maxprogress = pfinish - pstart
                    ftol = asimp.ftol_cont(progress/maxprogress)
                    #grtol = max(speed*sqrt(abs(second_derivative))/abs(first_derivative), sqrt(eps(T)))
                    @debug @show ftol
                    #MMA.grtol!(workspace.model, grtol)
                    #MMA.xtol!(workspace.model, grtol)
                    MMA.ftol!(workspace.model, ftol)                    
                #else
                #    @show grtol
                #    break
                #end
            end
        end
    end
    asimp.prev_f = workspace.f_x
    dCdp = dFdp(asimp.simp.optimizer.obj)
    @debug @show dCdp
    asimp.prev_dfdp = dCdp
    asimp.prev_ftol = ftol

    # Last tolerance: grtol
    # Last objective: workspace.f_x
    @debug @show workspace.f_x

    return mma_results
end
