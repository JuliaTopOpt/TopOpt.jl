# This module implements the MMA Algorithm in Julia
# as described in:
# AU  - Svanberg, Krister
# TI  - The method of moving asymptotes—a new method for structural optimization
# JO  - International Journal for Numerical Methods in Engineering
# JA  - Int. J. Numer. Meth. Engng.
module MMA

using Parameters, StructArrays, Setfield, TimerOutputs, GPUUtils, CuArrays

using LinearAlgebra

import Optim
import Optim: OnceDifferentiable, Fminbox, GradientDescent, update!, 
                MultivariateOptimizationResults, OptimizationTrace, maxdiff, 
                LineSearches, ConjugateGradient, LBFGS, AbstractOptimizer
import Base: min, max, show

export MMAModel, box!, ineq_constraint!, optimize

struct MMA87 <: AbstractOptimizer end
struct MMA02 <: AbstractOptimizer end
abstract type TopOptAlgorithm end

include("utils.jl")
include("model.jl")
include("primal.jl")
include("dual.jl")
include("lift.jl")
include("trace.jl")
include("workspace.jl")

const μ = 0.1
const ρmin = 1e-5

default_dual_caps(::MMA87, ::Type{T}) where T = (T(0.9), T(1.1))
#default_dual_caps(::MMA87, ::Type{T}) where T = (T(0.0), T(Inf))

#default_dual_caps(::MMA02, ::Type{T}) where T = (T(1), T(100))
default_dual_caps(::MMA02, ::Type{T}) where T = (T(1e6), T(1e6))

function optimize(model::MMAModel{T,TV}, x0::TV, optimizer=MMA02(), suboptimizer=Optim.ConjugateGradient(); s_init=T(0.5), s_incr=T(1.2), s_decr=T(0.7), dual_caps=default_dual_caps(optimizer, T)) where {T, TV}
    check_error(model, x0)
    workspace = MMAWorkspace(model, x0, optimizer, suboptimizer; s_init=s_init, 
        s_incr=s_incr, s_decr=s_decr, dual_caps=dual_caps)    
    optimize!(workspace)
end

struct MMAResult{TO, TX, T}
    optimizer::TO
    initial_x::TX
    minimizer::TX
    minimum::T
    iter::Int
    iteration_converged::Bool
    x_converged::Bool
    x_tol::T
    x_abschange::T
    f_converged::Bool
    f_tol::T
    f_abschange::T
    g_converged::Bool
    g_tol::T
    g_residual::T
    f_increased::Bool
    f_calls::Int
    g_calls::Int
    h_calls::Int
end

function optimize!(#=to, =#workspace::MMAWorkspace{T, TV, TM}) where {T, TV, TM}
    @unpack model, optimizer, suboptimizer, suboptions, x0, x, x1, x2, λ, l, u, ∇f_x, g, 
        ng_approx, ∇g, f_x, f_calls, g_calls, f_x_previous, primal_data, tr, tracing, 
        converged, x_converged, f_converged, gr_converged, f_increased, x_residual, 
        f_residual, gr_residual, asymptotes_updater, variable_bounds_updater, 
        cvx_grad_updater, lift_updater, lift_resetter, x_updater, dual_obj, 
        dual_obj_grad, dual_caps, outer_iter, iter = workspace

    TSubOptions = typeof(workspace.suboptions)
    n_i = length(constraints(model))
    n_j = dim(model)
    maxiter = model.maxiter[]
    while !converged && iter < maxiter
        outer_iter += 1
        asymptotes_updater(Iteration(outer_iter))

        # Track trial points two steps back
        copyto!(x2, x1)
        copyto!(x1, x)

        # Update convex approximation
        ## Update bounds on primal variables
        variable_bounds_updater()    

        ## Computes values and updates gradients of convex approximations of objective and constraints
        cvx_grad_updater()

        if optimizer isa MMA02
            lift_resetter(Iteration(outer_iter))
        end
        lift = true
        _suboptions::TSubOptions = @set suboptions.outer_iterations = model.maxiter[]
        _suboptions = @set suboptions.iterations = model.maxiter[]
        while lift && iter < model.maxiter[]
            iter += 1
            # Solve dual
            λ .= min.(dual_caps[2], max.(λ, dual_caps[1]))
            d = OnceDifferentiable(dual_obj, dual_obj_grad, λ)
            _minimizer = #=@timeit to "Fminbox"=# Optim.optimize(d, l, u, λ, Optim.Fminbox(suboptimizer), _suboptions).minimizer
            copyto!(λ, _minimizer)
            dual_obj_grad(ng_approx, λ)

            # Evaluate the objective function and its gradient
            f_x_previous, f_x = f_x, eval_objective(model, x, ∇f_x)
            f_calls, g_calls = f_calls + 1, g_calls + 1
            # Correct for functions whose gradients go to infinity at some points, e.g. √x
            while mapreduce((x)->(isinf(x) || isnan(x)), or, ∇f_x, init=false)
                map!((x1,x)->(T(0.01)*x1 + T(0.99)*x), x, x1, x)
                f_x = eval_objective(model, x, ∇f_x)
                f_calls, g_calls = f_calls + 1, g_calls + 1
            end
            primal_data.f_val[] = f_x

            # Evaluate the constraints and their Jacobian
            map!((i)->eval_constraint(model, i, x, @view(∇g[:,i])), g, 1:n_i)

            if optimizer isa MMA87
                lift = false
            else
                lift = lift_updater()
            end
        end

        # Assess convergence
        x_converged, f_converged, gr_converged, 
        x_residual, f_residual, gr_residual, 
        f_increased, converged = assess_convergence(x, x1, f_x, f_x_previous, ∇f_x, 
            xtol(model), ftol(model), grtol(model))

        converged = converged && all((x)->(x<=ftol(model)), g)
        # Print some trace if flag is on
        @mmatrace()
    end
    h_calls = 0

    @pack! workspace = model, optimizer, suboptimizer, x0, x, x1, x2, λ, l, u, ∇f_x, g, 
        ng_approx, ∇g, f_x, f_calls, g_calls, f_x_previous, primal_data, tr, tracing, 
        converged, x_converged, f_converged, gr_converged, f_increased, x_residual, 
        f_residual, gr_residual, asymptotes_updater, variable_bounds_updater, 
        cvx_grad_updater, lift_updater, lift_resetter, x_updater, dual_obj, 
        dual_obj_grad, dual_caps, outer_iter, iter

    results = MMAResult(optimizer,
            x0,
            x,
            f_x,
            iter,
            iter == model.maxiter[],
            x_converged,
            xtol(model),
            x_residual,
            f_converged,
            ftol(model),
            f_residual,
            gr_converged,
            grtol(model),
            gr_residual,
            f_increased,
            f_calls,
            g_calls,
            h_calls)
    return results
end

end # module
