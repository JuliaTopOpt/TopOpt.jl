# This module implements the MMA Algorithm in Julia
# as described in:
# AU  - Svanberg, Krister
# TI  - The method of moving asymptotes—a new method for structural optimization
# JO  - International Journal for Numerical Methods in Engineering
# JA  - Int. J. Numer. Meth. Engng.
module MMA

using Parameters, StructArrays, Setfield, TimerOutputs, Base.Threads
using ..GPUUtils, CuArrays, CUDAnative, KissThreading, ..Utilities
using GPUArrays: GPUVector
using CUDAdrv: CUDAdrv

const dev = CUDAdrv.device()
const ctx = CUDAdrv.CuContext(dev)

using LinearAlgebra

import Optim
import Optim: OnceDifferentiable, Fminbox, GradientDescent, update!, 
                MultivariateOptimizationResults, OptimizationTrace, maxdiff, 
                LineSearches, ConjugateGradient, LBFGS, AbstractOptimizer
import Base: min, max, show
import ..GPUUtils: whichdevice

export Model, box!, ineq_constraint!, optimize, ConvergenceState, Workspace

struct MMA87 <: AbstractOptimizer end
struct MMA02 <: AbstractOptimizer end
abstract type TopOptAlgorithm end

include("utils.jl")
include("model.jl")
include("dual.jl")
include("primal.jl")
include("lift.jl")
include("trace.jl")
include("workspace.jl")

const μ = 0.1
const ρmin = 1e-5

default_dual_caps(::Type{T}) where T = (eps(T), one(T))

function optimize(  model::Model, 
                    x0, 
                    optimizer = MMA02(), 
                    suboptimizer = Optim.ConjugateGradient(); 
                    options = Options()
                )
    check_error(model, x0)
    workspace = Workspace(model, x0, optimizer, suboptimizer; options = options)
    return optimize!(workspace)
end

@params mutable struct MMAResult{T}
    optimizer
    initial_x::AbstractVector{T}
    minimizer::AbstractVector{T}
    minimum::T
    iter::Int
    maxiter_reached::Bool
    tol::Tolerances
    convstate
    f_calls::Integer
    g_calls::Integer
    h_calls::Integer
end

abstract type ConvergenceCriteria end
struct DefaultCriteria <: ConvergenceCriteria end
struct KKTCriteria <: ConvergenceCriteria end

function get_kkt_residual(∇f_x, g, ∇g_x, c, x, lb, ub)
    r = mapreduce(max, 1:length(x), init = zero(eltype(x))) do j
        if lb[j] == x[j]
            return abs(min(0, ∇f_x[j] + dot(@view(∇g_x[j,:]), c)))
        elseif x[j] == ub[j]
            return abs(min(0, -∇f_x[j] - dot(@view(∇g_x[j,:]), c)))
        elseif  lb[j] < x[j] < ub[j]
            return zero(eltype(x))
        else
            throw("x is out of range.")
        end
    end
    r = mapreduce(max, 1:length(g), init = r) do i
        return max(abs(g[i] * c[i]), g[i], 0)
    end
    return r
end

function update_values!(w, _x = nothing)
    @unpack model, primal_data, f_calls, g_calls = w
    @unpack x1, x2, ∇f_x, f_x, f_x_previous, g, ∇g = primal_data
    n_i = length(constraints(model))
    n_j = dim(model)

    if _x === nothing
        x = primal_data.x
        f_x = eval_objective(model, x, ∇f_x)
    else
        x = _x
        x2 .= x1
        x1 .= primal_data.x
        primal_data.x .= x
        f_x_previous, f_x = f_x, eval_objective(model, x, ∇f_x)
    end
    T = eltype(x)
    f_calls, g_calls = f_calls + 1, g_calls + 1
    
    # Correct for functions whose gradients go to infinity at some points, e.g. √x
    while mapreduce(or, ∇f_x, init = false) do x
            isinf(x) || isnan(x) 
        end

        map!(x, x1, x) do x1, x
            T(0.01)*x1 + T(0.99)*x
        end
        f_x = eval_objective(model, x, ∇f_x)
        f_calls, g_calls = f_calls + 1, g_calls + 1
    end

    # Evaluate the constraints and their Jacobian
    map!(g, 1:n_i) do i
        @views eval_constraint(model, i, x, ∇g[:,i])
    end

    @pack! w = f_calls, g_calls
    @pack! primal_data = f_x, f_x_previous

    return w
end

function optimize!(workspace::Workspace{T, TV, TM}) where {T, TV, TM}
    @unpack model, optimizer, suboptimizer, options = workspace
    @unpack primal_data, dual_data = workspace
    @unpack asymptotes_updater, variable_bounds_updater = workspace 
    @unpack cvx_grad_updater, lift_updater, lift_resetter, x_updater = workspace
    @unpack dual_obj, dual_obj_grad, tracing, tr = workspace
    @unpack outer_iter, iter = workspace
    
    @unpack subopt_options, dual_caps = options 
    @unpack x0, x, x1, x2, ∇f_x, g, ∇g = primal_data
    @unpack ng_approx = lift_updater
    @unpack λ, l, u = dual_data 
    
    n_i = length(constraints(model))
    n_j = dim(model)
    maxiter = options.maxiter
    outer_maxiter = options.outer_maxiter

    while !(workspace.convstate.converged) && iter < maxiter && 
        outer_iter < outer_maxiter

        outer_iter += 1
        asymptotes_updater(Iteration(outer_iter))

        # Track trial points two steps back
        tmap!(identity, x2, x1)
        tmap!(identity, x1, x)

        # Update convex approximation
        ## Update bounds on primal variables
        variable_bounds_updater()    

        ## Computes values and updates gradients of convex approximations of objective and constraints
        cvx_grad_updater()

        if optimizer isa MMA02
            lift_resetter(Iteration(outer_iter))
        end
        lift = true
        while !workspace.convstate.converged && lift && iter < options.maxiter
            iter += 1

            # Solve dual
            λ.cpu .= min.(dual_caps[2], max.(λ.cpu, dual_caps[1]))

            d = OnceDifferentiable(dual_obj, dual_obj_grad, λ.cpu)
            minimizer = Optim.optimize(d, l, u, λ.cpu, Optim.Fminbox(suboptimizer), subopt_options).minimizer

            copyto!(λ.cpu, minimizer)
            dual_obj_grad(ng_approx, λ.cpu)

            update_values!(workspace)
            workspace.convstate = assess_convergence(workspace)
            lift = (optimizer isa MMA87) ? false : lift_updater()
        end

        # Print some trace if flag is on
        @mmatrace()
    end
    h_calls = 0
    @unpack f_calls, g_calls = workspace
    @unpack f_x, f_x_previous = primal_data
    @pack! workspace = outer_iter, iter, tr, tracing
    
    results = MMAResult(    optimizer,
                            x0,
                            x,
                            f_x,
                            iter,
                            iter == options.maxiter,
                            options.tol,
                            deepcopy(workspace.convstate),
                            f_calls,
                            g_calls,
                            h_calls
                        )
    return results
end

end # module
