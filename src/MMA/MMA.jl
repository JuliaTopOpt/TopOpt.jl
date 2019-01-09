# This module implements the MMA Algorithm in Julia
# as described in:
# AU  - Svanberg, Krister
# TI  - The method of moving asymptotes—a new method for structural optimization
# JO  - International Journal for Numerical Methods in Engineering
# JA  - Int. J. Numer. Meth. Engng.
module MMA

using Parameters, StructArrays, Setfield, TimerOutputs, Base.Threads
using ..GPUUtils, CuArrays, CUDAnative, KissThreading
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

default_dual_caps(::Type{T}) where T = (eps(T), T(Inf))

#default_dual_caps(::MMA87, ::Type{T}) where T = (T(0.9), T(1.1))
#default_dual_caps(::MMA02, ::Type{T}) where T = (T(1), T(100))
#default_dual_caps(::MMA02, ::Type{T}) where T = (T(1e6), T(1e6))

function optimize(  model::Model{T, TV}, 
                    x0::TV, 
                    optimizer = MMA02(), 
                    suboptimizer = Optim.ConjugateGradient(); 
                    options = Options()
                ) where {T, TV}
    check_error(model, x0)
    workspace = Workspace(model, x0, optimizer, suboptimizer; options = options)
    return optimize!(workspace)
end

struct Tolerances{T}
    x_tol::T
    f_tol::T
    g_tol::T
end

struct MMAResult{TO, TX, T, TState}
    optimizer::TO
    initial_x::TX
    minimizer::TX
    minimum::T
    iter::Int
    maxiter_reached::Bool
    tol::Tolerances{T}
    convstate::TState
    f_calls::Int
    g_calls::Int
    h_calls::Int
end

function optimize!(workspace::Workspace{T, TV, TM}) where {T, TV, TM}
    @unpack model, optimizer, suboptimizer, options = workspace
    @unpack primal_data, dual_data, convstate = workspace
    @unpack asymptotes_updater, variable_bounds_updater = workspace 
    @unpack cvx_grad_updater, lift_updater, lift_resetter, x_updater = workspace
    @unpack dual_obj, dual_obj_grad, tracing, tr, f_calls, g_calls = workspace
    @unpack outer_iter, iter = workspace
    
    @unpack subopt_options, dual_caps = options 
    @unpack x0, x, x1, x2, ∇f_x, g = primal_data
    @unpack ng_approx = lift_updater
    @unpack ∇g = primal_data
    @unpack λ, l, u = dual_data 
     
    @unpack converged, x_converged, f_converged, gr_converged = convstate
    @unpack f_increased, x_residual, f_residual, gr_residual = convstate

    f_x = primal_data.f_x[]
    f_x_previous = primal_data.f_x_previous[]

    n_i = length(constraints(model))
    n_j = dim(model)
    maxiter = model.maxiter[]
    while !converged && iter < maxiter
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
        while lift && iter < model.maxiter[]
            iter += 1

            # Solve dual
            λ.cpu .= min.(dual_caps[2], max.(λ.cpu, dual_caps[1]))
            d = OnceDifferentiable(dual_obj, dual_obj_grad, λ.cpu)
            minimizer = Optim.optimize(d, l, u, λ.cpu, Optim.Fminbox(suboptimizer), subopt_options).minimizer
            copyto!(λ.cpu, minimizer)
            dual_obj_grad(ng_approx, λ.cpu)

            # Evaluate the objective function and its gradient
            f_x_previous, f_x = f_x, eval_objective(model, x, ∇f_x)
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
            primal_data.f_x_previous[] = f_x[]
            primal_data.f_x[] = f_x

            # Evaluate the constraints and their Jacobian
            map!(g, 1:n_i) do i
                @views eval_constraint(model, i, x, ∇g[:,i])
            end
            lift = optimizer isa MMA87 ? false : lift_updater()
        end

        # Assess convergence
        convstate = assess_convergence(x, x1, f_x, f_x_previous, ∇f_x, 
            xtol(model), ftol(model), grtol(model))

        converged = converged && all(g) do x
            x <= ftol(model)
        end

        # Print some trace if flag is on
        @mmatrace()
    end
    h_calls = 0

    @pack! workspace = outer_iter, iter, tr, tracing, f_calls, g_calls, convstate

    results = MMAResult(    optimizer,
                            x0,
                            x,
                            f_x,
                            iter,
                            iter == model.maxiter[],
                            Tolerances(xtol(model), ftol(model), grtol(model)),
                            convstate,
                            f_calls,
                            g_calls,
                            h_calls
                        )
    return results
end

end # module
