# This module implements the MMA Algorithm in Julia
# as described in:
# AU  - Svanberg, Krister
# TI  - The method of moving asymptotes—a new method for structural optimization
# JO  - International Journal for Numerical Methods in Engineering
# JA  - Int. J. Numer. Meth. Engng.
module MMALag

using Parameters, Base.Threads
using KissThreading: tmap!
using ..TopOpt: TopOpt, CPU, AbstractDevice, Optimise
using LinearAlgebra
using Optim: Optim, AbstractOptimizer
using ..Functions: AbstractFunction, LinQuadAggregation
using ..MMA: MMA, MMA87, MMA02, AbstractModel, Workspace, constraints, constraint, dim, update_values!, assess_convergence, @mmatrace, Iteration, MMAResult, infsof, ninfsof

export MMALag20

struct MMALag20{T} <: AbstractOptimizer
    mma_alg::T
    aug::Bool
end
MMALag20(alg) = MMALag20(alg, false)

mutable struct Model{T, TV<:AbstractVector{T}, TC<:AbstractVector{<:Function}} <: AbstractModel{T, TV}
    dim::Int
    objective::Function
    ineq_constraints::TC
    box_max::TV
    box_min::TV
    aug::Bool
end
Model(args...; kwargs...) = Model{CPU}(args...; kwargs...)
Model{T}(args...; kwargs...) where T = Model(T(), args...; kwargs...) 
Model(::CPU, args...; kwargs...) = Model{Float64, Vector{Float64}, Vector{Function}}(args...; kwargs...)
function Model{T, TV, TC}(dim, objective::Function, aug::Bool) where {T, TV, TC}
    mins = ninfsof(TV, dim)
    maxs = infsof(TV, dim)
    Model{T, TV, TC}(dim, objective, Function[],
             mins, maxs, aug)
end

function MMA.ineq_constraint!(m::Model, f::AbstractFunction{T}; r0 = 0.01) where {T}
    d = TopOpt.dim(f)
    _r0 = m.aug ? T(r0) : zero(T)
    constr = LinQuadAggregation(f, ones(T, d), _r0; max=true)
    push!(m.ineq_constraints, constr)
end
function MMA.ineq_constraint!(m::Model, fs::Vector{<:AbstractFunction}; r0 = 0.01)
    _r0 = m.aug ? T(r0) : zero(T)
    for f in fs
        d = TopOpt.dim(f)
        constr = LinQuadAggregation(f, ones(T, d), _r0; max=true)
        push!(m.ineq_constraints, constr)
    end
end

const μ = 0.1
const ρmin = 1e-5

function MMA.optimize!(workspace::Workspace{T, TV, TM}) where {T, TV, TM <: Model}
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
    infeas = floatmax(T)
    
    n_i = length(constraints(model))
    n_j = dim(model)
    maxiter = options.maxiter
    outer_maxiter = options.outer_maxiter

    step = 0.1
    agg_constrs = constraints(model)
    #stoch_opt = Optimise.Momentum(step)
    #stoch_opt = Optimise.Nesterov(step)
    stoch_opt = Optimise.Descent(step)
    function callback()
        dot1 = zero(T)
        dot2 = zero(T)
        dot3 = zero(T)
        quad_weight = zero(T)
        for (i, constr) in enumerate(agg_constrs)
            if constr isa LinQuadAggregation
                normN = Inf
                multiple = 1
                @show norm(constr.fval, normN)
                new_dir = multiple*normalize(constr.fval, normN)
                old_weights = copy(constr.lin_weights)
                step = (sqrt(infeas) / (1 + sqrt(infeas))) * (0.8999) + 0.0001
                @show step
                stoch_opt = Optimise.Descent(step)
                Optimise.apply!(stoch_opt, old_weights, new_dir)
                new_weights = max.(0, constr.lin_weights .+ new_dir)
                n = norm(new_weights, normN)/multiple
                if n != 0
                    new_weights .= new_weights ./ n
                    dot1 += dot(new_weights, constr.lin_weights)
                    dot2 += dot(constr.lin_weights, constr.lin_weights)
                    dot3 += dot(new_weights, new_weights)
                    constr.lin_weights .= new_weights
                else
                    throw("Norm of the weights cannot be 0.")
                end
                if outer_iter == 1
                    constr.quad_weight = 0.001
                else
                    constr.quad_weight = min(1000.0, constr.quad_weight * 1.05)
                end
                quad_weight = max(quad_weight, constr.quad_weight)
            end
        end
        f = sqrt(dot2)*sqrt(dot3)
        angle = f == 0 ? zero(T) : acos(clamp(dot1/f, -1, 1))
        return angle, quad_weight
    end

    angle = one(T)
    while iter < maxiter && outer_iter < outer_maxiter
        @show workspace.convstate.converged, angle
        infeas, slackness = get_infeas_and_slackness(model, workspace.primal_data.g, workspace.dual_data.λ.cpu)
        println("max infeasibility = ", infeas)
        println("slackness = ", slackness)
        workspace.convstate.converged && infeas <= options.tol.kkttol && slackness <= options.tol.kkttol && break
        angle, quad_weight = callback()
        println("angle = ", angle)
        println("Quad pen = ", quad_weight)

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
        while lift && iter < options.maxiter
            iter += 1

            # Solve dual
            λ.cpu .= min.(dual_caps[2], max.(λ.cpu, dual_caps[1]))

            d = Optim.OnceDifferentiable(dual_obj, dual_obj_grad, λ.cpu)
            minimizer = Optim.optimize(d, l, u, λ.cpu, Optim.Fminbox(suboptimizer), subopt_options).minimizer

            copyto!(λ.cpu, minimizer)
            dual_obj_grad(ng_approx, λ.cpu)

            update_values!(workspace)
            workspace.convstate = assess_convergence(workspace)
            lift = (optimizer isa MMA87) ? false : lift_updater()
            workspace.convstate.converged && break
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

function get_infeas_and_slackness(model::Model, g, c)
    infeas = mapreduce(max, 1:length(c)) do i
        constr = constraint(model, i)
        #max(0, g[i])
        if constr isa LinQuadAggregation
            return max(0, maximum(constr.fval))
        else
            return max(0, g[i])
        end
    end
    @show maximum(g), maximum(c)
    comp_slackness = maximum(i -> abs(g[i]*c[i]), 1:length(c))
    return infeas, comp_slackness
end

function MMA.get_ipopt_residual(model::Model, ∇f_x, g, ∇g_x, c, x, lb, ub)
    n = length(x)
    m = sum(1:length(c)) do i
        constr = constraint(model, i)
        constr isa LinQuadAggregation ? length(constr.lin_weights) : 1
    end
    s = zero(eltype(x))
    r = mapreduce(max, 1:n, init = zero(eltype(x))) do j
        _r = ∇f_x[j] + dot(@view(∇g_x[j,:]), c)
        if lb[j] >= x[j]
            dj = _r
            s += max(dj, 0)
            return abs(min(0, dj))
        elseif x[j] >= ub[j]
            yj = -_r
            s += max(yj, 0)
            return abs(min(0, yj))
        else
            return abs(_r)
        end
    end
    param = 100
    sd = max(param, (sum(abs, c) + s) / (n + m)) / param
    infeas = mapreduce(max, 1:length(c)) do i
        constr = constraint(model, i)
        #max(0, g[i])
        constr isa LinQuadAggregation ? max(0, maximum(constr.fval)) : max(0, g[i])
    end
    #comp_slackness = maximum(i -> abs(g[i]*c[i]), 1:length(c))
    #r = max(r / sd, infeas, comp_slackness)
    #r = max(r / sd, infeas)
    r = r / sd
    println("ipopt residual: $r")
    return r
end

end # module
