@params mutable struct MeanCompliance{T, TC <: Compliance{T}, TM, TS} <: AbstractFunction{T}
    compliance::TC
    method::TM
    F::TS
    grad_temp::AbstractVector{T}
end
function MeanCompliance(problem::MultiLoad, solver::AbstractDisplacementSolver, args...; method = :exact_svd, sample_once = true, nv = nothing, V = nothing, sample_method = :hutch, kwargs...)
    if method == :exact
        method = ExactMean(problem.F)
    elseif method == :exact_svd
        method = ExactSVDMean(problem.F)
    elseif method == :trace || method == :approx
        if sample_method isa Symbol
            sample_method = sample_method == :hadamard ? hadamard! : hutch_rand!
        end
        if V === nothing
            nv = nv === nothing ? 1 : nv
            method = TraceEstimationMean(problem.F, nv, sample_once, sample_method)
        else
            nv = nv === nothing ? size(V, 2) : nv
            method = TraceEstimationMean(problem.F, view(V, :, 1:nv), sample_once, sample_method)
        end
    else
        if sample_method isa Symbol
            sample_method = sample_method == :hadamard ? hadamard! : hutch_rand!
        end
        if V === nothing
            nv = nv === nothing ? 1 : nv
            method = TraceEstimationSVDMean(problem.F, nv, sample_once, sample_method)
        else
            nv = nv === nothing ? size(V, 2) : nv
            method = TraceEstimationSVDMean(problem.F, view(V, :, 1:nv), sample_once, sample_method)
        end
    end
    comp = Compliance(whichdevice(solver), problem.problem, solver, args...; kwargs...)
    return MeanCompliance(comp, method, problem.F, similar(comp.grad))
end

function (ec::MeanCompliance{T})(x, grad = ec.grad) where {T}
    @unpack compliance, F, method = ec
    @unpack cell_comp, solver, tracing, topopt_trace = compliance
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes, black, white, varind = elementinfo
    @unpack cell_dofs = metadata

    @timeit to "Eval obj and grad" begin
        #if compliance.solver.vars ≈ x && getpenalty(ec).p ≈ getprevpenalty(ec).p
        #    grad .= o.grad
        #    return compliance.comp
        #end

        penalty = getpenalty(ec)    
        copyto!(solver.vars, x)
        compliance.fevals += 1
        setpenalty!(solver, penalty.p)
        obj = compute_mean_compliance(ec, method, solver.vars, grad)

        if compliance.logarithm
            compliance.comp = log(obj)
            grad ./= obj
        else
            compliance.comp = obj
        end
        if compliance.grad !== grad
            copyto!(compliance.grad, grad)
        end
        
        if compliance.tracing
            if ec.reuse
                ec.reuse = false
            else
                push!(topopt_trace.c_hist, obj)
                push!(topopt_trace.x_hist, copy(x))
                if length(topopt_trace.x_hist) == 1
                    push!(topopt_trace.add_hist, 0)
                    push!(topopt_trace.rem_hist, 0)
                else
                    push!(topopt_trace.add_hist, sum(topopt_trace.x_hist[end] .> topopt_trace.x_hist[end-1]))
                    push!(topopt_trace.rem_hist, sum(topopt_trace.x_hist[end] .< topopt_trace.x_hist[end-1]))
                end
            end
        end
    end
    return compliance.comp::T
end

function compute_mean_compliance(ec::MeanCompliance, ::ExactMean, x, grad)
    compute_exact_ec(ec, x, grad, ec.F, size(ec.F, 2))
end
function compute_exact_ec(ec, x, grad, F, n)
    @unpack compliance, grad_temp = ec
    @unpack cell_comp, solver = compliance
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes, black, white, varind = elementinfo
    @unpack cell_dofs = metadata
    penalty = getpenalty(compliance)    
    T = eltype(grad)
    obj = zero(T)
    grad .= 0
    for i in 1:size(F, 2)
        @views solver.rhs .= F[:,i]
        solver(assemble_f = false, reuse_chol = (i > 1))
        u = solver.lhs
        obj += compute_compliance(cell_comp, grad_temp, cell_dofs, Kes, u, 
                                    black, white, varind, x, penalty, xmin)
        grad .+= grad_temp
    end
    obj /= n
    grad ./= n
    return obj
end

function compute_mean_compliance(ec::MeanCompliance, ap::TraceEstimationMean, x, grad)
    compute_approx_ec(ec, x, grad, ap.F, ap.V, size(ap.F, 2))
end
function compute_approx_ec(ec, x, grad, F, V, n)
    nv = size(ec.method.V, 2)
    @unpack compliance, grad_temp = ec
    @unpack cell_comp, solver = compliance
    @unpack elementinfo, u, xmin = solver
    @unpack metadata, Kes, black, white, varind = elementinfo
    @unpack cell_dofs = metadata
    penalty = getpenalty(compliance)
    T = eltype(grad)
    obj = zero(T)
    grad .= 0
    ec.method.sample_once || ec.method.sample_method(V)
    for i in 1:nv
        @views mul!(solver.rhs, F, V[:,i])
        solver(assemble_f = false, reuse_chol = (i > 1))
        invKFv = solver.lhs
        obj += compute_compliance(cell_comp, grad_temp, cell_dofs, Kes, invKFv, 
                                    black, white, varind, x, penalty, xmin)
        grad .+= grad_temp
    end
    obj /= nv * n
    grad ./= nv * n
    return obj
end

function compute_mean_compliance(ec::MeanCompliance, ex::ExactSVDMean, x, grad)
    compute_exact_ec(ec, x, grad, ex.US, ex.n)
end

function compute_mean_compliance(ec::MeanCompliance, ax::TraceEstimationSVDMean, x, grad)
    compute_approx_ec(ec, x, grad, ax.US, ax.V, ax.n)
end

Utilities.getpenalty(c::MeanCompliance) = c.compliance |> getsolver |> getpenalty
@forward_property MeanCompliance compliance

hutch_rand!(x::Array) = x .= rand.(Ref(-1.0:2.0:1.0))
function hadamard3!(V)
	n, nv = size(V)
	H = ones(Int, 1, 1)
	while size(H, 1) < nv
		H = [H H ; H -H]
	end
	H = H[:, 1:nv]
	while size(H, 1) < n
		n1 = nv ÷ 2
		H1 = H[:, 1:n1]
		H2 = H[:, (n1+1):nv]
		H = [H1 H2; H1 -H2]
	end
	V .= H[1:n, :]
	return V
end
function hadamard2!(V)
	n, nv = size(V)
	H = ones(Int, 1, 1)
	while size(H, 1) < nv
		H = [H H ; H -H]
	end
	H = H[:, 1:nv]
	while size(H, 1) < n
		H = [H; -H]
	end
	V .= H[1:n, :]
	return V
end
function hadamard!(V)
	n, nv = size(V)
	H = ones(Int, 1, 1)
	while size(H, 1) < nv
		H = [H H ; H -H]
	end
	H = H[:, 1:nv]
	H = repeat(H, ceil(Int, n/nv), 1)
	V .= H[1:n, :]
	return V
end

function generate_scenarios(dof::Int, size::Tuple{Int, Int}, f, perturb = ()->(rand()-0.5))
    ndofs, nscenarios = size
    I = Int[]; J = Int[]; V = Float64[];
    V = [f * (1 + perturb()) for s in 1:nscenarios]
    I = [dof for s in 1:nscenarios]
    J = 1:nscenarios
    return sparse(I, J, V, ndofs, nscenarios)
end
Nonconvex.NonconvexCore.getdim(f::MeanCompliance) = 1
