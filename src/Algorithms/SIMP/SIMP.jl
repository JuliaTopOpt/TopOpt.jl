abstract type AbstractSIMP <: TopOptAlgorithm end

function setreuse!(optimizer, reuse)
    if isdefined(optimizer.obj.f, :reuse)
        optimizer.obj.f.reuse = reuse
    end
    if isdefined(optimizer.constr.f, :reuse)
        optimizer.constr.f.reuse = reuse
    end
    return
end
function getreuse(optimizer)
    if isdefined(optimizer.obj.f, :reuse)
        return optimizer.obj.f.reuse
    end
    if isdefined(optimizer.constr.f, :reuse)
        return optimizer.constr.f.reuse
    end
end
Functions.getfevals(optimizer) = FunctionEvaluations(optimizer)

# MMA wrapper
include("math_optimizers.jl")

# Basic SIMP
include("basic_simp.jl")

## Continuation SIMP
include("continuation_schemes.jl")
include("continuation_simp.jl")

## Adaptive SIMP
include("polynomials.jl")
include("polynomials2.jl")
include("adaptive_simp.jl")
