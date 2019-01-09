abstract type AbstractSIMP <: TopOptAlgorithm end

struct ReuseStatus{TC}
    obj::Bool
    constr::TC
end
Base.any(r::ReuseStatus) = r.obj || any(r.constr)
Base.all(r::ReuseStatus) = r.obj && all(r.constr)
function setreuse!(optimizer, reuse::Bool)
    optimizer.obj.f.reuse = reuse
    setproperty!.(optimizer.constr, :reuse, reuse)
    return getreuse(optimizer)
end
function setreuse!(optimizer, reuse::ReuseStatus)
    optimizer.obj.f.reuse = reuse.obj
    setproperty!.(optimizer.constr, :reuse, reuse.constr)
    return getreuse(optimizer)
end
function getreuse(optimizer)
    obj_reuse = optimizer.obj.reuse
    constr_reuse = getproperty.(optimizer.constr, :reuse)
    return ReuseStatus(obj_reuse, constr_reuse)
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
