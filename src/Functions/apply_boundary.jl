"""
    apply_zero!(Kσ, ch)

Apply boundary condition to the stress stiffness matrix. More info about this can be found at: 
https://github.com/JuliaTopOpt/TopOpt.jl/wiki/Applying-boundary-conditions-to-the-stress-stiffness-matrix
"""
function apply_zero!(Kσ, ch)
    # dummy f, applyzero=true
    apply!(Kσ, eltype(Kσ)[], ch, true)
    return Kσ
end
function ChainRulesCore.rrule(::typeof(apply_zero!), Kσ, ch)
    project_to = ChainRulesCore.ProjectTo(Kσ)
    return apply_zero!(Kσ, ch), Δ -> begin
        NoTangent(), apply_zero!(project_to(Δ), ch) , NoTangent()
    end
end

"""
rrule for the normal Ferrite apply! function.
"""
function ChainRulesCore.rrule(::typeof(apply!), K, ch)
    project_to = ChainRulesCore.ProjectTo(K)
    return apply!(K, ch), Δ -> begin
        NoTangent(), apply!(project_to(Δ), ch) , NoTangent()
    end
end