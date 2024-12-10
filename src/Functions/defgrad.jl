@params struct DefGradTensor{T} <: AbstractFunction{T}
    problem::Any
    solver::Any
    global_dofs::Vector{Int}
    cellvalues::Any
    cells::Any
    _::T
end
function DefGradTensor(solver)
    problem = solver.problem
    dh = problem.ch.dh
    n = ndofs_per_cell(dh)
    global_dofs = zeros(Int, n)
    cellvalues = solver.elementinfo.cellvalues
    return DefGradTensor(
        problem, solver, global_dofs, cellvalues, collect(CellIterator(dh)), 0.0
    )
end

function Ferrite.reinit!(s::DefGradTensor, cellidx)
    reinit!(s.cellvalues, s.cells[cellidx])
    celldofs!(s.global_dofs, s.problem.ch.dh, cellidx)
    return s
end
function ChainRulesCore.rrule(::typeof(reinit!), st::DefGradTensor, cellidx)
    return reinit!(st, cellidx), _ -> (NoTangent(), NoTangent(), NoTangent())
end

function (f::DefGradTensor)(dofs::DisplacementResult) # This is called in the PkgTest.ipynb by good(sim1_u) ----------------------------------------------------[C1] root for interation being studied
    return map(1:length(f.cells)) do cellidx 
        cf = f[cellidx] #  This runs C2 in order to makea ElementDefGradTensor struct for this cell with indiex cellidx
        return cf(dofs) # cf is a ElementDefGradTensor struct that will run C4 to yield a __________
    end
end

@params struct ElementDefGradTensor{T} <: AbstractFunction{T} # C2 creates one of these objects ----------------------------------------------------------------[C3]
    defgrad_tensor::DefGradTensor{T}
    cell::Any
    cellidx::Any
end
function Base.getindex(f::DefGradTensor{T}, cellidx) where {T} # This is encounter in C1 -----------------------------------------------------------------------[C2]
    reinit!(f, cellidx)
    return ElementDefGradTensor(f, f.cells[cellidx], cellidx)
end

function Ferrite.reinit!(s::ElementDefGradTensor, cellidx) 
    reinit!(s.defgrad_tensor, cellidx)
    return s
end
function ChainRulesCore.rrule(::typeof(reinit!), st::ElementDefGradTensor, cellidx)
    return reinit!(st, cellidx), _ -> (NoTangent(), NoTangent(), NoTangent())
end

function Fto3by3(F::Matrix{T}) where {T} 
    if size(F) == (2,2)
        F = [F zeros(T,2); zeros(T,1,2) T(1)] # plane strain assumption dictates that F₃₃ = 1
    elseif size(F) == (3,3)
        F
    else
        error("Invalid dimensions of deformation gradient tensor")
    end
end

function (f::ElementDefGradTensor)(u::DisplacementResult; element_dofs=false) #---------------------------------------------------------------------------------[C4] summing from C8
    st = f.defgrad_tensor
    reinit!(f, f.cellidx) # refreshing f
    if element_dofs # i think this is just choosing between local and global dofs
        cellu = u.u
    else
        cellu = u.u[copy(st.global_dofs)]
    end
    n_basefuncs = getnbasefunctions(st.cellvalues)
    n_quad = getnquadpoints(st.cellvalues)
    dim = TopOptProblems.getdim(st.problem)
    V = sum(st.cellvalues.detJdV)
    F = sum(map(1:n_quad) do  q_point
        dΩ = getdetJdV(st.cellvalues, q_point) 
        sum(map(1:n_basefuncs) do a   
            _u = cellu[dim * (a - 1) .+ (1:dim)]
            return tensor_kernel(f, q_point, a)(DisplacementResult(_u))
        end) * dΩ
    end) ./ V + I(dim)
    
    if dim == 2
        F = Fto3by3(F) 
    end
    return F
end

@params struct ElementDefGradTensorKernel{T} <: AbstractFunction{T} # ------------------------------------------------------------------------------------------------------------[C7]
    E::T
    ν::T
    q_point::Int
    a::Int
    cellvalues::Any
    dim::Int
end
function (f::ElementDefGradTensorKernel)(u::DisplacementResult) # ----------------------------------------------------------------------------------------------------------------[C8] ---- nifty
    @unpack E, ν, q_point, a, cellvalues = f
    ∇ϕ = Vector(shape_gradient(cellvalues, q_point, a))
    return u.u * ∇ϕ'
end
function ChainRulesCore.rrule(f::ElementDefGradTensorKernel, u::DisplacementResult)
    v, (∇,) = AD.value_and_jacobian(
        AD.ForwardDiffBackend(), u -> vec(f(DisplacementResult(u))), u.u
    )
    return reshape(v, f.dim, f.dim), Δ -> (NoTangent(), Tangent{typeof(u)}(; u=∇' * vec(Δ))) # Need to verify that this is true
end

function tensor_kernel(f::DefGradTensor, quad, basef) # -------------------------------------------------------------------------------------------------------------------------[C6* Altered ]
    #if string(typeof(f.problem))[1:12]!="InpStiffness"
        return ElementDefGradTensorKernel(
        f.problem.E,
        f.problem.ν,
        quad,
        basef,
        f.cellvalues,
        TopOptProblems.getdim(f.problem),
    )
    #else
    #    return ElementDefGradTensorKernel(
    #    f.problem.inp_content.E,
    #    f.problem.inp_content.ν,
    #    quad,
    #    basef,
    #    f.cellvalues,
    #    TopOptProblems.getdim(f.problem),
    #)
    #end
end
function tensor_kernel(f::ElementDefGradTensor, quad, basef) # --------------------------------------------------------------------------------------------------------------------[C5]
    return tensor_kernel(f.defgrad_tensor, quad, basef)
end