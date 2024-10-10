abstract type AbstractHyperelasticSolver <: AbstractFEASolver end

mutable struct HyperelasticCompressibleDisplacementSolver{
    T,
    dim,
    TP1<:AbstractPenalty{T},
    TP2<:StiffnessTopOptProblem{dim,T},
    TG<:GlobalFEAInfo_hyperelastic{T},
    TE<:ElementFEAInfo_hyperelastic{dim,T},
    Tu<:AbstractVector{T},
} <: AbstractHyperelasticSolver
    mp
    problem::TP2
    globalinfo::TG
    elementinfo::TE
    u::Tu # JGB: u --> u0
    vars::Tu
    penalty::TP1
    prev_penalty::TP1
    xmin::T
    tsteps::Int
    ntsteps::Int
end
mutable struct HyperelasticNearlyIncompressibleDisplacementSolver{
    T,
    dim,
    TP1<:AbstractPenalty{T},
    TP2<:StiffnessTopOptProblem{dim,T},
    TG<:GlobalFEAInfo_hyperelastic{T},
    TE<:ElementFEAInfo_hyperelastic{dim,T},
    Tu<:AbstractVector{T},
} <: AbstractHyperelasticSolver
    mp
    problem::TP2
    globalinfo::TG
    elementinfo::TE
    u::Tu # JGB: u --> u0
    vars::Tu
    penalty::TP1
    prev_penalty::TP1
    xmin::T
    tsteps::Int
    ntsteps::Int
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, x::HyperelasticCompressibleDisplacementSolver)
    return println("TopOpt compressible hyperelastic solver")
end
function Base.show(::IO, ::MIME{Symbol("text/plain")}, x::HyperelasticNearlyIncompressibleDisplacementSolver)
    return println("TopOpt nearly-incompressible hyperelastic solver")
end
function HyperelasticDisplacementSolver(
    mp, # JGB: add type later
    sp::StiffnessTopOptProblem{dim,T}; # JGB: eventually add ::HyperelaticParam type
    xmin=T(1)/1000,
    penalty=PowerPenalty{T}(1),
    prev_penalty=deepcopy(penalty),
    quad_order=default_quad_order(sp),
    tstep = 1,
    ntsteps = 20,
    nearlyincompressible=false
) where {dim,T}
    u = zeros(T, ndofs(sp.ch.dh))
    ts0 = tstep/ntsteps
    update!(sp.ch,ts0) # set initial time-step (adjusts dirichlet bcs)
    apply!(u,sp.ch) # apply dbc for initial guess
    elementinfo = ElementFEAInfo_hyperelastic(mp, sp, u, quad_order, Val{:Static}, nearlyincompressible; ts = ts0) # JGB: add u
    globalinfo = GlobalFEAInfo_hyperelastic(sp) # JGB: small issue this leads to symmetric K initialization
    #u = zeros(T, ndofs(sp.ch.dh)) # JGB
    vars = fill(one(T), getncells(sp.ch.dh.grid) - sum(sp.black) - sum(sp.white))
    varind = sp.varind
    if nearlyincompressible
        return HyperelasticNearlyIncompressibleDisplacementSolver(mp, sp, globalinfo, elementinfo, u, vars, penalty, prev_penalty, xmin, tstep, ntsteps) 
    else 
        return HyperelasticCompressibleDisplacementSolver(mp, sp, globalinfo, elementinfo, u, vars, penalty, prev_penalty, xmin, tstep, ntsteps)
    end
end
function (s::HyperelasticCompressibleDisplacementSolver{T})(
    ::Type{Val{safe}}=Val{false},
    ::Type{newT}=T;
    assemble_f=true,
    kwargs...,
) where {T,safe,newT}
    globalinfo = s.globalinfo
    dh = s.problem.ch.dh
    ch = s.problem.ch

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    #un = s.u
    un = zeros(_ndofs) # previous solution vector
    #apply!(un, ch)

    # Perform Newton iterations
    NEWTON_TOL = 1e-8
    NEWTON_MAXITER =30 # OG is 30
    #prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")

    ntsteps = s.ntsteps
    for tstep ∈ 1:ntsteps
        ts = tstep/ntsteps
        update!(ch,ts)
        apply!(un,ch)
        println(maximum(un))
        u  = zeros(_ndofs) 
        Δu = zeros(_ndofs)
        ΔΔu = zeros(_ndofs)

        newton_itr = -1
        while true; newton_itr += 1
            # Construct the current guess
            u .= un .+ Δu
            #u += Δu
            # Compute residual and tangent for current guess
            s.elementinfo = ElementFEAInfo_hyperelastic(s.mp, s.problem, u, default_quad_order(s.problem), Val{:Static}; ts=ts) # JGB: add u
            #s.globalinfo = GlobalFEAInfo_hyperelastic(s.problem) # JGB: small issue this leads to symmetric K initialization
            #assemble_global!(K, g, dh, cv, fv, mp, u, ΓN)
            assemble_hyperelastic!(s.globalinfo,s.problem,s.elementinfo,s.vars,getpenalty(s),s.xmin,assemble_f=assemble_f)
            #K = s.globalinfo.K
            #g = s.globalinfo.g
            # Apply boundary conditions
            #if newton_itr == 1
            #    BSON.bson("C:\\Users\\jbecktt\\.julia\\juliaup\\julia-1.10.5+0.x64.w64.mingw32\\dev\\TopOpt\\test2.bson",K=s.globalinfo.K,g=s.globalinfo.g)
            #end
            #apply_zero!(s.globalinfo.K, s.globalinfo.g, ch) # why is this not active currently!!
            #if newton_itr == 1
            #    BSON.bson("C:\\Users\\jbecktt\\.julia\\juliaup\\julia-1.10.5+0.x64.w64.mingw32\\dev\\TopOpt\\test2.bson",K1=s.globalinfo.K,g1=s.globalinfo.g)
            #end
            # Compute the residual norm and compare with tolerance
            normg = norm(s.globalinfo.g)
            println("Tstep: $tstep / $ntsteps. Iteration: $newton_itr. normg is equal to " * string(normg))
            if normg < NEWTON_TOL
                break
            elseif newton_itr > NEWTON_MAXITER
                error("Reached maximum Newton iterations, aborting")
            end

            # Compute increment using conjugate gradients
            IterativeSolvers.cg!(ΔΔu, s.globalinfo.K, s.globalinfo.g; maxiter=1000)

            apply_zero!(ΔΔu, ch)
            Δu .-= ΔΔu  #Δu = Δu - Δ(Δu)
            #u = un + Δu
            #println("Finished iteration $newton_itr of tstep $tstep of $ntsteps")
        end
        un = u
    end
    # worst case scenario I can save F here for now
    s.u .= un #, F_storage, F_avg
    return nothing
end

function (s::HyperelasticNearlyIncompressibleDisplacementSolver{T})(
    ::Type{Val{safe}}=Val{false},
    ::Type{newT}=T;
    assemble_f=true,
    reuse_fact=false,
    kwargs...,
) where {T,safe,newT}
    globalinfo = s.globalinfo
    dh = s.problem.ch.dh
    ch = s.problem.ch

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    un = s.u
    u  = zeros(_ndofs)
    Δu = zeros(_ndofs)
    ΔΔu = zeros(_ndofs)

    # Perform Newton iterations
    newton_itr = -1
    NEWTON_TOL = 1e-8
    NEWTON_MAXITER = 1000 # OG is 30
    #prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")

    while true; newton_itr += 1
        # Construct the current guess
        u .= un .+ Δu
        # Compute residual and tangent for current guess
        s.elementinfo = ElementFEAInfo_hyperelastic(s.mp, s.problem, u, default_quad_order(s.problem), Val{:Static}) # JGB: add u
        #s.globalinfo = GlobalFEAInfo_hyperelastic(s.problem) # JGB: small issue this leads to symmetric K initialization
        #assemble_global!(K, g, dh, cv, fv, mp, u, ΓN)
        assemble_hyperelastic!(s.globalinfo,s.problem,s.elementinfo,s.vars,getpenalty(s),s.xmin,assemble_f=assemble_f)
        #K = s.globalinfo.K
        #g = s.globalinfo.g
        # Apply boundary conditions
        #apply_zero!(s.globalinfo.K, s.globalinfo.g, ch) 
        # Compute the residual norm and compare with tolerance
        normg = norm(s.globalinfo.g)
        if normg < NEWTON_TOL
            break
        elseif newton_itr > NEWTON_MAXITER
            error("Reached maximum Newton iterations, aborting")
        end

        # Compute increment using conjugate gradients
        IterativeSolvers.cg!(ΔΔu, s.globalinfo.K, s.globalinfo.g; maxiter=1000)

        apply_zero!(ΔΔu, ch)
        Δu .-= ΔΔu  #Δu = Δu - Δ(Δu)
    end
    
    s.u .= u #, F_storage, F_avg
    return nothing
end


















# ALL OF THE NEW STUFF TO BE PUT PLACES
# ATTRIBUTION: the following code was primarily sourced from sample code provided in Ferrite.jl documentation
#=
struct NeoHooke # JGB: add to .ipynb
    μ::Float64
    λ::Float64
end

function Ψ(C, mp::NeoHooke) # JGB: add to .ipynb
    μ = mp.μ
    λ = mp.λ
    Ic = tr(C)
    J = sqrt(det(C))
    return μ / 2 * (Ic - 3) - μ * log(J) + λ / 2 * log(J)^2
end

function constitutive_driver(C, mp::NeoHooke)
    # Compute all derivatives in one function call
    ∂²Ψ∂C², ∂Ψ∂C = Tensors.hessian(y -> Ψ(y, mp), C, :all)
    S = 2.0 * ∂Ψ∂C
    ∂S∂C = 2.0 * ∂²Ψ∂C²
    return S, ∂S∂C
end;

function assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN, F_storage, F_avg)
    # Reinitialize cell values, and reset output arrays
    reinit!(cv, cell)
    fill!(ke, 0.0)
    fill!(ge, 0.0)

    b = Vec{3}((0.0, -0.5, 0.0)) # Body force
    tn = 0.1 # Traction (to be scaled with surface normal)
    ndofs = getnbasefunctions(cv)
    cell_id=cell.current_cellid.x

    for qp in 1:getnquadpoints(cv)
        dΩ = getdetJdV(cv, qp)
        # Compute deformation gradient F and right Cauchy-Green tensor C
        ∇u = function_gradient(cv, qp, ue)
        F = one(∇u) + ∇u #*note that one() is the identity tensor times whatever you put in
        C = tdot(F) # F' ⋅ F
        # Compute stress and tangent
        S, ∂S∂C = constitutive_driver(C, mp)
        P = F ⋅ S
        I = one(S)
        ∂P∂F =  otimesu(I, S) + 2 * otimesu(F, I) ⊡ ∂S∂C ⊡ otimesu(F', I)

        # Store F in our data structure
        #F_storage[(cell.current_cellid, qp)] = F
        F_storage[cell_id][qp] = F

        F_avg[cell_id] = F_avg[cell_id] + cv.qr_weights[1]*F

        # Loop over test functions
        for i in 1:ndofs
            # Test function and gradient
            δui = shape_value(cv, qp, i)
            ∇δui = shape_gradient(cv, qp, i)
            # Add contribution to the residual from this test function
            ge[i] += ( ∇δui ⊡ P - δui ⋅ b ) * dΩ

            ∇δui∂P∂F = ∇δui ⊡ ∂P∂F # Hoisted computation
            for j in 1:ndofs
                ∇δuj = shape_gradient(cv, qp, j)
                # Add contribution to the tangent
                ke[i, j] += ( ∇δui∂P∂F ⊡ ∇δuj ) * dΩ
            end
        end
    end

    # Surface integral for the traction
    for face in 1:nfaces(cell) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! dloads start
        if (cellid(cell), face) in ΓN
            reinit!(fv, cell, face)
            for q_point in 1:getnquadpoints(fv)
                t = tn * getnormal(fv, q_point)
                dΓ = getdetJdV(fv, q_point)
                for i in 1:ndofs
                    δui = shape_value(fv, q_point, i)
                    ge[i] -= (δui ⋅ t) * dΓ
                end
            end
        end
    end
end;

function assemble_global!(K, g, dh, cv, fv, mp, u, ΓN, F_storage, F_avg)
    n = ndofs_per_cell(dh)
    ke = zeros(n, n)
    ge = zeros(n)

    # start_assemble resets K and g
    assembler = start_assemble(K, g)

    # Loop over all cells in the grid
    for cell in CellIterator(dh)
        global_dofs = celldofs(cell)
        ue = u[global_dofs] # element dofs
        assemble_element!(ke, ge, cell, cv, fv, mp, ue, ΓN, F_storage, F_avg)
        assemble!(assembler, global_dofs, ge, ke)
    end
end;

function solve()
    # Generate a grid
    N = 10
    L = 1.0
    left = zero(Vec{3})
    right = L * ones(Vec{3})
    grid = generate_grid(Tetrahedron, (N, N, N), left, right)

    # Material parameters
    E = 10.0
    ν = 0.3
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHooke(μ, λ)

    # Finite element base
    ip = Lagrange{3, RefTetrahedron, 1}()
    qr = QuadratureRule{3, RefTetrahedron}(1)
    qr_face = QuadratureRule{2, RefTetrahedron}(1)
    cv = CellVectorValues(qr, ip)
    fv = FaceVectorValues(qr_face, ip)

    # DofHandler
    dh = DofHandler(grid)
    push!(dh, :u, 3) # Add a displacement field
    close!(dh)
    #F_storage = Dict{Tuple{Int, Int}, Tensor{2,3,Float64}}()
    ncells = Ferrite.getncells(dh.grid)
    nqpoints_per_cell = getnquadpoints(cv)
    #F_storage = [Vector{Matrix{Float64}(undef,3,3)}(undef, nqpoints_per_cell) for _ in 1:ncells]
    #F_storage = [Vector{Tensor{2,3,Float64}}(undef, nqpoints_per_cell) for _ in 1:ncells]
    
    # Creating a nested array where each cell will hold an array of Tensors
    F_storage = Vector{Vector{Tensor{2, 3, Float64}}}(undef, ncells)
    # Initialize the inner arrays in the nested array for each cell
    for i in 1:ncells
        F_storage[i] = Vector{Tensor{2, 3, Float64}}(undef, nqpoints_per_cell)
    end
    
    F_avg = Vector{Tensor{2, 3, Float64}}(undef, ncells)  # Array for averaged deformation gradients.
    for i in 1:ncells
        F_avg[i] = zero(Tensor{2, 3})
    end

    function rotation(X, t)
        θ = pi / 3 # 60°
        x, y, z = X
        return t * Vec{3}((
            0.0,
            L/2 - y + (y-L/2)*cos(θ) - (z-L/2)*sin(θ),
            L/2 - z + (y-L/2)*sin(θ) + (z-L/2)*cos(θ)
        ))
    end

    dbcs = ConstraintHandler(dh)
    # Add a homogeneous boundary condition on the "clamped" edge
    dbc = Dirichlet(:u, getfaceset(grid, "right"), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> rotation(x, t), [1, 2, 3])
    add!(dbcs, dbc)
    close!(dbcs)
    t = 0.5
    Ferrite.update!(dbcs, t)

    # Neumann part of the boundary
    ΓN = union(
        getfaceset(grid, "top"),
        getfaceset(grid, "bottom"),
        getfaceset(grid, "front"),
        getfaceset(grid, "back"),
    )

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    un = zeros(_ndofs) # previous solution vector
    u  = zeros(_ndofs)
    Δu = zeros(_ndofs)
    ΔΔu = zeros(_ndofs)
    apply!(un, dbcs)

    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh)
    g = zeros(_ndofs)

    # Perform Newton iterations
    newton_itr = -1
    NEWTON_TOL = 1e-8
    NEWTON_MAXITER = 30
    prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")

    while true; newton_itr += 1
        # Construct the current guess
        u .= un .+ Δu
        # Compute residual and tangent for current guess
        assemble_global!(K, g, dh, cv, fv, mp, u, ΓN, F_storage, F_avg)
        # Apply boundary conditions
        apply_zero!(K, g, dbcs)
        # Compute the residual norm and compare with tolerance
        normg = norm(g)
        ProgressMeter.update!(prog, normg; showvalues = [(:iter, newton_itr)])
        if normg < NEWTON_TOL
            break
        elseif newton_itr > NEWTON_MAXITER
            error("Reached maximum Newton iterations, aborting")
        end

        # Compute increment using conjugate gradients
        IterativeSolvers.cg!(ΔΔu, K, g; maxiter=1000)

        apply_zero!(ΔΔu, dbcs)
        Δu .-= ΔΔu  #Δu = Δu - Δ(Δu)
    end
    
    return u, F_storage, F_avg
end

u, F_storage, F_avg, LogEH = solve()=#