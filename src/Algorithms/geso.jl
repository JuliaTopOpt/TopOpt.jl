mutable struct GESOResult{T,Tt<:AbstractVector{T}}
    topology::Tt
    objval::T
    change::T
    converged::Bool
    fevals::Int
end

"""
The GESO algorithm, see [LiuYiLiShen2008](@cite).
"""
struct GESO{T<:Real,TF<:AbstractCheqFilter} <: TopOptAlgorithm
    comp::ComplianceFun
    vol::VolumeFun
    vol_limit::T
    filter::TF
    vars::AbstractVector{T}
    topology::AbstractVector{T}
    Pcmin::T
    Pcmax::T
    Pmmin::T
    Pmmax::T
    Pen::T
    string_length::Int
    var_volumes::AbstractVector{T}
    cum_var_volumes::AbstractVector{T}
    order::AbstractVector{Int}
    genotypes::BitArray{2}
    children::BitArray{2}
    var_black::BitVector
    maxiter::Int
    penalty::AbstractPenalty{T}
    sens::AbstractVector{T}
    old_sens::AbstractVector{T}
    obj_trace::MVector{10,T}
    tol::T
    sens_tol::T
    result::GESOResult{T,Vector{T}}
end
function Base.show(io::IO, ::MIME{Symbol("text/plain")}, ::GESO)
    return println(io, "TopOpt GESO algorithm")
end

function GESO(
    comp::ComplianceFun,
    vol::VolumeFun,
    vol_limit,
    filter;
    maxiter=1000,
    tol=0.001,
    p=3.0,
    Pcmin=0.6,
    Pcmax=1.0,
    Pmmin=0.5,
    Pmmax=1.0,
    Pen=3.0,
    sens_tol=tol / 100,
    string_length=4,
    k=10,
)
    penalty = comp.solver.penalty
    setpenalty!(penalty, p)
    solver = comp.solver
    T = eltype(solver.vars)
    nel = Ferrite.getncells(solver.problem.ch.dh.grid)
    # GESO now works with full design variables
    nvars = nel
    vars = zeros(T, nvars)

    topology = zeros(T, nel)
    result = GESOResult(topology, T(NaN), T(NaN), false, 0)
    sens = zeros(T, nvars)
    old_sens = zeros(T, nvars)
    obj_trace = zeros(MVector{k,T})
    var_volumes = vol.cellvolumes
    cum_var_volumes = zeros(T, nvars)
    order = zeros(Int, nvars)
    genotypes = trues(string_length, nvars)
    children = trues(string_length, nvars)
    var_black = trues(nvars)

    return GESO{T,typeof(filter)}(
        comp,
        vol,
        T(vol_limit),
        filter,
        vars,
        topology,
        T(Pcmin),
        T(Pcmax),
        T(Pmmin),
        T(Pmmax),
        T(Pen),
        string_length,
        var_volumes,
        cum_var_volumes,
        order,
        genotypes,
        children,
        var_black,
        maxiter,
        penalty,
        sens,
        old_sens,
        obj_trace,
        T(tol),
        T(sens_tol),
        result,
    )
end

function Utilities.setpenalty!(b::GESO, p::Number)
    b.penalty.p = p
    return b
end

function get_progress(current_volume, total_volume, design_volume)
    return clamp(
        min(
            (total_volume - current_volume) / (total_volume - design_volume),
            current_volume / design_volume,
        ),
        0,
        1,
    )
end

function get_probs(b::GESO, Prg)
    return (
        b.Pcmin + (b.Pcmax - b.Pcmin) * Prg^b.Pen, b.Pmmin + (b.Pmmax - b.Pmmin) * Prg^b.Pen
    )
end

function crossover!(children, genotypes, i, j)
    for k in axes(genotypes, 1)
        r = rand()
        if r < 0.5
            children[k, i] = genotypes[k, i]
        else
            children[k, i] = genotypes[k, j]
        end
    end
    return nothing
end

# Crossover over one sensitivity class: for each element in `self_class`, pick
# a partner from `self_class` (with probability `Pc`), `other1`, or `other2`
# and mix their genotypes. Black/white elements are skipped. When
# `self_class` has a single element the self-pick is impossible, so the choice
# between `other1` and `other2` uses `single_threshold` instead.
function crossover_class!(
    children, genotypes, self_class, other1, other2, Pc, black, white; single_threshold
)
    for i in self_class
        if !isempty(black) && black[i]
            continue
        end
        if !isempty(white) && white[i]
            continue
        end
        r = rand()
        j = i
        if length(self_class) > 1
            if r < Pc
                while i == j
                    j = rand(self_class)
                end
            elseif r < 0.5 + 0.5 * Pc
                j = rand(other1)
            else
                j = rand(other2)
            end
        else
            if r < single_threshold
                j = rand(other1)
            else
                j = rand(other2)
            end
        end
        crossover!(children, genotypes, i, j)
    end
    return nothing
end

# Mutation over one sensitivity class: each genotype bit flips with probability
# `Pm`. `flip_from_zero` selects whether zeros flip to ones (high-sensitivity
# elements gain material) or ones flip to zeros (mid/low-sensitivity elements
# lose material). Returns whether any element's black/white state changed.
function mutate_class!(
    genotypes, var_black, self_class, Pm, black, white; flip_from_zero::Bool
)
    topology_changed = false
    for i in self_class
        if !isempty(black) && black[i]
            continue
        end
        if !isempty(white) && white[i]
            continue
        end
        for j in axes(genotypes, 1)
            r = rand()
            if r < Pm && (flip_from_zero ? !genotypes[j, i] : genotypes[j, i])
                genotypes[j, i] = !genotypes[j, i]
            end
        end
        if any(@view genotypes[:, i]) != var_black[i]
            var_black[i] = !var_black[i]
            topology_changed = true
        end
    end
    return topology_changed
end

function update!(
    var_black,
    children,
    genotypes,
    Pc,
    Pm,
    high_class,
    mid_class,
    low_class,
    black=BitVector(),
    white=BitVector(),
)
    topology_changed = false
    while !topology_changed
        # Crossover for all classes
        crossover_class!(
            children,
            genotypes,
            high_class,
            mid_class,
            low_class,
            Pc,
            black,
            white;
            single_threshold=0.5,
        )
        crossover_class!(
            children,
            genotypes,
            mid_class,
            high_class,
            low_class,
            Pc,
            black,
            white;
            single_threshold=0.5 + 0.5 * Pc,
        )
        crossover_class!(
            children,
            genotypes,
            low_class,
            mid_class,
            high_class,
            Pc,
            black,
            white;
            single_threshold=0.5,
        )
        genotypes .= children

        # Mutation for all classes
        c1 = mutate_class!(
            genotypes, var_black, high_class, Pm, black, white; flip_from_zero=true
        )
        c2 = mutate_class!(
            genotypes, var_black, mid_class, Pm, black, white; flip_from_zero=false
        )
        c3 = mutate_class!(
            genotypes, var_black, low_class, Pm, black, white; flip_from_zero=false
        )
        topology_changed = c1 | c2 | c3
    end

    return var_black
end

function (b::GESO)(
    x0=copy(b.comp.solver.vars); seed=NaN, black=BitVector(), white=BitVector()
)
    @unpack sens, old_sens, tol, maxiter = b
    @unpack obj_trace, topology, sens_tol, vars = b
    @unpack Pcmin, Pcmax, Pmmin, Pmmax, Pen = b
    @unpack string_length, genotypes, children, var_black = b
    @unpack cum_var_volumes, var_volumes, order = b
    @unpack total_volume, cellvolumes = b.vol
    T = eltype(x0)
    V = b.vol_limit
    design_volume = V * total_volume

    nel = length(x0)
    nvars = length(vars)

    # Validate black and white vectors
    if !isempty(black)
        length(black) == nel ||
            throw(DimensionMismatch("black must have length $nel (got $(length(black)))"))
    end
    if !isempty(white)
        length(white) == nel ||
            throw(DimensionMismatch("white must have length $nel (got $(length(white)))"))
    end
    if !isempty(black) && !isempty(white)
        any(black .& white) &&
            throw(ArgumentError("elements cannot be both black and white"))
    end

    # Set seed
    isnan(seed) || Random.seed!(seed)

    # Initialize the topology (work with full design variables)
    for i in eachindex(topology)
        topology[i] = round(x0[i])
        vars[i] = topology[i]
    end

    # Initialize black elements to 1 (solid) and white elements to 0 (void)
    initialize_black_white!(topology, vars, black, white)
    # GESO additionally tracks each element's black/white genotype state.
    @inbounds for i in eachindex(topology)
        if !isempty(black) && black[i]
            var_black[i] = true
            for j in axes(genotypes, 1)
                genotypes[j, i] = true
            end
        elseif !isempty(white) && white[i]
            var_black[i] = false
            for j in axes(genotypes, 1)
                genotypes[j, i] = false
            end
        end
    end

    check(x) = x > design_volume
    current_volume = dot(vars, var_volumes)
    vol = current_volume / total_volume
    # Main loop
    change = T(1)
    iter = 0
    f = x -> b.comp(b.filter(PseudoDensities(x)))
    while (change > tol || vol > V) && iter < maxiter
        iter += 1
        if iter > 1
            old_sens .= sens
        end
        for j in max(2, 10 - iter + 2):10
            obj_trace[j - 1] = obj_trace[j]
        end
        obj_trace[10], pb = Zygote.pullback(f, vars)
        sens = pb(1.0)[1]
        rmul!(sens, -1)
        if iter > 1
            @. sens = (sens + old_sens) / 2
        end

        # Classify the cells by their sensitivities
        # Note: black and white elements are included in sensitivity sorting
        # but will be excluded from mutation via update! function
        sortperm!(order, sens; rev=true)
        accumulate!(+, cum_var_volumes, view(var_volumes, order))
        N1 = findfirst(check, cum_var_volumes) - 1
        N2 = (nel - N1) ÷ 2
        N3 = nvars - N1 - N2
        high_class = @view order[1:N1]
        mid_class = @view order[(N1 + 1):(N1 + N2)]
        low_class = @view order[(N1 + N2 + 1):end]

        # Crossover and mutation
        Prg = get_progress(current_volume, total_volume, design_volume)
        Pc, Pm = get_probs(b, Prg)
        vars .= update!(
            var_black,
            children,
            genotypes,
            Pc,
            Pm,
            high_class,
            mid_class,
            low_class,
            black,
            white,
        )

        # Update crossover and mutation probabilities
        current_volume = dot(vars, var_volumes)
        vol = current_volume / total_volume

        if iter >= 10
            l = sum(@view obj_trace[1:5])
            h = sum(@view obj_trace[6:10])
            change = abs(l - h) / h
        end
    end

    for i in eachindex(topology)
        topology[i] = vars[i]
    end

    b.result.objval = obj_trace[10]
    b.result.change = change
    b.result.converged = change <= tol
    b.result.fevals = iter

    return b.result
end
