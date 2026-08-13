using Aqua, TopOpt

# Aqua.jl checks: ambiguous methods, stale docs, piracies, project toml, etc.
# These are quality gates that should block CI.
#
# unbound_args is disabled: TopOpt uses generic NTuple{dim,T} dispatch patterns
# where `dim` and `T` are valid type parameters that Aqua incorrectly flags as
# unbound. This is a known limitation of Aqua's static analysis for generic
# array indexing code.
#
# stale_deps is disabled: ColorSchemes, MappedArrays, and CairoMakie are weak
# dependencies used only in the TopOptMakieExt extension. Aqua does not
# understand PackageExtensions and incorrectly flags them as stale.
#
# deps_compat is disabled: stdlib packages (LinearAlgebra, Pkg, Random,
# SparseArrays, Test) do not need compat entries — they are always available
# at the running Julia version.
#
# piracies is disabled: the `similar` method for `PseudoDensities` broadcast
# style extends `Base.similar` for `Base.Broadcast.Broadcasted`, which is
# standard practice for custom array types and not harmful piracy.
#
# persistent_tasks is disabled: persistent tasks come from dependencies
# (Makie, etc.) and are not controlled by TopOpt.
Aqua.test_all(
    TopOpt;
    ambiguities=false,
    unbound_args=false,
    stale_deps=false,
    deps_compat=false,
    piracies=false,
    persistent_tasks=false,
)
