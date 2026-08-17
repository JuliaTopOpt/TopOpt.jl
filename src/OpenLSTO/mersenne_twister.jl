# Mersenne-Twister random number generator (a port of
# `M2DO_LSM/include/mersenne_twister.h`). Wraps Julia's MT19937 with the same
# interface OpenLSTO uses for stochastic velocity extension and hole
# nucleation.

"""
    MersenneTwister()

A Mersenne-Twister random number generator. `rng()` draws a uniform value in
`[0, 1]`, [`integer`](@ref) draws a uniform integer, and [`normal`](@ref)
draws a normal deviate. Seeded from the system entropy source unless
[`set_seed!`](@ref) is called first.
"""
mutable struct MersenneTwister
    rng::Random.MersenneTwister
    seed::UInt32
end

function MersenneTwister()
    seed = rand(Random.RandomDevice(), UInt32)
    return MersenneTwister(Random.MersenneTwister(seed), seed)
end

(rng::MersenneTwister)() = rand(rng.rng)

"""
    integer(rng::MersenneTwister, min, max)

Draw a uniform random integer in `min:max` (inclusive).
"""
integer(rng::MersenneTwister, min::Integer, max::Integer) = rand(rng.rng, min:max)

"""
    normal(rng::MersenneTwister)
    normal(rng::MersenneTwister, mean, std_dev)

Draw a normally distributed random number with zero mean and unit standard
deviation (or the given `mean` and `std_dev`).
"""
normal(rng::MersenneTwister) = randn(rng.rng)
normal(rng::MersenneTwister, mean::Real, std_dev::Real) = mean + std_dev * randn(rng.rng)

"""
    get_seed(rng::MersenneTwister)

Return the generator seed.
"""
get_seed(rng::MersenneTwister) = rng.seed

"""
    set_seed!(rng::MersenneTwister, seed)

Reseed the generator.
"""
function set_seed!(rng::MersenneTwister, seed::Integer)
    rng.seed = UInt32(seed)
    rng.rng = Random.MersenneTwister(rng.seed)
    return rng
end
