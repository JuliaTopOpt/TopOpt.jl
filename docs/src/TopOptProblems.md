# `TopOptProblems`

This sub-module of `TopOpt` defines a number of standard topology optimization problems for the convenient testing of algorithms.

```@meta
CurrentModule = TopOpt.TopOptProblems
```

## Problem types

```@docs
StiffnessTopOptProblem
PointLoadCantilever
HalfMBB
LBeam
TieBeam
InpStiffness
```

## Finite element backend

Currently, `TopOpt` uses a [forked version](https://github.com/mohamed82008/JuAFEM.jl) of [`JuAFEM.jl`](https://github.com/KristofferC/JuAFEM.jl). This means that all the problems above are described in the language and types of `JuAFEM`.

The fork is needed for GPU support but the main package should also work on the CPU. The changes in the fork should make it back to the main repo at some point.

## Element matrices and vectors

Generating the element stiffness matrices and load vectors from a linear elasticity stiffness problem `sp::StiffnessTopOptProblem` using a Gaussian quadrature order `quad_order` is done using:
```julia
elementinfo = ElementFEAInfo(sp, quad_order)
```
