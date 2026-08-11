# Tutorials

The TopOpt.jl tutorials are rendered separately using Quarto.

- **Local build**: run `quarto render` in `docs/tutorials/` and open `_site/index.html`
- **CI build**: tutorials are rendered by the CI workflow and deployed with the docs site
- **Online**: visit the [Tutorials](tutorials/index.html) section of the deployed docs

## Available tutorials

| Tutorial | Topic |
|---|---|
| [SIMP](simp.qmd) | Solid Isotropic Material with Penalization |
| [Continuation SIMP](csimp.qmd) | Penalty ramp optimization |
| [BESO](beso.qmd) | Bi-directional Evolutionary Structural Optimization |
| [GESO](geso.qmd) | Gradient-based Evolutionary Structural Optimization |
| [Local Stress](local_stress.qmd) | Local stress-constrained optimization |
| [Global Stress](global_stress.qmd) | Global stress-constrained optimization |
| [Truss](truss.qmd) | Truss topology optimization |
| [Truss Problems](problems_truss.qmd) | Truss problem types |
| [Mixed-Integer Truss](mixed_integer_truss.qmd) | MINLP truss design |
| [Buckling](buckling.qmd) | Buckling-constrained truss optimization |
| [Heat Sink](heat_sink.qmd) | Thermal compliance optimization |
| [Heat Tree](heat_tree.qmd) | Tree-shaped heat conduction |
| [Multi-Material](multimaterial.qmd) | Multi-material topology optimization |
| [TOBS](tobs.qmd) | Topology Optimization of Binary Structures |
| [Neural (IPOPT)](neural.qmd) | Neural network parametrization with IPOPT |
| [Neural (Adam)](neural2.qmd) | Neural network parametrization with Adam |
| [Continuum Problems](problems_continuum.qmd) | Continuum problem types |
