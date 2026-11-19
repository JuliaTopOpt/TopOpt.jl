# Release notes

## 0.16.7

### Fixed

- Differentiated state functions (`DisplacementFun`, `TemperatureFun`,
  `MeanComplianceFun`, `BlockComplianceFun`, `TrussStressFun`) now keep
  per-evaluation state, so pullbacks are correct when the same function
  object is evaluated at several designs in one expression (#358).

### Changed

- The global-stress tutorial now starts its aggregation formulations from
  feasible designs, demonstrates the iteratively normalized lower-bound KS
  (Verbart et al., 2017), and drives the ε-relaxation with an ε
  continuation under the augmented Lagrangian; all designs are verified
  connected. The stochastic tutorial spreads its load ensemble over the tip
  and verifies loaded-node support and connectivity, and the remaining
  tutorials received feasibility and consistency fixes. The neural network
  tutorials were removed (#358).
## 0.16.6

### Changed

- Use NonconvexAugLagLab in the local stress, simultaneous analysis and design,
  and stochastic optimization tutorials (#356).

### Fixed

- Merge page-specific Quarto assets during documentation assembly while
  rejecting conflicting contents at shared asset paths (#356).
