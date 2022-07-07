# Differentiable functions

All the following functions are defined in a differentiable way and you can use them in the objectives or constraints in topology optimization formulation. In `TopOpt.jl`, you can build arbitrarily complex objective and constraint functions using the following building blocks as lego pieces chaining them in any arbitrary way. The gradient and jacobian of the aggregate Julia function defined is then obtained using [automatic differentiation](https://www.youtube.com/watch?v=UqymrMG-Qi4). Beside the following specific functions, arbitrary differentiable Julia functions such as `LinearAlgebra.norm` and `StatsFuns.logsumexp` are also supported which can for example be used in aggregating constraints.

## Density filter
  - **Function name**: `DensityFilter`
  - **Description**: Density chequerboard filter with a parameter `rmin`
  - **Input(s)**: Unfiltered design `x::Vector{<:Real}`
  - **Output**: Filtered design `y::Vector{<:Real}`
  - **Constructor example**: `flt = DensityFilter(solver, rmin = 3.0)`
  - **Usage example**: `y = flt(x)`

## Sensitivity filter
  - **Function name**: `SensFilter`
  - **Description**: Sensitivity chequerboard filter with a parameter `rmin`
  - **Input(s)**: Unfiltered design `x::Vector{<:Real}`
  - **Output**: Filtered design `y::Vector{<:Real}`
  - **Constructor example**: `flt = SensFilter(solver, rmin = 3.0)`
  - **Usage example**: `y = flt(x)`

## Heaviside projection
  - **Function name**: `HeavisideProjection`
  - **Description**: Heaviside projection function with a parameter `β` for producing near binary designs
  - **Input(s)**: Filtered design `x::Vector{<:Real}`
  - **Output**: Projected design `y::Vector{<:Real}`
  - **Constructor example**: `proj = HeavisideProjection(5.0)`
  - **Usage example**: `y = proj(x)`

## Compliance
  - **Function name**: `Compliance`
  - **Description**: Compliance function which applies the penalty and interpolation, solves the finite element analysis and calculates the compliance
  - **Input(s)**: Filtered and optionally projected design `x::Vector{<:Real}`
  - **Output**: Compliance value `comp::Real`
  - **Constructor example**: `compf = Compliance(problem, solver)`
  - **Usage example**: `comp = compf(x)`

## Volume
  - **Function name**: `Volume`
  - **Description**: Volume or volume fraction function depending on the value of the parameter `fraction` (default is `true`)
  - **Input(s)**: Filtered and optionally projected design `x::Vector{<:Real}`
  - **Output**: Volume or volume fracton `vol::Real`
  - **Constructor example**: `compf = Compliance(problem, solver)`
  - **Usage example**: `comp = compf(x)`

## Nodal displacements
  - **Function name**: `Displacement`
  - **Description**: Nodal displacements function which can be used to set a displacement constraint, minimize displacement or compute stresses and stress stiffness matrices
  - **Input(s)**: Filtered and optionally projected design `x::Vector{<:Real}`
  - **Output**: Displacements of all the nodes `u::Vector{<:Real}`
  - **Constructor example**: `disp = Displacement(solver)`
  - **Usage example**: `u = disp(x)`

## Element-wise microscopic stress tensor
  - **Function name**: `StressTensor`
  - **Description**: A function computing the element-wise microscopic stress tensor which is useful in stress-constrained optimization and machine learning for topology optimization. The microscopic stress tensor uses the base Young's modulus to compute the stiffness tensor and calculate the stress tensor from the strain tensor.
  - **Input(s)**: Nodal displacements vector `u::Vector{<:Real}`. This could be computed by the `Displacement` function above.
  - **Output**: Element-wise microscopic stress tensor `σ::Vector{<:Matrix{<:Real}}`. This is a vector of symmetric matrices, one matrix for each element.
  - **Constructor example**: `σf = StressTensor(solver)`
  - **Usage example**: `σ = σf(u)`

## Element-wise microscopic von Mises stress
  - **Function name**: `von_mises_stress_function`
  - **Description**: A function which applies the penalty and interpolation, solves the finite element analysis and computes the microscopic von Mises stress value for each element. The microscopic von Mises stress uses the base Young's modulus to compute the stiffness tensor and calculate the stress tensor from the strain tensor.
  - **Input(s)**: Filtered and optionally projected design `x::Vector{<:Real`
  - **Output**: Element-wise von Mises stress values `σv::Vector{<:Real}`
  - **Constructor example**: `σvf = von_mises_stress_function(solver)`
  - **Usage example**: `σv = σvf(x)`

## Element stiffness matrices
  - **Function name**: `ElementK`
  - **Description**: A function which computes the element stiffness matrices from the input design variables. The function applies the penalty and interpolation on inputs followed by computing the element stiffness matrices using a quadrature approximation of the discretized integral. This function is useful in buckling-constrained optimization.
  - **Input(s)**: Filtered and optionally projected design `x::Vector{<:Real}`
  - **Output**: Element-wise stiffness matrices `Kes::Vector{<:Matrix{<:Real}}`. This is a vector of symmetric positive (semi-)definite matrices, one matrix for each element.
  - **Constructor example**: `Kesf = ElementK(solver)`
  - **Usage example**: `Kes = Kesf(x)`

## Matrix assembly
  - **Function name**: `AssembleK`
  - **Description**: A function which assembles the element-wise matrices to a global sparse matrix. This function is useful in buckling-constrained optimization.
  - **Input(s)**: Element-wise matrices `Kes::Vector{<:Matrix{<:Real}}`. This is a vector of symmetric matrices, one matrix for each element.
  - **Output**: Global assembled sparse matrix `K::SparseMatrixCSC{<:Real}`.
  - **Constructor example**: `assemble = AssembleK(problem)`
  - **Usage example**: `K = assemble(Kes)`

## Applying Dirichlet boundary conditions with zeroing
  - **Function name**: `apply_boundary_with_zerodiag!`
  - **Description**: A function which zeroes out the columns and rows corresponding to degrees of freedom constrained by a Dirichlet boundary condition. This function is useful in buckling-constrained optimization.
  - **Input(s)**: Global assembled sparse matrix `Kin::SparseMatrixCSC{<:Real}` without boundary conditions applied.
  - **Output**: Global assembled sparse matrix `Kout::SparseMatrixCSC{<:Real}` with the boundary conditions applied.
  - **Constructor example**: NA
  - **Usage example**: `Kout = apply_boundary_with_zerodiag!(Kin, problem.ch)`

## Applying Dirichlet boundary conditions without causing singularity
  - **Function name**: `apply_boundary_with_meandiag!`
  - **Description**: A function which zeroes out the columns and rows corresponding to degrees of freedom constrained by a Dirichlet boundary condition followed by calculating the mean diagonal and assigning it to the zeroed diagonal entries. This function applies the boundary conditions while maintaining the non-singularity of the output matrix.
  - **Input(s)**: Global assembled sparse matrix `Kin::SparseMatrixCSC{<:Real}` without boundary conditions applied.
  - **Output**: Global assembled sparse matrix `Kout::SparseMatrixCSC{<:Real}` with the boundary conditions applied.
  - **Constructor example**: NA
  - **Usage example**: `Kout = apply_boundary_with_meandiag!(Kin, problem.ch)`

## Macroscopic truss element stress/geometric stiffness matrices
  - **Function name**: `TrussElementKσ`
  - **Description**: A function which computes the element-wise stress/geometric stiffness matrices for truss domains. This is useful in buckling-constrained truss optimization.
  - **Input(s)**: (1) The nodal displacement vector `u::Vector{<:Real}` computed from the `Displacement` function, and (2) the filtered, penalized, optionally projected and interpolated design `ρ::Vector{<:Real}`.
  - **Output**: The macroscopic element-wise stress/geometric stiffness matrices, `Kσs::Vector{<:Matrix{<:Real}}`. This is a vector of symmetric matrices, one matrix for each element.
  - **Constructor example**: `Kσsf = TrussElementKσ(problem, solver)`
  - **Usage example**: `Kσs = Kσsf(u, ρ)`

## Neural network re-parameterization
  - **Function name**: `NeuralNetwork`
  - **Description**: A function which re-parameterizes the design in terms of a neural network's weights and biases. The input to the neural network model is the coordinates of the centroid of an element. The output is the design variable associated with this element (from 0 to 1). The model is called once for each element in "prediction mode". When using the model in training however, the inputs to the training function will be the parameters of the model (to be optimized) and the elements' centroids' coordinates will be conditioned upon. The output of the training function will be the vector of element-wise design variables which can be passed on to any of the above functions, e.g. `Volume`, `DensityFilter`, etc. In the constructor example below, `nn` can be an almost arbitrary [`Flux.jl`](https://github.com/FluxML/Flux.jl) neural network model, `train_func` is what needs to be used to define the objective or constraints in the re-parameterized topology optimization formulation and `p0` is a vector of the neural network's initial weights and biases which can be used to initialize the optimization. The neural netowrk `nn` used must be one that can take 2 (or 3) input coordinates in the first layer for 2D (or 3D) problems and returns a scalar between 0 and 1 from the last layer. In prediction mode, this model will be called on each element using the centroid's coordinates as the input to neural network's first layer to compute the element's design variable. 
  - **Input(s)**: `train_func` below takes the vector of neural network weights and biases, `p::Vector{<:Real}`, as input.
  - **Output**: `train_func` below returns the vector of element-wise design variables, `x::Vector{<:Real}`, as outputs.
  - **Constructor example**:
  ```
  nn_model = NeuralNetwork(nn, problem)
  train_func = TrainFunction(nn_model)
  p0 = nn_model.init_params
  ```
  - **Usage example**: `x = train_func(p)`
