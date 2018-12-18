# Setup

run(`git clone https://github.com/mohamed82008/TopOpt.jl TopOpt`)
cd(() -> run(`git checkout cuda_error`), "TopOpt")
using Pkg
Pkg.activate("./TopOpt")
Pkg.instantiate()

# Error example

using Revise, TopOpt, CuArrays, LinearAlgebra
CuArrays.allowscalar(false)

function gpu_testcase(s)
	problem = PointLoadCantilever(Val{:Linear}, s, (1.0, 1.0), 1.0, 0.3, 1.0);
	solver = FEASolver(Displacement, CG, MatrixFree, problem);
	cusolver = cu(solver);
	cusolver.vars .= 1;
	cuop = TopOpt.buildoperator(cusolver)
	cux = CuArray(ones(size(cuop, 1)))
	cuy = similar(cux)
	mul!(cuy, cuop, cux)
end

function cpu_testcase(s)
	problem = PointLoadCantilever(Val{:Linear}, s, (1.0, 1.0), 1.0, 0.3, 1.0);
	solver = FEASolver(Displacement, CG, MatrixFree, problem);
	solver.vars .= 1;
	op = TopOpt.buildoperator(solver)
	x = ones(size(op, 1))
	y = similar(x)
	mul!(y, op, x)
end

s = (4, 4)
cpu_testcase(s) # works
gpu_testcase(s) # works

s = (10, 4) 
cpu_testcase(s) # works
gpu_testcase(s)
# Some times works and other times errors with:
# ERROR: CUDA error: unspecified launch failure (code #719, ERROR_LAUNCH_FAILED)

s = (100, 10) 
gpu_testcase(s) 
# Works the first time and then fails the second time with:
# ERROR: CUDA error: an illegal memory access was encountered (code #700, ERROR_ILLEGAL_ADDRESS)

#= 
The block-thread config is found using https://github.com/mohamed82008/TopOpt.jl/blob/cuda_error/src/GPUUtils/GPUUtils.jl#L101.

The number of blocks and threads can be overwritten in https://github.com/mohamed82008/TopOpt.jl/blob/cuda_error/src/GPUUtils/GPUUtils.jl#L95.
=#
