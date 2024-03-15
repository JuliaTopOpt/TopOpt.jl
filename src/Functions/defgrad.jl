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
    return sum(
        map(1:n_basefuncs, 1:n_quad) do a, q_point
            _u = cellu[dim * (a - 1) .+ (1:dim)]
            return tensor_kernel(f, q_point, a)(DisplacementResult(_u))
        end,
    ) + I(3)
end

@params struct ElementDefGradTensorKernel{T} <: AbstractFunction{T} # ------------------------------------------------------------------------------------------------------------[C7]
    E::T
    ν::T
    q_point::Int
    a::Int
    cellvalues::Any
    dim::Int
end
# function (f::ElementDefGradTensorKernel)(u::DisplacementResult) # ----------------------------------------------------------------------------------------------------------------[C8]
#     @unpack E, ν, q_point, a, cellvalues = f
#     ∇ϕ = Vector(shape_gradient(cellvalues, q_point, a))
#     ϵ = (u.u .* ∇ϕ' .+ ∇ϕ .* u.u') ./ 2
#     c1 = E * ν / (1 - ν^2) * sum(diag(ϵ))
#     c2 = E * ν * (1 + ν)
#     return c1 * I + c2 * ϵ
# end
function (f::ElementDefGradTensorKernel)(u::DisplacementResult) # ----------------------------------------------------------------------------------------------------------------[C8] ---- nifty
    @unpack E, ν, q_point, a, cellvalues = f
    ∇ϕ = Vector(shape_gradient(cellvalues, q_point, a))
    # ϵ = (u.u .* ∇ϕ' .+ ∇ϕ .* u.u') ./ 2
    # c1 = E * ν / (1 - ν^2) * sum(diag(ϵ))
    # c2 = E * ν * (1 + ν)
    F_quad = u.u * ∇ϕ' # should this I(3) go somewhere else
    return F_quad
end
function ChainRulesCore.rrule(f::ElementDefGradTensorKernel, u::DisplacementResult)
    v, (∇,) = AD.value_and_jacobian(
        AD.ForwardDiffBackend(), u -> vec(f(DisplacementResult(u))), u.u
    )
    return reshape(v, f.dim, f.dim), Δ -> (NoTangent(), Tangent{typeof(u)}(; u=∇' * vec(Δ)))
end

# function tensor_kernel(f::DefGradTensor, quad, basef) # -------------------------------------------------------------------------------------------------------------------------[C6]
#     return ElementDefGradTensorKernel(
#         f.problem.E,
#         f.problem.ν,
#         quad,
#         basef,
#         f.cellvalues,
#         TopOptProblems.getdim(f.problem),
#     )
# end

function tensor_kernel(f::DefGradTensor, quad, basef) # -------------------------------------------------------------------------------------------------------------------------[C6* Altered ]
    if string(typeof(f.problem))[1:12]!="InpStiffness"
        return ElementDefGradTensorKernel(
        f.problem.E,
        f.problem.ν,
        quad,
        basef,
        f.cellvalues,
        TopOptProblems.getdim(f.problem),
    )
    else
        return ElementDefGradTensorKernel(
        f.problem.inp_content.E,
        f.problem.inp_content.ν,
        quad,
        basef,
        f.cellvalues,
        TopOptProblems.getdim(f.problem),
    )
    end
end
function tensor_kernel(f::ElementDefGradTensor, quad, basef) # --------------------------------------------------------------------------------------------------------------------[C5]
    return tensor_kernel(f.defgrad_tensor, quad, basef)
end



# function von_mises(σ::AbstractMatrix)
#     if size(σ, 1) == 2
#         t1 = σ[1, 1]^2 - σ[1, 1] * σ[2, 2] + σ[2, 2]^2
#         t2 = 3 * σ[1, 2]^2
#     elseif size(σ, 1) == 3
#         t1 = ((σ[1, 1] - σ[2, 2])^2 + (σ[2, 2] - σ[3, 3])^2 + (σ[3, 3] - σ[1, 1])^2) / 2
#         t2 = 3 * (σ[1, 2]^2 + σ[2, 3]^2 + σ[3, 1]^2)
#     else
#         throw(ArgumentError("Unsupported stress tensor type."))
#     end
#     return sqrt(t1 + t2)
# end

# function von_mises_stress_function(solver::AbstractFEASolver)
#     st = StressTensor(solver)
#     dp = Displacement(solver)
#     return x -> von_mises.(st(dp(x)))
# end






















# function FToK2AndK3(F::Vector{Matrix{Float64}})
#     k1_list=[]
#     k2_list=[]
#     k3_list=[]
#     lambda_list=[]
#     #Catches potential error of an empty F value
#     if length(F)==0
#         throw("F is empty")
#     end
#     #Catches the error of F being the wrong size
#     x=[0 0 0; 0 0 0; 0 0 0]
#     for l in 1:length(F)
#         if size(F[l])!=size(x)
#             msg = "Error at index $l: Deformation gradient shape is not (3,3)"
#             throw(ArgumentError(msg))
#         end

#         #Here begins calculations to get k1, k2, and k3
#         C=transpose(F[l])*F[l]
#         #creates object R which holds eigenvalues and eigenvectors, which are then extracted
#         R=eigen(C)
#         lam2 = R.values
#         n = R.vectors

#         Q=n
#         g=[0.0,0.0,0.0]
#         lam=sqrt.(lam2)
#         #Tries to sort lam vector, and attempts to catch errors in trying to do so 
#         try
#             lam=sort([lam[1],lam[2],lam[3]],rev=true)
#         catch
#             has_complex=false
#             for s in 1:3
#                 if lam2[s]<0
#                     has_complex=true
#                 end
#             end
#             if has_complex==true
#                 throw("Lambda values include complex numbers")
#             else 
#                 throw("Lambda values cannot be sorted")
#             end
#         end


#         kproduct=lam[1]*lam[2]*lam[3]
#         k1=log(lam[1]*lam[2]*lam[3])

#         for r in 1:3
#             g[r] = float(log(lam[r])-(k1/3))
#         end

#         k2=sqrt(g[1]^2+g[2]^2+g[3]^2)
#         k3=3*sqrt(6)*g[1]*g[2]*g[3]/(k2^3);

#         #Adds k1, k2, and k3 valus for element to the lists of values of all elements
#         push!(k1_list,k1)
#         push!(k2_list,k2)
#         push!(k3_list,k3)

#         map(Float64,k1_list)
#         map(Float64,k2_list)
#         map(Float64,k3_list)

#     end
#     return(map(Float64,k1_list),map(Float64,k2_list),map(Float64,k3_list))
# end

# function Entropy_Calc(h::Matrix{Int64})
#     #Calcuates entropy based on h matrix given
#     p=transpose(h)./(sum(h))
#     ElementWise_Entropy=(p.*log.(p))
#     temp_p=deepcopy(ElementWise_Entropy)
#     replace!(temp_p, -NaN=>0)
#     H_pwise=-sum(temp_p)
#     #Entropy of Gaussian was not considered
#     ElementWise_Entropy*=-1
#     return (H_pwise,ElementWise_Entropy)
# end   

# function Entropy(F::Vector{Vector{Matrix{Float64}}},n_bins::Int64=100,offset::Float64=0.005,make_plot::Bool=false,save_plot::Bool=false,saveplot_name::String="",saveplot_path::String="")
#        #Creates a vector of entropy values, in the case multiple time signatures are given. If not and F is of length one, returns a vector of length one  
#        #Elemetnt-Wise value returned by function retains NaN values and is therefore not able to be differenciated without further tinkering 
#        Entropy_Value_List=[]
#        ElementWise_Entropy_List=[]
#        for t in 1:length(F)
#            #Pre-processing of F vector 
#                F_copy=deepcopy(F[t])
#                for i in 1:length(F_copy)
#                    no_nan=true
#                    for q in 1:3
#                        for w=1:3
#                            if isnan(F_copy[i][q,w])==true
#                                no_nan=false
#                            end
#                        end
#                    end
#                    F_copy[i]=reshape(F_copy[i],(3,3))
#                    F_copy[i]=F_copy[i]/(det(F_copy[i])^(1/3))
#                end
   
#            #Gets vectors of k1, k2, and k3
#            k1,k2,k3=FToK2AndK3(F_copy)
#            K3_Vector=k3
#            K2_Vector=k2
   
#            #Generates 2 histogram, and the h matrix which represents the counts in matrix form
#            K2_edges = (0.0-offset, 1.0-offset)
#            K3_edges=(-1-offset, 1.0-offset) 
#            Hist_K2_Edges=K2_edges[1]:((K2_edges[2]-K2_edges[1])/n_bins):K2_edges[2]
#            Hist_K3_Edges=K3_edges[1]:((K3_edges[2]-K3_edges[1])/n_bins):K3_edges[2]
#            Hist_Matrix=fit(Histogram, (map(Float64,K3_Vector), map(Float64,K2_Vector)), (Hist_K2_Edges, Hist_K3_Edges))
#        #Below line defines bins differently, in a way which doesn't quite work but it worth keep around
#            #Hist_Matrix=fit(Histogram,(map(Float64,K2_Vector),map(Float64,K3_Vector)),nbins=(n_bins,n_bins))
#            h=Hist_Matrix.weights
#            #Makes plots, and subsequently saves plots if inputs are set ot do so 
#            if make_plot==true
#        #Below line creates figure slightly differently, using a method that is not the same as what was used to get h
#                # hist = histogram2d(K2_Vector, K3_Vector, nbins=n_bins, color=:viridis, xlims=K2_edges, ylims=K3_edges, background="black")
#                p=(heatmap(h))
#                title!(p,"K2 v. K3");
#                xlabel!(p,"K2");
#                ylabel!(p,"K3");
#             #    xlims!(p,(K2_edges[1], K2_edges[2]));
#             #    ylims!(p,(K3_edges[1], K3_edges[2]));
#                display(p)
#                if save_plot==true
#                    name=string(saveplot_name,".png")
#                    path=(string(saveplot_path,name))
#                    savefig(p,path)
#                end
#            end
#        (Entropy_Value,ElementWise_Entropy)=Entropy_Calc(h)
#        push!(Entropy_Value_List,Entropy_Value)    
#        push!(ElementWise_Entropy_List,ElementWise_Entropy)        
#        end
#        if length(Entropy_Value_List)==1
#            Entropy_Value_List=Entropy_Value_List[1]
#            ElementWise_Entropy_List=ElementWise_Entropy_List[1]
#        end
#        return (map(Float64,Entropy_Value_List),map(Float64,ElementWise_Entropy_List))
#    end
   
   


#    function Entropy(F::Vector{Matrix{Float64}},n_bins::Int64=100,offset::Float64=0.005,make_plot::Bool=false,save_plot::Bool=false,saveplot_name::String="",saveplot_path::String="")
#     #Creates a vector of entropy values, in the case multiple time signatures are given. If not and F is of length one, returns a vector of length one  
#     F=[F]
#     Entropy_Value_List=[]
#     ElementWise_Entropy_List=[]
#     for t in 1:length(F)
#         #Pre-processing of F vector 
#             F_copy=deepcopy(F[t])
#             for i in 1:length(F_copy)
#                 no_nan=true
#                 for q in 1:3
#                     for w=1:3
#                         if isnan(F_copy[i][q,w])==true
#                             no_nan=false
#                         end
#                     end
#                 end
#                 F_copy[i]=reshape(F_copy[i],(3,3))
#                 F_copy[i]=F_copy[i]/(det(F_copy[i])^(1/3))
#             end

#         #Gets vectors of k1, k2, and k3
#         k1,k2,k3=FToK2AndK3(F_copy)
#         K3_Vector=k3
#         K2_Vector=k2

#         #Generates 2 histogram, and the h matrix which represents the counts in matrix form
#         K2_edges = (0.0-offset, 1.0-offset)
#         K3_edges=(-1-offset, 1.0-offset) 
#         Hist_K2_Edges=K2_edges[1]:((K2_edges[2]-K2_edges[1])/n_bins):K2_edges[2]
#         Hist_K3_Edges=K3_edges[1]:((K3_edges[2]-K3_edges[1])/n_bins):K3_edges[2]
#         Hist_Matrix=fit(Histogram, (map(Float64,K3_Vector), map(Float64,K2_Vector)), (Hist_K2_Edges, Hist_K3_Edges))
#     #Below line defines bins differently, in a way which doesn't quite work but it worth keep around
#         #Hist_Matrix=fit(Histogram,(map(Float64,K2_Vector),map(Float64,K3_Vector)),nbins=(n_bins,n_bins))
#         h=Hist_Matrix.weights
#         #Makes plots, and subsequently saves plots if inputs are set ot do so 
#         if make_plot==true
#     #Below line creates figure slightly differently, using a method that is not the same as what was used to get h
#             # hist = histogram2d(K2_Vector, K3_Vector, nbins=n_bins, color=:viridis, xlims=K2_edges, ylims=K3_edges, background="black")
#             p=(heatmap(h))
#             title!(p,"K2 v. K3");
#             xlabel!(p,"K2");
#             ylabel!(p,"K3");
#             # xlims!(p,(K2_edges[1], K2_edges[2]));
#             # ylims!(p,(K3_edges[1], K3_edges[2])); #Error on this line??
#             display(p)
#             if save_plot==true
#                 name=string(saveplot_name,".png")
#                 path=(string(saveplot_path,name))
#                 savefig(p,path)
#             end
#         end
#     (Entropy_Value,ElementWise_Entropy)=Entropy_Calc(h)
#     push!(Entropy_Value_List,Entropy_Value)    
#     push!(ElementWise_Entropy_List,ElementWise_Entropy)        
#     end
#     if length(Entropy_Value_List)==1
#         Entropy_Value_List=Entropy_Value_List[1]
#         ElementWise_Entropy_List=ElementWise_Entropy_List[1]
#     end
#     return (map(Float64,Entropy_Value_List),map.(Float64,ElementWise_Entropy_List))
# end


# function SMu_gen(k2_list::Vector{Float64},k3_list::Vector{Float64},alpha::Float64=2.0,mu::Float64=1.0)
#     #Takes in a list of k2 and k3 values, and returns both total and element-wise sensitivity in terms of mu
#     SMu_List=[]
#     for i in 1:length(k3_list)
#         k3=k3_list[i]
#         k2=k2_list[i]
#         g1 = sqrt(2/3)*sin(-asin(k3)/3+(2*pi/3))
#         g2 = sqrt(2/3)*sin(-asin(k3)/3)
#         g3 = sqrt(2/3)*sin(-asin(k3)/3-(2*pi/3))

#         val1=g1*exp(g1*alpha*k2)
#         val2=g2*exp((g2*alpha*k2))
#         val3=g3*exp((g3*alpha*k2))
#         SMu=val1+val2+val3

#         push!(SMu_List,SMu)
#     end
#     Sum=sum(SMu_List)
#     return map(Float64,SMu_List), Sum
# end

# function SAlpha_gen(k2_list::Vector{Float64},k3_list::Vector{Float64},alpha::Float64=2.0,mu::Float64=1.0)
#     #Takes in a list of k2 and k3 values, and returns both total and element-wise sensitivity in terms of Alpha
#     SAlpha_List=[]
#     for i in 1:length(k3_list)
#         k3=k3_list[i]
#         k2=k2_list[i]

#         g1 = sqrt(2/3)*sin(-asin(k3)/3+(2*pi/3))
#         g2 = sqrt(2/3)*sin(-asin(k3)/3)
#         g3 = sqrt(2/3)*sin(-asin(k3)/3-(2*pi/3))

#         val1=(g1^2)*k2*(exp(g1*alpha*k2))
#         val2=(g2^2)*k2*(exp(g2*alpha*k2))
#         val3=(g3^2)*k2*(exp(g3*alpha*k2))

#         SAlpha=mu*(val1+val2+val3)
#         SAlpha_List=vcat(SAlpha_List,[SAlpha])
#     end
#     Sum=sum(SAlpha_List)
#     return (map(Float64,SAlpha_List)), Sum
# end

# # function Sensitivity(F::Vector{Matrix{Float64}},alpha::Float64=2.0,mu::Float64=1.0)
