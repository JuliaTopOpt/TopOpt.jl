#= Orthogonal decomposition function to convert F to an invariant basis (K₁, K₂, K₃) for isotropic hyperelastic strain (Criscione JC et al. JMPS, 2000)
K = FToK123(F) and K = orth_decomp(F) 
In the current formulation orth_decomp is not autodiff compatible, but FToK123(F) ≈ orth_decomp(F)
Benchmarking suggests that FToK123 is also about twice as fast as orth_decomp
K is of dims a Vector{Vector{T}} of dim [1:3][1:nels]
Outputs: K = [K₁ K₂ K₃] where each term is a vector of length nels =#

function FToK123(F::AbstractMatrix{T}) where {T}
    @assert size(F) == (3, 3)
    g = Vector{T}(undef,3)
      
    C = transpose(F)*F  
    λ = sqrt.(eigen(C).values)

    K₁ = log(λ[1]*λ[2]*λ[3])
    g = log.(λ) .- (K₁/3)
    K₂ = sqrt(sum(g.^2))
    K₃ = 3*sqrt(6)*g[1]*g[2]*g[3]/(K₂^3)
    return [K₁ K₂ K₃]
end

function FToK123(F::AbstractVector{<:AbstractMatrix{T}}) where {T}
    nels = length(F)
    Kel = map(i -> FToK123(F[i]), 1:nels)
    K = [[Kel[i][j] for i in 1:nels] for j in 1:3]
    return K
end

function orth_decomp(F::AbstractMatrix{T}) where {T}
    @assert size(F) == (3, 3)
    K = Vector{T}(undef,3)
    
    J = det(F)
    B = F*transpose(F)
    V = sqrt(B)
    η = log(V)
    devη = η - (1/3)*(tr(η))*Matrix(I,3,3)

    K[1] = log(J)
    K[2] = sqrt(tr(devη^2))
    Φ = devη/K[2]
    K[3] = 3*sqrt(6)*det(Φ)
    return K # warning: final K values may need to be modified to ensure they are within physical limits (e.g., K₃ ∈ [-1 1])
end

function orth_decomp(F::AbstractVector{<:AbstractMatrix{T}}) where {T}
    nels = length(F)
    Kel =  map(i -> orth_decomp(F[i]), 1:nels)
    K = [[Kel[i][j] for i in 1:nels] for j in 1:3] # transposing the vector of vectors to yield K[1:3][el]
    return K
end

#= Function that produces stress and kinematic terms used to construct the virtual fields
stressTerms, kinTerms = sensitivityFieldFunctions(:ConstitutiveModel) 
stressTerms = fns[1]
kinTerms = fns[2]
Outputs: stressTerms[i,j] (i.e. ∂²W(K₁,K₂,K₃)/∂Kᵢ∂ξⱼ) and kinTerms[i,j,k] (this is equivalent to d^3W/dK_i/dXi_j/dK_kk) =#

function sensitivityFieldFncs(matlModel::Symbol)
    @variables α K[1:3] # K[1:3] are the orthogonal strain invariants
    λᵅ = Array{Any}(undef,3)
    λᵅ[1] = exp(α*K[2]*sqrt(2/3)*sin((-asin(K[3])+2π)/3))
    λᵅ[2] = exp(α*K[2]*sqrt(2/3)*sin((-asin(K[3]))/3))
    λᵅ[3] = exp(α*K[2]*sqrt(2/3)*sin((-asin(K[3])-2π)/3))

    # construct the work function for called constitutive model, W
    Ī₁ = sum(substitute(λᵅ[i], Dict(α => 2)) for i in 1:3)
    bulk = (exp(K[1])-1)^2 # equivalent to (J-1)²
    if matlModel in (:MooneyRivlin, :MR) # note: bulk modulus (κ) must be the final parameter in ξ for all matlModel
        ξ = @variables C₁₀ C₀₁ κ
        Ī₂ = sum(substitute(λᵅ[i], Dict(α => -2)) for i in 1:3)
        W = C₁₀*(Ī₁-3) + C₀₁*(Ī₂-3) + (κ/2)*bulk
    elseif matlModel in (:NeoHookean, :NH)
        ξ = @variables µ κ
        W = (µ/2)*(Ī₁-3) + (κ/2)*bulk
    elseif matlModel in (:Yeoh2, :Y2)
        ξ = @variables C₁₀ C₂₀ κ
        W = C₁₀*(Ī₁-3) + C₂₀*(Ī₁-3)^2 + (κ/2)*bulk
    elseif matlModel in (:Yeoh3, :Y3)
        ξ = @variables C₁₀ C₂₀ C₃₀ κ
        W = C₁₀*(Ī₁-3) + C₂₀*(Ī₁-3)^2 + C₃₀*(Ī₁-3)^3 + (κ/2)*bulk
    end

    # solve for terms used to in stress and kinematic field calculations
    stressTerms = Array{Any}(undef,length(K),length(ξ))
    for i = 1:length(K)
        for j = 1:length(ξ)
            ∂Kᵢ∂ξⱼ = Differential(K[i])*Differential(ξ[j]) # ∂²/∂Kᵢ∂ξⱼ
            if i == 1 && j == length(ξ)
                stressTerms[i,j] = eval(build_function(expand_derivatives(∂Kᵢ∂ξⱼ(W)),K[1])) # ∂²W(K₁)/∂K₁∂κ
            elseif i != 1 && j != length(ξ)
                stressTerms[i,j] = eval(build_function(expand_derivatives(∂Kᵢ∂ξⱼ(W)),K[2],K[3])) # ∂²W(K₂,K₃)/∂Kᵢ∂ξⱼ
            end
        end
    end
    kinTerms = Array{Any}(undef,length(K),length(ξ),length(K))
    for i = 1:length(K)
        for j = 1:length(ξ)
            for k = 1:length(K)
                ∂Kᵢ∂ξⱼ∂Kₖ = Differential(K[i])*Differential(ξ[j])*Differential(K[k]) # ∂³/∂Kᵢ∂ξⱼ∂Kₖ
                if i == 1 && j == length(ξ) && k == 1 
                    kinTerms[i,j,k] = eval(build_function(expand_derivatives(∂Kᵢ∂ξⱼ∂Kₖ(W)),K[1])) # ∂³W(K₁)/∂²K₁∂κ
                elseif i != 1 && j != length(ξ) && k != 1 
                    kinTerms[i,j,k] = eval(build_function(expand_derivatives(∂Kᵢ∂ξⱼ∂Kₖ(W)),K[2],K[3])) # ∂³W(K₂,K₃)/∂Kᵢ∂ξⱼ∂Kₖ
                end
            end
        end
    end
    return stressTerms, kinTerms
end

# Function that produces a volume-weighted element-wise sensitivity metric
# sens_IVW = sensitivity_PVW(x,K,V0,stressTerms,kinTerms,ξ)
    # x is the 
    # K₁, K₂, and K₃ are all vectors of size [nels] (acquired from FtoK123 or orth_decomp)
    # Where matl_props is a vector, e.g. for Mooney-Rivlin matl_props could be [0.04 0.001 0.5]
    # Vol is a vector of size [nels] that contains information of the volume for each element
    # stressTerms and kinTerms are acquired from sensitivityFieldFunctions
        # This function only needs to be run once!

# Outputs
# sens_IVW (sensitivity metric across all elements)

function sensitivityPVW(x,K,V0,stressTerms,kinTerms,ξ)
    # Stress term evaluation at K
    ξᵢ∂²W_∂K₁∂ξᵢ = ξ[end]*stressTerms[1,end].(K[1]) 
    ξᵢ∂²W_∂K₂∂ξᵢ = sum(map(i -> ξ[i]*stressTerms[2,i].(K[2],K[3]), 1:length(ξ)-1))
    ξᵢ∂²W_∂K₃∂ξᵢ = sum(map(i -> ξ[i]*stressTerms[3,i].(K[2],K[3]), 1:length(ξ)-1))

    # Internal virtual work (IVW) evaluation
    IVW_bar = zeros(length(x))
    for i = 1:length(ξ) # ξᵢ
        for j = 1:length(K) # Kⱼ
            if i == length(ξ) && j == 1
                IVW_bar += 3*V0.*(x.^2).*ξᵢ∂²W_∂K₁∂ξᵢ.*kinTerms[1,i,j].(K[1])
            elseif i != length(ξ) && j != 1 
                IVW_bar += V0.*(x.^2).*(ξᵢ∂²W_∂K₂∂ξᵢ.*kinTerms[2,i,j].(K[2],K[3]) + (9*(ones(length(x))-(K[3].^2))./(K[2].^2)).*ξᵢ∂²W_∂K₃∂ξᵢ.*kinTerms[3,i,j].(K[2],K[3]))
            end
        end
    end
    return IVW_bar/sum(V0.*x) # Volume-weighted element-wise sensitivity metric of all IVW fields
end

# NOTE: Functions below are legacy material. They should work properly, however, they are less well vetted than the sensitivity related functions.

function Entropy_Calc(h::Matrix{Int64})
    #Calcuates entropy based on h matrix given
    p=transpose(h)./(sum(h))
    ElementWise_Entropy=(p.*log.(p))
    temp_p=deepcopy(ElementWise_Entropy)
    replace!(temp_p, -NaN=>0)
    H_pwise=-sum(temp_p)
    #Entropy of Gaussian was not considered
    ElementWise_Entropy*=-1
    return (H_pwise,ElementWise_Entropy)
end   

function Entropy(F::Vector{Vector{Matrix{Float64}}},n_bins::Int64=100,offset::Float64=0.005,make_plot::Bool=false,save_plot::Bool=false,saveplot_name::String="",saveplot_path::String="")
       #Creates a vector of entropy values, in the case multiple time signatures are given. If not and F is of length one, returns a vector of length one  
       #Elemetnt-Wise value returned by function retains NaN values and is therefore not able to be differenciated without further tinkering 
       Entropy_Value_List=[]
       ElementWise_Entropy_List=[]
       for t in 1:length(F)
           #Pre-processing of F vector 
               F_copy=deepcopy(F[t])
               for i in 1:length(F_copy)
                   no_nan=true
                   for q in 1:3
                       for w=1:3
                           if isnan(F_copy[i][q,w])==true
                               no_nan=false
                           end
                       end
                   end
                   F_copy[i]=reshape(F_copy[i],(3,3))
                   F_copy[i]=F_copy[i]/(det(F_copy[i])^(1/3))
               end
   
           #Gets vectors of k1, k2, and k3
           k1,k2,k3=FToK2AndK3(F_copy)
           K3_Vector=k3
           K2_Vector=k2
   
           #Generates 2 histogram, and the h matrix which represents the counts in matrix form
           K2_edges = (0.0-offset, 1.0-offset)
           K3_edges=(-1-offset, 1.0-offset) 
           Hist_K2_Edges=K2_edges[1]:((K2_edges[2]-K2_edges[1])/n_bins):K2_edges[2]
           Hist_K3_Edges=K3_edges[1]:((K3_edges[2]-K3_edges[1])/n_bins):K3_edges[2]
           Hist_Matrix=fit(Histogram, (map(Float64,K3_Vector), map(Float64,K2_Vector)), (Hist_K2_Edges, Hist_K3_Edges))
       #Below line defines bins differently, in a way which doesn't quite work but it worth keep around
           #Hist_Matrix=fit(Histogram,(map(Float64,K2_Vector),map(Float64,K3_Vector)),nbins=(n_bins,n_bins))
           h=Hist_Matrix.weights
           #Makes plots, and subsequently saves plots if inputs are set ot do so 
           if make_plot==true
       #Below line creates figure slightly differently, using a method that is not the same as what was used to get h
               # hist = histogram2d(K2_Vector, K3_Vector, nbins=n_bins, color=:viridis, xlims=K2_edges, ylims=K3_edges, background="black")
               p=(heatmap(h))
               title!(p,"K2 v. K3");
               xlabel!(p,"K2");
               ylabel!(p,"K3");
            #    xlims!(p,(K2_edges[1], K2_edges[2]));
            #    ylims!(p,(K3_edges[1], K3_edges[2]));
               display(p)
               if save_plot==true
                   name=string(saveplot_name,".png")
                   path=(string(saveplot_path,name))
                   savefig(p,path)
               end
           end
       (Entropy_Value,ElementWise_Entropy)=Entropy_Calc(h)
       push!(Entropy_Value_List,Entropy_Value)    
       push!(ElementWise_Entropy_List,ElementWise_Entropy)        
       end
       if length(Entropy_Value_List)==1
           Entropy_Value_List=Entropy_Value_List[1]
           ElementWise_Entropy_List=ElementWise_Entropy_List[1]
       end
       return (map(Float64,Entropy_Value_List),map(Float64,ElementWise_Entropy_List))
   end
   
   function Entropy(F::Vector{Matrix{Float64}},n_bins::Int64=100,offset::Float64=0.005,make_plot::Bool=false,save_plot::Bool=false,saveplot_name::String="",saveplot_path::String="")
    #Creates a vector of entropy values, in the case multiple time signatures are given. If not and F is of length one, returns a vector of length one  
    F=[F]
    Entropy_Value_List=[]
    ElementWise_Entropy_List=[]
    for t in 1:length(F)
        #Pre-processing of F vector 
            F_copy=deepcopy(F[t])
            for i in 1:length(F_copy)
                no_nan=true
                for q in 1:3
                    for w=1:3
                        if isnan(F_copy[i][q,w])==true
                            no_nan=false
                        end
                    end
                end
                F_copy[i]=reshape(F_copy[i],(3,3))
                F_copy[i]=F_copy[i]/(det(F_copy[i])^(1/3))
            end

        #Gets vectors of k1, k2, and k3
        k1,k2,k3=FToK2AndK3(F_copy)
        K3_Vector=k3
        K2_Vector=k2

        #Generates 2 histogram, and the h matrix which represents the counts in matrix form
        K2_edges = (0.0-offset, 1.0-offset)
        K3_edges=(-1-offset, 1.0-offset) 
        Hist_K2_Edges=K2_edges[1]:((K2_edges[2]-K2_edges[1])/n_bins):K2_edges[2]
        Hist_K3_Edges=K3_edges[1]:((K3_edges[2]-K3_edges[1])/n_bins):K3_edges[2]
        Hist_Matrix=fit(Histogram, (map(Float64,K3_Vector), map(Float64,K2_Vector)), (Hist_K2_Edges, Hist_K3_Edges))
    #Below line defines bins differently, in a way which doesn't quite work but it worth keep around
        #Hist_Matrix=fit(Histogram,(map(Float64,K2_Vector),map(Float64,K3_Vector)),nbins=(n_bins,n_bins))
        h=Hist_Matrix.weights
        #Makes plots, and subsequently saves plots if inputs are set ot do so 
        if make_plot==true
    #Below line creates figure slightly differently, using a method that is not the same as what was used to get h
            # hist = histogram2d(K2_Vector, K3_Vector, nbins=n_bins, color=:viridis, xlims=K2_edges, ylims=K3_edges, background="black")
            p=(heatmap(h))
            title!(p,"K2 v. K3");
            xlabel!(p,"K2");
            ylabel!(p,"K3");
            # xlims!(p,(K2_edges[1], K2_edges[2]));
            # ylims!(p,(K3_edges[1], K3_edges[2])); #Error on this line??
            display(p)
            if save_plot==true
                name=string(saveplot_name,".png")
                path=(string(saveplot_path,name))
                savefig(p,path)
            end
        end
    (Entropy_Value,ElementWise_Entropy)=Entropy_Calc(h)
    push!(Entropy_Value_List,Entropy_Value)    
    push!(ElementWise_Entropy_List,ElementWise_Entropy)        
    end
    if length(Entropy_Value_List)==1
        Entropy_Value_List=Entropy_Value_List[1]
        ElementWise_Entropy_List=ElementWise_Entropy_List[1]
    end
    return (map(Float64,Entropy_Value_List),map.(Float64,ElementWise_Entropy_List))
end

function SMu_gen(k2_list::Vector{Float64},k3_list::Vector{Float64},alpha::Float64=2.0,mu::Float64=1.0)
    #Takes in a list of k2 and k3 values, and returns both total and element-wise sensitivity in terms of mu
    SMu_List=[]
    for i in 1:length(k3_list)
        k3=k3_list[i]
        k2=k2_list[i]
        g1 = sqrt(2/3)*sin(-asin(k3)/3+(2*pi/3))
        g2 = sqrt(2/3)*sin(-asin(k3)/3)
        g3 = sqrt(2/3)*sin(-asin(k3)/3-(2*pi/3))

        val1=g1*exp(g1*alpha*k2)
        val2=g2*exp((g2*alpha*k2))
        val3=g3*exp((g3*alpha*k2))
        SMu=val1+val2+val3

        push!(SMu_List,SMu)
    end
    Sum=sum(SMu_List)
    return map(Float64,SMu_List), Sum
end

function SAlpha_gen(k2_list::Vector{Float64},k3_list::Vector{Float64},alpha::Float64=2.0,mu::Float64=1.0)
    #Takes in a list of k2 and k3 values, and returns both total and element-wise sensitivity in terms of Alpha
    SAlpha_List=[]
    for i in 1:length(k3_list)
        k3=k3_list[i]
        k2=k2_list[i]

        g1 = sqrt(2/3)*sin(-asin(k3)/3+(2*pi/3))
        g2 = sqrt(2/3)*sin(-asin(k3)/3)
        g3 = sqrt(2/3)*sin(-asin(k3)/3-(2*pi/3))

        val1=(g1^2)*k2*(exp(g1*alpha*k2))
        val2=(g2^2)*k2*(exp(g2*alpha*k2))
        val3=(g3^2)*k2*(exp(g3*alpha*k2))

        SAlpha=mu*(val1+val2+val3)
        SAlpha_List=vcat(SAlpha_List,[SAlpha])
    end
    Sum=sum(SAlpha_List)
    return (map(Float64,SAlpha_List)), Sum
end