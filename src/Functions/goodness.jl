function FToK2AndK3(F::Vector{Matrix{Float64}})
    k1_list=Float64[]
    k2_list=Float64[]
    k3_list=Float64[]
    lambda_list=Float64[]
    #Catches potential error of an empty F value
    #if length(F)==0
    #    throw("F is empty")
    #end

    function Fto3by3(F)
        return Vector{Matrix{Float64}}(map(eachindex(F)) do i
            F_tmp = F[i]
            if size(F_tmp) == (2, 2)
                # Build an immutable 3x3 matrix from the 2x2 F_item
                [F_tmp zeros(2, 1); 0 0 1]
            elseif size(F_tmp) == (3, 3)
                F_tmp
            else
                error("Unexpected deformation gradient size at index $l")
            end
        end)
    end
    F3by3 = Fto3by3(F) # 3 by 3

    for l in 1:length(F3by3)       
        #Here begins calculations to get k1, k2, and k3
        C=transpose(F3by3[l])*F3by3[l]  
        #creates object R which holds eigenvalues and eigenvectors, which are then extracted
        R=eigen(C)
        lam2 = R.values
        n = R.vectors

        Q=n
        g=[0.0,0.0,0.0]
        lam=sqrt.(lam2)

        #Tries to sort lam vector, and attempts to catch errors in trying to do so
        #Not necessary 
        # try
        #     lam=sort([lam[1],lam[2],lam[3]],rev=true)
        # catch
        #     has_complex=false
        #     for s in 1:3
        #         if lam2[s]<0
        #             has_complex=true
        #         end
        #     end
        #     if has_complex==true
        #         throw("Lambda values include complex numbers")
        #     else 
        #         throw("Lambda values cannot be sorted")
        #     end
        # end


        kproduct=lam[1]*lam[2]*lam[3]
        k1=log(lam[1]*lam[2]*lam[3])

        # for r in 1:3
        #     g[r] = float(log(lam[r])-(k1/3))
        # end

        g1=float(log(lam[1])-(k1/3))
        g2=float(log(lam[2])-(k1/3))
        g3=float(log(lam[3])-(k1/3))
        g=[g1,g2,g3]

        k2=sqrt(g[1]^2+g[2]^2+g[3]^2)
        k3=3*sqrt(6)*g[1]*g[2]*g[3]/(k2^3)

        #Adds k1, k2, and k3 valus for element to the lists of values of all elements
        k1_list=vcat(k1_list,k1)
        k2_list=vcat(k2_list,k2)
        k3_list= vcat(k3_list,k3)

    end
    return(map(Float64,k1_list),map(Float64,k2_list),map(Float64,k3_list))
end

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

# function Sensitivity(F::Vector{Matrix{Float64}},alpha::Float64=2.0,mu::Float64=1.0)

function Sensitivity(x,dispfun,defgrad)
    sim1_u= dispfun(x)
    F = defgrad(sim1_u);
    k1,k2,k3=FToK2AndK3(F)
    S=SAlpha_gen(k2,k3);
    Sens_Corrected=ceil(x).*S[1]
    return Sens_Corrected
end