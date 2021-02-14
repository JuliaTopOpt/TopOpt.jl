@testset "Element stiffness matrix" begin

    # 1x1 linear quadrilateral cell
    # Isoparameteric
    # 2nd order quadrature rule

    E = 1
    nu = 0.3
    k = [1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8]
    Ke = E/(1-nu^2)*[k[1] k[2] k[3] k[4] k[5] k[6] k[7] k[8];
                     k[2] k[1] k[8] k[7] k[6] k[5] k[4] k[3];
                     k[3] k[8] k[1] k[6] k[7] k[4] k[5] k[2];
                     k[4] k[7] k[6] k[1] k[8] k[3] k[2] k[5];
                     k[5] k[6] k[7] k[8] k[1] k[2] k[3] k[4];
                     k[6] k[5] k[4] k[3] k[2] k[1] k[8] k[7];
                     k[7] k[4] k[5] k[2] k[3] k[8] k[1] k[6];
                     k[8] k[3] k[2] k[5] k[4] k[7] k[6] k[1]]
    
    problem = HalfMBB((2,2), (1.,1.), 1., 0.3, 1.);

    #@test ElementFEAInfo(problem).Kes[1] ≈ Ke
end
