@testset "Metadata" begin
    #HalfMBB - 2x2

    # 7   8   9              
    #  x--x--x      x--x--x
    #  |  |  |      |C3|C4|
    #  x--x--x      x--x--x
    # 4| 5|  |6     |C1|C2| 
    #  x--x--x      x--x--x
    # 1   2   3              

    #Node coords:
    #JuAFEM.Node{2,Float64}([0.0, 0.0])
    #JuAFEM.Node{2,Float64}([1.0, 0.0])
    #JuAFEM.Node{2,Float64}([2.0, 0.0])
    #JuAFEM.Node{2,Float64}([0.0, 1.0])
    #JuAFEM.Node{2,Float64}([1.0, 1.0])
    #JuAFEM.Node{2,Float64}([2.0, 1.0])
    #JuAFEM.Node{2,Float64}([0.0, 2.0])
    #JuAFEM.Node{2,Float64}([1.0, 2.0])
    #JuAFEM.Node{2,Float64}([2.0, 2.0])

    #Cells
    #JuAFEM.Cell{2,4,4}((1, 2, 5, 4))
    #JuAFEM.Cell{2,4,4}((2, 3, 6, 5))
    #JuAFEM.Cell{2,4,4}((4, 5, 8, 7))
    #JuAFEM.Cell{2,4,4}((5, 6, 9, 8))

    # 7   8   9              
    #  x--x--x      x--x--x
    #  |  |  |      |C3|C4|
    #  x--x--x      x--x--x
    # 4| 5|  |6     |C1|C2| 
    #  x--x--x      x--x--x
    # 1   2   3              

    #Node dofs
    ##1 #2 #3  #4 #5 #6  #7  #8  #9
    #1  3   9  7  5  11  15  13  17
    #2  4  10  8  6  12  16  14  18

    #Cell dofs
    # #1  #2  #3  #4
    # 1   3   7   5
    # 2   4   8   6
    # 3   9   5  11
    # 4  10   6  12
    # 5  11  13  17
    # 6  12  14  18
    # 7   5  15  13
    # 8   6  16  14

    problem = HalfMBB((2,2), (1.,1.), 1., 0.3, 1.);

    coords = [(0.0, 0.0),
              (1.0, 0.0),
              (2.0, 0.0),
              (0.0, 1.0),
              (1.0, 1.0),
              (2.0, 1.0),
              (0.0, 2.0),
              (1.0, 2.0),
              (2.0, 2.0)]

    for (i, node) in enumerate(problem.ch.dh.grid.nodes)
        @test node.x.data == coords[i]
    end

    cells = [JuAFEM.Cell{2,4,4}((1, 2, 5, 4)),
        JuAFEM.Cell{2,4,4}((2, 3, 6, 5)),
        JuAFEM.Cell{2,4,4}((4, 5, 8, 7)),
        JuAFEM.Cell{2,4,4}((5, 6, 9, 8))]
    @test problem.ch.dh.grid.cells == cells

    node_dofs = [1 3  9 7 5 11 15 13 17;
                 2 4 10 8 6 12 16 14 18]
    @test problem.metadata.node_dofs == node_dofs

    cell_dofs = [1  3  7 5;
                 2  4  8 6;
                 3  9  5 11;
                 4 10  6 12;
                 5 11 13 17;
                 6 12 14 18;
                 7  5 15 13;
                 8  6 16 14]
    @test problem.metadata.cell_dofs == cell_dofs

    offsets = problem.metadata.dof_cells_offset
    dof_cells = problem.metadata.dof_cells
    cell_dofs = problem.metadata.cell_dofs
    for i in 1:JuAFEM.ndofs(problem.ch.dh)
        r = offsets[i] : offsets[i+1] - 1
        for j in r
            (cellid, localdof) = dof_cells[j]
            @test i == cell_dofs[localdof, cellid]
        end
    end

    node_first_cells = [(1, 1),
                        (1, 2),
                        (2, 2),
                        (1, 4),
                        (1, 3),
                        (2, 3),
                        (3, 4),
                        (3, 3),
                        (4, 3)]
    @test problem.metadata.node_first_cells == node_first_cells

end
