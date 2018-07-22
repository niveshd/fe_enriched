#set up instructions. Note that all the instructions in setup are with respect to git repository home: "fe_enriched" 
mkdir build
cd build
#ensure that DEAL_II_DIR is correctly set to dealII build directory with the correct dealii version
#petsc should be installed
#see pull request https://github.com/dealii/dealii/pull/6277
cmake ..	
make

     
#for debugging p3m constraint line error with 50 items use
build/src/step tests/step/multiple_source_enriched.prm tests/step/atom_n3_50.data
#generates triangulation_debug.vtk file with a patch of cells around cell with a particular problematic dof (8611)
#the function relevant is debug_constraint() in the include/solver_trilinos.h
#the vtk file labels based fe indices, (if cells have dof 8611) and material id
#run the problem in debug mode to ensure the same dof gives the constraint line error
#initial tests reveal that both the cells cotaining dof (8611) have same fe index, material id!


#Unit tests.
cd build/tests/step && ctest



#The main executable is:
build/src/step



#usage with parameter files
#Note: one or two additional parameter files can be given to the main exectuable
#single parameter file is given. The file includes solver parameters and optionally number of sources, enrichment functions, boundary conditions.
#example usage with two source poisson's problem
build/src/step tests/step/two_source_no_overlap.prm
#other example parameter files are in tests/step directory:
single_source_known_solution_enriched.prm
single_source_known_solution.prm
two_source_overlap.prm
two_source_no_overlap.prm
three_source.prm
five_source.prm
five_source_3d.prm

#2. two parameter files are given inorder to solve P3M problems. 
#The first file gives same information as above. but the information about number of sources and their position is ignored. e.g. "tests/step/multiple_source.prm"
#The second file gives the number of sources, their charges and their position. e.g. "tests/step/atom_n3_25.data" with 25 atoms
build/src/step tests/step/multiple_source_enriched.prm tests/step/atom_n3_25.data
#for without enrichement use:
build/src/step tests/step/multiple_source.prm tests/step/atom_n3_25.data



