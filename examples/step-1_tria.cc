// ---------------------------------------------------------------------
//
// Copyright (C) 2004 - 2015 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------



// check GridTools::partition_triangulation using ZOLTAN as partitioner
// Test 1 (metis_01.cc) of metis is used as model for this test

#include "../tests.h"
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/sparsity_tools.h>


#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <vector>



template <int dim>
void test ()
{
  static Triangulation<dim> triangulation;
  GridGenerator::hyper_cube (triangulation);
  triangulation.refine_global (5);
  
  //get connectivity matrix
  DynamicSparsityPattern cell_connectivity;
  GridTools::get_face_connectivity_of_cells (triangulation, cell_connectivity);
  
  SparsityPattern sp_cell_connectivity;
  sp_cell_connectivity.copy_from(cell_connectivity);
  
  std::vector<int> color_indices (triangulation.n_active_cells());
  SparsityTools::color_sparsity_pattern (sp_cell_connectivity, color_indices);
  
  //color
    for (auto cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell) 
      cell->set_subdomain_id (color_indices[cell->active_cell_index()]);

  // subdivides into 5 subdomains
  deallog << "Partitioning" << std::endl;
  for (auto cell = triangulation.begin_active(); cell != triangulation.end(); ++cell)
    deallog << cell << ' ' << cell->subdomain_id() << std::endl;  
  
  //print triangulation
  Vector<float> subdomains(triangulation.n_active_cells());
  
  //dof handler
  DoFHandler<dim>        dof_handler(triangulation);
  static const FE_Q<dim> finite_element(1);             //need for DoFHandler. bilinear element for quads
  dof_handler.distribute_dofs (finite_element);         //distribute DoF's
  
  //data out
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);  
  
  //loop over cells
  unsigned int cell_index = 0;
  for (auto cell: dof_handler.active_cell_iterators())
    { 
       if (cell->is_locally_owned())
           subdomains(cell_index) = cell->subdomain_id();
      cell_index++;
    }

      data_out.add_data_vector (subdomains, "partion_id");
      data_out.build_patches ();
      std::ofstream output ("solution.vtu");
      data_out.write_vtu (output);

}



int main (int argc, char **argv)
{
  initlog();

  //Initialize MPI and Zoltan
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);

  //tests
//   test<1> ();
  test<2> ();
//   test<3> ();
  }
