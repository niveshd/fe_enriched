// ---------------------------------------------------------------------
//
// Copyright (C) 2016 - 2017 by the deal.II authors
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


/*
 * Test function - ColorEnriched::internal::set_cellwise_color_set_and_fe_index
 * for a set of predicates.
 * Check for each cell, if appropriate fe index and color-index map is set.
 * Color-index map associates different colors of different enrichment
 * functions with corresponding enrichment function index.
 */

#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/hp/fe_collection.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_enriched.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/slepc_solver.h>

#include <map>
#include "../tests.h"
#include "helper.h"

const unsigned int dim = 2;


int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  MPILogInitAll all;

  //Make basic grid
  Triangulation<dim>  triangulation;
  hp::DoFHandler<dim> dof_handler(triangulation);
  GridGenerator::hyper_cube (triangulation, -2, 2);
  triangulation.refine_global (2);

  //Make predicates. Predicate 0 and 1 overlap.
  //Predicate 2 is connected to 0.
  std::vector<EnrichmentPredicate<dim>> vec_predicates;
  vec_predicates.push_back
  ( EnrichmentPredicate<dim>(Point<dim>(0,1), 1) );
  vec_predicates.push_back
  ( EnrichmentPredicate<dim>(Point<dim>(-1,1), 1) );
  vec_predicates.push_back
  ( EnrichmentPredicate<dim>(Point<dim>(1.5,-1.5), 1) );

  //Do manual coloring since we are not testing coloring function here!
  std::vector<unsigned int> predicate_colors;
  predicate_colors.resize(vec_predicates.size());
  predicate_colors[0] = 1;
  predicate_colors[1] = 2;
  predicate_colors[2] = 1;

  //Make required objects to call function set_cellwise_color_set_and_fe_index
  std::map<unsigned int,
      std::map<unsigned int, unsigned int>> cellwise_color_predicate_map;
  std::vector <std::set<unsigned int>> fe_sets;
  ColorEnriched::internal::set_cellwise_color_set_and_fe_index
  (dof_handler,
   vec_predicates,
   predicate_colors,
   cellwise_color_predicate_map,
   fe_sets);

  /*
   * Run through active cells to check fe index, colors of
   * enrichment functions associated with it.
   *
   * A unique color set corresponds to an fe index.
   *
   * Eg: If an fe index 1 corresponds to color set {2,3},
   * means that a cell with fe index 1 has enrichment functions
   * which are colored 2 and 3. Here different enrichment function
   * have same color 2 but for a given cell only one of them would
   * be relevant. So all additional information we need is which
   * enrichment function is relevant that has color 2. We need to
   * do the same thing with color 2 as well.
   *
   * Each cell is assigned unique material id by the function
   * set_cellwise_color_set_and_fe_index. Now using material id,
   * each cell is associated with a map which assigns a color to a
   * particular enrichment function id.
   *
   */
  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (unsigned int cell_index=0; cell != endc; ++cell, ++cell_index)
    {
      //print true predicates for a cell as a binary code
      //0 1 0 indicates predicate with index 2 is true in the cell.
      deallog << cell->index() << ":predicates=";
      for (auto predicate : vec_predicates)
        deallog << predicate(cell) << ":";

      //print color predicate pairs for a cell
      //Note that here material id is used to identify cells
      //Here (1,2) indicates predicate 2 of color 1 is relavant for cell.
      deallog << "(color, enrichment_func_id):";
      for (auto color_predicate_pair :
           cellwise_color_predicate_map[cell->material_id()])
        {
          deallog << "("
                  << color_predicate_pair.first << ","
                  << color_predicate_pair.second << "):";
        }

      //For a cell, print fe active index and corresponding fe set.
      //{1,2} indicates 2 enrichment functions of color 1 and 2 are relevant.
      deallog << ":fe_active_index:" << cell->active_fe_index() << ":fe_set:";
      for (auto fe_set_element : fe_sets[cell->active_fe_index()])
        deallog << fe_set_element << ":";
      deallog << std::endl;
    }

  return 0;
}
