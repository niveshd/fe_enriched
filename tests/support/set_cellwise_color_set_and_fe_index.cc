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

#include "../tests.h"
#include "support.h"
#include "helper.h"

#include <map>

const unsigned int dim = 2;

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  MPILogInitAll all;

  Triangulation<dim>  triangulation;
  hp::DoFHandler<dim> dof_handler(triangulation);
  GridGenerator::hyper_cube (triangulation, -2, 2);
  triangulation.refine_global (2);

  std::vector<EnrichmentPredicate<dim>> vec_predicates;
  std::vector<unsigned int> predicate_colors;

  //connections between vec predicates : 0-1 (overlap connection),
  vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(0,1), 1) );
  vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(-1,1), 1) );
  vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(1.5,-1.5), 1) );

  predicate_colors.resize(vec_predicates.size());

  predicate_colors[0] = 1;
  predicate_colors[1] = 2;
  predicate_colors[2] = 1;

  std::map<unsigned int,
      std::map<unsigned int, unsigned int> > cellwise_color_predicate_map;
  std::vector <std::set<unsigned int>> fe_sets;

  ColorEnriched::internal::set_cellwise_color_set_and_fe_index
  (dof_handler,
   vec_predicates,
   predicate_colors,
   cellwise_color_predicate_map,
   fe_sets);

  auto cell = dof_handler.begin_active();
  auto endc = dof_handler.end();
  for (unsigned int cell_index=0; cell != endc; ++cell, ++cell_index)
    {
      //print true predicates for a cell as a binary code
      deallog << cell->index() << ":predicates=";
      for (auto predicate : vec_predicates)
        deallog << predicate(cell) << ":";

      //print color predicate pairs for a cell
      deallog << "(color, enrichment_func_id):";
      for (auto color_predicate_pair : cellwise_color_predicate_map[cell_index])
        {
          deallog << "("
                  << color_predicate_pair.first << ","
                  << color_predicate_pair.second << "):";
        }
      //print fe active index and fe set for a cell.
      //{1,2} indicates 2 enrichment functions of color 1 and 2
      deallog << ":fe_active_index:" << cell->active_fe_index() << ":fe_set:";
      for (auto fe_set_element : fe_sets[cell->active_fe_index()])
        deallog << fe_set_element << ":";
      deallog << std::endl;
    }

  return 0;
}
