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


//Test the function ColorEnriched::internal::color_predicates.
//Check if the function correctly colors vector of predicates
//using graph coloring.
//
//Two predicates are said to be connected if cells belonging to
//different predicates touch each other.

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

template <int dim>
using predicate_function = std::function< bool
                           (const typename hp::DoFHandler<dim>::cell_iterator &) >;

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  MPILogInitAll all;

  //Make basic grid
  Triangulation<dim>  triangulation;
  GridGenerator::hyper_cube (triangulation, -20, 20);
  triangulation.refine_global (4);
  hp::DoFHandler<dim> dof_handler(triangulation);

  //check the coloring function on different set of predicates.
  std::vector<predicate_function<dim>> vec_predicates;
  std::vector<unsigned int> predicate_colors;
  {
    //case 1: predicates are not connected
    vec_predicates.clear();
    vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(-10,10), 2) );
    vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(0,0), 2) );

    predicate_colors.resize(vec_predicates.size());

    ColorEnriched::internal::color_predicates<dim>
    (dof_handler, vec_predicates, predicate_colors);

    deallog << "Case 1" << std::endl;
    for (auto i:predicate_colors)
      {
        deallog << i << std::endl;
      }
  }

  {
    //case 2: Two predicates that are connected.
    vec_predicates.clear();
    vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(-10,10), 2) );
    vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(-7.5,7.5), 2) );

    predicate_colors.resize(vec_predicates.size());

    ColorEnriched::internal::color_predicates
    (dof_handler, vec_predicates, predicate_colors);

    deallog << "Case 2" << std::endl;
    for (auto i:predicate_colors)
      {
        deallog << i << std::endl;
      }
  }

  {
    //case 3: connections between predicates is as follows -
    //0-1 (overlap connection),
    //3-4 (edge connection)
    vec_predicates.clear();
    vec_predicates.push_back
    ( EnrichmentPredicate<dim>(Point<dim>(-10,10), 2) );
    vec_predicates.push_back
    ( EnrichmentPredicate<dim>(Point<dim>(-7.5,7.5), 2) );
    vec_predicates.push_back
    ( EnrichmentPredicate<dim>(Point<dim>(0,0), 2) );
    vec_predicates.push_back
    ( EnrichmentPredicate<dim>(Point<dim>(7.5,-7.5), 2) );
    vec_predicates.push_back
    ( EnrichmentPredicate<dim>(Point<dim>(12.5,-12.5), 2) );

    predicate_colors.resize(vec_predicates.size());

    ColorEnriched::internal::color_predicates
    (dof_handler, vec_predicates, predicate_colors);

    deallog << "Case 3" << std::endl;
    for (auto i:predicate_colors)
      {
        deallog << i << std::endl;
      }
  }
  return 0;
}
