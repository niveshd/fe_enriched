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

#include <map>

const unsigned int dim = 2;

//uncomment when debugging
// #define DATA_OUT_FE_ENRICHED

unsigned int patches = 10;

template <int dim>
void plot_shape_function
(hp::DoFHandler<dim> &dof_handler)
{
  deallog << "n_cells: "<< dof_handler.get_triangulation().n_active_cells()<<std::endl;

  ConstraintMatrix constraints;
  constraints.clear();
  dealii::DoFTools::make_hanging_node_constraints  (dof_handler, constraints);
  constraints.close ();

  // output to check if all is good:
  std::vector<Vector<double>> shape_functions;
  std::vector<std::string> names;
  for (unsigned int s=0; s < dof_handler.n_dofs(); s++)
    {
      Vector<double> shape_function;
      shape_function.reinit(dof_handler.n_dofs());
      shape_function[s] = 1.0;

      // if the dof is constrained, first output unconstrained vector
      if (constraints.is_constrained(s))
        {
          names.push_back(std::string("UN_") +
                          dealii::Utilities::int_to_string(s,2));
          shape_functions.push_back(shape_function);
        }

      names.push_back(std::string("N_") +
                      dealii::Utilities::int_to_string(s,2));

      // make continuous/constrain:
      constraints.distribute(shape_function);
      shape_functions.push_back(shape_function);
    }

  DataOut<dim,hp::DoFHandler<dim>> data_out;
  data_out.attach_dof_handler (dof_handler);

  // get material ids:
  Vector<float> fe_index(dof_handler.get_triangulation().n_active_cells());
  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active (),
  endc = dof_handler.end ();
  for (unsigned int index=0; cell!=endc; ++cell,++index)
    {
      fe_index[index] = cell->active_fe_index();
    }
  data_out.add_data_vector(fe_index, "fe_index");

  for (unsigned int i = 0; i < shape_functions.size(); i++)
    data_out.add_data_vector (shape_functions[i], names[i]);

  data_out.build_patches(patches);

  std::string filename = "hp-shape_functions_"
                         +dealii::Utilities::int_to_string(dim)+"D.vtu";
  std::ofstream output (filename.c_str ());
  data_out.write_vtu (output);
}


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
  vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(-1,1), 1) );
  vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(1.5,-1.5), 1) );

  //vector of enrichment functions
  std::vector<EnrichmentFunction<dim>> vec_enrichments;
  vec_enrichments.reserve( vec_predicates.size() );
  for (unsigned int i=0; i<vec_predicates.size(); ++i)
    {
      EnrichmentFunction<dim> func(10+i);  //constant function
      vec_enrichments.push_back( func );
    }

  //find colors for predicates
  predicate_colors.resize(vec_predicates.size());
  unsigned int num_colors
    = color_predicates (triangulation, vec_predicates, predicate_colors);

  std::map<unsigned int,
      std::map<unsigned int, unsigned int> > cellwise_color_predicate_map;
  std::vector <std::set<unsigned int>> fe_sets;

  set_cellwise_color_set_and_fe_index
  (dof_handler,
   vec_predicates,
   predicate_colors,
   cellwise_color_predicate_map,
   fe_sets);

  std::vector<
  std::function<const Function<dim>*
  (const typename Triangulation<dim>::cell_iterator &)> >
  color_enrichments;

  make_colorwise_enrichment_functions
  (num_colors,          //needs number of colors
   vec_enrichments,     //enrichment functions based on predicate id
   cellwise_color_predicate_map,
   color_enrichments);

  FE_Q<dim> fe_base(2);
  FE_Q<dim> fe_enriched(1);
  FE_Nothing<dim> fe_nothing(1,true);
  hp::FECollection<dim> fe_collection;
  make_fe_collection_from_colored_enrichments
  (num_colors,
   fe_sets,         //total list of color sets possible
   color_enrichments,  //color wise enrichment functions
   fe_base,            //basic fe element
   fe_enriched,        //fe element multiplied by enrichment function
   fe_nothing,
   fe_collection);

  deallog << "fe sets:" << std::endl;
  for (auto fe_set : fe_sets)
    {
      deallog << "color:";
      for (auto color : fe_set)
        deallog << ":" << color;
      deallog << std::endl;
    }

  deallog << "fe_collection[index] mapping:" << std::endl;
  for (unsigned int index = 0; index != fe_collection.size(); ++index)
    {
      deallog <<"name:"<<fe_collection[index].get_name() << std::endl;
      deallog <<"n_blocks:"<<fe_collection[index].n_blocks()<<std::endl;
      deallog <<"n_comp:"<< fe_collection[index].n_components() << std::endl;
      deallog <<"n_dofs:"<< fe_collection[index].n_dofs_per_cell() << std::endl;
    }

  GridTools::partition_triangulation
  (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
   triangulation);
  dof_handler.distribute_dofs(fe_collection);

#ifdef DATA_OUT_FE_ENRICHED
  plot_shape_function<dim>(dof_handler);
#endif

  dof_handler.clear();
  return 0;
}