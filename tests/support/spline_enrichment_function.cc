// ..........................................-
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
// ..........................................-


#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/grid/grid_out.h>

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
#include <deal.II/base/parameter_handler.h>

#include <string>
#include <sstream>
#include <fstream>

#include "support.h"
#include "estimate_enrichment.h"
#include "solver.h"
#include "../tests.h"

const unsigned int dim = 2;

template <int dim>
class Problem : public Step1::LaplaceProblem<dim>
{
public:
  Problem (int argc,char **argv):
    Step1::LaplaceProblem<dim>(argc, argv) {}

  void run_pre_solution_steps()
  {
    this->build_fe_space();
    this->setup_system();

#ifdef DATA_OUT
    plot_shape_function<dim>(this->dof_handler);
#endif
  }

private:
  void make_enrichment_function();
};


template <int dim>
void Problem<dim>::make_enrichment_function ()
{
  this->pcout << "!!! Inherited make enrichment function called" << std::endl;

  //1.original function
  for (unsigned int i=0; i<this->vec_predicates.size(); ++i)
    {
      //formulate a 1d problem with x coordinate and radius (i.e sigma)
      double x = this->points_enrichments[i][0];
      double sigma = this->radii_enrichments[i];
      EstimateEnrichmentFunction<1> problem_1d(Point<1>(x),
                                               sigma,
                                               50);
      problem_1d.run();

      //make points at which solution needs to interpolated
      std::vector<double> interpolation_points_1D, interpolation_values_1D;
      double factor = 2;
      interpolation_points_1D.push_back(0);
      for (double x = 0.25; x < 2*sigma; x*=factor)
        interpolation_points_1D.push_back(x);
      interpolation_points_1D.push_back(2*sigma);

      problem_1d.evaluate_at_x_values(interpolation_points_1D,interpolation_values_1D);
      std::cout << "solved problem with "
                << "(x, sigma): "
                << x << ", " << sigma << std::endl;

      //construct enrichment function and push
      EnrichmentFunction<dim> func(this->points_enrichments[i],
                                   this->radii_enrichments[i],
                                   interpolation_points_1D,
                                   interpolation_values_1D);
      this->vec_enrichments.push_back(func);

      double step = 0.1;
      AssertDimension (dim,2);
      for (double x=0; x <= 1; x+=step)
        {
          Point<dim> p(x,0);
          std::cout << x << " "
                    << func.value(p) << " "
                    << func.value(p)*(1-x) << " "
                    << 1-x << " "
                    << func.value(p)*(x) << " "
                    << x
                    << std::endl;
        }
    }



//  //2.const enrichment functions!
//  for (unsigned int i=0; i<this->vec_predicates.size(); ++i)
//    {
//      EnrichmentFunction<dim> func(Point<2> (0,0),
//                                   2,
//                                   0);  //constant function
//      this->vec_enrichments.push_back( func );
//    }



//  //3.function
//  for (unsigned int i=0; i<this->vec_predicates.size(); ++i)
//    {
//      //formulate a 1d problem with x coordinate and radius (i.e sigma)
//      double x = this->points_enrichments[i][0];
//      double sigma = this->radii_enrichments[i];

//      //make points at which solution needs to interpolated
//      std::vector<double> interpolation_points_1D, interpolation_values_1D;
//      double factor = 2;
//      interpolation_points_1D.push_back(0);
//      for (double x = 0.25; x < 2*sigma; x*=factor)
//        interpolation_points_1D.push_back(x);
//      interpolation_points_1D.push_back(2*sigma);

//      for (auto y: interpolation_points_1D)
//        interpolation_values_1D.push_back(1-y*y);

//      //construct enrichment function and push
//      EnrichmentFunction<dim> func(this->points_enrichments[i],
//                                   this->radii_enrichments[i],
//                                   interpolation_points_1D,
//                                   interpolation_values_1D);
//      this->vec_enrichments.push_back(func);
//    }
}



int main (int argc,char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  {
    Problem<dim> step_test(argc,argv);
    step_test.run_pre_solution_steps();
  }
}
