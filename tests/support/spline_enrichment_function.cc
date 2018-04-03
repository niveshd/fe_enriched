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



//inherit from solver class to test just the pre-assembly steps
template <int dim>
class Problem : public Step1::LaplaceProblem<dim>
{
public:
  using Step1::LaplaceProblem<dim>::LaplaceProblem;

  void run_pre_solution_steps()
  {
    this->initialize();
    this->build_fe_space();
    this->setup_system();

    if (this->prm.debug_level == 9)
      plot_shape_function<dim>(this->dof_handler);
  }

private:
  void make_enrichment_functions();
};



//modify enrichment function to be used!
template <int dim>
void Problem<dim>::make_enrichment_functions ()
{
  this->pcout << "!!! Inherited make enrichment function called" << std::endl;


  //1.original function
  for (unsigned int i=0; i<this->vec_predicates.size(); ++i)
    {
      //formulate a 1d problem with x coordinate and radius (i.e sigma)
      //always solve problem with
      //center = 0, domain size = 100, sigma = 0.05, coefficient 100, refinement 7
      double center = 0;
      double domain_size = 100;
      double sigma = 0.05;
      Step1::EstimateEnrichmentFunction<1> problem_1d
      (Point<1>(center),
       domain_size,
       sigma,
       "1.0/(2*pi*sigma*sigma)*exp(-(x*x)/(2*sigma*sigma))",
       "0",
       11);
      problem_1d.run();


      //make points at which solution needs to interpolated
      std::vector<double> interpolation_points_1D, interpolation_values_1D;
      double range = 2;
      double cut_point = 0.6;
      unsigned int n1 = 3, n2 = 3;
      double right_bound = center + range;
      double h1 = cut_point/n1, h2 = (range-cut_point)/n2;
      for (double p = center; p < cut_point; p+=h1)
        interpolation_points_1D.push_back(p);
      for (double p = cut_point; p < right_bound; p+=h2)
        interpolation_points_1D.push_back(p);
      interpolation_points_1D.push_back(right_bound);

      problem_1d.evaluate_at_x_values(interpolation_points_1D,interpolation_values_1D);
      this->pcout << "solved problem with "
                  << "(x, sigma): "
                  << center << ", " << sigma << std::endl;


      //construct enrichment function and push
      SplineEnrichmentFunction<dim> func(this->prm.points_enrichments[i],
                                         this->prm.radii_predicates[i],
                                         interpolation_points_1D,
                                         interpolation_values_1D);
      this->vec_enrichments.push_back(func);


      //check how well spline approximates function
      {
        double max_error=0;
        std::ofstream file("spline_accuracy.csv", std::ios::out);
        double h = sigma/100;
        file << "x spline_value solution_value relative_error" << std::endl;
        for (double x=0; x < right_bound; x+=h)
          {
            double spline_value = func.value(Point<2> (x,0));
            double solution_value = problem_1d.value(Point<1> (x));
            double relative_error = (spline_value != 0) ? (spline_value-solution_value)/spline_value :
                                    0;
            file << x
                 << " " << spline_value
                 << " " << solution_value
                 << " " << relative_error
                 << std::endl;
            max_error = (max_error < relative_error)?
                        relative_error :
                        max_error;
          }
        this->pcout << "Max error due to spline approximation: " << max_error << std::endl;
        file.close();
      }


      //print shape function values after enrichment
      double step = 0.1;
      AssertDimension (dim,2);
      for (double x=0; x <= 2; x+=step)
        {
          Point<dim> p(x,0);
          deallog << x << ":"
                  << func.value(p) << ":"
                  << func.value(p)*(1-x) << ":"
                  << 1-x << ":"
                  << func.value(p)*(x) << ":"
                  << x
                  << std::endl;
        }
    }


  //TODO push these out to separate tests
//  //2.const enrichment functions!
//  for (unsigned int i=0; i<this->vec_predicates.size(); ++i)
//    {
//      SplineEnrichmentFunction<dim> func(Point<2> (0,0),
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
//      SplineEnrichmentFunction<dim> func(this->points_enrichments[i],
//                                   this->radii_enrichments[i],
//                                   interpolation_points_1D,
//                                   interpolation_values_1D);
//      this->vec_enrichments.push_back(func);
//    }
}


//change right hand side value function

//change what 1d problem solves

int main (int argc,char **argv)
{
  /**
   * Run initial steps of Laplace problem with enrichment.
   * A simple enrichment function is chosen and shape function
   * values are checked.
   */

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll all;
  {
    Problem<dim> step_test
    (4,   //domain size
     1,   //cube shape
     2,   //global refinement
     0,
     1,   //fe base degree
     1,   //fe enriched degree
     10000, //max iterations
     1e-8,  //tolerance
     "1.0/(2*pi*sigma*sigma)*exp(-(x*x + y*y)/(2*sigma*sigma))",
     "0",
     "1.0/(2*pi*sigma*sigma)*exp(-(x*x)/(2*sigma*sigma))",
     "0",
     "",
     true,
     15,   //patches
     9,   //debug level
     1,   //num enrichments
    {Point<dim>()}, //enrichment points
    {1},    //predicate radii
    {0.05}); //sigmas

    step_test.run_pre_solution_steps();
  }
}
