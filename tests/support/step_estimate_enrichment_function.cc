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

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <iostream>

#include "../tests.h"
#include "estimate_enrichment.h"
#include "functions.h"
#include "support.h"

/*
 * Verify if enrichment function is correctly calculated.
 *
 * Calculate error by solving an equivalent radial problem
 * and then compare with exact solution.
 *
 * Calculate spline approximation of the radial solution
 * and compare with exact solution.
 */

using namespace dealii;

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv);
  MPILogInitAll all;

  // calculate vector of points at which solution needs to be interpolated
  double sigma = 0.1;
  double size = 2;
  double center = 0;

  // solve 1d problem
  std::string rhs_expr = "-(exp(-(x*x)/(2*sigma*sigma))*(-2*sigma*sigma + "
                         "x*x))/(2*sigma*sigma*sigma*sigma*sigma*sigma*pi)";
  std::string boundary_expr =
    "1.0/(2*pi*sigma*sigma)*exp(-(x*x)/(2*sigma*sigma))";
  Step1::EstimateEnrichmentFunction radial_problem(
    Point<1>(center), size, sigma, rhs_expr, boundary_expr, 10);
  radial_problem.run();

  /*
   * Evaluate value of the solution at the sample points
   * and approximate using a spline function
   *
   * Divide domain into two regions, center to center + cut point;
   * center + cut point to boundary. The first region is smaller
   * but needs to have smaller step size to capture the variation.
   */
  std::vector<double> interpolation_points, interpolation_values;
  double cut_point = 3 * sigma;
  unsigned int n1 = 15, n2 = 15;
  double radius = size / 2;
  double right_bound = center + radius;
  double h1 = cut_point / n1, h2 = (radius - cut_point) / n2;
  for (double p = center; p < center + cut_point; p += h1)
    interpolation_points.push_back(p);
  for (double p = center + cut_point; p < right_bound; p += h2)
    interpolation_points.push_back(p);
  interpolation_points.push_back(right_bound);
  radial_problem.evaluate_at_x_values(interpolation_points,
                                      interpolation_values);

  Step1::SigmaFunction<1> exact_solution;
  std::string exact_soln_expr(boundary_expr);
  exact_solution.initialize(Point<1>(), sigma, exact_soln_expr);

  // Test if 1d solution is correct
  double exact_value, error;
  for (unsigned int i = 0; i != interpolation_points.size(); ++i)
    {
      exact_value = exact_solution.value(Point<1>(interpolation_points[i]));
      error = abs(exact_value - interpolation_values[i]);
      deallog << "Error in radial solution < 1e-3::" << (error < 1e-3)
              << std::endl;
    }

  // cspline function within enrichment function
  SplineEnrichmentFunction<2> spline_approx(Point<2>(), interpolation_points,
                                            interpolation_values);

  // check how well spline approximates function
  double max_error = 0;
  std::ofstream file("spline_accuracy.data", std::ios::out);
  double h = size / 100;
  for (double x = 0; x < size / 2; x += h)
    {
      double approx_value = spline_approx.value(Point<2>(x, 0));
      double exact_value = exact_solution.value(Point<1>(x));
      double error = approx_value - exact_value;

      file << x << "\t" << approx_value << "\t" << exact_value << "\t" << error
           << std::endl;

      max_error = (max_error < error) ? error : max_error;
    }

  std::cout << max_error << std::endl;
  deallog << "Max spline approx. error < 1e-2 : " << (max_error < 1e-2)
          << std::endl;
  file.close();
  return 0;
}
