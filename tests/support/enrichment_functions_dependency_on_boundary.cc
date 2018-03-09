#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/base/logstream.h>

#include "estimate_enrichment.h"
#include "support.h"
#include "../tests.h"
#include <sstream>
#include <algorithm>

using namespace dealii;

int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  MPILogInitAll all;
//  deallog.precision(2);

  double sigma = 0.05;

  //make vector determining the combinations used
  std::vector<double> left_bound = {-50,-60,-70,-80};
  std::vector<double> right_bound = {50,60,70,40};
  std::vector<double> center = {0,0,0,0};
  AssertDimension(left_bound.size(), center.size());
  AssertDimension(left_bound.size(), right_bound.size());

  for (unsigned int i=0; i!=center.size(); ++i)
    {

      //calculate vector of points at which solution needs to be interpolated
      //make points at which solution needs to be evaluated
      std::vector<double> interpolation_points, interpolation_values;
      for (double x = left_bound[i];
           x < right_bound[i];
           x+=0.1)
        interpolation_points.push_back(x);
      interpolation_points.push_back(right_bound[i]);
      interpolation_values.resize(interpolation_points.size());



      //solve 1d problem
      Step1::EstimateEnrichmentFunction<1> problem_1d
      (Point<1>(center[i]),
       left_bound[i],
       right_bound[i],
       sigma,
       "1.0/(2*pi*sigma*sigma)*exp(-(x*x)/(2*sigma*sigma))",
       "0",
       7);
      std::cout << "Solving with size,origin: " << left_bound[i]
                << "," << right_bound[i]
                << "," << center[i] << std::endl;
      problem_1d.run();

      //evaluate solution at values
      problem_1d.evaluate_at_x_values(interpolation_points,interpolation_values);
      double max_value = *max_element(interpolation_values.begin(),
                                      interpolation_values.end());

      //print points as output with sigma and coefficient
      std::stringstream name;
      name << "left_bound" << left_bound[i]
           << "right_bound" << right_bound[i]
           << "center" << center[i] << ".csv";
      std::ofstream file(name.str(), std::ios::out);
      file << "x value" << std::endl;
      for (unsigned int i=0; i!=interpolation_points.size(); ++i)
        {
          //output normalized values to file
          file << interpolation_points[i] << " "
               << interpolation_values[i]/max_value << std::endl;
          deallog << interpolation_points[i] << " "
                  << interpolation_values[i] << std::endl;
        }
      file.close();
    }
  return 0;
}
