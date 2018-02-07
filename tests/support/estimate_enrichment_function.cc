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

using namespace dealii;

template <int dim>
class PoissonSolver
{
public:
  PoissonSolver (Point<dim> center,
                 double domain_size,
                 double sigma,
                 double coeff);
  void run ();
  void evaluate_at_x_values
  (std::vector< double >  &interpolation_points,
   std::vector< double >   &interpolation_values);
  double evaluate (const Point<dim> &p);
private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void output_results () const;
  Point<dim> center;
  double domain_size;
  double sigma;
  double coeff;
  Triangulation<dim>   triangulation;
  unsigned int refinement;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       solution;
  Vector<double>       system_rhs;
};


template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide (Point<dim> center, double sigma, double coeff=1)
    : Function<dim>(),
      center(center),
      sigma(sigma),
      coeff(coeff)
  {}
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
  Point<dim> center;
  double sigma;
  double coeff;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
  double return_value = 0.0;
  return_value = coeff*exp(-p.distance_square(center)/(sigma*sigma));
  return return_value;
}



template <int dim>
class BoundaryValues: public Function<dim>
{
public:
  BoundaryValues () : Function<dim>() {}
  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
  return 0; //TODO is the boundary value 0? just use constant function!
}



template <int dim>
PoissonSolver<dim>::PoissonSolver
(Point<dim> center,
 double domain_size,
 double sigma,
 double coeff)
  :
  center(center),
  domain_size(domain_size),
  sigma(sigma),
  coeff(coeff),
  refinement(5),
  fe (1),
  dof_handler (triangulation)
{}
template <int dim>
void PoissonSolver<dim>::make_grid ()
{
  //TODO domain 50 times original radius enough?
  GridGenerator::hyper_ball (triangulation, center, domain_size/2);
  triangulation.set_all_manifold_ids_on_boundary(0);
  static SphericalManifold<dim> spherical_manifold(center);
  triangulation.set_manifold(0,spherical_manifold);

  triangulation.refine_global (5);
}

template <int dim>
void PoissonSolver<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit (sparsity_pattern);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}

template <int dim>
void PoissonSolver<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(2);
  const RightHandSide<dim> right_hand_side(center,sigma,coeff);
  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);
  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();
  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  for (const auto &cell: dof_handler.active_cell_iterators())
    {
      fe_values.reinit (cell);
      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                                   fe_values.shape_grad (j, q_index) *
                                   fe_values.JxW (q_index));
            cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                            right_hand_side.value (fe_values.quadrature_point (q_index)) *
                            fe_values.JxW (q_index));
          }
      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }
  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            BoundaryValues<dim>(),
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}
template <int dim>
void PoissonSolver<dim>::solve ()
{
  SolverControl           solver_control (10000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());
}

template <int dim>
void PoissonSolver<dim>::output_results () const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();
  std::ofstream output ("solution-2d.vtk");
  data_out.write_vtk (output);
}

template <int dim>
void PoissonSolver<dim>::run ()
{
//  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
  make_grid();

  double old_value=0, value=1, relative_change = 1;
  int cycles = 0;
  bool start = true;
  do
    {
      triangulation.refine_global (1);
      ++refinement;
      ++cycles;
      setup_system ();
      assemble_system ();
      solve ();

      value = VectorTools::point_value(dof_handler, solution, Point<dim>(0,0));
      if (!start)
        {
          relative_change = fabs((old_value - value)/old_value);
        }
      start = false;
      old_value = value;
    }
  while (relative_change > 0.001);

  std::cout << "2D solution at origin = "
            << value
            << " after global refinement "
            << refinement
            << std::endl;

  output_results ();
}



template <int dim>
void PoissonSolver<dim>::evaluate_at_x_values
(std::vector< double >  &interpolation_points,
 std::vector< double >   &interpolation_values)
{
  AssertDimension(interpolation_values.size(),0);
  for (auto x:interpolation_points)
    {
      double value = VectorTools::point_value(dof_handler, solution, Point<dim>(x,0));
      interpolation_values.push_back(value);
    }
}


template <int dim>
double PoissonSolver<dim>::evaluate
(const Point<dim> &p)
{
  return VectorTools::point_value(dof_handler, solution, p);
}


int main (int argc, char **argv)
{
  //calculate vector of points at which solution needs to be interpolated
  double sigma = 0.5;
  double domain_size = 100;
  double coeff = 2;
  std::vector<double> interpolation_points, interpolation_values_2D;
  double factor = 2;
  interpolation_points.push_back(0);
  for (double x = 0.25; x < 2*sigma; x*=factor)
    interpolation_points.push_back(x);
  interpolation_points.push_back(2*sigma);
//  for (double x = 10*sigma; x < right_bound; x+=10*sigma)
//     interpolation_points.push_back(x);
//  interpolation_points.push_back(right_bound);

  //solve 2d problem
  PoissonSolver<2> problem_2d(Point<2>(0,0), //center and size
                              domain_size,
                              sigma,        //right hand side
                              coeff);
  problem_2d.run ();
  problem_2d.evaluate_at_x_values(interpolation_points,interpolation_values_2D);

  //solve 1d problem
  EstimateEnrichmentFunction<1> problem_1d(Point<1>(0),
                                           domain_size,
                                           sigma,
                                           coeff);
  problem_1d.run();
  std::vector<double> interpolation_values_1D;
  problem_1d.evaluate_at_x_values(interpolation_points,interpolation_values_1D);

  //Test if 1d and 2d solutions match
  AssertDimension(interpolation_values_1D.size(),interpolation_values_2D.size());
  for (unsigned int i = 0; i != interpolation_points.size(); ++i)
    {
      double error = interpolation_values_1D[i] - interpolation_values_2D[i];
      deallog << "error_ok::" << (error<1e-3) << std::endl;
    }


  //cspline function within enrichment function
  EnrichmentFunction<2> func(Point<2>(),
                             sigma,
                             interpolation_points,
                             interpolation_values_1D);


  //check how well spline approximates function
  double max_error=0;
  std::ofstream file("spline_accuracy.data", std::ios::out);
  double h = sigma/100;
  for (double x=0; x <= 2*sigma; x+=h)
    {
      double func_value = func.value(Point<2> (x,0));
      double prob_value = problem_1d.value(Point<1> (x));
      double relative_error = (func_value != 0) ? (func_value-prob_value)/func_value :
                              0;
      file << x
           << " " << func_value
           << " " << prob_value
           << " " << relative_error
           << std::endl;
      max_error = (max_error < relative_error)?
                  relative_error :
                  max_error;
    }
  std::cout << "Max error due to spline approximation: " << max_error << std::endl;
  file.close();

  return 0;
}
