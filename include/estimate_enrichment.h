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
#include <deal.II/lac/solver_cg.h>  //TODO REMOVE
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include <deal.II/base/logstream.h>

using namespace dealii;

//TODO remove template and make it for 1d
template <int dim>
class EstimateEnrichmentFunction
{
public:
  EstimateEnrichmentFunction (Point<dim> center,
                              double domain_size,
                              double sigma,
                              double coeff);
  EstimateEnrichmentFunction  (Point<dim> center,
                               double left_bound,
                               double right_bound,
                               double sigma,
                               double coeff,
                               double refinement=7);
  ~EstimateEnrichmentFunction();
  void run ();
  void evaluate_at_x_values
  (std::vector< double >  &interpolation_points,
   std::vector< double >   &interpolation_values);
  double value (const Point<dim> &p);
private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void refine_grid();
  void output_results () const;
  Point<dim> center;
  double domain_size;
  double left_bound, right_bound;
  double sigma;
  double coeff;
public:
  unsigned int debug_level;
private:
  Triangulation<dim>   triangulation;
  unsigned int refinement;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double>       solution;
  Vector<double>       system_rhs;
};
