#ifndef ESTIMATE_ENRICHMENT_H
#define ESTIMATE_ENRICHMENT_H

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <fstream>
#include <functions.h>
#include <iostream>
#include <math.h>

using namespace dealii;

namespace Step1
{
  /*
   * EstimateEnrichmentFunction is used to estimate enrichment function by
   * solveing a 1D poisson problem with right hand side and boundary
   * expression provided as a string of single variable 'x', to be
   * interpreted as distance from @par center.
   *
   * Eg: For a 2D poisson problem with right hand side expression R and boundary
   * expression B given by functions R( x^2 + y^2) and B( x^2 + y^2 ), an
   * equivalent radial problem can be solved using this class by setting
   * @par rhs_expr = R( x^2)
   * @par boundary_expr = B (x^2)
   * Note that the original poisson problem is defined by right hand side
   * and boundary expression dependent on square of distance (x^2 + y^2)
   * from center.
   */
  class EstimateEnrichmentFunction
  {
  public:
    EstimateEnrichmentFunction(const Point<1> &center, const double &domain_size,
                               const double &sigma, const std::string &rhs_expr,
                               const std::string &boundary_expr,
                               const std::map<std::string,double> &constants={},
                               const double &refinement = 11);
    EstimateEnrichmentFunction(const Point<1> &center, const double &left_bound,
                               const double &right_bound, const double &sigma,
                               const std::string &rhs_expr,
                               const std::string &boundary_expr,
                               const std::map<std::string,double> &constants={},
                               const double &refinement = 11);
    ~EstimateEnrichmentFunction();
    void run();
    void evaluate_at_x_values(std::vector<double> &interpolation_points,
                              std::vector<double> &interpolation_values);
    double value(const Point<1> &p, const unsigned int &component = 0);

  private:
    void make_grid();
    void setup_system();
    void assemble_system();
    void solve();
    void refine_grid();
    void output_results() const;
    Point<1> center;
    double domain_size;
    double left_bound, right_bound;
    double sigma;
    std::string rhs_expr;
    std::string boundary_expr;
    const std::map<std::string,double> constants;
    std::vector<double> rhs_values;

  public:
    unsigned int debug_level;

  private:
    Triangulation<1> triangulation;
    unsigned int refinement;
    FE_Q<1> fe;
    DoFHandler<1> dof_handler;
    SparsityPattern sparsity_pattern;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> system_rhs;
  };

  EstimateEnrichmentFunction::EstimateEnrichmentFunction(
    const Point<1> &center, const double &domain_size, const double &sigma,
    const std::string &rhs_expr, const std::string &boundary_expr,
    const std::map<std::string,double> &constants,
    const double &refinement)
    : center(center), domain_size(domain_size), sigma(sigma),
      rhs_expr(rhs_expr), boundary_expr(boundary_expr),
      constants(constants),
      debug_level(0),
      refinement(refinement), fe(1), dof_handler(triangulation)
  {
    left_bound = center[0] - domain_size / 2;
    right_bound = center[0] + domain_size / 2;
  }

  EstimateEnrichmentFunction::EstimateEnrichmentFunction(
    const Point<1> &center, const double &left_bound, const double &right_bound,
    const double &sigma, const std::string &rhs_expr,
    const std::string &boundary_expr,
    const std::map<std::string,double> &constants,
    const double &refinement)
    : center(center), left_bound(left_bound), right_bound(right_bound),
      sigma(sigma), rhs_expr(rhs_expr), boundary_expr(boundary_expr),
      constants(constants),
      debug_level(0), refinement(refinement), fe(1),
      dof_handler(triangulation)
  {
    domain_size = right_bound - left_bound;
  }

  void EstimateEnrichmentFunction::make_grid()
  {
    GridGenerator::hyper_cube(triangulation, left_bound, right_bound);
    triangulation.refine_global(refinement);
  }

  void EstimateEnrichmentFunction::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  void EstimateEnrichmentFunction::assemble_system()
  {
    QGauss<1> quadrature_formula(2);
    SigmaFunction<1> rhs;
    rhs.initialize(center, sigma, rhs_expr,constants);
    FEValues<1> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                          update_quadrature_points | update_JxW_values);
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);
        cell_matrix = 0;
        cell_rhs = 0;
        rhs_values.resize(n_q_points);
        rhs.value_list(fe_values.get_quadrature_points(), rhs_values);

        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
          {
            double radius = center.distance(fe_values.quadrature_point(q_index));

            //-1/r (r*u_r) = f form converts to
            // r(u_r, v_r) = (r*f,v)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  cell_matrix(i, j) += (radius * fe_values.shape_grad(i, q_index) *
                                        fe_values.shape_grad(j, q_index)) *
                                       fe_values.JxW(q_index);
                cell_rhs(i) += radius * (fe_values.shape_value(i, q_index) *
                                         rhs_values[q_index] * fe_values.JxW(q_index));
              }
          }
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                                cell_matrix(i, j));
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }
    std::map<types::global_dof_index, double> boundary_values;
    SigmaFunction<1> boundary_func;
    boundary_func.initialize(center, sigma, boundary_expr,constants);

    VectorTools::interpolate_boundary_values(dof_handler, 0, boundary_func,
                                             boundary_values);
    VectorTools::interpolate_boundary_values(dof_handler, 1, boundary_func,
                                             boundary_values);

    MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution,
                                       system_rhs);
  }

  void EstimateEnrichmentFunction::solve()
  {
    SolverControl solver_control(50000, 1e-12, false, false);
    SolverCG<> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
  }

  void EstimateEnrichmentFunction::refine_grid()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<1>::estimate(dof_handler, QGauss<1 - 1>(3),
                                     typename FunctionMap<1>::type(), solution,
                                     estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number(
      triangulation, estimated_error_per_cell, 0.2, 0.01);
    triangulation.execute_coarsening_and_refinement();
  }

  void EstimateEnrichmentFunction::output_results() const
  {
    DataOut<1> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.build_patches();
    std::ofstream output("radial_solution.vtk");
    data_out.write_vtk(output);
  }

  void EstimateEnrichmentFunction::run()
  {
    if (debug_level >= 1)
      std::cout << "Solving problem in 1.: " << 1 << " with center: " << center
                << ", size: " << domain_size << ", sigma: " << sigma << std::endl;

    make_grid();

    double old_value = 0, value = 1, relative_change = 1;
    bool start = true;
    do
      {
        if (!start)
          {
            refine_grid();
            ++refinement;
          }

        if (debug_level >= 1)
          std::cout << "Refinement level: " << refinement << std::endl;

        setup_system();
        assemble_system();
        solve();

        value = VectorTools::point_value(dof_handler, solution, center);
        if (!start)
          {
            relative_change = fabs((old_value - value) / old_value);
          }
        start = false;
        old_value = value;
      }
    while (relative_change > 0.005);

    if (debug_level >= 1)
      std::cout << "Radial solution at origin = " << value
                << " after global refinement " << refinement << std::endl;

    if (debug_level >= 1)
      output_results();
  }

  void EstimateEnrichmentFunction::evaluate_at_x_values(
    std::vector<double> &interpolation_points,
    std::vector<double> &interpolation_values)
  {
    if (interpolation_values.size() != interpolation_points.size())
      interpolation_values.resize(interpolation_points.size());

    // x varies from 0 to 2*sigma.
    // factor 2 because once a cell is decided to be enriched based on its center,
    // its quadrature points can cause x to be twice!
    for (unsigned int i = 0; i != interpolation_values.size(); ++i)
      {
        double value = VectorTools::point_value(dof_handler, solution,
                                                Point<1>(interpolation_points[i]));
        interpolation_values[i] = value;
      }
  }

  double EstimateEnrichmentFunction::value(const Point<1> &p,
                                           const unsigned int &component)
  {
    return VectorTools::point_value(dof_handler, solution, p);
  }

  EstimateEnrichmentFunction::~EstimateEnrichmentFunction()
  {
    triangulation.clear();
  }
} // namespace Step1

#endif
