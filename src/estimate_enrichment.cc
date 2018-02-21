#include <estimate_enrichment.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <functions.h>

//unnamed namespace
namespace EstimateEnrichment
{
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

}

template <int dim>
EstimateEnrichmentFunction<dim>::EstimateEnrichmentFunction
(Point<dim> center, double domain_size, double sigma)
  :
  Function<dim>(1),
  center(center),
  domain_size(domain_size),
  sigma(sigma),
  debug_level(0),
  refinement(11),
  fe (1),
  dof_handler (triangulation)
{
  AssertDimension(dim,1);
  left_bound = center[0] - domain_size/2;
  right_bound = center[0] + domain_size/2;
}

template <int dim>
EstimateEnrichmentFunction<dim>::EstimateEnrichmentFunction
(Point<dim> center, double left_bound,
 double right_bound, double sigma, double refinement)
  :
  Function<dim>(1),
  center(center),
  left_bound(left_bound),
  right_bound(right_bound),
  sigma(sigma),
  debug_level(0),
  refinement(refinement),
  fe (1),
  dof_handler (triangulation)
{
  AssertDimension(dim,1);
  domain_size = right_bound - left_bound;
}

template <int dim>
void EstimateEnrichmentFunction<dim>::make_grid ()
{
  //TODO domain 50 times original radius enough?
  GridGenerator::hyper_cube (triangulation,
                             left_bound,
                             right_bound);
  triangulation.refine_global (refinement);
}

template <int dim>
void EstimateEnrichmentFunction<dim>::setup_system ()
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
void EstimateEnrichmentFunction<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(2);
  const GaussianFunction<dim> rhs(center,sigma);
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
      rhs_values.resize(n_q_points);
      rhs.value_list (fe_values.get_quadrature_points(),
                      rhs_values);

      AssertDimension(dim,1);
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          double radius = center.distance(fe_values.quadrature_point(q_index));

          //-1/r (r*u_r) = f form converts to
          // r(u_r, v_r) = (r*f,v)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += (radius*
                                     fe_values.shape_grad (i, q_index) *
                                     fe_values.shape_grad (j, q_index)
                                    )*fe_values.JxW (q_index);
              cell_rhs(i) += radius*(fe_values.shape_value (i, q_index) *
                                     rhs_values[q_index] *
                                     fe_values.JxW (q_index));
            }
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
                                            EstimateEnrichment::BoundaryValues<dim>(),
                                            boundary_values);
  //TODO find a better way to do this!
  AssertDimension(dim,1);   //Here we assume a 1D problem
  VectorTools::interpolate_boundary_values (dof_handler,
                                            1,
                                            EstimateEnrichment::BoundaryValues<dim>(),
                                            boundary_values);

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}

template <int dim>
void EstimateEnrichmentFunction<dim>::solve ()
{
  SolverControl           solver_control (50000,1e-12,false,false);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());
}

template <int dim>
void EstimateEnrichmentFunction<dim>::refine_grid ()
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(3),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimated_error_per_cell);
  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.2, 0.01);
  triangulation.execute_coarsening_and_refinement ();
}

template <int dim>
void EstimateEnrichmentFunction<dim>::output_results () const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();
  std::ofstream output ("radial_solution.vtk");
  data_out.write_vtk (output);
}

template <int dim>
void EstimateEnrichmentFunction<dim>::run ()
{
//  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

  std::cout << "Solving problem in dim.: " << dim
            << " with center: " << center
            << ", size: " << domain_size
            << ", sigma: " << sigma
            << std::endl;

  make_grid();

  double old_value=0, value=1, relative_change = 1;
  int cycles = 0;
  bool start = true;
  do
    {
//      std::cout << "Refinement level: " << refinement << std::endl;

      if (cycles!=0)
        {
          refine_grid ();
          ++refinement;
        }
      ++cycles;
      setup_system ();
      assemble_system ();
      solve ();

      value = VectorTools::point_value(dof_handler, solution, center);
      if (!start)
        {
          relative_change = fabs((old_value - value)/old_value);
        }
      start = false;
      old_value = value;
    }
  while (relative_change > 0.005);

  std::cout << "Radial solution at origin = "
            << value
            << " after global refinement "
            << refinement
            << std::endl;

  if (debug_level >= 1)
    output_results ();
}

template <int dim>
void EstimateEnrichmentFunction<dim>::evaluate_at_x_values
(std::vector< double >  &interpolation_points,
 std::vector< double >   &interpolation_values)
{
  if (interpolation_values.size() != interpolation_points.size())
    interpolation_values.resize(interpolation_points.size());

  //x varies from 0 to 2*sigma.
  //factor 2 because once a cell is decided to be enriched based on its center,
  //its quadrature points can cause x to be twice!
  for (unsigned int i=0; i!=interpolation_values.size(); ++i)
    {
      double value = VectorTools::point_value(dof_handler, solution,
                                              Point<dim>(interpolation_points[i]));
      interpolation_values[i] = value;
    }
}

template <int dim>
double EstimateEnrichmentFunction<dim>::value
(const Point<dim> &p,
 const unsigned int & component)
{
  return VectorTools::point_value(dof_handler, solution, p);
}

template <int dim>
EstimateEnrichmentFunction<dim>::~EstimateEnrichmentFunction()
{
  triangulation.clear();
}
//instantiations
template struct EstimateEnrichmentFunction<1>;
template struct EstimateEnrichmentFunction<2>;
