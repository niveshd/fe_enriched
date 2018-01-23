#include <estimate_enrichment.h>

//unnamed namespace
namespace
{
  template <int dim>
  class RightHandSide : public Function<dim>
  {
  public:
    RightHandSide (Point<dim> center, double sigma)
      : Function<dim>(),
        center(center),
        sigma(sigma)
    {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    Point<dim> center;
    double sigma;
  };

  template <int dim>
  double RightHandSide<dim>::value (const Point<dim> &p,
                                    const unsigned int /*component*/) const
  {
    double return_value = 0.0;
    return_value = exp(-p.distance_square(center)/(sigma*sigma));

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

}

template <int dim>
EstimateEnrichmentFunction<dim>::EstimateEnrichmentFunction
(Point<dim> center, double sigma)
  :
  center(center),
  sigma(sigma),
  fe (1),
  dof_handler (triangulation)
{}

template <int dim>
void EstimateEnrichmentFunction<dim>::make_grid ()
{
  //TODO domain 50 times original radius enough?
  GridGenerator::hyper_cube (triangulation, -50*sigma, 50*sigma);
  triangulation.refine_global (10);
  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;
}

template <int dim>
void EstimateEnrichmentFunction<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;
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
  const RightHandSide<dim> right_hand_side(center,sigma);
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

      AssertDimension(dim,1);
      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          double radius = center.distance(fe_values.quadrature_point(q_index));
          AssertThrow(radius != 0, ExcMessage("Radius is zero"));
          double factor = -(1.0/radius);

          //-u_xx-1/x*u_x = f form converts to
          // (-u_xx, v) - (1/x) * (u_x,v) = (f,v)
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += (radius*
                                    fe_values.shape_grad (i, q_index) *
                                    fe_values.shape_grad (j, q_index)
                                    )*fe_values.JxW (q_index);
              cell_rhs(i) += radius*(fe_values.shape_value (i, q_index) *
                              right_hand_side.value (fe_values.quadrature_point (q_index)) *
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
                                            BoundaryValues<dim>(),
                                            boundary_values);
  //TODO find a better way to do this!
  AssertDimension(dim,1);   //Here we assume a 1D problem
  VectorTools::interpolate_boundary_values (dof_handler,
                                            1,
                                            BoundaryValues<dim>(),
                                            boundary_values);

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}

template <int dim>
void EstimateEnrichmentFunction<dim>::solve ()
{
  SolverControl           solver_control (10000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;
}

template <int dim>
void EstimateEnrichmentFunction<dim>::output_results () const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();
  std::ofstream output ("solution-estimate.vtk");
  data_out.write_vtk (output);
}

template <int dim>
void EstimateEnrichmentFunction<dim>::run ()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
  make_grid();
  setup_system ();
  assemble_system ();
  solve ();
  output_results ();
}


//instantiations
template struct EstimateEnrichmentFunction<1>;
