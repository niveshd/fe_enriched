#ifndef SOLVER_TRILINOS_H
#define SOLVER_TRILINOS_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/grid/grid_out.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/hp/dof_handler.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/fe/fe_enriched.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/utilities.h>

#include <deal.II/base/parameter_handler.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/numerics/vector_tools.h>

#include "estimate_enrichment.h"
#include "functions.h"
#include "paramater_reader.h"
#include "support.h"
#include <deal.II/fe/fe_enriched.h>
#include <math.h>
#include <set>

#include "../tests/tests.h"

namespace LA
{
  using namespace dealii::LinearAlgebraTrilinos;
}

template <int dim>
using predicate_function =
  std::function<bool(const typename Triangulation<dim>::cell_iterator &)>;

namespace Step1
{
  /**
   * Main class
   */
  template <int dim> class LaplaceProblem_t
  {
  public:
    LaplaceProblem_t();
    LaplaceProblem_t(const ParameterCollection &prm);
    virtual ~LaplaceProblem_t();
    void run();

  protected:
    void initialize();
    void build_fe_space();
    virtual void make_enrichment_functions();
    void setup_system();

  private:
    void build_tables();
    void
    output_cell_attributes(const unsigned int &cycle); // change to const later
    void assemble_system();
    unsigned int solve();
    void refine_grid();
    void output_results(const unsigned int cycle);
    void process_solution();

  protected:
    ParameterCollection prm;
    unsigned int n_enriched_cells;

    Triangulation<dim> triangulation;
    hp::DoFHandler<dim> dof_handler;

    typedef LA::MPI::SparseMatrix matrix_t;
    typedef LA::MPI::Vector vector_t;

    std::shared_ptr<const hp::FECollection<dim>> fe_collection;
    hp::QCollection<dim> q_collection;

    FE_Q<dim> fe_base;
    FE_Q<dim> fe_enriched;
    FE_Nothing<dim> fe_nothing;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    matrix_t system_matrix;
    vector_t solution;
    Vector<double> localized_solution;
    vector_t system_rhs;

    ConstraintMatrix constraints;

    std::shared_ptr<ColorEnriched::Helper<dim>> fe_space;

    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    ConditionalOStream pcout;

    std::vector<SigmaFunction<dim>> vec_rhs;

    using cell_iterator_function = std::function<Function<dim> *(
                                     const typename Triangulation<dim>::cell_iterator &)>;

    std::vector<std::shared_ptr<Function<dim>>> vec_enrichments;
    std::vector<predicate_function<dim>> vec_predicates;
    std::vector<cell_iterator_function> color_enrichments;

    // output vectors. size triangulation.n_active_cells()
    // change to Vector
    std::vector<Vector<float>> predicate_output;
    Vector<float> color_output;
    Vector<float> vec_fe_index;
    Vector<float> mat_id;

    //required functions
    Vec_SigmaFunction<dim> v_sol_func;
  };

  template <int dim>
  LaplaceProblem_t<dim>::LaplaceProblem_t()
    : prm(), n_enriched_cells(0), dof_handler(triangulation),
      fe_base(prm.fe_base_degree), fe_enriched(prm.fe_enriched_degree),
      fe_nothing(1, true), mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      pcout(std::cout, (this_mpi_process == 0))
  {
    prm.print();

    pcout << "...default parameters set" << std::endl;
  }

  template <int dim>
  LaplaceProblem_t<dim>::LaplaceProblem_t(const ParameterCollection &_par)
    : prm(_par), n_enriched_cells(0), dof_handler(triangulation),
      fe_base(prm.fe_base_degree), fe_enriched(prm.fe_enriched_degree),
      fe_nothing(1, true), mpi_communicator(MPI_COMM_WORLD),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
      this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
      pcout(std::cout, (this_mpi_process == 0)  &&(prm.debug_level >= 1))
  {

    AssertThrow(prm.dim == dim, ExcMessage("parameter file dim != problem dim"));
    prm.print();
    pcout << "...parameters set" << std::endl;
  }

  /*
   * Construct basic grid, vector of predicate functions and
   * right hand side function of the problem.
   */
  template <int dim> void LaplaceProblem_t<dim>::initialize()
  {
    pcout << "...Start initializing trilinos solver" << std::endl;

    /*
     * set up basic grid which is a hyper cube or hyper ball based on
     * parameter file. Refine as per the global refinement value in the
     * parameter file.
     *
     */
    if (prm.shape == 1)
      GridGenerator::hyper_cube(triangulation, -prm.size / 2.0, prm.size / 2.0);
    else if (prm.shape == 0)
      {
        Point<dim> center = Point<dim>();
        GridGenerator::hyper_ball(triangulation, center, prm.size / 2.0);
        triangulation.set_all_manifold_ids_on_boundary(0);
        static SphericalManifold<dim> spherical_manifold(center);
        triangulation.set_manifold(0, spherical_manifold);
      }
    else
      AssertThrow(false, ExcMessage("Shape not implemented."));

    if (prm.global_refinement > 0)
      triangulation.refine_global(prm.global_refinement);

    /*
     * Ensure that num of radii, sigma and focal points of enrichment domains
     * provided matches the number of predicates. Since focal points of
     * enrichment domains are stored as vector of numbers, the dimension of
     * points_enrichments is dim times the n_enrichments in prm file.
     */
    Assert(prm.points_enrichments.size() / dim == prm.n_enrichments,
           ExcMessage("Number of enrichment points, predicate radii and sigmas "
                      "should be equal"));

    /*
     * Construct vector of predicate functions, where a function at index i
     * returns true for a given cell if it belongs to enrichment domain i.
     * The decision is based on cell center being at a distance within radius
     * of the domain from focal point of the domain.
     */
    for (unsigned int i = 0; i != prm.n_enrichments; ++i)
      {
        Point<dim> p;
        prm.set_enrichment_point(p, i);
        vec_predicates.push_back(
          EnrichmentPredicate<dim>(p, prm.predicate_radius));
      }

    /*
     * Construct a vector of right hand side functions where the actual right hand
     * side of problem is evaluated through summation of contributions from each
     * of the functions in the vector. Each function i at a given point is
     * evaluated with respect to point's position {x_i, y_i, z_i} relative to
     * focal point of enrichment domain i. The value is then a function of x_i,
     * y_i, z_i and sigma given by the rhs_value_expr[i] in parameter file.
     */
    vec_rhs.resize(prm.n_enrichments);
    for (unsigned int i = 0; i != prm.n_enrichments; ++i)
      {
        Point<dim> p;
        prm.set_enrichment_point(p, i);
        std::map<std::string,double> constants;

        if (prm.coefficients.size() != 0)
          constants.insert({"c",prm.coefficients[i]});

        vec_rhs[i].initialize(p, prm.sigma, prm.rhs_value_expr, constants);
      }

    pcout << "...finish initializing" << std::endl;
  }

  /*
   * Create a enrichment function associated with each enrichment domain.
   * Here the approximate solution is assumed to be only dependent on
   * distance (r) from the focal point of enrichment domain. This is
   * true for poisson problem (a linear PDE) provided the right hand side is a
   * radial function and drops exponentially for points farther from focal
   * point of enrichment domain.
   */
  template <int dim> void LaplaceProblem_t<dim>::make_enrichment_functions()
  {
    pcout << "!!! Make enrichment function called" << std::endl;

    for (unsigned int i = 0; i < vec_predicates.size(); ++i)
      {
        /*
         * Radial problem is solved only when enrichment domain has a positive
         * radius and hence non-empty domain.
         */
        if (prm.predicate_radius != 0)
          {
            /*
             * Formulate a 1d/radial problem with center and size appropriate
             * to cover the enrichment domain. The function determining
             * right hand side and boundary value of the problem is provided in
             * parameter file. The sigma governing this function is the same as
             * sigma provided for the corresponding enrichment domain and predicate
             * function for which the enrichment function is to be estimated.
             *
             * The center is 0 since the function is evaluated with respect to
             * relative position from focal point anyway.
             *
             * For hexahedral cells, dimension can extend upto sqrt(3) < 2 times!
             * So take a factor of 4 as size of the problem. This ensures that
             * enrichment function can be evaluated at all points in the enrichment
             * domain
             */
            double sub_domain_size = prm.predicate_radius * 4;
            double radius = sub_domain_size / 2;

            // make points from 0 to radius with cutpoint at 0.25 times the radius
            std::vector<double> interpolation_points, interpolation_values;
            double cut_point = 0.25 * prm.sigma;
            unsigned int n1 = 15, n2 = 15;
            double h1 = cut_point/n1, h2 = (radius - cut_point) / n2;
            for (double p = 0; p < cut_point; p += h1)
              interpolation_points.push_back(p);
            for (double p = cut_point; p < radius; p += h2)
              interpolation_points.push_back(p);
            interpolation_points.push_back(radius);

            // if exact solution is available use it for enrichment function
            if (prm.estimate_exact_soln)
              {
                pcout << "...estimating enrichment function for predicate: "
                      << i
                      << std::endl;
                std::map<std::string,double> constants;

                if (prm.coefficients.size() != 0)
                  constants.insert({"c",prm.coefficients[i]});

                EstimateEnrichmentFunction radial_problem(Point<1>(0),
                                                          prm.size,
                                                          prm.sigma,
                                                          prm.rhs_radial_problem,
                                                          prm.boundary_radial_problem,
                                                          constants);
                radial_problem.debug_level = prm.debug_level; // print output
                radial_problem.run();
                pcout << "solved problem with "
                      << "x and sigma : " << 0 << ", " << prm.sigma << std::endl;

                // add enrichment function only when predicate radius is non-zero
                radial_problem.evaluate_at_x_values(interpolation_points,
                                                    interpolation_values);

                // construct enrichment function and push
                Point<dim> p;
                prm.set_enrichment_point(p, i);
                SplineEnrichmentFunction<dim> func(p,
                                                   interpolation_points,
                                                   interpolation_values);
                vec_enrichments.push_back(
                  std::make_shared<SplineEnrichmentFunction<dim>>(func));
              }
            else
              {
                pcout << "...using exact enrichment expression" << std::endl;
                SigmaFunction<dim> enr;
                Point<dim> p;
                prm.set_enrichment_point(p, i);
                // take only x coordinate
                std::map<std::string,double> constants;
                if (prm.coefficients.size() != 0)
                  constants.insert({"c",prm.coefficients[i]});
                enr.initialize(p, prm.sigma, prm.boundary_value_expr,constants);

                vec_enrichments.push_back(std::make_shared<SigmaFunction<dim>>(enr));
              }
          }
        else
          {
            pcout << "Dummy function added at " << i << std::endl;
            ConstantFunction<dim> func(0);
            vec_enrichments.push_back(std::make_shared<ConstantFunction<dim>>(func));
          }
      }
  }

  /*
   * Since each enrichment domain has different enrichment function
   * associated with it and the cells common to different enrichment
   * domains need to treated differently, we use helper function in
   * ColorEnriched namespace to construct finite element space and necessary
   * data structures required by it. The helper function is also used to
   * set FE indices of dof handler cells. We also set the cell with a unique
   * material id for now, which is used to map cells with a pairs of
   * color and corresponding enrichment function. All this is internally
   * used to figure out the correct set of enrichment functions to be used
   * for a cell.
   *
   * The quadrature point collection, with size equal to FE collection
   * is also constructed here.
   */
  template <int dim> void LaplaceProblem_t<dim>::build_fe_space()
  {
    pcout << "...building fe space" << std::endl;

    make_enrichment_functions();
    fe_space = std::make_shared<ColorEnriched::Helper<dim>>
               (ColorEnriched::Helper<dim>(fe_base, fe_enriched,
                                           vec_predicates, vec_enrichments));
    fe_collection = std::make_shared<const hp::FECollection<dim>>(
                      fe_space->build_fe_collection(dof_handler));
    std::cout << "size of fe collection: " << fe_collection->size() << std::endl;

    if (prm.debug_level == 9)
      {
        if (triangulation.n_active_cells() < 100)
          {
            pcout << "...start print fe indices" << std::endl;

            // print fe index
            const std::string base_filename =
              "fe_indices" + dealii::Utilities::int_to_string(dim) + "_p" +
              dealii::Utilities::int_to_string(0);
            const std::string filename = base_filename + ".gp";
            std::ofstream f(filename.c_str());

            f << "set terminal png size 400,410 enhanced font \"Helvetica,8\""
              << std::endl
              << "set output \"" << base_filename << ".png\"" << std::endl
              << "set size square" << std::endl
              << "set view equal xy" << std::endl
              << "unset xtics" << std::endl
              << "unset ytics" << std::endl
              << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 "
              "offset 1,1 notitle"
              << std::endl;
            GridOut().write_gnuplot(triangulation, f);
            f << "e" << std::endl;

            for (auto it : dof_handler.active_cell_iterators())
              f << it->center() << " \"" << it->active_fe_index() << "\"\n";

            f << std::flush << "e" << std::endl;
            pcout << "...finished print fe indices" << std::endl;
          }

        if (triangulation.n_active_cells() < 100)
          {
            pcout << "...start print cell indices" << std::endl;

            // print cell ids
            const std::string base_filename =
              "cell_id" + dealii::Utilities::int_to_string(dim) + "_p" +
              dealii::Utilities::int_to_string(0);
            const std::string filename = base_filename + ".gp";
            std::ofstream f(filename.c_str());

            f << "set terminal png size 400,410 enhanced font \"Helvetica,8\""
              << std::endl
              << "set output \"" << base_filename << ".png\"" << std::endl
              << "set size square" << std::endl
              << "set view equal xy" << std::endl
              << "unset xtics" << std::endl
              << "unset ytics" << std::endl
              << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 "
              "offset 1,1 notitle"
              << std::endl;
            GridOut().write_gnuplot(triangulation, f);
            f << "e" << std::endl;

            for (auto it : dof_handler.active_cell_iterators())
              f << it->center() << " \"" << it->index() << "\"\n";

            f << std::flush << "e" << std::endl;

            pcout << "...end print cell indices" << std::endl;
          }
      }

    // q collections the same size as different material identities
    q_collection.push_back(QGauss<dim>(prm.q_degree));
    for (unsigned int i = 1; i < fe_collection->size(); ++i)
      q_collection.push_back(QGauss<dim>(prm.q_degree));

    pcout << "...building fe space" << std::endl;
  }

  template <int dim> void LaplaceProblem_t<dim>::setup_system()
  {
    pcout << "...start setup system" << std::endl;

    GridTools::partition_triangulation(n_mpi_processes, triangulation);

    dof_handler.distribute_dofs(*fe_collection);

    DoFRenumbering::subdomain_wise(dof_handler);
    std::vector<IndexSet> locally_owned_dofs_per_proc =
      DoFTools::locally_owned_dofs_per_subdomain(dof_handler);
    locally_owned_dofs = locally_owned_dofs_per_proc[this_mpi_process];
    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    // if multiple sources are present
        std::vector<SigmaFunction<dim>> vec_bnd_func;
        vec_bnd_func.resize(prm.n_enrichments);

        for (unsigned int i = 0; i < vec_bnd_func.size(); ++i)
          {
            Point<dim> p;
            prm.set_enrichment_point(p, i);
            std::map<std::string,double> constants;
            if (prm.coefficients.size() > 0)
              constants.insert({"c",prm.coefficients[i]});
            vec_bnd_func[i].initialize(p, prm.sigma, prm.boundary_value_expr,constants);
          }

        v_sol_func.initialize(vec_bnd_func);
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 v_sol_func,
                                                 constraints);

//    if (prm.debug_level >= 1)
//    {
//      pcout << "print cells with dof" << std::endl;
//      //set dof id
//      unsigned int match_dof = 8617;

//      Vector<float> cells_match_dof;
//      cells_match_dof.reinit(triangulation.n_active_cells());

//      //replace all strings. used to better represent fe enriched elements
//      auto replace = [](std::string& str, const std::string& from, const std::string& to) {
//      if(from.empty())
//          return;
//      size_t start_pos = 0;
//      while((start_pos = str.find(from, start_pos)) != std::string::npos) {
//          str.replace(start_pos, from.length(), to);
//          start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
//      }
//      };

//      //loop through cells
//      int i = 0;
//      for (auto cell:dof_handler.active_cell_iterators()){

//        //if the dof belongs to it
//        std::vector< types::global_dof_index > 	dof_indices(cell->get_fe().dofs_per_cell);
//        cell->get_dof_indices(dof_indices);
//        for (auto dof:dof_indices)
//          if (dof == match_dof){
//             std::string fe_name(cell->get_fe().get_name());
//             replace(fe_name, "FE_Q<"+dealii::Utilities::int_to_string(dim)+">(1)", "1");
//             replace(fe_name, "FE_Nothing<"+dealii::Utilities::int_to_string(dim)+">(dominating)", "0");

//             cells_match_dof[cell->active_cell_index()] = ++i;

//            //print FE enriched for the cell
//            pcout << cell->id() << " "
//                  << cell->active_fe_index() << " "
//                  << fe_name << std::endl;
//          }
//        }

//      //geometry of the cell (adaptable for 3d!)
//      std::ofstream output("dof_debug.vtk");
//      DataOut<dim, hp::DoFHandler<dim>> data_out;
//      data_out.attach_dof_handler(dof_handler);
//      data_out.add_data_vector(cells_match_dof, "match_dof");
//      data_out.build_patches(prm.patches);
//      data_out.write_vtk(output);
//      output.close();
//    }

    constraints.close();

    // Initialise the stiffness and mass matrices
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    std::vector<types::global_dof_index> n_locally_owned_dofs(n_mpi_processes);
    for (unsigned int i = 0; i < n_mpi_processes; ++i)
      n_locally_owned_dofs[i] = locally_owned_dofs_per_proc[i].n_elements();

    SparsityTools::distribute_sparsity_pattern(
      dsp, n_locally_owned_dofs, mpi_communicator, locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                         mpi_communicator);

    solution.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    pcout << "...finished setup system" << std::endl;
  }

  template <int dim> void LaplaceProblem_t<dim>::assemble_system()
  {
    pcout << "...assemble system" << std::endl;

    system_matrix = 0;
    system_rhs = 0;

    FullMatrix<double> cell_system_matrix;
    Vector<double> cell_rhs;

    std::vector<types::global_dof_index> local_dof_indices;

    std::vector<double> rhs_value, tmp_rhs_value;

    hp::FEValues<dim> fe_values_hp(*fe_collection, q_collection,
                                   update_values | update_gradients |
                                   update_quadrature_points |
                                   update_JxW_values);

    for (auto cell : dof_handler.active_cell_iterators())
      if (cell->subdomain_id() == this_mpi_process)
        {
          fe_values_hp.reinit(cell);
          const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();

          const unsigned int &dofs_per_cell = cell->get_fe().dofs_per_cell;
          const unsigned int &n_q_points = fe_values.n_quadrature_points;

          /*
           * Initialize rhs values vector to zero. Add values calculated
           * from each of different rhs functions (vec_rhs).
           */
          rhs_value.assign(n_q_points, 0);
          tmp_rhs_value.assign(n_q_points, 0);
          for (unsigned int i = 0; i < vec_rhs.size(); ++i)
            {
              vec_rhs[i].value_list(fe_values.get_quadrature_points(), tmp_rhs_value);

              // add tmp to the total one at quadrature points
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                  rhs_value[q_point] += tmp_rhs_value[q_point];
                }
            }

          local_dof_indices.resize(dofs_per_cell);
          cell_system_matrix.reinit(dofs_per_cell, dofs_per_cell);
          cell_rhs.reinit(dofs_per_cell);

          cell_system_matrix = 0;
          cell_rhs = 0;

          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                for (unsigned int j = i; j < dofs_per_cell; ++j)
                  cell_system_matrix(i, j) +=
                    (fe_values.shape_grad(i, q_point) *
                     fe_values.shape_grad(j, q_point) * fe_values.JxW(q_point));

                cell_rhs(i) +=
                  (rhs_value[q_point] * fe_values.shape_value(i, q_point) *
                   fe_values.JxW(q_point));
              }

          // exploit symmetry
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int j = i; j < dofs_per_cell; ++j)
              cell_system_matrix(j, i) = cell_system_matrix(i, j);

          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_system_matrix, cell_rhs,
                                                 local_dof_indices, system_matrix,
                                                 system_rhs);
        }

    system_matrix.compress(VectorOperation::add);
    system_rhs.compress(VectorOperation::add);

    pcout << "...finished assemble system" << std::endl;
  }

  template <int dim> unsigned int LaplaceProblem_t<dim>::solve()
  {
    pcout << "...solving" << std::endl;
    SolverControl solver_control(prm.max_iterations, prm.tolerance, false, false);
    SolverCG<vector_t> cg(solver_control);

    // choose preconditioner
    if (prm.solver == prm.trilinos_amg)
      {
        TrilinosWrappers::PreconditionAMG::AdditionalData Amg_data;
        Amg_data.elliptic = true;
        Amg_data.higher_order_elements = true;
        Amg_data.smoother_sweeps = 2;
        Amg_data.aggregation_threshold = prm.threshold_amg;
        TrilinosWrappers::PreconditionAMG preconditioner;
        preconditioner.initialize(system_matrix, Amg_data);
        cg.solve(system_matrix, solution, system_rhs, preconditioner);
      }
    else if (prm.solver == prm.jacobi)
      {
        TrilinosWrappers::PreconditionJacobi preconditioner;
        preconditioner.initialize(system_matrix);
        cg.solve(system_matrix, solution, system_rhs, preconditioner);
      }
    else
      {
        AssertThrow(false, ExcMessage("Improper preconditioner. Value given " +
                                      prm.solver));
      }

    Vector<double> local_soln(solution);

    constraints.distribute(local_soln);
    solution = local_soln;
    pcout << "...finished solving" << std::endl;
    return solver_control.last_step();
  }

  template <int dim> void LaplaceProblem_t<dim>::refine_grid()
  {
    const Vector<double> localized_solution(solution);
    Vector<float> local_error_per_cell(triangulation.n_active_cells());

    hp::QCollection<dim - 1> q_collection_face;
    for (unsigned int i = 0; i < q_collection.size(); ++i)
      q_collection_face.push_back(QGauss<dim - 1>(1));

    KellyErrorEstimator<dim>::estimate(
      dof_handler, q_collection_face, typename FunctionMap<dim>::type(),
      localized_solution, local_error_per_cell, ComponentMask(), nullptr,
      n_mpi_processes, this_mpi_process);
    const unsigned int n_local_cells =
      GridTools::count_cells_with_subdomain_association(triangulation,
                                                        this_mpi_process);
    PETScWrappers::MPI::Vector distributed_all_errors(
      mpi_communicator, triangulation.n_active_cells(), n_local_cells);
    for (unsigned int i = 0; i < local_error_per_cell.size(); ++i)
      if (local_error_per_cell(i) != 0)
        distributed_all_errors(i) = local_error_per_cell(i);
    distributed_all_errors.compress(VectorOperation::insert);
    const Vector<float> localized_all_errors(distributed_all_errors);
    GridRefinement::refine_and_coarsen_fixed_fraction(
      triangulation, localized_all_errors, 0.85, 0);
    triangulation.execute_coarsening_and_refinement();
    ++prm.global_refinement;
  }

  template <int dim>
  void LaplaceProblem_t<dim>::output_results(const unsigned int cycle)
  {
    pcout << "...output results" << std::endl;
    pcout << "Patches used: " << prm.patches << std::endl;

    Vector<double> exact_soln_vector, error_vector;

    AssertThrow(prm.estimate_exact_soln == true,
                ExcMessage("Error norms cannot be calculated"));

    // create exact solution vector
    exact_soln_vector.reinit(dof_handler.n_dofs());

    // if the exact solution is a sum of functions
    // evaluate function using v_sol_func
    VectorTools::project(dof_handler, constraints, q_collection, v_sol_func,
                         exact_soln_vector);
    // create error vector
    error_vector.reinit(dof_handler.n_dofs());
    Vector<double> full_solution(localized_solution);
    error_vector += full_solution;
    error_vector -= exact_soln_vector;

    Assert(cycle < 10, ExcNotImplemented());
    if (this_mpi_process == 0)
      {
        std::string filename = "solution-";
        filename += Utilities::to_string(cycle);
        filename += ".vtk";
        std::ofstream output(filename.c_str());

        DataOut<dim, hp::DoFHandler<dim>> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(localized_solution, "solution");
        if (prm.estimate_exact_soln)
          {
            data_out.add_data_vector(exact_soln_vector, "exact_solution");
            data_out.add_data_vector(error_vector, "error_vector");
          }
        data_out.build_patches(prm.patches);
        data_out.write_vtk(output);
        output.close();
      }
    pcout << "...finished output results" << std::endl;
  }

// use this only when exact solution is known
  template <int dim> void LaplaceProblem_t<dim>::process_solution()
  {
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    double L2_error, H1_error;

    if (prm.estimate_exact_soln)
      {
        pcout << "...using exact solution for error calculation" << std::endl;

        VectorTools::integrate_difference(dof_handler, localized_solution,
                                          v_sol_func, difference_per_cell,
                                          q_collection, VectorTools::L2_norm);
        L2_error = VectorTools::compute_global_error(
                     triangulation, difference_per_cell, VectorTools::L2_norm);

        VectorTools::integrate_difference(dof_handler, localized_solution,
                                          v_sol_func, difference_per_cell,
                                          q_collection, VectorTools::H1_norm);
        H1_error = VectorTools::compute_global_error(
                     triangulation, difference_per_cell, VectorTools::H1_norm);
      }

    pcout << "refinement h_smallest Dofs L2_norm H1_norm" << std::endl;
    pcout << prm.global_refinement << " "
          << prm.size / std::pow(2.0, prm.global_refinement) << " "
          << dof_handler.n_dofs() << " " << L2_error << " " << H1_error
          << std::endl;
  }

  template <int dim>
  void LaplaceProblem_t<dim>::output_cell_attributes(const unsigned int &cycle)
  {
    pcout << "...output pre-solution" << std::endl;

    std::string file_name = "pre_solution_" + Utilities::to_string(cycle);

    // find number of enriched cells and set active fe index
    vec_fe_index.reinit(triangulation.n_active_cells());
    typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                              endc = dof_handler.end();
    for (unsigned int index = 0; cell != endc; ++cell, ++index)
      {
        vec_fe_index[index] = cell->active_fe_index();
      }

    unsigned int n_pred_outputs = (10 <= vec_predicates.size())?10:vec_predicates.size();
    /*
     * set predicate vector. This will change with each refinement.
     * But since the fe indices, fe mapping and enrichment functions don't
     * change with refinement, we are not changing the enriched space.
     * Enrichment functions don't change because color map, fe indices and
     * material id don't change.
     */
    {

      predicate_output.resize(n_pred_outputs);
      for (unsigned int i = 0; i < n_pred_outputs; ++i)
        {
          predicate_output[i].reinit(triangulation.n_active_cells());
        }

      for (auto cell : dof_handler.active_cell_iterators())
        {
          for (unsigned int i = 0; i < n_pred_outputs; ++i)
            if (vec_predicates[i](cell))
              predicate_output[i][cell->active_cell_index()] = i + 1;
        }
    }

    // make color index
    {
      std::vector<unsigned int> predicate_colors;
      ColorEnriched::internal::color_predicates(dof_handler, vec_predicates,
                                                predicate_colors);

      color_output.reinit(triangulation.n_active_cells());
      for (auto cell : dof_handler.active_cell_iterators())
        for (unsigned int i = 0; i < vec_predicates.size(); ++i)
          if (vec_predicates[i](cell))
            color_output[cell->active_cell_index()] = predicate_colors[i];
    }

    // make material id
    mat_id.reinit(triangulation.n_active_cells());
    {
      for (auto cell : dof_handler.active_cell_iterators())
        mat_id[cell->active_cell_index()] = cell->material_id();
    }

    // print fe_index, colors and predicate
    if (this_mpi_process == 0)
      {
        file_name += ".vtk";
        std::ofstream output(file_name.c_str());
        DataOut<dim, hp::DoFHandler<dim>> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(vec_fe_index, "fe_index");
        data_out.add_data_vector(color_output, "colors");
        data_out.add_data_vector(mat_id, "mat_id");

        for (unsigned int i = 0; i < predicate_output.size(); ++i)
          data_out.add_data_vector(predicate_output[i],
                                   "predicate_" + std::to_string(i));
        data_out.build_patches();
        data_out.write_vtk(output);
      }
    pcout << "...finished output pre-solution" << std::endl;
  }

  template <int dim> void LaplaceProblem_t<dim>::run()
  {
    pcout << "...run problem" << std::endl;
    double norm_soln_old(0), norm_rel_change_old(1);

    // Run making grids and building fe space only once.
    initialize();
    build_fe_space();

    for (unsigned int cycle = 0; cycle <= prm.cycles; ++cycle)
      {
        pcout << "Cycle " << cycle << std::endl;

        if (prm.debug_level >= 5)
          output_cell_attributes(cycle);

        setup_system();

        pcout << "Number of active cells:       " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

        if (prm.debug_level == 9 && this_mpi_process == 0)
          plot_shape_function<dim>(dof_handler);

        //calculate number of enriched cells
        typename hp::DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                                  .begin_active(),
                                                                  endc = dof_handler.end();
        n_enriched_cells = 0;
        for (unsigned int index = 0; cell != endc; ++cell, ++index)
          {
            if (cell->active_fe_index() != 0)
              ++n_enriched_cells;
          }
        pcout << "Number of enriched cells: " << n_enriched_cells << std::endl;

        assemble_system();

        if (prm.solve_problem)
          {
            Timer timer;
            timer.start();
            auto n_iterations = solve();
            timer.stop();
            pcout << "time: " << timer.wall_time() << std::endl;
            timer.reset();

            pcout << "iterations: " << n_iterations << std::endl;

            localized_solution.reinit(dof_handler.n_dofs());
            localized_solution = solution;
            double value = VectorTools::point_value(dof_handler, localized_solution,
                                                    Point<dim>());
            pcout << "Solution at origin:   " << value << std::endl;

            // calculate L2 norm of solution
            if (this_mpi_process == 0)
              {
                pcout << "calculating L2 norm of soln" << std::endl;
                double norm_soln_new(0), norm_rel_change_new(0);
                Vector<float> difference_per_cell(triangulation.n_active_cells());
                VectorTools::integrate_difference(
                  dof_handler, localized_solution, ZeroFunction<dim>(),
                  difference_per_cell, q_collection, VectorTools::H1_norm);
                norm_soln_new = VectorTools::compute_global_error(
                                  triangulation, difference_per_cell, VectorTools::H1_norm);
                // relative change can only be calculated for cycle > 0
                if (cycle > 0)
                  {
                    norm_rel_change_new =
                      std::abs((norm_soln_new - norm_soln_old) / norm_soln_old);
                    pcout << "relative change of solution norm " << norm_rel_change_new
                          << std::endl;
                  }

                // moniter relative change of norm in later stages
                if (cycle > 1)
                  {
                    deallog << (norm_rel_change_new < norm_rel_change_old) << std::endl;
                  }

                norm_soln_old = norm_soln_new;

                // first sample of relative change of norm comes only cycle = 1
                if (cycle > 0)
                  norm_rel_change_old = norm_rel_change_new;

                pcout << "End of L2 calculation" << std::endl;
              }

            if (prm.estimate_exact_soln && this_mpi_process == 0)
              process_solution();

            if (prm.debug_level >= 5 && this_mpi_process == 0)
              output_results(cycle);

            // Donot refine if loop is at the end
            if (cycle != prm.cycles)
              refine_grid();
          }
        pcout << "...step run complete" << std::endl;
      }
    pcout << "...finished run problem" << std::endl;
  }

  template <int dim> LaplaceProblem_t<dim>::~LaplaceProblem_t()
  {
    dof_handler.clear();
  }
} // namespace Step1
#endif
