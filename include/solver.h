#ifndef SOLVER_H
#define SOLVER_H

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

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>

#include <set>
#include <math.h>
#include "functions.h"
#include "support.h"
#include "paramater_reader.h"
#include "estimate_enrichment.h"


template <int dim>
void plot_shape_function
(hp::DoFHandler<dim> &dof_handler,
 unsigned int patches=5)
{
  std::cout << "...start plotting shape function" << std::endl;
  std::cout << "Patches for output: " << patches << std::endl;

  ConstraintMatrix constraints;
  constraints.clear();
  dealii::DoFTools::make_hanging_node_constraints  (dof_handler, constraints);
  constraints.close ();

  //find set of dofs which belong to enriched cells
  std::set<unsigned int> enriched_cell_dofs;
  for (auto cell : dof_handler.active_cell_iterators())
    if (cell->active_fe_index() != 0)
      {
        unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell->get_dof_indices(local_dof_indices);
        enriched_cell_dofs.insert(local_dof_indices.begin(), local_dof_indices.end());
      }

  // output to check if all is good:
  std::vector<Vector<double>> shape_functions;
  std::vector<std::string> names;
  for (auto dof : enriched_cell_dofs)
    {
      Vector<double> shape_function;
      shape_function.reinit(dof_handler.n_dofs());
      shape_function[dof] = 1.0;

      // if the dof is constrained, first output unconstrained vector
      names.push_back(std::string("C_") +
                      dealii::Utilities::int_to_string(dof,2));
      shape_functions.push_back(shape_function);

//      names.push_back(std::string("UC_") +
//                      dealii::Utilities::int_to_string(s,2));

//      // make continuous/constraint:
//      constraints.distribute(shape_function);
//      shape_functions.push_back(shape_function);
    }

  if (dof_handler.n_dofs() < 100)
    {
      std::cout << "...start printing support points" << std::endl;

      std::map<types::global_dof_index, Point<dim> > support_points;
      MappingQ1<dim> mapping;
      hp::MappingCollection<dim> hp_mapping;
      for (unsigned int i = 0; i < dof_handler.get_fe_collection().size(); ++i)
        hp_mapping.push_back(mapping);
      DoFTools::map_dofs_to_support_points(hp_mapping, dof_handler, support_points);

      const std::string base_filename =
        "DOFs" + dealii::Utilities::int_to_string(dim) + "_p" + dealii::Utilities::int_to_string(0);

      const std::string filename = base_filename + ".gp";
      std::ofstream f(filename.c_str());

      f << "set terminal png size 400,410 enhanced font \"Helvetica,8\"" << std::endl
        << "set output \"" << base_filename << ".png\"" << std::endl
        << "set size square" << std::endl
        << "set view equal xy" << std::endl
        << "unset xtics                                                                                   " << std::endl
        << "unset ytics" << std::endl
        << "unset grid" << std::endl
        << "unset border" << std::endl
        << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 offset 1,1 notitle" << std::endl;
      GridOut grid_out;
      grid_out.write_gnuplot (dof_handler.get_triangulation(), f);
      f << "e" << std::endl;

      DoFTools::write_gnuplot_dof_support_point_info(f,
                                                     support_points);

      f << "e" << std::endl;

      std::cout << "...finished printing support points" << std::endl;
    }

  DataOut<dim,hp::DoFHandler<dim>> data_out;
  data_out.attach_dof_handler (dof_handler);

  // get material ids:
  Vector<float> fe_index(dof_handler.get_triangulation().n_active_cells());
  typename hp::DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active (),
  endc = dof_handler.end ();
  for (unsigned int index=0; cell!=endc; ++cell,++index)
    {
      fe_index[index] = cell->active_fe_index();
    }
  data_out.add_data_vector(fe_index, "fe_index");

  for (unsigned int i = 0; i < shape_functions.size(); i++)
    data_out.add_data_vector (shape_functions[i], names[i]);

  data_out.build_patches(patches);

  std::string filename = "shape_functions.vtu";
  std::ofstream output (filename.c_str ());
  data_out.write_vtu (output);

  std::cout << "...finished plotting shape functions" << std::endl;
}



namespace Step1
{
  /**
   * Main class
   */
  template <int dim>
  class LaplaceProblem
  {
  public:
    LaplaceProblem ();
    LaplaceProblem (const ParameterCollection<dim> &prm);
    LaplaceProblem
    (const double &size,
     const unsigned int &shape,
     const unsigned int &global_refinement,
     const unsigned int &cycles,
     const unsigned int &fe_base_degree,
     const unsigned int &fe_enriched_degree,
     const unsigned int &max_iterations,
     const double &tolerance,
     const std::string &rhs_value_expr,
     const std::string &boundary_value_expr,
     const std::string &rhs_radial_problem,
     const std::string &boundary_radial_problem,
     const std::string &exact_soln_expr,
     const bool &estimate_exact_soln,
     const unsigned int &patches,
     const unsigned int &debug_level,
     const unsigned int &n_enrichments,
     const std::vector<Point<dim>> &points_enrichments,
     const std::vector<double> &radii_predicates,
     const std::vector<double> &sigmas);

    virtual ~LaplaceProblem();

    void run ();

  protected:
    void initialize();
    void build_fe_space();
    virtual void make_enrichment_function();
    void setup_system ();

  private:
    void build_tables ();
    void output_cell_attributes (const unsigned int &cycle);  //change to const later
    void assemble_system ();
    unsigned int solve ();
    void refine_grid ();
    void output_results (const unsigned int cycle);
    void process_solution();

  protected:
    ParameterCollection<dim> prm;
    unsigned int n_enriched_cells;

    Triangulation<dim>  triangulation;
    hp::DoFHandler<dim> dof_handler;

    hp::FECollection<dim> fe_collection;
    hp::QCollection<dim> q_collection;

    FE_Q<dim> fe_base;
    FE_Q<dim> fe_enriched;
    FE_Nothing<dim> fe_nothing;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    PETScWrappers::MPI::SparseMatrix        system_matrix;
    PETScWrappers::MPI::Vector              solution;
    PETScWrappers::MPI::Vector              system_rhs;

    ConstraintMatrix constraints;

    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    ConditionalOStream pcout;

    std::vector<SigmaFunction<dim>> vec_rhs;

    using cell_function = std::function<const Function<dim>*
                          (const typename Triangulation<dim>::cell_iterator &)>;

    std::vector<SplineEnrichmentFunction<dim>> vec_enrichments;
    std::vector<EnrichmentPredicate<dim>> vec_predicates;
    std::vector<cell_function>  color_enrichments;

    //each predicate is assigned a color depending on overlap of vertices.
    std::vector<unsigned int> predicate_colors;
    unsigned int num_colors;

    //cell wise mapping of color with enrichment functions
    //vector size = number of cells;
    //map:  < color , corresponding predicate_function >
    //(only one per color acceptable). An assertion checks this
    std::map<unsigned int,
        std::map<unsigned int, unsigned int> > cellwise_color_predicate_map;

    //vector of size num_colors + 1
    //different color combinations that a cell can contain
    std::vector <std::set<unsigned int>> fe_sets;

    //output vectors. size triangulation.n_active_cells()
    //change to Vector
    std::vector<Vector<float>> predicate_output;
    Vector<float> color_output;
    Vector<float> vec_fe_index;
    Vector<float> mat_id;
  };



  template <int dim>
  LaplaceProblem<dim>:: LaplaceProblem ()
    :
    prm(),
    n_enriched_cells(0),
    dof_handler (triangulation),
    fe_base(prm.fe_base_degree),
    fe_enriched(prm.fe_enriched_degree),
    fe_nothing(1,true),
    mpi_communicator(MPI_COMM_WORLD),
    n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout (std::cout, (this_mpi_process == 0))
  {
    prm.print();

    pcout << "...default parameters set" << std::endl;
  }



  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem
  (const ParameterCollection<dim> &_par)
    :
    prm(_par),
    n_enriched_cells(0),
    dof_handler (triangulation),
    fe_base(prm.fe_base_degree),
    fe_enriched(prm.fe_enriched_degree),
    fe_nothing(1,true),
    mpi_communicator(MPI_COMM_WORLD),
    n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout (std::cout, (this_mpi_process == 0)   &&(prm.debug_level >= 1))
  {

    AssertThrow (prm.dim == dim,
                 ExcMessage("parameter file dim != problem dim"));
    prm.print();

    pcout << "...parameters set" << std::endl;
  }


  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem
  (const double &size,
   const unsigned int &shape,
   const unsigned int &global_refinement,
   const unsigned int &cycles,
   const unsigned int &fe_base_degree,
   const unsigned int &fe_enriched_degree,
   const unsigned int &max_iterations,
   const double &tolerance,
   const std::string &rhs_value_expr,
   const std::string &boundary_value_expr,
   const std::string &rhs_radial_problem,
   const std::string &boundary_radial_problem,
   const std::string &exact_soln_expr,
   const bool &estimate_exact_soln,
   const unsigned int &patches,
   const unsigned int &debug_level,
   const unsigned int &n_enrichments,
   const std::vector<Point<dim>> &points_enrichments,
   const std::vector<double> &radii_predicates,
   const std::vector<double> &sigmas)
    :
    prm
    (dim,
     size,
     shape,
     global_refinement,
     cycles,
     fe_base_degree,
     fe_enriched_degree,
     max_iterations,
     tolerance,
     rhs_value_expr,
     boundary_value_expr,
     rhs_radial_problem,
     boundary_radial_problem,
     exact_soln_expr,
     estimate_exact_soln,
     patches,
     debug_level,
     n_enrichments,
     points_enrichments,
     radii_predicates,
     sigmas),
    n_enriched_cells(0),
    dof_handler (triangulation),
    fe_base(prm.fe_base_degree),
    fe_enriched(prm.fe_enriched_degree),
    fe_nothing(1,true),
    mpi_communicator(MPI_COMM_WORLD),
    n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout (std::cout, (this_mpi_process == 0)  &&(prm.debug_level >= 1))

  {
    prm.print();

    pcout << "...parameters set" << std::endl;
  }

  template <int dim>
  void LaplaceProblem<dim>::initialize()
  {
    pcout << "...Start initializing" << std::endl;
    //set up basic grid
    if (prm.shape == 1)
      GridGenerator::hyper_cube (triangulation, -prm.size/2.0, prm.size/2.0);
    else if (prm.shape == 0)
      {
        Point<dim> center = Point<dim>();
        GridGenerator::hyper_ball (triangulation, center, prm.size/2.0);
        triangulation.set_all_manifold_ids_on_boundary(0);
        static SphericalManifold<dim> spherical_manifold(center);
        triangulation.set_manifold(0,spherical_manifold);
      }
    else
      AssertThrow(false,ExcMessage("Shape not implemented."));

    triangulation.refine_global (prm.global_refinement);

    Assert(prm.points_enrichments.size()==prm.n_enrichments &&
           prm.radii_predicates.size()==prm.n_enrichments &&
           prm.sigmas.size()==prm.n_enrichments,
           ExcMessage
           ("Number of enrichment points, predicate radii and sigmas should be equal"));

    //initialize vector of predicate functions, f: assign cell --> 1 or 0
    for (unsigned int i=0; i != prm.n_enrichments; ++i)
      {
        vec_predicates.push_back( EnrichmentPredicate<dim>(prm.points_enrichments[i],
                                                           prm.radii_predicates[i]) );
      }

    //set a vector of right hand side functions
    vec_rhs.resize(prm.n_enrichments);
    for (unsigned int i=0; i != prm.n_enrichments; ++i)
      {
        vec_rhs[i].initialize(prm.points_enrichments[i],
                              prm.sigmas[i],
                              prm.rhs_value_expr);
      }

    pcout << "...finish initializing" << std::endl;
  }

  template <int dim>
  void LaplaceProblem<dim>::build_fe_space()
  {
    pcout << "...building fe space" << std::endl;

    make_enrichment_function();

    //make a sparsity pattern based on connections between regions
    if (vec_predicates.size() != 0)
      num_colors = color_predicates (dof_handler, vec_predicates, predicate_colors);
    else
      num_colors = 0;

    {
      //print color_indices
      for (unsigned int i=0; i<predicate_colors.size(); ++i)
        pcout << "predicate " << i << " : " << predicate_colors[i] << std::endl;
    }

    //build fe table. should be called everytime number of cells change!
    build_tables();

    if (prm.debug_level == 9)
      {
        if (triangulation.n_active_cells() < 100)
          {
            pcout << "...start print fe indices" << std::endl;

            //print fe index
            const std::string base_filename =
              "fe_indices" + dealii::Utilities::int_to_string(dim) + "_p" + dealii::Utilities::int_to_string(0);
            const std::string filename =  base_filename + ".gp";
            std::ofstream f(filename.c_str());

            f << "set terminal png size 400,410 enhanced font \"Helvetica,8\"" << std::endl
              << "set output \"" << base_filename << ".png\"" << std::endl
              << "set size square" << std::endl
              << "set view equal xy" << std::endl
              << "unset xtics" << std::endl
              << "unset ytics" << std::endl
              << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 offset 1,1 notitle" << std::endl;
            GridOut().write_gnuplot (triangulation, f);
            f << "e" << std::endl;

            for (auto it : dof_handler.active_cell_iterators())
              f << it->center() << " \"" << it->active_fe_index() << "\"\n";

            f << std::flush << "e" << std::endl;
            pcout << "...finished print fe indices" << std::endl;
          }

        if (triangulation.n_active_cells() < 100)
          {
            pcout << "...start print cell indices" << std::endl;

            //print cell ids
            const std::string base_filename =
              "cell_id" + dealii::Utilities::int_to_string(dim) + "_p" + dealii::Utilities::int_to_string(0);
            const std::string filename =  base_filename + ".gp";
            std::ofstream f(filename.c_str());

            f << "set terminal png size 400,410 enhanced font \"Helvetica,8\"" << std::endl
              << "set output \"" << base_filename << ".png\"" << std::endl
              << "set size square" << std::endl
              << "set view equal xy" << std::endl
              << "unset xtics" << std::endl
              << "unset ytics" << std::endl
              << "plot '-' using 1:2 with lines notitle, '-' with labels point pt 2 offset 1,1 notitle" << std::endl;
            GridOut().write_gnuplot (triangulation, f);
            f << "e" << std::endl;

            for (auto it : dof_handler.active_cell_iterators())
              f << it->center() << " \"" << it->index() << "\"\n";

            f << std::flush << "e" << std::endl;

            pcout << "...end print cell indices" << std::endl;
          }
      }

    //q collections the same size as different material identities
    //TODO in parameter file
    //TODO clear so that multiple runs will work
    q_collection.push_back(QGauss<dim>(4));
    for (unsigned int i=1; i!=fe_sets.size(); ++i)
      q_collection.push_back(QGauss<dim>(10));


    //setup color wise enrichment functions
    //i'th function corresponds to (i+1) color!
    make_colorwise_enrichment_functions (num_colors,
                                         vec_enrichments,
                                         cellwise_color_predicate_map,
                                         color_enrichments);

    //TODO clear fe_collection such that multiple calls will work
    make_fe_collection_from_colored_enrichments (num_colors,
                                                 fe_sets,
                                                 color_enrichments,
                                                 fe_base,
                                                 fe_enriched,
                                                 fe_nothing,
                                                 fe_collection);

    pcout << "...building fe space" << std::endl;
  }



  template <int dim>
  void LaplaceProblem<dim>::make_enrichment_function()
  {
    pcout << "!!! Make enrichment function called" << std::endl;

    for (unsigned int i=0; i<vec_predicates.size(); ++i)
      {
        //formulate a 1d/radial problem with x coordinate and radius (i.e sigma)
        double center = 0;
        double sigma = prm.sigmas[i];
        //Actual size is twice the radius. For hexahedral cells, dimension can
        //extend upto sqrt(3) times! so take a factor of 4 in total.
        double size = prm.radii_predicates[i]*4;

        if (prm.radii_predicates[i] != 0)
          {
            EstimateEnrichmentFunction<1> radial_problem(Point<1>(center),
                                                         size,
                                                         sigma,
                                                         prm.rhs_radial_problem,
                                                         prm.boundary_radial_problem);
            radial_problem.debug_level = prm.debug_level; //print output
            radial_problem.run();
            pcout << "solved problem with "
                  << "(x, sigma): "
                  << center << ", " << sigma << std::endl;

            //make points at which solution needs to interpolated
            std::vector<double> interpolation_points_1D, interpolation_values_1D;
            double radius = size/2; //only half side is important
            unsigned int n = 5;
            double right_bound(center + radius);
            double h = radius/n;

            for (double p = center; p < right_bound; p+=h)
              interpolation_points_1D.push_back(p);
            interpolation_points_1D.push_back(right_bound);

            //add enrichment function only when predicate radius is non-zero

            radial_problem.evaluate_at_x_values(interpolation_points_1D,interpolation_values_1D);


            //construct enrichment function and push
            SplineEnrichmentFunction<dim> func(prm.points_enrichments[i],
                                               prm.radii_predicates[i],
                                               interpolation_points_1D,
                                               interpolation_values_1D);
            vec_enrichments.push_back(func);
          }
        else
          {
            pcout << "Dummy function added at " << i << std::endl;
            SplineEnrichmentFunction<dim> func(Point<dim>(),1,0);
            vec_enrichments.push_back(func);
          }
      }
  }


  template <int dim>
  void LaplaceProblem<dim>::build_tables ()
  {
    pcout << "...start build tables" << std::endl;
    set_cellwise_color_set_and_fe_index (dof_handler,
                                         vec_predicates,
                                         predicate_colors,
                                         cellwise_color_predicate_map,
                                         fe_sets);
    {
      //print material table
      pcout << "\nMaterial table : " << std::endl;
      for ( unsigned int j=0; j<fe_sets.size(); ++j)
        {
          pcout << j << " ( ";
          for ( auto i = fe_sets[j].begin(); i != fe_sets[j].end(); ++i)
            pcout << *i << " ";
          pcout << " ) " << std::endl;
        }
    }

    pcout << "...finish build tables" << std::endl;


  }

  template <int dim>
  void LaplaceProblem<dim>::setup_system ()
  {
    pcout << "...start setup system" << std::endl;

    GridTools::partition_triangulation (n_mpi_processes, triangulation);

    dof_handler.distribute_dofs (fe_collection);

    //TODO renumbering screws up numbering of cell ids?
    DoFRenumbering::subdomain_wise (dof_handler);
    //...
    //?
    std::vector<IndexSet> locally_owned_dofs_per_proc
      = DoFTools::locally_owned_dofs_per_subdomain (dof_handler);
    locally_owned_dofs = locally_owned_dofs_per_proc[this_mpi_process];
    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);
    //?

    constraints.clear();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints  (dof_handler, constraints);

    SigmaFunction<dim> boundary_value_func;
    boundary_value_func.initialize(prm.points_enrichments[0],
                                   prm.sigmas[0],
                                   prm.boundary_value_expr);


    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              boundary_value_func,
                                              constraints);
    constraints.close ();

    // Initialise the stiffness and mass matrices
    DynamicSparsityPattern dsp (locally_relevant_dofs);
    DoFTools::make_sparsity_pattern (dof_handler, dsp,
                                     constraints,
                                     false);
    std::vector<types::global_dof_index> n_locally_owned_dofs(n_mpi_processes);
    for (unsigned int i = 0; i < n_mpi_processes; ++i)
      n_locally_owned_dofs[i] = locally_owned_dofs_per_proc[i].n_elements();

    SparsityTools::distribute_sparsity_pattern
    (dsp,
     n_locally_owned_dofs,
     mpi_communicator,
     locally_relevant_dofs);

    system_matrix.reinit (locally_owned_dofs,
                          locally_owned_dofs,
                          dsp,
                          mpi_communicator);

    solution.reinit (locally_owned_dofs, mpi_communicator);
    system_rhs.reinit (locally_owned_dofs, mpi_communicator);

    pcout << "...finished setup system" << std::endl;
  }


  template <int dim>
  void
  LaplaceProblem<dim>::assemble_system()
  {
    pcout << "...assemble system" << std::endl;

    system_matrix = 0;
    system_rhs = 0;

    FullMatrix<double> cell_system_matrix;
    Vector<double>     cell_rhs;

    std::vector<types::global_dof_index> local_dof_indices;

    std::vector<double> rhs_value, tmp_rhs_value;

    hp::FEValues<dim> fe_values_hp(fe_collection, q_collection,
                                   update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values);


    typename hp::DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active (),
    endc = dof_handler.end ();
    for (; cell!=endc; ++cell)
      if (cell->subdomain_id() == this_mpi_process )
        {
          fe_values_hp.reinit (cell);
          const FEValues<dim> &fe_values = fe_values_hp.get_present_fe_values();

          const unsigned int &dofs_per_cell = cell->get_fe().dofs_per_cell;
          const unsigned int &n_q_points    = fe_values.n_quadrature_points;

          /*
           * Initialize rhs values vector to zero. Add values calculated
           * from each of different rhs functions (vec_rhs).
           */
          rhs_value.assign(n_q_points,0);
          tmp_rhs_value.assign(n_q_points,0);
          for (unsigned int i=0; i<vec_rhs.size(); ++i)
            {
              vec_rhs[i].value_list (fe_values.get_quadrature_points(),
                                     tmp_rhs_value);

              // add tmp to the total one at quadrature points
              for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
                {
                  rhs_value[q_point] += tmp_rhs_value[q_point];
                }
            }

          local_dof_indices.resize     (dofs_per_cell);
          cell_system_matrix.reinit (dofs_per_cell,dofs_per_cell);
          cell_rhs.reinit (dofs_per_cell);

          cell_system_matrix = 0;
          cell_rhs = 0;

          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                for (unsigned int j=i; j<dofs_per_cell; ++j)
                  cell_system_matrix(i,j) += (fe_values.shape_grad(i,q_point) *
                                              fe_values.shape_grad(j,q_point) *
                                              fe_values.JxW(q_point));

                cell_rhs(i) += (rhs_value[q_point] *
                                fe_values.shape_value(i,q_point) *
                                fe_values.JxW(q_point));
              }

          // exploit symmetry
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=i; j<dofs_per_cell; ++j)
              cell_system_matrix (j, i) = cell_system_matrix (i, j);

          cell->get_dof_indices (local_dof_indices);

          constraints.distribute_local_to_global (cell_system_matrix,
                                                  cell_rhs,
                                                  local_dof_indices,
                                                  system_matrix,
                                                  system_rhs);
        }

    system_matrix.compress (VectorOperation::add);
    system_rhs.compress (VectorOperation::add);

    pcout << "...finished assemble system" << std::endl;

  }

  template <int dim>
  unsigned int LaplaceProblem<dim>::solve()
  {
    pcout << "...solving" << std::endl;
    SolverControl           solver_control (prm.max_iterations,
                                            prm.tolerance,
                                            false,
                                            false);
    PETScWrappers::SolverCG cg (solver_control,
                                mpi_communicator);

    //choose preconditioner
#define amg
#ifdef amg
    PETScWrappers::PreconditionBoomerAMG::AdditionalData additional_data;
    additional_data.symmetric_operator = true;
    PETScWrappers::PreconditionBoomerAMG preconditioner(system_matrix,
                                                        additional_data);
#else
    PETScWrappers::PreconditionJacobi preconditioner(system_matrix);
#endif

    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);

    Vector<double> localized_solution (solution);

    constraints.distribute (localized_solution);
    solution = localized_solution;

    pcout << "...finished solving" << std::endl;

    return solver_control.last_step();
  }

  template <int dim>
  LaplaceProblem<dim>::~LaplaceProblem()
  {
    dof_handler.clear ();
  }

  template <int dim>
  void LaplaceProblem<dim>::refine_grid ()
  {
    const Vector<double> localized_solution (solution);
    Vector<float> local_error_per_cell (triangulation.n_active_cells());

    hp::QCollection<dim-1> q_collection_face;
    for (unsigned int i=0; i < q_collection.size(); ++i)
      q_collection_face.push_back(QGauss<dim-1>(q_collection[i].size()));

    KellyErrorEstimator<dim>::estimate (dof_handler,
                                        q_collection_face,
                                        typename FunctionMap<dim>::type(),
                                        localized_solution,
                                        local_error_per_cell,
                                        ComponentMask(),
                                        nullptr,
                                        n_mpi_processes,
                                        this_mpi_process);
    const unsigned int n_local_cells
      = GridTools::count_cells_with_subdomain_association (triangulation,
                                                           this_mpi_process);
    PETScWrappers::MPI::Vector
    distributed_all_errors (mpi_communicator,
                            triangulation.n_active_cells(),
                            n_local_cells);
    for (unsigned int i=0; i<local_error_per_cell.size(); ++i)
      if (local_error_per_cell(i) != 0)
        distributed_all_errors(i) = local_error_per_cell(i);
    distributed_all_errors.compress (VectorOperation::insert);
    const Vector<float> localized_all_errors (distributed_all_errors);
    GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                     localized_all_errors,
                                                     0.3, 0.03);
    triangulation.execute_coarsening_and_refinement ();
    ++prm.global_refinement;
  }



  template <int dim>
  void LaplaceProblem<dim>::output_results (const unsigned int cycle)
  {
    pcout << "...output results" << std::endl;
    pcout << "Patches used: " << prm.patches << std::endl;

    Vector<double> exact_soln_vector, error_vector;
    SigmaFunction<dim> exact_solution;

    if (prm.exact_soln_expr != "")
      {
        //create exact solution vector
        exact_soln_vector.reinit(dof_handler.n_dofs());
        exact_solution.initialize(Point<dim>(),
                                  prm.sigmas[0],
                                  prm.exact_soln_expr);
        VectorTools::project(dof_handler,
                             constraints,
                             q_collection,
                             exact_solution,
                             exact_soln_vector);

        //create error vector
        error_vector.reinit(dof_handler.n_dofs());
        Vector<double> full_solution(solution);
        error_vector += full_solution;
        error_vector -= exact_soln_vector;
      }

    Assert (cycle < 10, ExcNotImplemented());
    if (this_mpi_process==0)
      {
        std::string filename = "solution-";
        filename += Utilities::to_string(cycle);
        filename += ".vtk";
        std::ofstream output (filename.c_str());

        DataOut<dim,hp::DoFHandler<dim> > data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, "solution");
        if (prm.exact_soln_expr != "")
          {
            data_out.add_data_vector (exact_soln_vector, "exact_solution");
            data_out.add_data_vector (error_vector, "error_vector");
          }
        data_out.build_patches (prm.patches);
        data_out.write_vtk (output);
        output.close();
      }

    pcout << "...finished output results" << std::endl;
  }


  //use this only when exact solution is known or can be solved
  template <int dim>
  void LaplaceProblem<dim>::process_solution()
  {
    Vector<float> difference_per_cell (triangulation.n_active_cells());
    double L2_error, H1_error;

    if (!prm.exact_soln_expr.empty())
      {
        pcout << "...using exact solution for error calculation" << std::endl;

        SigmaFunction<dim> exact_solution;
        exact_solution.initialize(Point<dim>(),
                                  prm.sigmas[0],
                                  prm.exact_soln_expr);


        VectorTools::integrate_difference (dof_handler,
                                           solution,
                                           exact_solution,
                                           difference_per_cell,
                                           q_collection,
                                           VectorTools::L2_norm);
        L2_error = VectorTools::compute_global_error(triangulation,
                                                     difference_per_cell,
                                                     VectorTools::L2_norm);

        VectorTools::integrate_difference (dof_handler,
                                           solution,
                                           exact_solution,
                                           difference_per_cell,
                                           q_collection,
                                           VectorTools::H1_norm);
        H1_error = VectorTools::compute_global_error(triangulation,
                                                     difference_per_cell,
                                                     VectorTools::H1_norm);
      }
    else if (prm.estimate_exact_soln)
      {
        //Not very accurate estimation
        AssertThrow(prm.shape == 0,
                    ExcMessage("solution only for circular domain can be estimated"));
        AssertThrow(prm.n_enrichments == 1 &&
                    prm.points_enrichments[0] == Point<dim>(),
                    ExcMessage("solution only for single source at origin"));

        pcout << "...estimate exact solution for error calculation" << std::endl;

        //Make enrichment function with spline ranging over whole domain.
        //Gradient of enrichment function is related to gradient of the spline.
        double center = 0;
        double sigma = prm.sigmas[0];
        double size = prm.size;
        //Ensure the radial problem is solved for diagonal incase shape is square
        if (prm.shape==1) size = size*sqrt(dim);
        EstimateEnrichmentFunction<1> radial_problem(Point<1>(center),
                                                     size,
                                                     sigma,
                                                     prm.rhs_radial_problem,
                                                     prm.boundary_radial_problem);
        radial_problem.debug_level = prm.debug_level; //print output
        radial_problem.run();
        pcout << "solving radial problem for error calculation "
              << "(x, sigma): "
              << center << ", " << sigma << std::endl;

        //make points at which solution needs to interpolated
        std::vector<double> interpolation_points_1D, interpolation_values_1D;
        double cut_point = 1;
        unsigned int n1 = 10, n2 = 200;
        double right_bound = size/2;
        double h1 = cut_point/n1, h2 = (prm.size-cut_point)/n2;
        for (double p = center; p < cut_point; p+=h1)
          interpolation_points_1D.push_back(p);
        for (double p = cut_point; p < right_bound; p+=h2)
          interpolation_points_1D.push_back(p);
        interpolation_points_1D.push_back(right_bound);

        radial_problem.evaluate_at_x_values(interpolation_points_1D,interpolation_values_1D);


        //construct enrichment function and make spline function
        SplineEnrichmentFunction<dim> exact_solution(Point<dim>(),
                                                     1, //TODO is not need. remove
                                                     interpolation_points_1D,
                                                     interpolation_values_1D);


        VectorTools::integrate_difference (dof_handler,
                                           solution,
                                           exact_solution,
                                           difference_per_cell,
                                           q_collection,
                                           VectorTools::L2_norm);
        L2_error = VectorTools::compute_global_error(triangulation,
                                                     difference_per_cell,
                                                     VectorTools::L2_norm);

        VectorTools::integrate_difference (dof_handler,
                                           solution,
                                           exact_solution,
                                           difference_per_cell,
                                           q_collection,
                                           VectorTools::H1_norm);
        H1_error = VectorTools::compute_global_error(triangulation,
                                                     difference_per_cell,
                                                     VectorTools::H1_norm);
      }
    else
      {
        pcout << "...error can't be calculated" << std::endl;
        return;
      }

    pcout << "refinement h_smallest Dofs L2_norm H1_norm" << std::endl;
    pcout << prm.global_refinement << " "
          << prm.size/std::pow(2.0,prm.global_refinement) << " "
          << dof_handler.n_dofs() << " "
          << L2_error << " "
          << H1_error << std::endl;

    deallog << "refinement Dofs L2_norm H1_norm" << std::endl;
    deallog << prm.global_refinement << " "
            << dof_handler.n_dofs() << " "
            << L2_error << " "
            << H1_error << std::endl;
  }

  template <int dim>
  void LaplaceProblem<dim>::output_cell_attributes (const unsigned int &cycle)
  {
    pcout << "...output pre-solution" << std::endl;

    std::string file_name = "pre_solution_" + Utilities::to_string(cycle);

    //find number of enriched cells and set active fe index
    vec_fe_index.reinit(triangulation.n_active_cells());
    typename hp::DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    n_enriched_cells = 0;
    for (unsigned int index=0; cell!=endc; ++cell, ++index)
      {
        vec_fe_index[index] = cell->active_fe_index();
        if (vec_fe_index[index] != 0)
          ++n_enriched_cells;
      }
    pcout << "Number of enriched cells: " << n_enriched_cells << std::endl;

    /*
     * set predicate vector. This will change with each refinement.
     * But since the fe indices, fe mapping and enrichment functions don't
     * change with refinement, we are not changing the enriched space.
     * Enrichment functions don't change because color map, fe indices and
     * material id don't change.
     */
    {
      predicate_output.resize(vec_predicates.size());
      for (unsigned int i = 0; i < vec_predicates.size(); ++i)
        {
          predicate_output[i].reinit(triangulation.n_active_cells());
        }

      unsigned int index = 0;
      for (typename hp::DoFHandler<dim>::cell_iterator cell = dof_handler.begin_active();
           cell != dof_handler.end(); ++cell, ++index)
        {
          for (unsigned int i=0; i<vec_predicates.size(); ++i)
            if ( vec_predicates[i](cell) )
              predicate_output[i][index] = i+1;
        }
    }

    //make color index
    {
      color_output.reinit(triangulation.n_active_cells());
      unsigned int index = 0;
      for (typename hp::DoFHandler<dim>::cell_iterator cell= dof_handler.begin_active();
           cell != dof_handler.end();
           ++cell, ++index)
        for (unsigned int i=0; i<vec_predicates.size(); ++i)
          if ( vec_predicates[i](cell) )
            color_output[index] = predicate_colors[i];
    }

    //make material id
    mat_id.reinit(triangulation.n_active_cells());
    {
      unsigned int index = 0;
      for (typename hp::DoFHandler<dim>::cell_iterator cell= dof_handler.begin_active();
           cell != dof_handler.end();
           ++cell, ++index)
        mat_id[index] = cell->material_id();
    }

    // print fe_index, colors and predicate
    if (this_mpi_process==0)
      {
        file_name += ".vtk";
        std::ofstream output (file_name.c_str());
        DataOut<dim,hp::DoFHandler<dim> > data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (vec_fe_index, "fe_index");
        data_out.add_data_vector (color_output, "colors");
        data_out.add_data_vector (mat_id, "mat_id");
        for (unsigned int i = 0; i < predicate_output.size(); ++i)
          data_out.add_data_vector (predicate_output[i], "predicate_" + std::to_string(i));
        data_out.build_patches ();
        data_out.write_vtk (output);
      }
    pcout << "...finished output pre-solution" << std::endl;
  }


  template <int dim>
  void
  LaplaceProblem<dim>::run()
  {
    pcout << "...run problem" << std::endl;

    //Run making grids and building fe space only once.
    initialize();
    build_fe_space();


    //TODO need cyles?
    for (unsigned int cycle = 0; cycle <= prm.cycles; ++cycle)
      {
        pcout << "Cycle "<< cycle <<std::endl;

        if (prm.debug_level >= 3)
          output_cell_attributes(cycle);

        setup_system ();

        pcout << "Number of active cells:       "
              << triangulation.n_active_cells ()
              << std::endl
              << "Number of degrees of freedom: "
              << dof_handler.n_dofs ()
              << std::endl;

        if (prm.debug_level == 9)
          plot_shape_function<dim>(dof_handler);

        assemble_system ();
        auto n_iterations = solve ();
        pcout << "Number of iterations: " << n_iterations << std::endl;
        pcout << "Solution at origin:   "
              << VectorTools::point_value(dof_handler,
                                          solution,
                                          Point<dim>())
              << std::endl;
        if (prm.exact_soln_expr != "" || prm.estimate_exact_soln==true)
          process_solution();

        if (prm.debug_level >= 2)
          output_results(cycle);

        //Donot refine if loop is at the end
        if (cycle != prm.cycles)
          refine_grid ();

        pcout << "...step run complete" << std::endl;
      }
    pcout << "...finished run problem" << std::endl;
  }
}
#endif
