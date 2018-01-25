// -------------------------------------------
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
// -------------------------------------------


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

const unsigned int dim = 2;
unsigned int patches = 10;

template <int dim>
void plot_shape_function
(hp::DoFHandler<dim> &dof_handler)
{
  deallog << "n_cells: "<< dof_handler.get_triangulation().n_active_cells()<<std::endl;

  ConstraintMatrix constraints;
  constraints.clear();
  dealii::DoFTools::make_hanging_node_constraints  (dof_handler, constraints);
  constraints.close ();

  // output to check if all is good:
  std::vector<Vector<double>> shape_functions;
  std::vector<std::string> names;
  for (unsigned int s=0; s < dof_handler.n_dofs(); s++)
    {
      Vector<double> shape_function;
      shape_function.reinit(dof_handler.n_dofs());
      shape_function[s] = 1.0;

      // if the dof is constrained, first output unconstrained vector
      if (constraints.is_constrained(s))
        {
          names.push_back(std::string("UN_") +
                          dealii::Utilities::int_to_string(s,2));
          shape_functions.push_back(shape_function);
        }

      names.push_back(std::string("N_") +
                      dealii::Utilities::int_to_string(s,2));

      // make continuous/constrain:
      constraints.distribute(shape_function);
      shape_functions.push_back(shape_function);

      //print support points
      {
        std::map<types::global_dof_index, Point<dim> > support_points;
        MappingQ1<dim> mapping;
        hp::MappingCollection<dim> hp_mapping;
        for (unsigned int i = 0; i < dof_handler.get_fe_collection().size(); ++i)
          hp_mapping.push_back(mapping);
        DoFTools::map_dofs_to_support_points(hp_mapping, dof_handler, support_points);

        const std::string base_filename =
          "grid" + dealii::Utilities::int_to_string(dim) + "_p" + dealii::Utilities::int_to_string(0);

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
      }
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
    LaplaceProblem (int argc,char **argv);
    virtual ~LaplaceProblem();
    void run ();

  private:
    void read_parameters();
    void build_tables ();
    void setup_system ();
    void assemble_system ();
    unsigned int solve ();
    void estimate_error ();
    void refine_grid ();
    void output_results (const unsigned int cycle) const;
    void output_test ();  //change to const later

    int argc;
    char **argv;
    int size;
    unsigned int global_refinement;
    unsigned int n_enrichments;
    std::vector<Point<dim>> points_enrichments;
    std::vector<unsigned int> radii_enrichments;
    ParameterHandler prm;

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

    RightHandSide<dim> right_hand_side;

    using cell_function = std::function<const Function<dim>*
                          (const typename Triangulation<dim>::cell_iterator &)>;

    std::vector<EnrichmentFunction<dim>> vec_enrichments;
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

    const FEValuesExtractors::Scalar fe_extractor;
    const FEValuesExtractors::Scalar pou_extractor;

    //output vectors. size triangulation.n_active_cells()
    //change to Vector
    std::vector<Vector<float>> predicate_output;
    Vector<float> color_output;
    Vector<float> active_fe_index;

    //solver parameters
    unsigned int max_iterations;
    double tolerance;

  };

  template <int dim>
  void LaplaceProblem<dim>::read_parameters()
  {
    pcout << "---Reading parameters" << std::endl;

    //declare parameters
    prm.enter_subsection("geometry");
    prm.declare_entry("size",
                      "1",
                      Patterns::Double(0));
    prm.declare_entry("Global refinement",
                      "1",
                      Patterns::Integer(1));
    prm.leave_subsection();

    prm.enter_subsection("solver");
    prm.declare_entry("max iterations",
                      "1000",
                      Patterns::Integer(1));
    prm.declare_entry("tolerance",
                      "1e-8",
                      Patterns::Double(0));
    prm.leave_subsection();

    //parse parameter file
    AssertThrow(argc >= 2, ExcMessage("Parameter file not given"));
    prm.parse_input(argv[1], "#end-of-dealii parser");

    //get parameters
    prm.enter_subsection("geometry");
    size = prm.get_integer("size");
    global_refinement = prm.get_integer("Global refinement");
    prm.leave_subsection();

    prm.enter_subsection("solver");
    max_iterations = prm.get_integer("max iterations");
    tolerance = prm.get_double("tolerance");
    prm.leave_subsection();

    pcout << "Size : "<< size << std::endl;
    pcout << "Global refinement : " << global_refinement << std::endl;
    pcout << "Max Iterations : " << max_iterations << std::endl;
    pcout << "Tolerance : " << tolerance << std::endl;

    //manual parsing
    //open parameter file
    AssertThrow(argc >= 2, ExcMessage("Parameter file not given"));
    std::ifstream prm_file(argv[1]);

    //read lines until "#end-of-dealii parser" is reached
    std::string line;
    while (getline(prm_file,line))
      if (line == "#end-of-dealii parser")
        break;

    AssertThrow(line == "#end-of-dealii parser",
                ExcMessage("line missing in parameter file = \'#end-of-dealii parser\' "));

    //function to read next line not starting with # or empty
    auto skiplines = [&] ()
    {
      while (getline(prm_file,line))
        {
          if (line.size()==0 || line[0] == '#' || line[0] == ' ')
            continue;
          else
            break;
        }
    };

    std::stringstream s_stream;

    //read num of enrichement points
    skiplines();
    s_stream.str(line);
    s_stream >> n_enrichments;
    pcout << "Number of enrichments : " << n_enrichments << std::endl;

    //note vector of points
    for (unsigned int i=0; i!=n_enrichments; ++i)
      {
        skiplines();
        s_stream.clear();
        s_stream.str(line);

        if (dim==2)
          {
            double x,y;
            s_stream >> x >> y;
            points_enrichments.push_back({x,y});
          }
        else if (dim==3)
          {
            double x,y,z;
            s_stream << x << y << z;
            points_enrichments.push_back({x,y,z});
          }
        else
          AssertThrow(false, ExcMessage("Dimension not implemented"));
      }

    pcout << "Enrichment points : " << std::endl;
    for (auto p:points_enrichments)
      pcout << p << std::endl;

    //note vector of coefficients
    for (unsigned int i=0; i!=n_enrichments; ++i)
      {
        skiplines();
        s_stream.clear();
        s_stream.str(line);

        unsigned int r;
        s_stream >> r;
        radii_enrichments.push_back(r);
      }

    pcout << "Enrichment radii : " << std::endl;
    for (auto r:radii_enrichments)
      pcout << r << std::endl;

    pcout << "---Parameter reading complete." << std::endl;
  }

  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem (int argc,char **argv)
    :
    argc(argc),
    argv(argv),
    dof_handler (triangulation),
    fe_base(2),
    fe_enriched(1),
    fe_nothing(1,true),
    mpi_communicator(MPI_COMM_WORLD),
    n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout (std::cout, (this_mpi_process == 0))
  {
    read_parameters();

    GridGenerator::hyper_cube (triangulation, -size, size);
    triangulation.refine_global (global_refinement);

    Assert(points_enrichments.size()==n_enrichments &&
           radii_enrichments.size()==n_enrichments,
           ExcMessage("Incorrect number of enrichment points and radii"));
    //initialize vector of vec_predicates
    for (unsigned int i=0; i != n_enrichments; ++i)
      {
        vec_predicates.push_back( EnrichmentPredicate<dim>(points_enrichments[i],
                                                           radii_enrichments[i]) );
      }

//     const enrichment functions!
//    for (unsigned int i=0; i<vec_predicates.size(); ++i)
//      {
//        EnrichmentFunction<dim> func(10+i);  //constant function
//        vec_enrichments.push_back( func );
//      }

    for (unsigned int i=0; i<vec_predicates.size(); ++i)
      {
        //formulate a 1d problem with x coordinate and radius (i.e sigma)
        double x = points_enrichments[i][0];
        double radius = radii_enrichments[i];
        EstimateEnrichmentFunction<1> problem_1d(Point<1>(x), radius);
        problem_1d.run();
        std::vector<double> interpolation_points_1d, interpolation_values_1d;
        problem_1d.interpolate(interpolation_points_1d,interpolation_values_1d);
        std::cout << "solved problem with " << x << ":" << radius << std::endl;

        //construct enrichment function and push
        EnrichmentFunction<dim> func(points_enrichments[i],
                                     radii_enrichments[i],
                                     interpolation_points_1d,
                                     interpolation_values_1d);
        vec_enrichments.push_back(func);

      }

    {
      //print predicates
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


    //build fe table. should be called everytime number of cells change!
    build_tables();

    {
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
    }

    {
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
    }

    //q collections the same size as different material identities
    q_collection.push_back(QGauss<dim>(4));

    for (unsigned int i=1; i!=fe_sets.size(); ++i)
      q_collection.push_back(QGauss<dim>(10));


    //setup color wise enrichment functions
    //i'th function corresponds to (i+1) color!
    make_colorwise_enrichment_functions (num_colors,
                                         vec_enrichments,
                                         cellwise_color_predicate_map,
                                         color_enrichments);

    make_fe_collection_from_colored_enrichments (num_colors,
                                                 fe_sets,
                                                 color_enrichments,
                                                 fe_base,
                                                 fe_enriched,
                                                 fe_nothing,
                                                 fe_collection);
    pcout << "---constructor complete" << std::endl;
  }

  template <int dim>
  void LaplaceProblem<dim>::build_tables ()
  {
    set_cellwise_color_set_and_fe_index (dof_handler,
                                         vec_predicates,
                                         predicate_colors,
                                         cellwise_color_predicate_map,
                                         fe_sets);

    output_test();

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

    pcout << "---build tables complete" << std::endl;


  }

  template <int dim>
  void LaplaceProblem<dim>::setup_system ()
  {

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
    //TODO  crashing here!
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              Functions::ZeroFunction<dim> (1),
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

    pcout << "---system set up complete" << std::endl;
  }


  template <int dim>
  void
  LaplaceProblem<dim>::assemble_system()
  {
    system_matrix = 0;
    system_rhs = 0;

    FullMatrix<double> cell_system_matrix;
    Vector<double>     cell_rhs;

    std::vector<types::global_dof_index> local_dof_indices;

    std::vector<double> rhs_values;

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

          rhs_values.resize(n_q_points);

          local_dof_indices.resize     (dofs_per_cell);
          cell_system_matrix.reinit (dofs_per_cell,dofs_per_cell);
          cell_rhs.reinit (dofs_per_cell);

          cell_system_matrix = 0;
          cell_rhs = 0;

          right_hand_side.value_list (fe_values.get_quadrature_points(),
                                      rhs_values);

          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              for (unsigned int j=i; j<dofs_per_cell; ++j)
                {
                  cell_system_matrix(i,j) += (fe_values.shape_grad(i,q_point) *
                                              fe_values.shape_grad(j,q_point) *
                                              fe_values.JxW(q_point));
                  cell_rhs(i) += (rhs_values[q_point] *
                                  fe_values.shape_value(i,q_point) *
                                  fe_values.JxW(q_point));
                }

          // exploit symmetry
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=i; j<dofs_per_cell; ++j)
              {
                cell_system_matrix (j, i) = cell_system_matrix (i, j);
              }

          cell->get_dof_indices (local_dof_indices);

          constraints
          .distribute_local_to_global (cell_system_matrix,
                                       cell_rhs,
                                       local_dof_indices,
                                       system_matrix,
                                       system_rhs);
        }

    system_matrix.compress (VectorOperation::add);
    system_rhs.compress (VectorOperation::add);

    pcout << "---assemble_system complete" << std::endl;

  }

  template <int dim>
  unsigned int LaplaceProblem<dim>::solve()
  {
    SolverControl           solver_control (max_iterations,
                                            tolerance);
    PETScWrappers::SolverCG cg (solver_control,
                                mpi_communicator);
    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);

    Vector<double> localized_solution (solution);
//    pcout << "local solution " << localized_solution.size() << ":" << solution.size() << std::endl;

    constraints.distribute (localized_solution);
    solution = localized_solution;

    return solver_control.last_step();

    pcout << "---solve step complete" << std::endl;
  }

  template <int dim>
  LaplaceProblem<dim>::~LaplaceProblem()
  {
    dof_handler.clear ();
  }

  template <int dim>
  void LaplaceProblem<dim>::estimate_error ()
  {
    {
      const PETScWrappers::MPI::Vector *sol;
      Vector<float>   *error;

      hp::QCollection<dim-1> face_quadrature_formula;
      face_quadrature_formula.push_back (QGauss<dim-1>(3));
      face_quadrature_formula.push_back (QGauss<dim-1>(3));

//       KellyErrorEstimator<dim>::estimate (dof_handler,
//                                           face_quadrature_formula,
//                                           typename FunctionMap<dim>::type(),
//                                           sol,
//                                           error);
    }
  }

  template <int dim>
  void LaplaceProblem<dim>::refine_grid ()
  {
//     const double threshold = 0.9 * estimated_error_per_cell.linfty_norm();
//     GridRefinement::refine (triangulation,
//                             estimated_error_per_cell,
//                             threshold);

//     triangulation.prepare_coarsening_and_refinement ();
//     triangulation.execute_coarsening_and_refinement ();
  }

  template <int dim>
  void LaplaceProblem<dim>::output_results (const unsigned int cycle) const
  {
    Vector<float> fe_index(triangulation.n_active_cells());
    {
      typename hp::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
      for (unsigned int index=0; cell!=endc; ++cell, ++index)
        {
          fe_index(index) = cell->active_fe_index();
        }
    }

    Assert (cycle < 10, ExcNotImplemented());
    if (this_mpi_process==0)
      {
        std::string filename = "solution-";
        filename += ('0' + cycle);
        filename += ".vtk";
        std::ofstream output (filename.c_str());

        DataOut<dim,hp::DoFHandler<dim> > data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, "solution");
        data_out.build_patches (6);
        data_out.write_vtk (output);
        output.close();
      }
  }

  template <int dim>
  void LaplaceProblem<dim>::output_test ()
  {
    std::string file_name = "output";

//     std::vector<Vector<float>> shape_funct(dof_handler.n_dofs(),
//                                            Vector<float>(dof_handler.n_dofs()));
//
//     for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)
//     {
//       shape_funct[i] = i;
//       constraints.distribute(shape_funct[i]);
//     }

    active_fe_index.reinit(triangulation.n_active_cells());
    typename hp::DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
    for (unsigned int index=0; cell!=endc; ++cell, ++index)
      {
        active_fe_index[index] = cell->active_fe_index();
      }

    // set the others correctly

    // second output without making mesh finer
    if (this_mpi_process==0)
      {
        file_name += ".vtk";
        std::ofstream output (file_name.c_str());


        DataOut<dim,hp::DoFHandler<dim> > data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (active_fe_index, "fe_index");
        data_out.add_data_vector (color_output, "colors");
        for (unsigned int i = 0; i < predicate_output.size(); ++i)
          data_out.add_data_vector (predicate_output[i], "predicate_" + std::to_string(i));

        /*
        for (unsigned int i = 0; i < shape_funct.size(); ++i)
          if (shape_funct[i].l_infty() > 0)
            data_out.add_data_vector (shape_funct[i], "shape_funct_" + std::to_string(i));
        */

        data_out.build_patches (); // <--- this is the one to additionally refine cells for output only
        data_out.write_vtk (output);
      }

  }


  template <int dim>
  void
  LaplaceProblem<dim>::run()
  {
    //TODO need cyles?
    for (unsigned int cycle = 0; cycle < 1; ++cycle)
      {
        pcout << "Cycle "<< cycle <<std::endl;
        setup_system ();

        pcout << "   Number of active cells:       "
              << triangulation.n_active_cells ()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs ()
              << std::endl;

        plot_shape_function<dim>(dof_handler);

        assemble_system ();
        auto n_iterations = solve ();

        //TODO Uncomment. correct function body
//        estimate_error ();

        output_results(cycle);

        //TODO UNCOMMENT. correct function body
//        refine_grid ();

        //TODO COMMENT after uncommenting refine grid function
        triangulation.refine_global(1);

        pcout << "Number of iterations " << n_iterations << std::endl;

        pcout << "---step run complete" << std::endl;
      }
  }
}


int main (int argc,char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      {
        Step1::LaplaceProblem<dim> step1(argc,argv);
//         PETScWrappers::set_option_value("-eps_target","-1.0");
//         PETScWrappers::set_option_value("-st_type","sinvert");
//         PETScWrappers::set_option_value("-st_ksp_type","cg");
//         PETScWrappers::set_option_value("-st_pc_type", "jacobi");
        step1.run();
      }

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl   << std::endl
                << "--------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "--------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "--------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "--------------------------------"
                << std::endl;
      return 1;
    };
}
