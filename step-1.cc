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


// 

#include "../tests.h"

#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

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

#include <map>

//TODO need 3d test example?
const unsigned int dim = 2;

using namespace dealii;

template <int dim>
struct EnrichmentPredicate
{
    EnrichmentPredicate(const Point<dim> origin, const int radius)
    :origin(origin),radius(radius){}

    template <class Iterator>
    bool operator () (const Iterator &i)
    { 
        return ( (i->center() - origin).norm_square() < radius*radius);
    }    
    
    const Point<dim> &get_origin() { return origin; }
    const int &get_radius() { return radius; }

    private:
        const Point<dim> origin;
        const int radius;   
};



template <int dim>
class RightHandSide :  public Function<dim>
{
public:
  RightHandSide ();
  virtual void value (const Point<dim> &p,
                      double   &values) const;
  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double >           &value_list) const;
};


template <int dim>
RightHandSide<dim>::RightHandSide () :
  Function<dim> ()
{}


template <int dim>
inline
void RightHandSide<dim>::value (const Point<dim> &p,
                                double           &values) const
{
  Assert (dim >= 2, ExcInternalError());
  
  values = 1;
}


template <int dim>
void RightHandSide<dim>::value_list (const std::vector<Point<dim> > &points,
                                     std::vector<double >           &value_list) const
{
  const unsigned int n_points = points.size();

  AssertDimension(points.size(), value_list.size());
  
  for (unsigned int p=0; p<n_points; ++p)
    RightHandSide<dim>::value (points[p],
                                      value_list[p]);
}


template <int dim>
class EnrichmentFunction : public Function<dim>
{
public:
  EnrichmentFunction(const Point<dim> &origin,
                     const double     &Z,
                     const double     &radius)
    : Function<dim>(1),
      origin(origin),
      Z(Z),
      radius(radius)
  {}

  virtual double value(const Point<dim> &point,
                       const unsigned int component = 0) const
  {
    Tensor<1,dim> dist = point-origin;
    const double r = dist.norm();
    return std::exp(-Z*r);
  }

  //TODO remove?
  bool is_enriched(const Point<dim> &point) const
  {
    if (origin.distance(point) < radius)
      return true;
    else
      return false;
  }

  virtual Tensor< 1, dim> gradient (const Point<dim > &p,
                                    const unsigned int component=0) const
  {
    Tensor<1,dim> dist = p-origin;
    const double r = dist.norm();
    Assert (r > 0.,
            ExcDivideByZero());
    dist/=r;
    return -Z*std::exp(-Z*r)*dist;
  }

private:
  /**
   * origin
   */
  const Point<dim> origin;

  /**
   * charge
   */
  const double Z;

  /**
   * enrichment radius
   */
  const double radius;
};

  

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
    virtual ~LaplaceProblem();
    void run ();

  private:
    void build_tables ();
    void setup_system ();
    void assemble_system ();
    unsigned int solve ();
    void estimate_error ();
    void refine_grid ();
    void output_results (const unsigned int cycle) const;
    void output_test (std::string file_name) const;

    Triangulation<dim>  triangulation;
    hp::DoFHandler<dim> dof_handler;
    hp::FECollection<dim> fe_collection;
    hp::QCollection<dim> q_collection;
    
    FE_Q<dim> fe_base;
    FE_Q<dim> fe_enriched;

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

    std::vector<EnrichmentFunction<dim>> vec_enrichments;     
    std::vector<EnrichmentPredicate<dim>> vec_predicates;
    
    //each predicate is assigned a color depending on overlap of vertices.
    std::vector<unsigned int> predicate_colors;
    size_t num_colors;
    
    //cell wise mapping of color with enrichment functions
    //vector size = number of cells;
    //map:  < color , corresponding predicate_function >
    //(only one per color acceptable). An assertion checks this
    std::vector <std::map<size_t, size_t>> color_predicate_table;
    
    //vector of size num_colors + 1
    //different color combinations that a cell can contain
    std::vector <std::set<size_t>> material_table;
    
    const FEValuesExtractors::Scalar fe_extractor;
    const FEValuesExtractors::Scalar pou_extractor;

    std::vector<Vector<float> > vec_estimated_error_per_cell;
    Vector<float> estimated_error_per_cell;    
  };

  template <int dim>
  LaplaceProblem<dim>::LaplaceProblem ()
    :
    dof_handler (triangulation),
    fe_base(2),
    fe_enriched(1),
    mpi_communicator(MPI_COMM_WORLD),
    n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process(Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout (std::cout, (this_mpi_process == 0)),
    fe_extractor(/*dofs start at...*/0),
    //TODO not used anywhere!
    pou_extractor(/*dofs start at (scalar fields!)*/1)
  {
    GridGenerator::hyper_cube (triangulation, -20, 20);
    triangulation.refine_global (4);
    
    //TODO undo? no refinement according to material.
//     for (auto cell= triangulation.begin_active(); cell != triangulation.end(); ++cell)
//       if (std::sqrt(cell->center().square()) < 5.0)
//         cell->set_refine_flag();

//     triangulation.execute_coarsening_and_refinement(); // 120 cells
    
    //initialize vector of vec_predicates
    vec_predicates.reserve(5);
    vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(-7.5,7.5), 2) );
    vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(-5,5), 2) );
    vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(0,0), 2) );
    vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(5,-5), 2) );
    vec_predicates.push_back( EnrichmentPredicate<dim>(Point<dim>(10,-10), 2) );
    
    //vector of enrichment functions
    vec_enrichments.reserve( vec_predicates.size() );
    for (size_t i=0; i<vec_predicates.size(); ++i)
    {
        EnrichmentFunction<dim> func( vec_predicates[i].get_origin(),
                                      1,
                                      vec_predicates[i].get_radius() );
        vec_enrichments.push_back( func );
            
    }    

    {
    //set material id based on predicate functions. TODO comment
    for (typename hp::DoFHandler<dim>::cell_iterator cell= dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
    {
      cell->set_active_fe_index(0);
        for (size_t i=0; i<vec_predicates.size(); ++i)
            if ( vec_predicates[i](cell) )
                cell->set_active_fe_index(i+1);
    }
    
    output_test("predicate");
    }
             
    //make a sparsity pattern based on connections between regions
    unsigned int num_indices = vec_predicates.size();
    DynamicSparsityPattern dsp;
    dsp.reinit ( num_indices, num_indices );
    for (size_t i = 0; i < num_indices; ++i)
        for (size_t j = i+1; j < num_indices; ++j)
            if ( GridTools::find_connection_between_subdomains(dof_handler, vec_predicates[i], vec_predicates[j]) )
                dsp.add(i,j);
    
    dsp.symmetrize();
    
    //color different regions (defined by predicate)
    SparsityPattern sp_graph;
    sp_graph.copy_from(dsp);
    
    Assert( num_indices == sp_graph.n_rows() , ExcInternalError() );
    predicate_colors.resize(num_indices);
    num_colors = SparsityTools::color_sparsity_pattern (sp_graph, predicate_colors);
    
    {//TODO comment
    //print color_indices
    for (size_t i=0; i<num_indices; ++i)
        pcout << "predicate " << i << " : " << predicate_colors[i] << std::endl;
    

    //set material id based on color.
    for (typename hp::DoFHandler<dim>::cell_iterator cell= dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
    {
    cell->set_active_fe_index(0);
        for (size_t i=0; i<vec_predicates.size(); ++i)
            if ( vec_predicates[i](cell) )
                cell->set_active_fe_index (predicate_colors[i]);
    }
    
    output_test("color"); 
    }
    
    //build fe table. should be called everytime number of cells change!
    build_tables();
        
   
    //q collections the same size as different material identities
    q_collection.push_back(QGauss<dim>(4));
    
    for (size_t i=1; i!=material_table.size(); ++i)
        q_collection.push_back(QGauss<dim>(10));
    
    // usual elements (active_fe_index ==0):
    fe_collection.push_back (FE_Enriched<dim> (fe_base));
    
    for (size_t i=1; i !=material_table.size(); ++i)   
    {
        //set vector base elements for fe enrichment
        //if {1,2} are color of fe element, 2 fe elements need to be enriched.
        std::vector<const FiniteElement<dim> *> vec_fe_enriched;          
        vec_fe_enriched.assign (material_table[i].size(), &fe_enriched);
            
        //set functions based on cell accessor
        //cell accessor --> cell id --> color --> appropriate enrichment function
        std::vector<std::vector<std::function<const Function<dim> *
            (const typename Triangulation<dim, dim>::cell_iterator &) > > >
                functions(material_table[i].size());
        
        size_t ind = 0;
        for (auto it=material_table[i].begin();
             it != material_table[i].end();
             ++it, ++ind)
        {
            auto func = [&] 
            (const typename Triangulation<dim, dim>::cell_iterator & cell)
            {
                size_t id = cell->index();
                return &vec_enrichments[color_predicate_table[id][i]];
            };
            functions[ind].push_back(func);               
            
        }
        
        AssertDimension(vec_fe_enriched.size(), functions.size());
        
        {
          pcout << "\nenrichment:"
                << "fe enriched - " << vec_fe_enriched.size() << std::endl;
          
          for (size_t i = 0; i < functions.size(); ++i)
          {
            pcout << "size " << functions[i].size() << std::endl;
          }
          
        }
                                            
        fe_collection.push_back (FE_Enriched<dim> (&fe_base,
                                                   vec_fe_enriched,
                                                   functions));
    }      
  }

  template <int dim>
  void LaplaceProblem<dim>::build_tables ()
  {
    bool found = false;
    color_predicate_table.resize( triangulation.n_cells() ); 
    
    //set first element of material_table size to empty
    material_table.resize(1);
    
    
    //loop throught cells and build fe table
    auto cell= triangulation.begin_active();
    size_t cell_index = 0;
    auto cell_test = triangulation.begin_active();
    for (typename hp::DoFHandler<dim>::cell_iterator cell= dof_handler.begin_active();
         cell != dof_handler.end(); ++cell, ++cell_index)
    {
        cell->set_active_fe_index (0);  //No enrichment at all
        std::set<size_t> color_list;
        
        //loop through predicate function to find connected subdomains
        //connections between same color regions is checked again.
        for (size_t i=0; i<vec_predicates.size(); ++i)
        {        
            //add if predicate true to vector of functions
            if (vec_predicates[i](cell))
            {
                //add color and predicate pair to each cell if predicate is true.
                auto ret = color_predicate_table[cell_index].insert
                    (std::pair <size_t, size_t> (predicate_colors[i], i));  
                
                color_list.insert(predicate_colors[i]);
                
                //A single predicate for a single color! repeat addition not accepted.
                Assert( ret.second == true, ExcInternalError () );                           
                
//                 pcout << " - " << predicate_colors[i] << "(" << i << ")" ;
            }            
        }
        
        if (!color_list.empty())
//                 pcout << std::endl;
        
        found = false;
        //check if color combination is already added
        if ( !color_list.empty() )
        {
            for ( size_t j=0; j<material_table.size(); j++)
            {
                if (material_table[j] ==  color_list)
                {
//                     pcout << "color combo set found at " << j << std::endl;
                    found=true;
                    cell->set_active_fe_index(j);                    
                    break;
                }
            }
            
            if (!found){
                material_table.push_back(color_list);
                cell->set_active_fe_index(material_table.size()-1);
//                 pcout << "color combo set pushed at " << material_table.size()-1 << std::endl;
            }     
        }
    }
    
    output_test("final");
    
    //print material table
    pcout << "\nMaterial table : " << std::endl;
    for ( size_t j=0; j<material_table.size(); j++)
    {
        pcout << j << " ( ";
        for ( auto i = material_table[j].begin(); i != material_table[j].end(); i++)
            pcout << *i << " ";
        pcout << " ) " << std::endl;
    }
  }

  template <int dim>
  void LaplaceProblem<dim>::setup_system ()
  {

    GridTools::partition_triangulation (n_mpi_processes, triangulation);
    dof_handler.distribute_dofs (fe_collection);
    DoFRenumbering::subdomain_wise (dof_handler);
    
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
    for (unsigned int i = 0; i < n_mpi_processes; i++)
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

    estimated_error_per_cell.reinit (triangulation.n_active_cells());
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
  }

  template <int dim>
  unsigned int LaplaceProblem<dim>::solve()
  {
    SolverControl           solver_control (solution.size(),
                                          1e-8*system_rhs.l2_norm());
    PETScWrappers::SolverCG cg (solver_control,
                                mpi_communicator);
    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);
    
    Vector<double> localized_solution (solution);
    pcout << "local solution" << localized_solution.size() << ":" << solution << std::endl;
    
    constraints.distribute (localized_solution);
    solution = localized_solution;    
    
    return solver_control.last_step();
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
      const PETScWrappers::MPI::Vector * sol;
      Vector<float> *  error;

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
    const double threshold = 0.9 * estimated_error_per_cell.linfty_norm();
    GridRefinement::refine (triangulation,
                            estimated_error_per_cell,
                            threshold);

    triangulation.prepare_coarsening_and_refinement ();
    triangulation.execute_coarsening_and_refinement ();
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
  void LaplaceProblem<dim>::output_test (std::string file_name) const
  {
    Vector<float> active_fe_index(triangulation.n_active_cells());
    {
      typename hp::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
      for (unsigned int index=0; cell!=endc; ++cell, ++index)
      {
        active_fe_index(index) = cell->active_fe_index();
      }
    }

    // second output without making mesh finer
    if (this_mpi_process==0)
      {
        file_name += ".vtk";
        std::ofstream output (file_name.c_str());
        

        DataOut<dim,hp::DoFHandler<dim> > data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (active_fe_index, "fe_index");
        data_out.build_patches ();
        data_out.write_vtk (output);
        output.close();
      }

  }


  template <int dim>
  void
  LaplaceProblem<dim>::run()
  {
    //TODO cycle to be set to 4 after testing
    for (unsigned int cycle = 0; cycle < 1; cycle++)
      {
        pcout << "Cycle "<<cycle <<std::endl;
//         setup_system ();

        pcout << "   Number of active cells:       "
              << triangulation.n_active_cells ()
              << std::endl
              << "   Number of degrees of freedom: "
              << dof_handler.n_dofs ()
              << std::endl;

//         assemble_system ();

    //TODO uncomment - do not solve for 2d
//         auto n_iterations = solve ();

//         estimate_error ();
//         output_results(cycle);
        //refine_grid ();

//         pcout << "number of iterations" << n_iterations << std::endl;
      }
  }
}


int main (int argc,char **argv)
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
      {
        Step1::LaplaceProblem<dim> step1;
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
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    };   
}

