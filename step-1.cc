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


// test FE_Enriched in real-life application on eigenvalue problem similar
// to Step-36. That involves assembly (shape values and gradients) and
// error estimator (Kelly - > face gradients) and MPI run.

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
        return ( (i->center() - origin).norm() < radius);
    }    
    
    const Point<dim> &get_origin() { return origin; }
    const int &get_radius() { return radius; }

    private:
        const Point<dim> origin;
        const int radius;   
};

/**
 * Coulomb potential
 */
template <int dim>
class PotentialFunction : public Function<dim>
{
public:
  PotentialFunction()
    : Function<dim>(1)
  {}

  virtual double value(const Point<dim> &point,
                       const unsigned int component = 0 ) const;
};

template <int dim>
double PotentialFunction<dim>::value(const Point<dim> &p,
                                     const unsigned int ) const
{
  Assert (p.square() > 0.,
          ExcDivideByZero());
  return -1.0 / std::sqrt(p.square());
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
    bool cell_is_pou(const typename hp::DoFHandler<dim>::cell_iterator &cell) const;
    
    void build_tables ();
    std::pair<unsigned int, unsigned int> setup_system ();
    void assemble_system ();
    std::pair<unsigned int, double> solve ();
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

    std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
    std::vector<PETScWrappers::MPI::Vector> eigenfunctions_locally_relevant;
    std::vector<PetscScalar>                eigenvalues;
    PETScWrappers::MPI::SparseMatrix        stiffness_matrix, mass_matrix;

    ConstraintMatrix constraints;

    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;

    ConditionalOStream pcout;

    const unsigned int number_of_eigenvalues;

    PotentialFunction<dim> potential;

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
    const unsigned int               fe_fe_index;
    const unsigned int               fe_material_id;
    const FEValuesExtractors::Scalar pou_extractor;
    const unsigned int               pou_fe_index;
    const unsigned int               pou_material_id;

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
    pcout (std::cout,
           (this_mpi_process == 0)),
    number_of_eigenvalues(1),
    fe_extractor(/*dofs start at...*/0),
    fe_fe_index(0),
    fe_material_id(0),
    pou_extractor(/*dofs start at (scalar fields!)*/1),
    pou_fe_index(1),
    pou_material_id(1)
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
    //set material id based on predicate functions. TODO undo
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
    
    {//TODO remove
    //print color_indices
    for (size_t i=0; i<num_indices; ++i)
        std::cout << "predicate " << i << " : " << predicate_colors[i] << std::endl;
    

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
        
        Assert (vec_fe_enriched.size() == functions.size(), 
                ExcDimensionMismatch(vec_fe_enriched.size(), functions.size()));
                                            
        fe_collection.push_back (FE_Enriched<dim> (&fe_base,
                                                   vec_fe_enriched,
                                                   functions));
    }
    
    //TODO Assert q collection and fe collection are of same size

      
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
    for (typename hp::DoFHandler<dim>::cell_iterator cell= dof_handler.begin_active();
         cell != dof_handler.end(); ++cell, ++cell_index)
    {
        cell->set_active_fe_index (0);  //No enrichment at all
        std::set<size_t> color_list;
        
        //loop through predicate function. connections between same color regions is also done.
        //doesn't matter though.
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
                
                std::cout << " - " << predicate_colors[i] << "(" << i << ")" ;
            }            
        }
        
        if (!color_list.empty())
                std::cout << std::endl;
        
        found = false;
        //check if color combination is already added
        if ( !color_list.empty() )
        {
            for ( size_t j=0; j<material_table.size(); j++)
            {
                if (material_table[j] ==  color_list)
                {
                    std::cout << "color combo set found at " << j << std::endl;
                    found=true;
                    cell->set_active_fe_index(j);                    
                    break;
                }
            }
            
            if (!found){
                material_table.push_back(color_list);
                cell->set_active_fe_index(material_table.size()-1);
                std::cout << "color combo set pushed at " << material_table.size()-1 << std::endl;
            }     
        }
    }
    
    output_test("final");
    
    //print material table
    std::cout << "Material table " << std::endl;
    for ( size_t j=0; j<material_table.size(); j++)
    {
        std::cout << j << " ( ";
        for ( auto i = material_table[j].begin(); i != material_table[j].end(); i++)
            std::cout << *i << " ";
        std::cout << " ) " << std::endl;
    }
  }

  template <int dim>
  std::pair<unsigned int, unsigned int>
  LaplaceProblem<dim>::setup_system ()
  {
    for (typename hp::DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      {
        if (cell->material_id() == fe_material_id)
          cell->set_active_fe_index (fe_fe_index);
        else if (cell->material_id() == pou_material_id)
          cell->set_active_fe_index (pou_fe_index);
        else
          Assert (false, ExcNotImplemented());
      }

    GridTools::partition_triangulation (n_mpi_processes, triangulation);
    dof_handler.distribute_dofs (fe_collection);
    DoFRenumbering::subdomain_wise (dof_handler);
    std::vector<IndexSet> locally_owned_dofs_per_processor
      = DoFTools::locally_owned_dofs_per_subdomain (dof_handler);
    locally_owned_dofs = locally_owned_dofs_per_processor[this_mpi_process];
    locally_relevant_dofs.clear();
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);

    constraints.clear();
    constraints.reinit (locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints  (dof_handler, constraints);
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              Functions::ZeroFunction<dim> (1),
                                              constraints);
    constraints.close ();

    // Initialise the stiffness and mass matrices
    DynamicSparsityPattern csp (locally_relevant_dofs);
    DoFTools::make_sparsity_pattern (dof_handler, csp,
                                     constraints,
                                     /* keep constrained dofs */ false);

    std::vector<types::global_dof_index> n_locally_owned_dofs(n_mpi_processes);
    for (unsigned int i = 0; i < n_mpi_processes; i++)
      n_locally_owned_dofs[i] = locally_owned_dofs_per_processor[i].n_elements();

    SparsityTools::distribute_sparsity_pattern
    (csp,
     n_locally_owned_dofs,
     mpi_communicator,
     locally_relevant_dofs);

    stiffness_matrix.reinit (locally_owned_dofs,
                             locally_owned_dofs,
                             csp,
                             mpi_communicator);

    mass_matrix.reinit (locally_owned_dofs,
                        locally_owned_dofs,
                        csp,
                        mpi_communicator);

    // reinit vectors
    eigenfunctions.resize (number_of_eigenvalues);
    eigenfunctions_locally_relevant.resize(number_of_eigenvalues);
    vec_estimated_error_per_cell.resize(number_of_eigenvalues);
    for (unsigned int i=0; i<eigenfunctions.size (); ++i)
      {
        eigenfunctions[i].reinit (locally_owned_dofs, mpi_communicator);//without ghost dofs
        eigenfunctions_locally_relevant[i].reinit (locally_owned_dofs,
                                                   locally_relevant_dofs,
                                                   mpi_communicator);

        vec_estimated_error_per_cell[i].reinit(triangulation.n_active_cells());
      }

    eigenvalues.resize (eigenfunctions.size ());

    estimated_error_per_cell.reinit (triangulation.n_active_cells());

    unsigned int n_pou_cells = 0,
                 n_fem_cells = 0;

    for (typename hp::DoFHandler<dim>::cell_iterator cell= dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
      if (cell_is_pou(cell))
        n_pou_cells++;
      else
        n_fem_cells++;

    return std::make_pair (n_fem_cells,
                           n_pou_cells);
  }

  template <int dim>
  bool
  LaplaceProblem<dim>::cell_is_pou(const typename hp::DoFHandler<dim>::cell_iterator &cell) const
  {
    return cell->material_id() == pou_material_id;
  }


  template <int dim>
  void
  LaplaceProblem<dim>::assemble_system()
  {
    stiffness_matrix = 0;
    mass_matrix = 0;

    FullMatrix<double> cell_stiffness_matrix;
    FullMatrix<double> cell_mass_matrix;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<double> potential_values;
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

          potential_values.resize(n_q_points);

          local_dof_indices.resize     (dofs_per_cell);
          cell_stiffness_matrix.reinit (dofs_per_cell,dofs_per_cell);
          cell_mass_matrix.reinit      (dofs_per_cell,dofs_per_cell);

          cell_stiffness_matrix = 0;
          cell_mass_matrix      = 0;

          potential.value_list (fe_values.get_quadrature_points(),
                                potential_values);

          for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              for (unsigned int j=i; j<dofs_per_cell; ++j)
                {
                  cell_stiffness_matrix (i, j)
                  += (fe_values[fe_extractor].gradient (i, q_point) *
                      fe_values[fe_extractor].gradient (j, q_point) *
                      0.5
                      +
                      potential_values[q_point] *
                      fe_values[fe_extractor].value (i, q_point) *
                      fe_values[fe_extractor].value (j, q_point)
                     ) * fe_values.JxW (q_point);

                  cell_mass_matrix (i, j)
                  += (fe_values[fe_extractor].value (i, q_point) *
                      fe_values[fe_extractor].value (j, q_point)
                     ) * fe_values.JxW (q_point);
                }

          // exploit symmetry
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=i; j<dofs_per_cell; ++j)
              {
                cell_stiffness_matrix (j, i) = cell_stiffness_matrix (i, j);
                cell_mass_matrix (j, i)      = cell_mass_matrix (i, j);
              }

          cell->get_dof_indices (local_dof_indices);

          constraints
          .distribute_local_to_global (cell_stiffness_matrix,
                                       local_dof_indices,
                                       stiffness_matrix);
          constraints
          .distribute_local_to_global (cell_mass_matrix,
                                       local_dof_indices,
                                       mass_matrix);
        }

    stiffness_matrix.compress (VectorOperation::add);
    mass_matrix.compress (VectorOperation::add);
  }

  template <int dim>
  std::pair<unsigned int, double>
  LaplaceProblem<dim>::solve()
  {
    SolverControl solver_control (dof_handler.n_dofs(), 1e-9,false,false);
    SLEPcWrappers::SolverKrylovSchur eigensolver (solver_control,
                                                  mpi_communicator);

    eigensolver.set_which_eigenpairs (EPS_SMALLEST_REAL);
    eigensolver.set_problem_type (EPS_GHEP);

    eigensolver.solve (stiffness_matrix, mass_matrix,
                       eigenvalues, eigenfunctions,
                       eigenfunctions.size());

    for (unsigned int i = 0; i < eigenfunctions.size(); i++)
      {
        constraints.distribute(eigenfunctions[i]);
        eigenfunctions_locally_relevant[i] = eigenfunctions[i];
      }

    return std::make_pair (solver_control.last_step (),
                           solver_control.last_value ());

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
      std::vector<const PETScWrappers::MPI::Vector *> sol (number_of_eigenvalues);
      std::vector<Vector<float> *>   error (number_of_eigenvalues);

      for (unsigned int i = 0; i < number_of_eigenvalues; i++)
        {
          sol[i] = &eigenfunctions_locally_relevant[i];
          error[i] = &vec_estimated_error_per_cell[i];
        }

      hp::QCollection<dim-1> face_quadrature_formula;
      face_quadrature_formula.push_back (QGauss<dim-1>(3));
      face_quadrature_formula.push_back (QGauss<dim-1>(3));

      KellyErrorEstimator<dim>::estimate (dof_handler,
                                          face_quadrature_formula,
                                          typename FunctionMap<dim>::type(),
                                          sol,
                                          error);
    }

    // sum up for a global:
    for (unsigned int c=0; c < estimated_error_per_cell.size(); c++)
      {
        double er = 0.0;
        for (unsigned int i = 0; i < number_of_eigenvalues; i++)
          er += vec_estimated_error_per_cell[i][c] * vec_estimated_error_per_cell[i][c];

        estimated_error_per_cell[c] = sqrt(er);
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
    Vector<float> material_id(triangulation.n_active_cells());
    {
      typename hp::DoFHandler<dim>::active_cell_iterator
      cell = dof_handler.begin_active(),
      endc = dof_handler.end();
      for (unsigned int index=0; cell!=endc; ++cell, ++index)
      {
        material_id(index) = cell->material_id();
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
        data_out.add_data_vector (eigenfunctions_locally_relevant[0], "solution");
        data_out.build_patches (6);
        data_out.write_vtk (output);
        output.close();
      }

    // second output without making mesh finer
    if (this_mpi_process==0)
      {
        std::string filename = "mesh-";
        filename += ('0' + cycle);
        filename += ".vtk";
        std::ofstream output (filename.c_str());

        DataOut<dim,hp::DoFHandler<dim> > data_out;
        data_out.attach_dof_handler (dof_handler);
//         data_out.add_data_vector (fe_index, "fe_index");
        data_out.add_data_vector (material_id, "material_id");
//         data_out.add_data_vector (estimated_error_per_cell, "estimated_error");
        data_out.build_patches ();
        data_out.write_vtk (output);
        output.close();
      }

//     scalar data for plotting
//     output scalar data (eigenvalues, energies, ndofs, etc)
    if (this_mpi_process == 0)
      {
        const std::string scalar_fname = "scalar-data.txt";

        std::ofstream output (scalar_fname.c_str(),
                              std::ios::out |
                              (cycle==0
                               ?
                               std::ios::trunc : std::ios::app));

        std::streamsize max_precision  = std::numeric_limits<double>::digits10;

        std::string sep("  ");
        output << cycle << sep
               << std::setprecision(max_precision)
               << triangulation.n_active_cells() << sep
               << dof_handler.n_dofs () << sep
               << std::scientific;

        for (unsigned int i = 0; i < eigenvalues.size(); i++)
          output << eigenvalues[i] << sep;

        output<<std::endl;
        output.close();
      } //end scope

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
//         const std::pair<unsigned int, unsigned int> n_cells = setup_system ();

//         pcout << "   Number of active cells:       "
//               << triangulation.n_active_cells ()
//               << std::endl
//               << "     FE / POUFE :                "
//               << n_cells.first << " / " << n_cells.second
//               << std::endl
//               << "   Number of degrees of freedom: "
//               << dof_handler.n_dofs ()
//               << std::endl;

//         assemble_system ();

    //TODO uncomment - do not solve for 2d
//         const std::pair<unsigned int, double> res = solve ();
//         AssertThrow(res.second < 5e-8,
//                     ExcInternalError());

//         estimate_error ();
//         output_results(cycle);
//         refine_grid ();
//
//         pcout << std::endl;
//         for (unsigned int i=0; i<eigenvalues.size(); ++i)
//           pcout << "      Eigenvalue " << i
//                 << " : " << eigenvalues[i]
//                 << std::endl;
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
      std::cerr << std::endl << std::endl
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