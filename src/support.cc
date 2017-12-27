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

#include <vector>
#include <map>
#include "support.h"

template <int dim, class MeshType>
unsigned int color_predicates
  (const MeshType &mesh,
   const std::vector<EnrichmentPredicate<dim>> &vec_predicates,
   std::vector<unsigned int> &predicate_colors) 
{
  unsigned int num_indices = vec_predicates.size();
  
  DynamicSparsityPattern dsp;
  dsp.reinit ( num_indices, num_indices );
  
  //find connections between subdomains defined by predicates
  for (unsigned int i = 0; i < num_indices; ++i)
      for (unsigned int j = i+1; j < num_indices; ++j)
          if ( GridTools::find_connection_between_subdomains
                  (mesh, vec_predicates[i], vec_predicates[j]) )
              dsp.add(i,j);
  
  dsp.symmetrize();
  
  //color different regions (defined by predicate)
  SparsityPattern sp_graph;
  sp_graph.copy_from(dsp);
  
  Assert( num_indices == sp_graph.n_rows() , ExcInternalError() );
  predicate_colors.resize(num_indices);
  
  //return num of colors and assign each predicate with a color
  return SparsityTools::color_sparsity_pattern (sp_graph, predicate_colors); 
}



template <int dim, class MeshType>
void 
set_cellwise_color_set_and_fe_index
  (MeshType &mesh,
   const std::vector<EnrichmentPredicate<dim>> &vec_predicates,
   const std::vector<unsigned int> &predicate_colors,
   std::map<unsigned int,
      std::map<unsigned int, unsigned int> > 
        &cellwise_color_predicate_map,
   std::vector <std::set<unsigned int>> &color_sets)
{
    //set first element of color_sets size to empty
    color_sets.resize(1);    
    
    //loop throught cells and build fe table
    unsigned int cell_index = 0;
    
    auto cell = mesh.begin_active();
    auto endc = mesh.end();
    for (unsigned int cell_index=0;
         cell != endc; ++cell, ++cell_index)
    {
        cell->set_active_fe_index (0);  //No enrichment at all
        std::set<unsigned int> color_list;
        
        //loop through predicate function to find connected subdomains
        //connections between same color regions is checked again.
        for (unsigned int i=0; i<vec_predicates.size(); ++i)
        {        
            //add if predicate true to vector of functions
            if (vec_predicates[i](cell))
            {
                //add color and predicate pair to each cell if predicate is true.
                auto ret = cellwise_color_predicate_map[cell_index].insert
                    (std::pair <unsigned int, unsigned int> (predicate_colors[i], i));  
                
                color_list.insert(predicate_colors[i]);
                
                //A single predicate for a single color! repeat addition not accepted.
                Assert( ret.second == true, ExcInternalError () );                           
                
//                 pcout << " - " << predicate_colors[i] << "(" << i << ")" ;
            }            
        }
        
//         if (!color_list.empty())
//                 pcout << std::endl;
        
        bool found = false;
        //check if color combination is already added
        if ( !color_list.empty() )
        {
            for ( unsigned int j=0; j<color_sets.size(); ++j)
            {
                if (color_sets[j] ==  color_list)
                {
//                     pcout << "color combo set found at " << j << std::endl;
                    found=true;
                    cell->set_active_fe_index(j);                    
                    break;
                }
            }


            if (!found){
                color_sets.push_back(color_list);
                cell->set_active_fe_index(color_sets.size()-1);
                /*
                num_colors+1 = (num_colors+1 > color_list.size())?
                                       num_colors+1:
                                       color_list.size();
//                 pcout << "color combo set pushed at " << color_sets.size()-1 << std::endl;
                */
            } 
            
        }
    }
}


template <int dim>
void make_colorwise_enrichment_functions
  (const unsigned int &num_colors,          //needs number of colors
   
   const std::vector<EnrichmentFunction<dim>> 
    &vec_enrichments,     //enrichment functions based on predicate id
   
   std::map<unsigned int,
    std::map<unsigned int, unsigned int> >
      &cellwise_color_predicate_map,
   
   std::vector<
    std::function<const Function<dim>*
      (const typename Triangulation<dim>::cell_iterator&)> >
        &color_enrichments)   //colorwise enrichment functions indexed from 0!
    //color_enrichments[0] is color 1 enrichment function
  {
    color_enrichments.resize (num_colors);
    
    for (unsigned int i = 0; i < num_colors; ++i)
    {
      color_enrichments[i] =  
        [&,i] (const typename Triangulation<dim, dim>::cell_iterator & cell)
            {
                unsigned int id = cell->index();
                
                //i'th color corresponds to i+1 function
                return &vec_enrichments[cellwise_color_predicate_map[id][i+1]];
            };
    }    
  }


template <int dim>  
void make_fe_collection_from_colored_enrichments
  (
    const unsigned int &num_colors,
    const std::vector <std::set<unsigned int>> 
      &color_sets,         //total list of color sets possible
   
    const std::vector<
      std::function<const Function<dim>*
        (const typename Triangulation<dim>::cell_iterator&)> >
          &color_enrichments,  //color wise enrichment functions
   
    const FE_Q<dim> &fe_base,            //basic fe element
    const FE_Q<dim> &fe_enriched,        //fe element multiplied by enrichment function
    const FE_Nothing<dim> &fe_nothing,
    hp::FECollection<dim> &fe_collection
  )
{
  std::vector<const FiniteElement<dim> *> vec_fe_enriched;
  std::vector<std::vector<std::function<const Function<dim> *
          (const typename Triangulation<dim, dim>::cell_iterator &) >>>
          functions;
          
  for (unsigned int color_set_id=0; color_set_id!=color_sets.size(); ++color_set_id)   
    {
        vec_fe_enriched.assign(num_colors, &fe_nothing);
        functions.assign(num_colors, {nullptr});
                            
        //ind = 0 means color id         
        unsigned int ind = 0;
        for (auto it=color_sets[color_set_id].begin();
             it != color_sets[color_set_id].end();
             ++it)
        {
            ind = *it-1;
            AssertIndexRange(ind, vec_fe_enriched.size());
            
            vec_fe_enriched[ind] = &fe_enriched;

            AssertIndexRange(ind, functions.size());
            AssertIndexRange(ind, color_enrichments.size());
            
            //color_set_id'th color function is (color_set_id-1) element of color wise enrichments
            functions[ind].assign(1,color_enrichments[ind]); 
        }
        
        AssertDimension(vec_fe_enriched.size(), functions.size());
        
        FE_Enriched<dim> fe_component(&fe_base,
                                     vec_fe_enriched,
                                     functions);
      
      {//TODO delete after testing
      ConditionalOStream pcout
        (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
        
      pcout << "Function set : \t ";
      for (auto enrichment_function_array : functions )
        for (auto func_component : enrichment_function_array)
          if (func_component)
            pcout << " X ";
          else
            pcout << " O ";
          
      pcout << std::endl;
      }
                                            
        fe_collection.push_back (fe_component);
    }  
}


//template instantiations  
template unsigned int color_predicates
  (const hp::DoFHandler<2,2> &mesh,
   const std::vector<EnrichmentPredicate<2>> &,
   std::vector<unsigned int> &);
  
  
template unsigned int color_predicates
  (const hp::DoFHandler<3,3> &mesh,
   const std::vector<EnrichmentPredicate<3>> &,
   std::vector<unsigned int> &);
  
  
template
void
set_cellwise_color_set_and_fe_index
  (hp::DoFHandler<2,2> &mesh,
   const std::vector<EnrichmentPredicate<2>> &vec_predicates,
   const std::vector<unsigned int> &predicate_colors,
   std::map<unsigned int,
      std::map<unsigned int, unsigned int> > 
        &cellwise_color_predicate_map,
   std::vector <std::set<unsigned int>> &color_sets); 
  
  
template
void
set_cellwise_color_set_and_fe_index
  (hp::DoFHandler<3,3> &mesh,
   const std::vector<EnrichmentPredicate<3>> &vec_predicates,
   const std::vector<unsigned int> &predicate_colors,
   std::map<unsigned int,
      std::map<unsigned int, unsigned int> > 
        &cellwise_color_predicate_map,
   std::vector <std::set<unsigned int>> &color_sets); 
  
  
  
template
void make_colorwise_enrichment_functions
  (const unsigned int &num_colors,          //needs number of colors
   
   const std::vector<EnrichmentFunction<2>> 
    &vec_enrichments,     //enrichment functions based on predicate id
   
   std::map<unsigned int,
    std::map<unsigned int, unsigned int> >
      &cellwise_color_predicate_map,
   
   std::vector<
    std::function<const Function<2>*
      (const typename Triangulation<2>::cell_iterator&)> >
        &color_enrichments);
  
template
void make_colorwise_enrichment_functions
  (const unsigned int &num_colors,          //needs number of colors
   
   const std::vector<EnrichmentFunction<3>> 
    &vec_enrichments,     //enrichment functions based on predicate id
   
   std::map<unsigned int,
    std::map<unsigned int, unsigned int> >
      &cellwise_color_predicate_map,
   
   std::vector<
    std::function<const Function<3>*
      (const typename Triangulation<3>::cell_iterator&)> >
        &color_enrichments);
  
  
template  
void make_fe_collection_from_colored_enrichments
  (
    const unsigned int &num_colors,
    const std::vector <std::set<unsigned int>> 
      &color_sets,         //total list of color sets possible
   
    const std::vector<
      std::function<const Function<2>*
        (const typename Triangulation<2>::cell_iterator&)> >
          &color_enrichments,  //color wise enrichment functions
   
    const FE_Q<2> &fe_base,            //basic fe element
    const FE_Q<2> &fe_enriched,        //fe element multiplied by enrichment function
    const FE_Nothing<2> &fe_nothing,
    hp::FECollection<2> &fe_collection
  );
  
  
template  
void make_fe_collection_from_colored_enrichments
  (
    const unsigned int &num_colors,
    const std::vector <std::set<unsigned int>> 
      &color_sets,         //total list of color sets possible
   
    const std::vector<
      std::function<const Function<3>*
        (const typename Triangulation<3>::cell_iterator&)> >
          &color_enrichments,  //color wise enrichment functions
   
    const FE_Q<3> &fe_base,            //basic fe element
    const FE_Q<3> &fe_enriched,        //fe element multiplied by enrichment function
    const FE_Nothing<3> &fe_nothing,
    hp::FECollection<3> &fe_collection
  );