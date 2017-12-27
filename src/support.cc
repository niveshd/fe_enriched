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
  
  return SparsityTools::color_sparsity_pattern (sp_graph, predicate_colors); 
}
  
template unsigned int color_predicates
  (const hp::DoFHandler<2,2> &mesh,
   const std::vector<EnrichmentPredicate<2>> &,
   std::vector<unsigned int> &);
  
template unsigned int color_predicates
  (const hp::DoFHandler<3,3> &mesh,
   const std::vector<EnrichmentPredicate<3>> &,
   std::vector<unsigned int> &);