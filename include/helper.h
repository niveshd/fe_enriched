#include <set>
#include <map>
#include <vector>
#include "functions.h"  //TODO remove
#include "support.h"    //TODO REMOVE
#include "estimate_enrichment.h"    //TODO REMOVE

namespace fe_enriched
{
  namespace internal
  {

  }

  template<int dim>
  struct helper
  {

    helper(const FE_Q<dim> &fe_base,
           const FE_Q<dim> &fe_enriched,
           const std::vector<EnrichmentPredicate<dim>> &vec_predicates,
           const std::vector<SplineEnrichmentFunction<dim>> &vec_enrichments);

    //TODO need initialize function

    void set(hp::DoFHandler<dim> &dof_handler);
    hp::FECollection<dim> get_fe_collection()
    {
      return fe_collection;
    }

  private:
    using cell_function = std::function<const Function<dim>*
                          (const typename Triangulation<dim>::cell_iterator &)>;

    hp::FECollection<dim> fe_collection;
    FE_Q<dim> fe_base;
    FE_Q<dim> fe_enriched;
    FE_Nothing<dim> fe_nothing;
    std::vector<EnrichmentPredicate<dim>> vec_predicates;
    std::vector<SplineEnrichmentFunction<dim>> vec_enrichments;
    std::vector<cell_function>  color_enrichments;
    std::vector<unsigned int> predicate_colors;
    unsigned int num_colors;
    std::map<unsigned int,
        std::map<unsigned int, unsigned int> > cellwise_color_predicate_map;
    std::vector <std::set<unsigned int>> fe_sets;
  };



  template<int dim>
  helper<dim>::helper
  (const FE_Q<dim> &fe_base,
   const FE_Q<dim> &fe_enriched,
   const std::vector<EnrichmentPredicate<dim>> &vec_predicates,
   const std::vector<SplineEnrichmentFunction<dim>> &vec_enrichments)
    :
    fe_base(fe_base),
    fe_enriched(fe_enriched),
    fe_nothing(1,true),
    vec_predicates(vec_predicates),
    vec_enrichments(vec_enrichments)
  {
  }

  template<int dim>
  void helper<dim>::set(hp::DoFHandler<dim> &dof_handler)
  {
      if (vec_predicates.size() != 0)
        num_colors = color_predicates (dof_handler, vec_predicates, predicate_colors);
      else
        num_colors = 0;

      {
        //print color_indices
        for (unsigned int i=0; i<predicate_colors.size(); ++i)
          std::cout << "predicate " << i << " : " << predicate_colors[i] << std::endl;
      }

    set_cellwise_color_set_and_fe_index (dof_handler,
                                         vec_predicates,
                                         predicate_colors,
                                         cellwise_color_predicate_map,
                                         fe_sets);
    {
      //print material table
      std::cout << "\nMaterial table : " << std::endl;
      for ( unsigned int j=0; j<fe_sets.size(); ++j)
        {
          std::cout << j << " ( ";
          for ( auto i = fe_sets[j].begin(); i != fe_sets[j].end(); ++i)
            std::cout << *i << " ";
          std::cout << " ) " << std::endl;
        }
    }

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
  }
}
