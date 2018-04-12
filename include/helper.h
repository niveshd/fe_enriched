#ifndef HELPER_H
#define HELPER_H

#include <set>
#include <map>
#include <vector>
#include "functions.h"  //TODO remove
#include "support.h"    //TODO REMOVE
#include "estimate_enrichment.h"    //TODO REMOVE

namespace ColorEnriched
{
  namespace internal
  {
    /**
       * Function returns true if there is a connection between subdomains in the
       * @p mesh i.e the subdomains share at least a vertex. The two subdomains
       * are defined by predicates provided by @p predicate_1 and @p predicate_2.
       * Predicates are functions or objects with operator() which take in a
       * cell in @p mesh and return true if the cell is in subdomain.
       *
       * An example of a custom predicate is one that checks distance from a fixed
       * point. Note that the operator() takes in a cell iterator. Using the
       * object constructor, the fixed point and the distance can be chosen.
       * @code
       * <int dim>
       * struct predicate
       * {
       *     predicate(const Point<dim> p, const int radius)
       *     :p(p),radius(radius){}
       *
       *     template <class Iterator>
       *     bool operator () (const Iterator &i)
       *     {
       *         return ( (i->center() - p).norm() < radius);
       *     }
       *
       * private:
       *     Point<dim> p;
       *     int radius;
       *
       * };
       * @endcode
       * and then the function can be used as follows to find if the subdomains
       * are connected.
       * @code
       * find_connection_between_subdomains
       * (tria,
       *  predicate<dim>(Point<dim>(0,0), 1)
       *  predicate<dim>(Point<dim>(2,2), 1));
       * @endcode
       *
       * @tparam MeshType A type that satisfies the requirements of the
       * @ref ConceptMeshType "MeshType concept".
       * @param[in] mesh A mesh (i.e. objects of type Triangulation, DoFHandler,
       * or hp::DoFHandler).
       * @param[in] predicate_1 A function  (or object of a type with an operator())
       * defining the subdomain 1. The function takes in an active cell and returns a boolean.
       * @param[in] predicate_2 Same as @p predicate_1 but defines subdomain 2.
       * @return A boolean "true" if the subdomains share atleast a vertex i.e cells including
       * halo or ghost cells of subdomain 1 overlap with subdomain 2.
       */
    template <class MeshType>
    bool find_connection_between_subdomains
    (const MeshType                                                              &mesh,
     const std::function<bool (const typename MeshType::active_cell_iterator &)> &predicate_1,
     const std::function<bool (const typename MeshType::active_cell_iterator &)> &predicate_2
    );



    /*
     * Assign colors to predicates. No two predicates which are
     * both active on a cell have the same color. Predicates that
     * share cell in this sense are said to be connected.
     */
    template <int dim, class MeshType>
    unsigned int color_predicates
    (const MeshType &mesh,
     const std::vector<EnrichmentPredicate<dim>> &vec_predicates,
     std::vector<unsigned int> &predicate_colors);



    /*
     * create a vector of fe sets. The index of the vector represents
     * active_fe_index and each fe index is associated with a set of colors.
     * A cell with an fe index associated with colors {a,b} means that
     * predicates active in the cell have colors a or b. All the
     * fe indices and their associated sets are in @par fe_sets.
     * Active_fe_index of cell is added as well here.
     * Eg: fe_sets = { {}, {1}, {2}, {1,2} } means
     * cells have no predicates or predicates with colors 1 or 2 or (1 and 2)
     * cell with active fe index 2 has predicates with color 2,
     * with active fe index 3 has predicates with color 1 and 2.
     */
    template <int dim, class MeshType>
    void
    set_cellwise_color_set_and_fe_index
    (MeshType &mesh,
     const std::vector<EnrichmentPredicate<dim>> &vec_predicates,
     const std::vector<unsigned int> &predicate_colors,
     std::map<unsigned int,
     std::map<unsigned int, unsigned int> >
     &cellwise_color_predicate_map,
     std::vector <std::set<unsigned int>> &fe_sets);



    /*
     * colorwise enrichment functions indexed from 0!
     * color_enrichments[0] is color 1 enrichment function
     * Enrichement function corresponding to predicate
     */
    template <int dim>
    void make_colorwise_enrichment_functions
    (const unsigned int &num_colors,

     const std::vector<SplineEnrichmentFunction<dim>>
     &vec_enrichments,

     const std::map<unsigned int,
     std::map<unsigned int, unsigned int> >
     &cellwise_color_predicate_map,

     std::vector<
     std::function<const Function<dim>*
     (const typename Triangulation<dim>::cell_iterator &)> >
     &color_enrichments);



    template <int dim>
    using EnrichmentFunctionArray = std::vector<std::vector<std::function<const Function<dim> *
                                    (const typename Triangulation<dim, dim>::cell_iterator &) >>>;



    template <int dim>
    void make_fe_collection_from_colored_enrichments
    (const unsigned int &num_colors,
     const std::vector <std::set<unsigned int>>
     &fe_sets,         //total list of color sets possible

     const std::vector<
     std::function<const Function<dim>*
     (const typename Triangulation<dim>::cell_iterator &)> >
     &color_enrichments,  //color wise enrichment functions

     const FE_Q<dim> &fe_base,            //basic fe element
     const FE_Q<dim> &fe_enriched,        //fe element multiplied by enrichment function
     const FE_Nothing<dim> &fe_nothing,
     hp::FECollection<dim> &fe_collection);
  }



  template<int dim>
  struct Helper
  {

    Helper(const FE_Q<dim> &fe_base,
           const FE_Q<dim> &fe_enriched,
           const std::vector<EnrichmentPredicate<dim>> &vec_predicates,
           const std::vector<SplineEnrichmentFunction<dim>> &vec_enrichments);
    void set(hp::DoFHandler<dim> &dof_handler);
    hp::FECollection<dim> get_fe_collection() const;

  private:
    using cell_function = std::function<const Function<dim>*
                          (const typename Triangulation<dim>::cell_iterator &)>;
    hp::FECollection<dim> fe_collection;
    const FE_Q<dim> &fe_base;
    const FE_Q<dim> &fe_enriched;
    const FE_Nothing<dim> fe_nothing;
    std::vector<EnrichmentPredicate<dim>> vec_predicates;
    std::vector<SplineEnrichmentFunction<dim>> vec_enrichments;
    std::vector<cell_function>  color_enrichments;
    std::vector<unsigned int> predicate_colors;
    unsigned int num_colors;
    std::map<unsigned int,
        std::map<unsigned int, unsigned int> > cellwise_color_predicate_map;
    std::vector <std::set<unsigned int>> fe_sets;
  };
}
#endif
