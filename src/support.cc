#include "support.h"

template <int dim, class MeshType>
unsigned int color_predicates
(const MeshType &mesh,
 const std::vector<EnrichmentPredicate<dim>> &vec_predicates,
 std::vector<unsigned int> &predicate_colors)
{
  const unsigned int num_indices = vec_predicates.size();

  DynamicSparsityPattern dsp;
  dsp.reinit ( num_indices, num_indices );

  /* find connections between subdomains defined by predicates
   * if the connection exists, add it to a graph object represented
   * by dynamic sparsity pattern
   */
  for (unsigned int i = 0; i < num_indices; ++i)
    for (unsigned int j = i+1; j < num_indices; ++j)
      if ( find_connection_between_subdomains
           (mesh, vec_predicates[i], vec_predicates[j]) )
        dsp.add(i,j);

  dsp.symmetrize();
  SparsityPattern sp_graph;
  sp_graph.copy_from(dsp);
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
 std::vector <std::set<unsigned int>> &fe_sets)
{
  //clear output variables first
  fe_sets.clear();
  cellwise_color_predicate_map.clear();

  /*
   * set first element of fe_sets size to empty. This means that
   * the default, fe index = 0 is associated with an empty set, since no
   * predicate is active in these regions.
   */
  fe_sets.resize(1);

  /*
   * loop through cells and create new fe index if needed.
   * Each fe index is associated with a set of colors. A cell
   * with an fe index associated with colors {a,b} means that
   * predicates active in the cell have colors a or b. All the
   * fe indices and their associated sets are in @par fe_sets.
   * Active_fe_index of cell is added as well here.
   * Eg: fe_sets = { {}, {1}, {2}, {1,2} } means
   * Cells have no predicates or predicates with colors 1 or 2 or (1 and 2)
   * active. Cell with active fe index 2 has predicates with color 2,
   * with active fe index 3 has predicates with color 1 and 2.
   *
   * Associate each cell_id with set of pairs. The number of pairs
   * is equal to the number of predicates active in the given cell.
   * The pair represents predicate color and the active predicate with
   * that color in that cell. Each color can only correspond to a single
   * predicate since predicates with the same color are disjoint domains.
   */
  unsigned int map_index(0);
  auto cell = mesh.begin_active();
  auto endc = mesh.end();
  for (;
       cell != endc; ++cell)
    {
      //set default fe index ==> no enrichment
      cell->set_active_fe_index (0);
      cell->set_material_id(map_index);
      std::set<unsigned int> color_list;

      //loop through active predicates in a cell
      for (unsigned int i=0; i<vec_predicates.size(); ++i)
        {
          if (vec_predicates[i](cell))
            {
              /*
               * create a pair predicate color and predicate id and add it
               * to a map associated with each enriched cell
               */
              auto ret = cellwise_color_predicate_map[map_index].insert
                         (std::pair <unsigned int, unsigned int> (predicate_colors[i], i));

              color_list.insert(predicate_colors[i]);

              //TODO check if single color doesn't have multiple predicates!

//              pcout << " - " << predicate_colors[i] << "(" << i << ")" ;
            }
        }

//         if (!color_list.empty())
//                 pcout << std::endl;

      /*
       * check if color combination is already added.
       * If already added, set the active fe index based on
       * its index in the fe_sets. If the combination doesn't
       * exist, add the set to fe_sets and once again set the
       * active fe index as last index in fe_sets.
       */
      bool found = false;
      if ( !color_list.empty() )
        {
          for ( unsigned int j=0; j<fe_sets.size(); ++j)
            {
              if (fe_sets[j] ==  color_list)
                {
//                     pcout << "color combo set found at " << j << std::endl;
                  found=true;
                  cell->set_active_fe_index(j);
                  break;
                }
            }


          if (!found)
            {
              fe_sets.push_back(color_list);
              cell->set_active_fe_index(fe_sets.size()-1);
              /*
              num_colors+1 = (num_colors+1 > color_list.size())?
                                     num_colors+1:
                                     color_list.size();
              //                 pcout << "color combo set pushed at " << fe_sets.size()-1 << std::endl;
              */
            }
        }
      ++map_index;
    }
}


template <int dim>
void make_colorwise_enrichment_functions
(const unsigned int &num_colors,          //needs number of colors

 const std::vector<SplineEnrichmentFunction<dim>>
 &vec_enrichments,     //enrichment functions based on predicate id

 const std::map<unsigned int,
 std::map<unsigned int, unsigned int> >
 &cellwise_color_predicate_map,

 std::vector<
 std::function<const Function<dim>*
 (const typename Triangulation<dim>::cell_iterator &)> >
 &color_enrichments)   //colorwise enrichment functions indexed from 0!
//color_enrichments[0] is color 1 enrichment function
{
  color_enrichments.clear();
  color_enrichments.resize (num_colors); // <<-- return by value and keep as a class member

  for (unsigned int i = 0; i < num_colors; ++i)
    {
      color_enrichments[i] =
        [ &,i] (const typename Triangulation<dim, dim>::cell_iterator & cell)
      {
        unsigned int id = cell->material_id();

        //i'th function corresponds to i+1 color
        return &vec_enrichments[cellwise_color_predicate_map.at(id).at(i+1)];
      };
    }
}

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
 hp::FECollection<dim> &fe_collection)
{
  //define dummy function
  using cell_function = std::function<const Function<dim>*
                        (const typename Triangulation<dim>::cell_iterator &)>;
  cell_function dummy_function;

  dummy_function = [=] (const typename Triangulation<dim>::cell_iterator &)
                   -> const Function<dim> *
  {
    AssertThrow(false, ExcMessage("Called enrichment function for FE_Nothing"));
    return nullptr;
  };

  //loop through color sets ignore starting empty sets
  //TODO remove volatile
  for (volatile unsigned int color_set_id=0; color_set_id!=fe_sets.size(); ++color_set_id)
    {
      std::vector<const FiniteElement<dim> *> vec_fe_enriched (num_colors, &fe_nothing);
      EnrichmentFunctionArray<dim> functions(num_colors, {dummy_function});

      // FIXME: remove
      std::set<unsigned int> set_components;
      for (auto it=fe_sets[color_set_id].begin();
           it != fe_sets[color_set_id].end();
           ++it)
        {
          const unsigned int ind = *it-1;
          set_components.insert(ind);

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

      {
        //TODO delete after testing
        ConditionalOStream pcout
        (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
      }

      fe_collection.push_back (fe_component);
    }
}



template <class MeshType>
bool find_connection_between_subdomains
(const MeshType                                                              &mesh,
 const std::function<bool (const typename MeshType::active_cell_iterator &)> &predicate_1,
 const std::function<bool (const typename MeshType::active_cell_iterator &)> &predicate_2
)
{
  std::vector<bool> locally_active_vertices_on_subdomain (mesh.get_triangulation().n_vertices(),
                                                          false);

  //Mark vertices in subdomain (1) defined by predicate 1
  for (typename MeshType::active_cell_iterator
       cell = mesh.begin_active();
       cell != mesh.end(); ++cell)
    if (predicate_1(cell)) // True predicate --> Part of region 1
      for (unsigned int v=0; v<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
        locally_active_vertices_on_subdomain[cell->vertex_index(v)] = true;

  //Find if cells in subdomain (2) defined by predicate 2 share vertices with region 1.
  for (typename MeshType::active_cell_iterator
       cell = mesh.begin_active();
       cell != mesh.end(); ++cell)
    if (predicate_2(cell)) // True predicate --> Potential connection between subdomains
      for (unsigned int v=0; v<GeometryInfo<MeshType::dimension>::vertices_per_cell; ++v)
        if (locally_active_vertices_on_subdomain[cell->vertex_index(v)] == true)
          {
            return true;
          }
  return false;
}


//template instantiations
template unsigned int color_predicates
(const Triangulation<2,2> &mesh,
 const std::vector<EnrichmentPredicate<2>> &,
 std::vector<unsigned int> &);

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
 std::vector <std::set<unsigned int>> &fe_sets);


template
void
set_cellwise_color_set_and_fe_index
(hp::DoFHandler<3,3> &mesh,
 const std::vector<EnrichmentPredicate<3>> &vec_predicates,
 const std::vector<unsigned int> &predicate_colors,
 std::map<unsigned int,
 std::map<unsigned int, unsigned int> >
 &cellwise_color_predicate_map,
 std::vector <std::set<unsigned int>> &fe_sets);



template
void
set_cellwise_color_set_and_fe_index
(DoFHandler<2,2> &mesh,
 const std::vector<EnrichmentPredicate<2>> &vec_predicates,
 const std::vector<unsigned int> &predicate_colors,
 std::map<unsigned int,
 std::map<unsigned int, unsigned int> >
 &cellwise_color_predicate_map,
 std::vector <std::set<unsigned int>> &fe_sets);


template
void
set_cellwise_color_set_and_fe_index
(DoFHandler<3,3> &mesh,
 const std::vector<EnrichmentPredicate<3>> &vec_predicates,
 const std::vector<unsigned int> &predicate_colors,
 std::map<unsigned int,
 std::map<unsigned int, unsigned int> >
 &cellwise_color_predicate_map,
 std::vector <std::set<unsigned int>> &fe_sets);



template
void make_colorwise_enrichment_functions
(const unsigned int &num_colors,          //needs number of colors

 const std::vector<SplineEnrichmentFunction<2>>
 &vec_enrichments,     //enrichment functions based on predicate id

 const std::map<unsigned int,
 std::map<unsigned int, unsigned int> >
 &cellwise_color_predicate_map,

 std::vector<
 std::function<const Function<2>*
 (const typename Triangulation<2>::cell_iterator &)> >
 &color_enrichments);

template
void make_colorwise_enrichment_functions
(const unsigned int &num_colors,          //needs number of colors

 const std::vector<SplineEnrichmentFunction<3>>
 &vec_enrichments,     //enrichment functions based on predicate id

 const std::map<unsigned int,
 std::map<unsigned int, unsigned int> >
 &cellwise_color_predicate_map,

 std::vector<
 std::function<const Function<3>*
 (const typename Triangulation<3>::cell_iterator &)> >
 &color_enrichments);


template
void make_fe_collection_from_colored_enrichments
(
  const unsigned int &num_colors,
  const std::vector <std::set<unsigned int>>
  &fe_sets,         //total list of color sets possible

  const std::vector<
  std::function<const Function<2>*
  (const typename Triangulation<2>::cell_iterator &)> >
  &color_enrichments,  //color wise enrichment functions

  const FE_Q<2> &fe_base,            //basic fe element
  const FE_Q<2> &fe_enriched,        //fe element multiplied by enrichment function
  const FE_Nothing<2> &fe_nothing,
  hp::FECollection<2> &fe_collection);


template
void make_fe_collection_from_colored_enrichments
(
  const unsigned int &num_colors,
  const std::vector <std::set<unsigned int>>
  &fe_sets,         //total list of color sets possible

  const std::vector<
  std::function<const Function<3>*
  (const typename Triangulation<3>::cell_iterator &)> >
  &color_enrichments,  //color wise enrichment functions

  const FE_Q<3> &fe_base,            //basic fe element
  const FE_Q<3> &fe_enriched,        //fe element multiplied by enrichment function
  const FE_Nothing<3> &fe_nothing,
  hp::FECollection<3> &fe_collection);


template
bool find_connection_between_subdomains
(const hp::DoFHandler<2> &,
 const std::function<bool (const dealii::internal::ActiveCellIterator<2, 2, hp::DoFHandler<2>>::type &)> &,
 const std::function<bool (const dealii::internal::ActiveCellIterator<2, 2, hp::DoFHandler<2>>::type &)> &);


 template
 bool find_connection_between_subdomains
 (const hp::DoFHandler<3> &,
  const std::function<bool (const dealii::internal::ActiveCellIterator<3, 3, hp::DoFHandler<3>>::type &)> &,
  const std::function<bool (const dealii::internal::ActiveCellIterator<3, 3, hp::DoFHandler<3>>::type &)> &);


  template
  bool find_connection_between_subdomains
  (const Triangulation<2> &,
   const std::function<bool (const dealii::internal::ActiveCellIterator<2, 2, Triangulation<2>>::type &)> &,
   const std::function<bool (const dealii::internal::ActiveCellIterator<2, 2, Triangulation<2>>::type &)> &);


   template
   bool find_connection_between_subdomains
   (const Triangulation<3> &,
    const std::function<bool (const dealii::internal::ActiveCellIterator<3, 3, Triangulation<3>>::type &)> &,
    const std::function<bool (const dealii::internal::ActiveCellIterator<3, 3, Triangulation<3>>::type &)> &);
