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
 std::vector <std::set<unsigned int>> &fe_sets)
{
  //set first element of fe_sets size to empty
  fe_sets.resize(1);

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
    }
}


template <int dim>
void make_colorwise_enrichment_functions
(const unsigned int &num_colors,          //needs number of colors

 const std::vector<EnrichmentFunction<dim>>
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
  color_enrichments.resize (num_colors); // <<-- return by value and keep as a class member

  for (unsigned int i = 0; i < num_colors; ++i)
    {
      color_enrichments[i] =
        [ &,i] (const typename Triangulation<dim, dim>::cell_iterator & cell)
      {
        unsigned int id = cell->index();

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
void make_colorwise_enrichment_functions
(const unsigned int &num_colors,          //needs number of colors

 const std::vector<EnrichmentFunction<2>>
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

 const std::vector<EnrichmentFunction<3>>
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
