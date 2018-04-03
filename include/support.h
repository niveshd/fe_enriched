#ifndef SUPPORT_H
#define SUPPORT_H

#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function_cspline.h>

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

using namespace dealii;

template <int dim>
struct EnrichmentPredicate
{
  EnrichmentPredicate(const Point<dim> origin, const double radius)
    :origin(origin),radius(radius) {}

  template <class Iterator>
  bool operator () (const Iterator &i) const
  {
    return ( (i->center() - origin).norm_square() < radius*radius);
  }

  const Point<dim> &get_origin()
  {
    return origin;
  }
  const double &get_radius()
  {
    return radius;
  }

private:
  const Point<dim> origin;
  const double radius;
};


template <int dim>
class SplineEnrichmentFunction : public Function<dim>
{
public:
  SplineEnrichmentFunction(const Point<dim> &origin,
                           const double     &sigma,
                           const std::vector<double> &interpolation_points_1d,
                           const std::vector<double> &interpolation_values_1d)
    : Function<dim>(1),
      origin(origin),
      sigma(sigma),
      interpolation_points(interpolation_points_1d),
      interpolation_values(interpolation_values_1d),
      cspline(interpolation_points, interpolation_values)
  {}

  //To be used only for debugging
  SplineEnrichmentFunction(const Point<dim> &origin,
                           const double &sigma,
                           const double &constant)
    :
    Function<dim>(1),
    origin(origin),
    sigma(sigma),
    interpolation_points({origin[0],
                         origin[0]+25*sigma,
                         origin[0]+50*sigma
  }),
  interpolation_values({constant,
                        constant,
                        constant
                       }),
  cspline(interpolation_points,
          interpolation_values)
  {}

  SplineEnrichmentFunction(SplineEnrichmentFunction &&other)
    :
    Function<dim>(1),
    origin(other.origin),
    sigma(other.sigma),
    interpolation_points(other.interpolation_points),
    interpolation_values(other.interpolation_values),
    cspline(interpolation_points,interpolation_values)
  {
  }

  SplineEnrichmentFunction(const SplineEnrichmentFunction &other)
    :
    Function<dim>(1),
    origin(other.origin),
    sigma(other.sigma),
    interpolation_points(other.interpolation_points),
    interpolation_values(other.interpolation_values),
    cspline(interpolation_points,interpolation_values)
  {
  }



  virtual double value(const Point<dim> &point,
                       const unsigned int component = 0) const
  {
    Tensor<1,dim> dist = point-origin;
    const double r = dist.norm();
    return cspline.value(Point<1>(r));
  }

  virtual Tensor< 1, dim> gradient (const Point<dim > &p,
                                    const unsigned int component=0) const
  {
    Tensor<1,dim> dist = p-origin;
    const double r = dist.norm();
    Assert (r > 0.,
            ExcDivideByZero());
    dist/=r;
    Assert(component == 0, ExcMessage("Not implemented"));
    return cspline.gradient(Point<1>(r))[0]*dist;
  }

private:
  /**
   * origin
   */
  const Point<dim> origin;
  /**
   * enrichment radius
   */
  const double sigma;
  //
  std::vector<double> interpolation_points;
  std::vector<double> interpolation_values;
  //enrichment function as CSpline based on radius
  Functions::CSpline<1> cspline;
};



template <int dim>
using EnrichmentFunctionArray = std::vector<std::vector<std::function<const Function<dim> *
                                (const typename Triangulation<dim, dim>::cell_iterator &) >>>;


/*
 * Assign colors to predicates. No two predicates which are
 * both active on a cell have the same color. Predicates that
 * share cell in this sense are said to be connected.
 */
template <int dim, class MeshType>
unsigned int color_predicates
(const MeshType &mesh,
 const std::vector<EnrichmentPredicate<dim>> &,
 std::vector<unsigned int> &);

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
 &color_enrichments);



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
 const std::function<bool (const typename MeshType::active_cell_iterator &)> &predicate_2);

#endif
