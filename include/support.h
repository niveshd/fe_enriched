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
  EnrichmentPredicate(const Point<dim> origin, const int radius)
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
  const int &get_radius()
  {
    return radius;
  }

private:
  const Point<dim> origin;
  const int radius;
};


template <int dim>
class EnrichmentFunction : public Function<dim>
{
public:
  EnrichmentFunction(const Point<dim> &origin,
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
  EnrichmentFunction(const Point<dim> &origin,
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

  EnrichmentFunction(EnrichmentFunction &&other)
    :
    origin(other.origin),
    sigma(other.sigma),
    interpolation_points(other.interpolation_points),
    interpolation_values(other.interpolation_values),
    cspline(interpolation_points,interpolation_values)
  {
  }

  EnrichmentFunction(const EnrichmentFunction &other)
    :
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

  //TODO remove?
  bool is_enriched(const Point<dim> &point) const
  {
    if (origin.distance(point) < sigma)
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
  //enrichement function as CSpline based on radius
  Functions::CSpline<1> cspline;
};



template <int dim>
using EnrichmentFunctionArray = std::vector<std::vector<std::function<const Function<dim> *
                                (const typename Triangulation<dim, dim>::cell_iterator &) >>>;



template <int dim, class MeshType>
unsigned int color_predicates
(const MeshType &mesh,
 const std::vector<EnrichmentPredicate<dim>> &,
 std::vector<unsigned int> &);


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

 const std::vector<EnrichmentFunction<dim>>
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
