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
    return cspline.value(Point<1>(r), component);
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
#endif
