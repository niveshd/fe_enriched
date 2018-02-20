#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <deal.II/base/exceptions.h>

template <int dim>
class GaussianFunction :  public Function<dim>
{
  Point<dim> center;
  double sigma;
  double coeff;
public:
  GaussianFunction ();
  GaussianFunction (const Point<dim> &center, const double &sigma);
  void set_point(const Point<dim> &points);
  void set_sigma(const double &sigma);
  virtual void value (const Point<dim> &p,
                      double   &value) const;
  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double >           &value_list) const;
};

template <int dim>
GaussianFunction<dim>::GaussianFunction ()
  :
  Function<dim> (),
  center(Point<dim>()),
  sigma(1)
{
  coeff = 1.0/(2*M_PI*sigma*sigma);   //2D gaussian
}

template <int dim>
GaussianFunction<dim>::GaussianFunction (const Point<dim> &center,
                                         const double &sigma)
  :
  Function<dim> (),
  center(center),
  sigma(sigma)
{
  coeff = 1.0/(2*M_PI*sigma*sigma);   //2D gaussian
}

template <int dim>
inline
void GaussianFunction<dim>::set_point(const Point<dim> &p)
{
  //TODO change to vector
  center = p;
}

template <int dim>
inline
void GaussianFunction<dim>::set_sigma(const double &value)
{
  //TODO change to vector
  sigma = value;
}

template <int dim>
inline
void GaussianFunction<dim>::value (const Point<dim> &p,
                                   double           &value) const
{
  double r_squared = p.distance_square(center);
  value = coeff*exp(-r_squared/(2*sigma*sigma));
}

template <int dim>
void GaussianFunction<dim>::value_list
(const std::vector<Point<dim> > &points,
 std::vector<double >           &value_list) const
{
  const unsigned int n_points = points.size();

  AssertDimension(points.size(), value_list.size());

  for (unsigned int p=0; p<n_points; ++p)
    GaussianFunction::value (points[p],
                             value_list[p]);
}

#endif
