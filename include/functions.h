#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/parsed_function.h>

using namespace dealii;

namespace Step1
{

  template <int dim>
  class RightHandSide :  public Function<dim>
  {
    Point<dim> center;
    FunctionParser<dim> func;
  public:
    RightHandSide():
      Function<dim>(),
      func(1) {}

    //to help with resize function. doesn't copy function parser(func)!
    RightHandSide(RightHandSide &&other)
      :
      center(other.center),
      func(1) {}

    void initialize (const Point<dim> &center,
                     const double &sigma,
                     const std::string &func_expr);
    virtual void value (const Point<dim> &p,
                        double   &value) const;
    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double >           &value_list) const;
  };

  template <int dim>
  void RightHandSide<dim>::initialize (const Point<dim> &_center,
                                       const double &sigma,
                                       const std::string &func_expr)
  {
    center = _center;
    std::string variables;
    std::map<std::string,double> constants =
    {
      {"sigma",sigma},
      {"pi", numbers::PI}
    };

    AssertThrow(dim == 1 || dim == 2 || dim == 3,
                ExcMessage("Dimension not implemented"));
    switch (dim)
      {
      case 1 :
        variables = "x";
        break;
      case 2 :
        variables = "x, y";
        break;
      case 3 :
        variables = "x, y, z";
        break;
      }

    func.initialize(variables,
                    func_expr,
                    constants);
  }

  template <int dim>
  inline
  void RightHandSide<dim>::value (const Point<dim> &p,
                                  double           &value) const
  {
    const Point<dim> d(p - center);
    value = func.value(d);
  }

  template <int dim>
  void RightHandSide<dim>::value_list
  (const std::vector<Point<dim> > &points,
   std::vector<double >           &value_list) const
  {
    const unsigned int n_points = points.size();

    AssertDimension(points.size(), value_list.size());

    for (unsigned int p=0; p<n_points; ++p)
      RightHandSide::value (points[p],
                            value_list[p]);
  }

}
#endif
