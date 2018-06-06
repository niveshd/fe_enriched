#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/point.h>

using namespace dealii;

namespace Step1
{

  template <int dim> class SigmaFunction : public Function<dim>
  {
    Point<dim> center;
    double sigma;
    std::map<std::string, double>  constants;
    std::string func_expr;
    FunctionParser<dim> func;

  public:
    SigmaFunction() : Function<dim>(),
      func(1)
    {}

    SigmaFunction(SigmaFunction &other)
      : Function<dim>(), center(other.center), sigma(other.sigma),
        constants(other.constants),
        func_expr(other.func_expr)
    {
      this->initialize(center, sigma, func_expr, constants);
    }

    void operator=(const SigmaFunction &other)
    {
      center = other.center;
      sigma = other.sigma;
      constants = other.constants;
      func_expr = other.func_expr;
      this->initialize(center, sigma, func_expr, constants);
    }

    // to help with resize function. doesn't copy function parser(func)!
    SigmaFunction(SigmaFunction &&other)
      : center(other.center), sigma(other.sigma), constants(other.constants), func_expr(other.func_expr)
    {
      this->initialize(center, sigma, func_expr, constants);
    }

    void initialize(const Point<dim> &center, const double &sigma,
                    const std::string &func_expr,
                    const std::map<std::string, double> &constants = {});
    double value(const Point<dim> &p, const unsigned int component = 0) const;
    Tensor<1, dim> gradient(const Point<dim> &p,
                            const unsigned int component = 0) const;
    virtual void value_list(const std::vector<Point<dim>> &points,
                            std::vector<double> &value_list) const;
  };

  template <int dim>
  void SigmaFunction<dim>::initialize(const Point<dim> &_center,
                                      const double &_sigma,
                                      const std::string &_func_expr,
                                      const std::map<std::string, double> & _constants)
  {
    center = _center;
    sigma = _sigma;
    func_expr = _func_expr;
    std::string variables;
    constants.insert({"sigma",sigma});
    constants.insert({"pi",numbers::PI});
    constants.insert(_constants.begin(), _constants.end());

    AssertThrow(dim == 1 || dim == 2 || dim == 3,
                ExcMessage("Dimension not implemented"));
    switch (dim)
      {
      case 1:
        variables = "x";
        break;
      case 2:
        variables = "x, y";
        break;
      case 3:
        variables = "x, y, z";
        break;
      }

    func.initialize(variables, func_expr, constants);
  }

  template <int dim>
  inline double SigmaFunction<dim>::value(const Point<dim> &p,
                                          const unsigned int component) const
  {
    const Point<dim> d(p - center);
    return func.value(d, component);
  }

  template <int dim>
  inline Tensor<1, dim>
  SigmaFunction<dim>::gradient(const Point<dim> &p,
                               const unsigned int component) const
  {
    const Point<dim> d(p - center);
    return func.gradient(d, component);
  }

  template <int dim>
  void SigmaFunction<dim>::value_list(const std::vector<Point<dim>> &points,
                                      std::vector<double> &value_list) const
  {
    const unsigned int n_points = points.size();

    AssertDimension(points.size(), value_list.size());

    for (unsigned int p = 0; p < n_points; ++p)
      value_list[p] = value(points[p]);
  }



  template <int dim> struct Vec_SigmaFunction : public Function<dim>
  {
    Vec_SigmaFunction() : Function<dim>() {}
    void initialize(const std::vector<SigmaFunction<dim>>& _vec_func)
    {
      vec_func.resize(_vec_func.size());
      for (unsigned int i=0; i<vec_func.size(); ++i)
          vec_func[i] = _vec_func[i];
    }
    double value(const Point<dim> &p, const unsigned int component = 0) const;
    Tensor<1, dim> gradient(const Point<dim> &p, const unsigned int component=0) const;

  private:
     std::vector<SigmaFunction<dim>> vec_func;
  };

  template <int dim>
  inline double Vec_SigmaFunction<dim>::value(const Point<dim> &p,
                                              const unsigned int component) const
  {
    double res=0;
    for (unsigned int i=0; i<vec_func.size(); ++i)
      {
        res += vec_func[i].value(p,component);
      }
    return res;
  }

  template <int dim>
  inline Tensor<1, dim>
  Vec_SigmaFunction<dim>::gradient(const Point<dim> &p,
                               const unsigned int component) const
  {
    Tensor<1, dim> grad;
    for (unsigned int i=0; i<vec_func.size(); ++i)
      {
        grad += vec_func[i].gradient(p,component);
      }
    return grad;
  }

} // namespace Step1
#endif
