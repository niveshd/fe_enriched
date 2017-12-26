
using namespace dealii;

template <int dim>
struct EnrichmentPredicate
{
    EnrichmentPredicate(const Point<dim> origin, const int radius)
    :origin(origin),radius(radius){}

    template <class Iterator>
    bool operator () (const Iterator &i)
    { 
        return ( (i->center() - origin).norm_square() < radius*radius);
    }    
    
    const Point<dim> &get_origin() { return origin; }
    const int &get_radius() { return radius; }

    private:
        const Point<dim> origin;
        const int radius;   
};



template <int dim>
class RightHandSide :  public Function<dim>
{
public:
  RightHandSide ();
  virtual void value (const Point<dim> &p,
                      double   &values) const;
  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double >           &value_list) const;
};



template <int dim>
RightHandSide<dim>::RightHandSide () :
  Function<dim> ()
{}


template <int dim>
inline
void RightHandSide<dim>::value (const Point<dim> &p,
                                double           &values) const
{
  Assert (dim >= 2, ExcInternalError());
  
  values = 1;
}



template <int dim>
void RightHandSide<dim>::value_list (const std::vector<Point<dim> > &points,
                                     std::vector<double >           &value_list) const
{
  const unsigned int n_points = points.size();

  AssertDimension(points.size(), value_list.size());
  
  for (unsigned int p=0; p<n_points; ++p)
    RightHandSide<dim>::value (points[p],
                                      value_list[p]);
}


template <int dim>
class EnrichmentFunction : public Function<dim>
{
public:
  EnrichmentFunction(const Point<dim> &origin,
                     const double     &Z,
                     const double     &radius)
    : Function<dim>(1),
      origin(origin),
      Z(Z),
      radius(radius)
  {}

  virtual double value(const Point<dim> &point,
                       const unsigned int component = 0) const
  {
    Tensor<1,dim> dist = point-origin;
    const double r = dist.norm();
    return std::exp(-Z*r);
  }

  //TODO remove?
  bool is_enriched(const Point<dim> &point) const
  {
    if (origin.distance(point) < radius)
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
    return -Z*std::exp(-Z*r)*dist;
  }

private:
  /**
   * origin
   */
  const Point<dim> origin;

  /**
   * charge
   */
  const double Z;

  /**
   * enrichment radius
   */
  const double radius;
};