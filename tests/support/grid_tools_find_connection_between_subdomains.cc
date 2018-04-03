// ---------------------------------------------------------------------
//
// Copyright (C) 2001 - 2015 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------



#include "../tests.h"
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>

#include "support.h"
#include <vector>

using namespace dealii;


template <int dim>
struct predicate_template
{
  predicate_template(const Point<dim> p, const int radius)
    :p(p),radius(radius) {}

  template <class Iterator>
  bool operator () (const Iterator &i)
  {
    return ( (i->center() - p).norm() < radius);
  }

private:
  Point<dim> p;
  int radius;
};



template <int dim>
void test ()
{
  deallog << "dim = " << dim << std::endl;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, -20, 20);
  tria.refine_global(4);

  Assert ( dim==2 || dim==3, ExcDimensionMismatch2(dim, 2, 3) );

  //Vector of predicates for testing
  typedef std::function<bool (const typename Triangulation<dim>::active_cell_iterator &)> predicate_function;
  std::vector<predicate_function> predicates;
  predicates.resize(5);
  if ( dim==2 )
    {
      //Radius set such that every region has 2^2 = 4 cells
      predicates[1] = predicate_template<dim>(Point<dim>(7.5,7.5), 2);
      predicates[2] = predicate_template<dim>(Point<dim>(5,5), 2);
      predicates[0] = predicate_template<dim>(Point<dim>(), 2);
      predicates[3] = predicate_template<dim>(Point<dim>(-5,-5), 2);
      predicates[4] = predicate_template<dim>(Point<dim>(-10,-10), 2);
    }
  else
    {
      //Radius set such that every region has 2^3 = 8 cells
      predicates[1] = predicate_template<dim>(Point<dim>(7.5,7.5,7.5), 3);
      predicates[2] = predicate_template<dim>(Point<dim>(5,5,5), 3);
      predicates[0] = predicate_template<dim>(Point<dim>(), 3);
      predicates[3] = predicate_template<dim>(Point<dim>(-5,-5,-5), 3);
      predicates[4] = predicate_template<dim>(Point<dim>(-10,-10,-10), 3);
    }

  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      {
        deallog << i << ":" << j << "="
                << find_connection_between_subdomains (tria,
                                                       predicates[i],
                                                       predicates[j])
                << std::endl;
      }
}


int main ()
{
  initlog();

  test<2> ();
  test<3> ();

  return 0;
}
