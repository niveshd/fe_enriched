#ifndef PARAMETER_READER_H
#define PARAMETER_READER_H

#include <deal.II/base/point.h>
#include <string>
#include <vector>
#include <iostream>
#include <deal.II/base/parameter_handler.h>


template<int dim>
struct ParameterCollection
{
  ParameterCollection(const std::string &file_name);

  ParameterCollection
  (const double size,
   const unsigned int shape,
   const unsigned int global_refinement,
   const unsigned int cycles,
   const unsigned int fe_base_degree,
   const unsigned int fe_enriched_degree,
   const unsigned int max_iterations,
   const double tolerance,
   const unsigned int patches,
   const unsigned int debug_level,
   const unsigned int n_enrichments,
   const std::vector<Point<dim>> points_enrichments,
   const std::vector<double> radii_predicates,
   const std::vector<double> sigmas_rhs)
    :
    size(size),
    shape(shape),
    global_refinement(global_refinement),
    cycles(cycles),
    fe_base_degree(fe_base_degree),
    fe_enriched_degree(fe_enriched_degree),
    max_iterations(max_iterations),
    tolerance(tolerance),
    patches(patches),
    debug_level(debug_level),
    n_enrichments(n_enrichments),
    points_enrichments(points_enrichments),
    radii_predicates(radii_predicates),
    sigmas_rhs(sigmas_rhs)
  {}

  void print()
  {
    std::cout << "Size : "<< size << std::endl;
    std::cout << "Shape : " << shape << std::endl;
    std::cout << "Global refinement : " << global_refinement << std::endl;
    std::cout << "Cycles : " << cycles << std::endl;
    std::cout << "FE base degree : " << fe_base_degree << std::endl;
    std::cout << "FE enriched degree : " << fe_enriched_degree << std::endl;
    std::cout << "Max Iterations : " << max_iterations << std::endl;
    std::cout << "Tolerance : " << tolerance << std::endl;
    std::cout << "Patches used for output: " << patches << std::endl;
    std::cout << "Debug level: " << debug_level << std::endl;
    std::cout << "Number of enrichments: " << n_enrichments << std::endl;

    std::cout << "Enrichment points : " << std::endl;
    for (auto p:points_enrichments)
      std::cout << p << std::endl;

    std::cout << "Enrichment radii : " << std::endl;
    for (auto r:radii_predicates)
      std::cout << r << std::endl;

    std::cout << "Sigma : " << std::endl;
    for (auto r:sigmas_rhs)
      std::cout << r << std::endl;
  }


  double size;
  unsigned int shape; //0 = ball, 1 = cube
  unsigned int global_refinement;
  unsigned int cycles;
  unsigned int fe_base_degree;
  unsigned int fe_enriched_degree;
  unsigned int max_iterations;
  double tolerance;
  unsigned int patches;
  //debug level = 0(output nothing),
  //1 (print statements)
  //2 (output solution)
  //3 (+ output grid data as well)
  //9 (+ shape functions as well)
  unsigned int debug_level;
  unsigned int n_enrichments;
  std::vector<Point<dim>> points_enrichments;
  std::vector<double> radii_predicates;
  std::vector<double> sigmas_rhs;
  std::vector<std::string> rhs_expressions;
};


template<int dim>
ParameterCollection<dim>::ParameterCollection
(const std::string &file_name)
{

  //std::cout << "...reading parameters" << std::endl;

  ParameterHandler prm;

  //declare parameters
  prm.enter_subsection("geometry");
  prm.declare_entry("size",
                    "1",
                    Patterns::Double(0));
  prm.declare_entry("shape",
                    "1",
                    Patterns::Integer(0));
  prm.declare_entry("Global refinement",
                    "1",
                    Patterns::Integer(1));
  prm.declare_entry("cycles",
                    "0",
                    Patterns::Integer(0));
  prm.leave_subsection();

  prm.enter_subsection("solver");
  prm.declare_entry("fe base degree",
                    "1",
                    Patterns::Integer(1));
  prm.declare_entry("fe enriched degree",
                    "1",
                    Patterns::Integer(1));
  prm.declare_entry("max iterations",
                    "1000",
                    Patterns::Integer(1));
  prm.declare_entry("tolerance",
                    "1e-8",
                    Patterns::Double(0));
  prm.leave_subsection();

  prm.enter_subsection("output");
  prm.declare_entry("patches",
                    "1",
                    Patterns::Integer(1));
  prm.declare_entry("debug level",
                    "0",
                    Patterns::Integer(0,9));
  prm.leave_subsection();

  //parse parameter file
  prm.parse_input(file_name, "#end-of-dealii parser");


  //get parameters
  prm.enter_subsection("geometry");
  size = prm.get_double("size");
  shape = prm.get_integer("shape");
  global_refinement = prm.get_integer("Global refinement");
  cycles = prm.get_integer("cycles");
  prm.leave_subsection();

  prm.enter_subsection("solver");
  fe_base_degree = prm.get_integer("fe base degree");
  fe_enriched_degree = prm.get_integer("fe enriched degree");
  max_iterations = prm.get_integer("max iterations");
  tolerance = prm.get_double("tolerance");
  prm.leave_subsection();

  prm.enter_subsection("output");
  patches = prm.get_integer("patches");
  debug_level = prm.get_integer("debug level");
  prm.leave_subsection();


  //std::cout << "Size : "<< size << std::endl;
  //std::cout << "Global refinement : " << global_refinement << std::endl;
  //std::cout << "Max Iterations : " << max_iterations << std::endl;
  //std::cout << "Tolerance : " << tolerance << std::endl;
  //std::cout << "Patches used for output: " << patches << std::endl;
  //std::cout << "Debug level: " << debug_level << std::endl;

  //manual parsing
  //open parameter file
  std::ifstream prm_file(file_name);

  //read lines until "#end-of-dealii parser" is reached
  std::string line;
  while (getline(prm_file,line))
    if (line == "#end-of-dealii parser")
      break;

  AssertThrow(line == "#end-of-dealii parser",
              ExcMessage("line missing in parameter file = \'#end-of-dealii parser\' "));

  //function to read next line not starting with # or empty
  auto read_next_proper_line = [&] (std::string &line)
  {
    while (getline(prm_file,line))
      {
        if (line.size()==0 || line[0] == '#' || line[0] == ' ')
          continue;
        else
          break;
      }
  };

  std::stringstream s_stream;

  //read num of enrichement points
  read_next_proper_line(line);
  s_stream.str(line);
  s_stream >> n_enrichments;
  //std::cout << "Number of enrichments: " << n_enrichments << std::endl;

  //note vector of points
  for (unsigned int i=0; i!=n_enrichments; ++i)
    {
      read_next_proper_line(line);
      s_stream.clear();
      s_stream.str(line);

      if (dim==2)
        {
          double x,y;
          s_stream >> x >> y;
          points_enrichments.push_back({x,y});
        }
      else if (dim==3)
        {
          double x,y,z;
          s_stream << x << y << z;
          points_enrichments.push_back({x,y,z});
        }
      else
        AssertThrow(false, ExcMessage("Dimension not implemented"));
    }

  //std::cout << "Enrichment points : " << std::endl;
//  for (auto p:points_enrichments)
  //std::cout << p << std::endl;

  //note vector of radii for predicates
  for (unsigned int i=0; i!=n_enrichments; ++i)
    {
      read_next_proper_line(line);
      s_stream.clear();
      s_stream.str(line);

      double r;
      s_stream >> r;
      radii_predicates.push_back(r);
    }

  //std::cout << "Enrichment radii : " << std::endl;
//  for (auto r:radii_predicates)
  //std::cout << r << std::endl;

  //note vector of sigmas for rhs
  for (unsigned int i=0; i!=n_enrichments; ++i)
    {
      read_next_proper_line(line);
      s_stream.clear();
      s_stream.str(line);

      double r;
      s_stream >> r;
      sigmas_rhs.push_back(r);
    }

  //std::cout << "Sigma : " << std::endl;
//  for (auto r:sigmas_rhs)
  //std::cout << r << std::endl;

  //read right hand side function expressions
  for (unsigned int i=0; i!=n_enrichments; ++i)
    {

      read_next_proper_line(line);
      s_stream.clear();

      rhs_expressions.push_back(line);
    }

  std::cout << "rhs expressions : " << std::endl;
  for (auto r:rhs_expressions)
    std::cout << r << std::endl;


  //std::cout << "...finished parameter reading from file." << std::endl;
}

#endif
