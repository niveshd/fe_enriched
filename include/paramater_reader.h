#ifndef PARAMETER_READER_H
#define PARAMETER_READER_H

#include <deal.II/base/point.h>
#include <string>
#include <vector>
#include <iostream>
#include <deal.II/base/parameter_handler.h>


struct ParameterCollection
{
  ParameterCollection(const std::string &file_name);

  ParameterCollection
  (const int &dim,
   const double &size,
   const unsigned int &shape,
   const unsigned int &global_refinement,
   const unsigned int &cycles,
   const unsigned int &fe_base_degree,
   const unsigned int &fe_enriched_degree,
   const unsigned int &max_iterations,
   const double &tolerance,
   const std::string &rhs_value_expr,
   const std::string &boundary_value_expr,
   const std::string &rhs_radial_problem,
   const std::string &boundary_radial_problem,
   const std::string &exact_soln_expr,
   const unsigned int &patches,
   const unsigned int &debug_level,
   const unsigned int &n_enrichments,
   const std::vector<double> &points_enrichments,
   const std::vector<double> &radii_predicates,
   const std::vector<double> &sigmas);

  void print();

  void set_enrichment_point(Point <2> &p, const unsigned int i)
  {
    AssertDimension(dim,2);
    p(0) = points_enrichments[2*i];
    p(1) = points_enrichments[2*i + 1];
  }
  void set_enrichment_point(Point <3> &p, const unsigned int i)
  {
    AssertDimension(dim,3);
    p(0) = points_enrichments[3*i];
    p(1) = points_enrichments[3*i + 1];
    p(2) = points_enrichments[3*i + 2];
  }

  int dim;
  double size;
  unsigned int shape; //0 = ball, 1 = cube
  unsigned int global_refinement;
  unsigned int cycles;
  unsigned int fe_base_degree;
  unsigned int fe_enriched_degree;
  unsigned int max_iterations;
  double tolerance;

  //parameters related to exact solution
  std::string rhs_value_expr;
  std::string boundary_value_expr;

  //value = true ==> estimate exact solution from radial problem
  std::string rhs_radial_problem;
  std::string boundary_radial_problem;

  std::string exact_soln_expr;

  unsigned int patches;
  //debug level = 0(output nothing),
  //1 (print statements)
  //2 (output solution)
  //3 (+ output grid data as well)
  //9 (+ shape functions as well)
  unsigned int debug_level;
  unsigned int n_enrichments;
  std::vector<double> points_enrichments;
  std::vector<double> radii_predicates;
  std::vector<double> sigmas;
};



ParameterCollection::ParameterCollection
(const std::string &file_name)
{

  //std::cout << "...reading parameters" << std::endl;

  ParameterHandler prm;

  //declare parameters
  prm.enter_subsection("geometry");
  prm.declare_entry("dim",
                    "2",
                    Patterns::Integer());
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


  prm.enter_subsection("expressions");
  prm.declare_entry("rhs value",
                    "0",
                    Patterns::Anything());
  prm.declare_entry("boundary value",
                    "0",
                    Patterns::Anything());
  prm.declare_entry("rhs value radial problem",
                    "0",
                    Patterns::Anything());
  prm.declare_entry("boundary value radial problem",
                    "0",
                    Patterns::Anything());
  prm.declare_entry("exact solution expression",
                    "",
                    Patterns::Anything());
  prm.declare_entry("estimate exact solution",
                    "false",
                    Patterns::Bool());
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
  dim = prm.get_integer("dim");
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

  prm.enter_subsection("expressions");
  rhs_value_expr = prm.get("rhs value");
  boundary_value_expr = prm.get("boundary value");
  rhs_radial_problem = prm.get("rhs value radial problem");
  boundary_radial_problem = prm.get("boundary value radial problem");
  exact_soln_expr = prm.get("exact solution expression");
  prm.leave_subsection();

  prm.enter_subsection("output");
  patches = prm.get_integer("patches");
  debug_level = prm.get_integer("debug level");
  prm.leave_subsection();


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

  //note vector of points
  for (unsigned int i=0; i!=n_enrichments; ++i)
    {
      read_next_proper_line(line);
      s_stream.clear();
      s_stream.str(line);

      points_enrichments.resize(dim*n_enrichments);

      if (dim==2)
        {
          double x,y;
          s_stream >> x >> y;
          points_enrichments[2*i]=x;
          points_enrichments[2*i+1]=y;
        }
      else if (dim==3)
        {
          double x,y,z;
          s_stream >> x >> y >> z;
          points_enrichments[3*i]=x;
          points_enrichments[3*i+1]=y;
          points_enrichments[3*i+2]=z;
        }
      else
        AssertThrow(false, ExcMessage("Dimension not implemented"));
    }

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

  //note vector of sigmas for rhs
  for (unsigned int i=0; i!=n_enrichments; ++i)
    {
      read_next_proper_line(line);
      s_stream.clear();
      s_stream.str(line);

      double r;
      s_stream >> r;
      sigmas.push_back(r);
    }
}



ParameterCollection::ParameterCollection
(const int &dim,
 const double &size,
 const unsigned int &shape,
 const unsigned int &global_refinement,
 const unsigned int &cycles,
 const unsigned int &fe_base_degree,
 const unsigned int &fe_enriched_degree,
 const unsigned int &max_iterations,
 const double &tolerance,
 const std::string &rhs_value_expr,
 const std::string &boundary_value_expr,
 const std::string &rhs_radial_problem,
 const std::string &boundary_radial_problem,
 const std::string &exact_soln_expr,
 const unsigned int &patches,
 const unsigned int &debug_level,
 const unsigned int &n_enrichments,
 const std::vector<double> &points_enrichments,
 const std::vector<double> &radii_predicates,
 const std::vector<double> &sigmas)
  :
  dim(dim),
  size(size),
  shape(shape),
  global_refinement(global_refinement),
  cycles(cycles),
  fe_base_degree(fe_base_degree),
  fe_enriched_degree(fe_enriched_degree),
  max_iterations(max_iterations),
  tolerance(tolerance),
  rhs_value_expr(rhs_value_expr),
  boundary_value_expr(boundary_value_expr),
  rhs_radial_problem(rhs_radial_problem),
  boundary_radial_problem(boundary_radial_problem),
  exact_soln_expr(exact_soln_expr),
  patches(patches),
  debug_level(debug_level),
  n_enrichments(n_enrichments),
  points_enrichments(points_enrichments),
  radii_predicates(radii_predicates),
  sigmas(sigmas)
{}



void ParameterCollection::print()
{
  std::cout << "Dim : " << dim << std::endl
            << "Size : "<< size << std::endl
            << "Shape : " << shape << std::endl
            << "Global refinement : " << global_refinement << std::endl
            << "Cycles : " << cycles << std::endl
            << "FE base degree : " << fe_base_degree << std::endl
            << "FE enriched degree : " << fe_enriched_degree << std::endl
            << "Max Iterations : " << max_iterations << std::endl
            << "Tolerance : " << tolerance << std::endl
            << "rhs - main problem : "
            << rhs_value_expr << std::endl
            << "boundary value - main problem : "
            << boundary_value_expr << std::endl
            << "rhs of radial problem : "
            << rhs_radial_problem << std::endl
            << "boundary value of radial problem : "
            << boundary_radial_problem << std::endl
            << "exact solution expr : "
            << exact_soln_expr << std::endl
            << "estimate exact solution using radial problem : "
            << "Patches used for output: " << patches << std::endl
            << "Debug level: " << debug_level << std::endl
            << "Number of enrichments: " << n_enrichments << std::endl;

  std::cout << "Enrichment points : " << std::endl;
  for (unsigned int i = 0; i < points_enrichments.size(); i = i+dim)
    {
      for (int d = 0; d < dim; ++d)
        std::cout << points_enrichments[i+d] << " ";

      std::cout << std::endl;
    }

  std::cout << "Enrichment radii : " << std::endl;
  for (auto r:radii_predicates)
    std::cout << r << std::endl;

  std::cout << "Sigma values of different sources : " << std::endl;
  for (auto r:sigmas)
    std::cout << r << std::endl;
}

#endif
