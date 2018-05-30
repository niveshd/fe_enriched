// ..........................................-
//
// Copyright (C) 2016 - 2017 by the deal.II authors
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
// ..........................................-


#include <deal.II/base/utilities.h>
#include <deal.II/base/function.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/grid/grid_out.h>

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
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/base/parameter_handler.h>

#include <string>
#include <sstream>
#include <fstream>

#include "support.h"
#include "estimate_enrichment.h"
#include "paramater_reader.h"
#include "solver.h"
#include "solver_trilinos.h"
#include "../tests/tests.h"


int main (int argc,char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  MPILogInitAll all;
  {
    AssertThrow(argc>=2, ExcMessage("Parameter file not given."));
    ParameterCollection prm(argv[1]);

    if (prm.dim == 2)
      {
        const int d = 2;
        if (prm.solver == prm.trilinos_amg){
          Step1::LaplaceProblem_t<d> step1(prm);
          step1.run();
          } else {
          Step1::LaplaceProblem<d> step1(prm);
          step1.run();
          }
      }
    else if (prm.dim == 3)
      {
        const int d = 3;
        if (prm.solver == prm.trilinos_amg){
          Step1::LaplaceProblem_t<d> step1(prm);
          step1.run();
          } else {
          Step1::LaplaceProblem<d> step1(prm);
          step1.run();
          }
      }
    else
      AssertThrow(false, ExcMessage("Dimension incorect. dim can be 2 or 3"));
  }
}
