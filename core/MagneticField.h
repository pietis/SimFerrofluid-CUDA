#pragma once

#include "SGridData.h"
#include "StaggeredGrid.h"
#include "SurfaceMesh.h"

#include <amgcl/amg.hpp>
#include <amgcl/backend/eigen.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>

namespace Pivot {
class MagneticField {
  private:
    using Solver = amgcl::make_solver<
        amgcl::amg<amgcl::backend::eigen<double>,
                   amgcl::coarsening::smoothed_aggregation,
                   amgcl::relaxation::spai0>,
        amgcl::solver::cg<amgcl::backend::eigen<double>>>;

  public:
    MagneticField(StaggeredGrid const &sgrid, double susceptibility,
                  double bandWidth);

    void SetSusceptibility(double susceptibility) {
        m_Susceptibility = susceptibility;
    }

    void SetBandWidth(double bandWidth) { m_BandWidth = bandWidth; }

    void Solve(GridData<double> const &levelSet,
               SGridData<double> const &externalField);

    std::vector<double> CalculateVertexPressures(SurfaceMesh const &mesh) const;

  private:
    void BuildMatrix(GridData<double> const &levelSet);
    void AssembleRHS(SGridData<double> const &externalField);
    void SolveLinearSystem();
    void AssignTotalField(SGridData<double> const &externalField);

    double Heaviside(double phi) const {
        if (phi < -m_BandWidth) {
            return 0.;
        }
        if (phi > m_BandWidth) {
            return 1.;
        }
        return .5 * (1. + phi / m_BandWidth +
                     std::sin(std::numbers::pi * phi / m_BandWidth) /
                         std::numbers::pi);
    }

  private:
    StaggeredGrid m_SGridExt;

    SGridData<double> m_Coeff;
    SGridData<double> m_TotalField;
    GridData<double> m_Potential;

    double m_Susceptibility;
    double m_BandWidth;

    SparseMatrix<double, RowMajor> m_MatL;

    VectorXd m_RHS;
    VectorXd m_Sol;

    std::unique_ptr<Solver> m_Solver;
};
} // namespace Pivot