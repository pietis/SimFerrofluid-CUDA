#include "MagneticField.h"

#include "TriLerp.h"

namespace Pivot {
namespace {
constexpr double c_Pi = 3.141592653589793238462643383279502884;
constexpr double c_Mu0 = 4e-7 * c_Pi;
}

MagneticField::MagneticField(StaggeredGrid const &sgrid,
                             double susceptibility, double bandWidth)
    : m_SGridExt(0, sgrid.GetSpacing(), sgrid.GetResolution(),
                 sgrid.GetOrigin() +
                     sgrid.GetResolution().cast<double>() *
                         sgrid.GetSpacing() / 2.),
      m_Coeff(m_SGridExt.GetFaceGrids(), Vector3d::Constant(1.)),
      m_TotalField(m_SGridExt.GetFaceGrids()),
      m_Potential(m_SGridExt.GetCellGrid()),
      m_Susceptibility(susceptibility), m_BandWidth(bandWidth) {
    int const numCells = m_Potential.GetGrid().GetNumVertices();
    m_MatL.resize(numCells, numCells);
    m_RHS.resize(numCells);
    m_Sol.resize(numCells);
    m_RHS.setZero();
    m_Sol.setZero();
}

void MagneticField::Solve(GridData<double> const &levelSet,
                          SGridData<double> const &externalField) {
    BuildMatrix(levelSet);
    AssembleRHS(externalField);
    SolveLinearSystem();
    AssignTotalField(externalField);
}

std::vector<double>
MagneticField::CalculateVertexPressures(SurfaceMesh const &mesh) const {
    std::vector<double> pressures(mesh.size());
    tbb::parallel_for(0, mesh.size(), [&](int i) {
        Vector3d const H = TriLerp::Interpolate(m_TotalField, mesh.Positions[i]);
		double coeff = H.squaredNorm();
		double Hn = H.dot(mesh.Normals[i]);
		coeff += m_Susceptibility * m_Susceptibility / (4 * m_Susceptibility + 4) * Hn * Hn;
		pressures[i] = c_Mu0 * 0.5 * m_Susceptibility * coeff;
    });
    return pressures;
}

void MagneticField::BuildMatrix(GridData<double> const &levelSet) {
    ParallelForEach(m_Coeff.GetGrids(), [&](int axis, Vector3i const &face) {
        if (m_SGridExt.IsBoundaryFace(axis, face)) {
            m_Coeff[axis][face] = 1.;
        } else {
            Vector3d const pos = m_Coeff[axis].GetGrid().PositionOf(face);
            m_Coeff[axis][face] =
                1. + (1. - Heaviside(TriLerp::Interpolate(levelSet, pos))) *
                         m_Susceptibility;
        }
    });

    auto const &grid = m_Potential.GetGrid();
    std::vector<Triplet<double>> elements;
    elements.reserve(static_cast<std::size_t>(grid.GetNumVertices()) * 7);

    for (int r = 0; r < grid.GetNumVertices(); r++) {
        if (r == 0) {
            elements.emplace_back(r, r, 1.);
            continue;
        }

        Vector3i const cell = grid.CoordOf(r);
        double diagCoeff = 0.;
        for (int i = 0; i < Grid::GetNumNeighbors(); i++) {
            Vector3i const nbCell = Grid::NeighborOf(cell, i);
            if (!grid.IsValid(nbCell)) {
                continue;
            }
            auto const [axis, face] = StaggeredGrid::FaceOfCell(cell, i);
            double const weight = m_Coeff[axis][face];
            int const col = grid.IndexOf(nbCell);
            if (col != 0) {
                elements.emplace_back(r, col, -weight);
            }
            diagCoeff += weight;
        }
        elements.emplace_back(r, r, diagCoeff);
    }

    m_MatL.setFromTriplets(elements.begin(), elements.end());

    Solver::params prm;
    prm.solver.tol = 1e-6;
    m_Solver = std::make_unique<Solver>(m_MatL, prm);
}

void MagneticField::AssembleRHS(SGridData<double> const &externalField) {
    auto const &grid = m_Potential.GetGrid();
    m_RHS.setZero();

    for (int r = 0; r < grid.GetNumVertices(); r++) {
        if (r == 0) {
            continue;
        }
        Vector3i const cell = grid.CoordOf(r);
        for (int i = 0; i < Grid::GetNumNeighbors(); i++) {
            Vector3i const nbCell = Grid::NeighborOf(cell, i);
            if (!grid.IsValid(nbCell)) {
                continue;
            }
            auto const [axis, face] = StaggeredGrid::FaceOfCell(cell, i);
            int const side = StaggeredGrid::FaceSideOfCell(i);
            double const weight = m_Coeff[axis][face];
            m_RHS[r] += (weight - 1.) * externalField[axis][face] * (-side) *
                        grid.GetSpacing();
        }
    }
}

void MagneticField::SolveLinearSystem() {
    auto const [iters, error] = m_Solver->operator()(m_RHS, m_Sol);
    fmt::print("{:>6} iters ({:.2e}) ", iters, error);
}

void MagneticField::AssignTotalField(SGridData<double> const &externalField) {
    ParallelForEach(m_Potential.GetGrid(), [&](Vector3i const &cell) {
        m_Potential[cell] = m_Sol[m_Potential.GetGrid().IndexOf(cell)];
    });

    ParallelForEach(m_TotalField.GetGrids(), [&](int axis, Vector3i const &face) {
        Vector3i const cell0 = StaggeredGrid::AdjCellOfFace(axis, face, 0);
        Vector3i const cell1 = StaggeredGrid::AdjCellOfFace(axis, face, 1);
        if (m_Potential.GetGrid().IsValid(cell0) &&
            m_Potential.GetGrid().IsValid(cell1)) {
            m_TotalField[axis][face] =
                (m_Potential[cell0] - m_Potential[cell1]) *
                m_Potential.GetGrid().GetInvSpacing();
        } else {
            m_TotalField[axis][face] = 0.;
        }
        m_TotalField[axis][face] += externalField[axis][face];
    });
}
} // namespace Pivot