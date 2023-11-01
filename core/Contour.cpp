#include "Contour.h"

#include "StaggeredGrid.h"

#include "FiniteDiff.h"
#include "LagrangeDiff.h"

#include "fraction.hpp"

namespace Pivot {
static inline std::uint8_t GetCellType(GridData<double> const &grData,
                                       Vector3i const &cell, double value) {
    std::uint8_t type = 0;
    for (int i = 0; i < StaggeredGrid::GetNumNodesPerCell(); i++) {
        Vector3i const node = StaggeredGrid::NodeOfCell(cell, i);
        if (grData[node] <= value)
            type |= 1 << i;
    }
    return type;
}

Contour::Contour(Grid const &nodeGrid)
    : m_NodeGrid{nodeGrid},
      m_CellGrid(m_NodeGrid.GetSpacing(),
                 m_NodeGrid.GetSize() - Vector3i::Ones(),
                 m_NodeGrid.GetOrigin() +
                     Vector3d::Constant(m_NodeGrid.GetSpacing() / 2)),
      m_EdgeGrids{
          Grid(m_NodeGrid.GetSpacing(),
               m_NodeGrid.GetSize() - Vector3i::Unit(0),
               m_NodeGrid.GetOrigin() +
                   Vector3d::Unit(0) * m_NodeGrid.GetSpacing() / 2),
          Grid(m_NodeGrid.GetSpacing(),
               m_NodeGrid.GetSize() - Vector3i::Unit(1),
               m_NodeGrid.GetOrigin() +
                   Vector3d::Unit(1) * m_NodeGrid.GetSpacing() / 2),
          Grid(m_NodeGrid.GetSpacing(),
               m_NodeGrid.GetSize() - Vector3i::Unit(2),
               m_NodeGrid.GetOrigin() +
                   Vector3d::Unit(2) * m_NodeGrid.GetSpacing() / 2),
      },
      m_EdgeMark{
          GridData<int>(m_EdgeGrids[0]),
          GridData<int>(m_EdgeGrids[1]),
          GridData<int>(m_EdgeGrids[2]),
      } {}

void Contour::Generate(GridData<double> const &grData, double value) {
    if (m_NodeGrid != grData.GetGrid()) {
        spdlog::critical(
            "Failed to generate contour because of incompatible grids");
        std::exit(EXIT_FAILURE);
    }

    m_EdgeMark[0].SetConstant(-1);
    m_EdgeMark[1].SetConstant(-1);
    m_EdgeMark[2].SetConstant(-1);
    m_Mesh.Clear();

    ForEach(m_CellGrid, [&](Vector3i const &cell) {
        auto const cellType = GetCellType(grData, cell, value);
        auto edgeState = c_EdgeStateTable3[cellType];
        for (int i = 0; i < StaggeredGrid::GetNumEdgesPerCell(); i++) {
            if (edgeState >> i & 1) {
                auto const [axis, edge] = StaggeredGrid::EdgeOfCell(cell, i);
                if (m_EdgeMark[axis][edge] < 0) {
                    Vector3i const node0 =
                        StaggeredGrid::NodeOfEdge(axis, edge, 0);
                    Vector3i const node1 =
                        StaggeredGrid::NodeOfEdge(axis, edge, 1);
                    double const theta = (grData[node0] - value) /
                                         (grData[node0] - grData[node1]);
                    Vector3d const pos =
                        (1 - theta) * m_NodeGrid.PositionOf(node0) +
                        theta * m_NodeGrid.PositionOf(node1);
                    m_EdgeMark[axis][edge] =
                        static_cast<int>(m_Mesh.Positions.size());
                    m_Mesh.Positions.push_back(pos);
                }
            }
        }
        for (auto *it = c_EdgeOrdsTable3[cellType]; *it != -1; it++) {
            auto const [axis, edge] = StaggeredGrid::EdgeOfCell(cell, *it);
            m_Mesh.Indices.push_back(
                static_cast<std::uint32_t>(m_EdgeMark[axis][edge]));
        }
    });
}

void Contour::ComputeVertexInfos() {
    m_Mesh.ComputeNormals();
    m_Mesh.ComputeAreas();
    m_Mesh.ComputeMeanCurvatures();
}
void Contour::ComputeVertexInfosFromLS(GridData<double> const &levelSet) {
    m_Mesh.ComputeAreas();
    m_Mesh.Normals.resize(m_Mesh.size());
    m_Mesh.MeanCurvatures.resize(m_Mesh.size());

    ForEach(m_CellGrid, [&](Vector3i const &cell) {
        for (int i = 0; i < StaggeredGrid::GetNumEdgesPerCell(); i++) {
            auto const [axis, edge] = StaggeredGrid::EdgeOfCell(cell, i);
            int index = VertexIndexOf(axis, edge);
            if (index >= 0) {
                Vector3i const cell0 = StaggeredGrid::NodeOfEdge(axis, edge, 0);
                Vector3i const cell1 = StaggeredGrid::NodeOfEdge(axis, edge, 1);
                double const theta =
                    levelSet[cell0] / (levelSet[cell0] - levelSet[cell1]);
                Vector3d grad0 =
                    Vector3d(FiniteDiff::CalcFirstDrv(levelSet, cell0, 0),
                             FiniteDiff::CalcFirstDrv(levelSet, cell0, 1),
                             FiniteDiff::CalcFirstDrv(levelSet, cell0, 2))
                        .normalized();
                Vector3d grad1 =
                    Vector3d(FiniteDiff::CalcFirstDrv(levelSet, cell1, 0),
                             FiniteDiff::CalcFirstDrv(levelSet, cell1, 1),
                             FiniteDiff::CalcFirstDrv(levelSet, cell1, 2))
                        .normalized();
                Vector3d grad =
                    ((1 - theta) * grad0 + theta * grad1).normalized();
                m_Mesh.Normals[index] = grad;
                // double const kappa0 =
                //     FiniteDiff::CalcCurvature(levelSet, cell0);
                // double const kappa1 =
                //     FiniteDiff::CalcCurvature(levelSet, cell1);
                // double const kappa = (1 - theta) * kappa0 + theta * kappa1;
                // m_Mesh.MeanCurvatures[index] = kappa;
                // double phi_x =
                //     (1 - theta) * FiniteDiff::CalcFirstDrv(levelSet, cell0, 0) +
                //     theta * FiniteDiff::CalcFirstDrv(levelSet, cell1, 0);
                // double phi_y =
                //     (1 - theta) * FiniteDiff::CalcFirstDrv(levelSet, cell0, 1) +
                //     theta * FiniteDiff::CalcFirstDrv(levelSet, cell1, 1);
                // double phi_z =
                //     (1 - theta) * FiniteDiff::CalcFirstDrv(levelSet, cell0, 2) +
                //     theta * FiniteDiff::CalcFirstDrv(levelSet, cell1, 2);
                // double phi_xx =
                //     (1 - theta) *
                //         FiniteDiff::CalcSecondDrv(levelSet, cell0, 0, 0) +
                //     theta * FiniteDiff::CalcSecondDrv(levelSet, cell1, 0, 0);
                // double phi_xy =
                //     (1 - theta) *
                //         FiniteDiff::CalcSecondDrv(levelSet, cell0, 0, 1) +
                //     theta * FiniteDiff::CalcSecondDrv(levelSet, cell1, 0, 1);
                // fmt::print("{:.3e} {:.3e} {:.3e} {:.3e} {:.3e} {:.3e}\n", phi_x,
                //            phi_y, phi_z, phi_xx, phi_xy, kappa);
                Vector3d pos = m_Mesh.Positions[index];
                m_Mesh.MeanCurvatures[index] =
                    LagrangeDiff::CalcCurvature<7>(levelSet, pos);
            }
        }
    });
};

void Contour::ComputeVolumeFromLS(GridData<double> const &levelSet) {
    double vol = 0;
    ForEach(m_CellGrid, [&](Vector3i const &cell) {
        std::array<double, 8> phi3d{
            levelSet.At(cell + Vector3i(0, 0, 0)),
            levelSet.At(cell + Vector3i(1, 0, 0)),
            levelSet.At(cell + Vector3i(1, 1, 0)),
            levelSet.At(cell + Vector3i(0, 1, 0)),
            levelSet.At(cell + Vector3i(0, 0, 1)),
            levelSet.At(cell + Vector3i(1, 0, 1)),
            levelSet.At(cell + Vector3i(1, 1, 1)),
            levelSet.At(cell + Vector3i(0, 1, 1)),
        };
        vol += Fraction::get_mc_vol(phi3d);
    });
    m_Mesh.TotalVolume = vol * levelSet.GetGrid().GetSpacing() *
                         levelSet.GetGrid().GetSpacing() *
                         levelSet.GetGrid().GetSpacing();
};

} // namespace Pivot
