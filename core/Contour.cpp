#include "Contour.h"

#include "StaggeredGrid.h"

namespace Pivot {
	static inline std::uint8_t GetCellType(GridData<double> const &grData, Vector3i const &cell, double value) {
		std::uint8_t type = 0;
		for (int i = 0; i < StaggeredGrid::GetNumNodesPerCell(); i++) {
			Vector3i const node = StaggeredGrid::NodeOfCell(cell, i);
			if (grData[node] <= value) type |= 1 << i;
		}
		return type;
	}

	Contour::Contour(Grid const &nodeGrid) :
		m_NodeGrid { nodeGrid },
		m_CellGrid(m_NodeGrid.GetSpacing(), m_NodeGrid.GetSize() - Vector3i::Ones(), m_NodeGrid.GetOrigin() + Vector3d::Constant(m_NodeGrid.GetSpacing() / 2)),
		m_EdgeGrids {
			Grid(m_NodeGrid.GetSpacing(), m_NodeGrid.GetSize() - Vector3i::Unit(0), m_NodeGrid.GetOrigin() + Vector3d::Unit(0) * m_NodeGrid.GetSpacing() / 2),
			Grid(m_NodeGrid.GetSpacing(), m_NodeGrid.GetSize() - Vector3i::Unit(1), m_NodeGrid.GetOrigin() + Vector3d::Unit(1) * m_NodeGrid.GetSpacing() / 2),
			Grid(m_NodeGrid.GetSpacing(), m_NodeGrid.GetSize() - Vector3i::Unit(2), m_NodeGrid.GetOrigin() + Vector3d::Unit(2) * m_NodeGrid.GetSpacing() / 2),
		},
		m_EdgeMark {
			GridData<int>(m_EdgeGrids[0]),
			GridData<int>(m_EdgeGrids[1]),
			GridData<int>(m_EdgeGrids[2]),
		} {
	}

	void Contour::Generate(GridData<double> const &grData, double value) {
		if (m_NodeGrid != grData.GetGrid()) {
			spdlog::critical("Failed to generate contour because of incompatible grids");
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
						Vector3i const node0 = StaggeredGrid::NodeOfEdge(axis, edge, 0);
						Vector3i const node1 = StaggeredGrid::NodeOfEdge(axis, edge, 1);
						double const theta = (grData[node0] - value) / (grData[node0] - grData[node1]);
						Vector3d const pos = (1 - theta) * m_NodeGrid.PositionOf(node0) + theta * m_NodeGrid.PositionOf(node1);
						m_EdgeMark[axis][edge] = static_cast<int>(m_Mesh.Positions.size());
						m_Mesh.Positions.push_back(pos);
					}
				}
			}
			for (auto *it = c_EdgeOrdsTable3[cellType]; *it != -1; it++) {
				auto const [axis, edge] = StaggeredGrid::EdgeOfCell(cell, *it);
				m_Mesh.Indices.push_back(static_cast<std::uint32_t>(m_EdgeMark[axis][edge]));
			}
		});
	}

	void Contour::ComputeVertexInfos() {
		m_Mesh.ComputeNormals();
	}
}
