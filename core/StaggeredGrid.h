#pragma once

#include "Grid.h"

namespace Pivot {
	class StaggeredGrid {
	public:
		StaggeredGrid(int boundaryWidth, double spacing, Vector3i resolution, Vector3d center = Vector3d::Zero()) :
			m_BoundaryWidth { boundaryWidth },
			m_Spacing { spacing },
			m_InvSpacing { 1 / m_Spacing },
			m_Resolution { resolution },
			m_Origin { center - m_Resolution.cast<double>() * m_Spacing / 2 },
			m_NodeGrid(m_Spacing, m_Resolution + Vector3i::Ones(), m_Origin),
			m_CellGrid(m_Spacing, m_Resolution, m_Origin + Vector3d::Constant(m_Spacing / 2)),
			m_FaceGrids {
				Grid(m_Spacing, m_Resolution + Vector3i(1, 0, 0), m_Origin + Vector3d(0, 1, 1) * m_Spacing / 2),
				Grid(m_Spacing, m_Resolution + Vector3i(0, 1, 0), m_Origin + Vector3d(1, 0, 1) * m_Spacing / 2),
				Grid(m_Spacing, m_Resolution + Vector3i(0, 0, 1), m_Origin + Vector3d(1, 1, 0) * m_Spacing / 2),
			} {
		}

		int      GetBoundaryWidth() const { return m_BoundaryWidth; }
		double   GetSpacing      () const { return m_Spacing; }
		double   GetInvSpacing   () const { return m_InvSpacing; }
		Vector3i GetResolution   () const { return m_Resolution; }
		Vector3d GetOrigin       () const { return m_Origin; }

		int GetNumNodes() const { return m_NodeGrid.GetNumVertices(); }
		int GetNumCells() const { return m_CellGrid.GetNumVertices(); }
		int GetNumFaces() const { return m_FaceGrids[0].GetNumVertices() + m_FaceGrids[1].GetNumVertices() + m_FaceGrids[2].GetNumVertices(); }

		Grid                const &GetNodeGrid () const { return m_NodeGrid; }
		Grid                const &GetCellGrid () const { return m_CellGrid; }
		std::array<Grid, 3> const &GetFaceGrids() const { return m_FaceGrids; }

		Grid CreateRefinedCellGrid(int factor) const { return Grid(m_Spacing / factor, m_Resolution * factor, m_Origin + Vector3d::Constant(m_Spacing / factor / 2)); }

		Vector3d GetDomainOrigin () const { return m_Origin + Vector3d::Constant(m_BoundaryWidth * m_Spacing); }
		Vector3d GetDomainLengths() const { return (m_Resolution - Vector3i::Constant(m_BoundaryWidth * 2)).cast<double>() * m_Spacing; }
		double   GetDomainRadius () const { return GetDomainLengths().norm() * .5; }

		bool IsInsideNode  (Vector3i const &node)           const { return m_NodeGrid.IsInside(node, m_BoundaryWidth); }
		bool IsBoundaryNode(Vector3i const &node)           const { return !m_NodeGrid.IsInside(node, m_BoundaryWidth + 1); }
		bool IsInsideCell  (Vector3i const &cell)           const { return m_CellGrid.IsInside(cell, m_BoundaryWidth); }
		bool IsBoundaryCell(Vector3i const &cell)           const { return !IsInsideCell(cell); }
		bool IsInsideFace  (int axis, Vector3i const &face) const { return m_FaceGrids[axis].IsInside(face, m_BoundaryWidth); }
		bool IsBoundaryFace(int axis, Vector3i const &face) const { return face[axis] <= m_BoundaryWidth || face[axis] >= m_Resolution[axis] - m_BoundaryWidth || !IsInsideFace(axis, face); }

		static constexpr int GetNumFacesPerCell() { return 6; }
		static constexpr int GetNumEdgesPerCell() { return 12; }
		static constexpr int GetNumNodesPerCell() { return 8; }
		static constexpr int GetNumNodesPerFace() { return 4; }

		static constexpr int FaceAxisOfCell(int ord) { return ord >> 1; }
		static constexpr int FaceSideOfCell(int ord) { return ord & 1 ? 1: -1; }
		static std::pair<int, Vector3i> FaceOfCell(Vector3i const &cell, int ord) { return { FaceAxisOfCell(ord), cell + Vector3i::Unit(FaceAxisOfCell(ord)) * (ord & 1) }; }
		static Vector3i AdjCellOfFace(int axis, Vector3i const &face, int ord) { return face - Vector3i::Unit(axis) * (ord & 1 ^ 1); }

		static constexpr int EdgeAxisOfCell(int ord) { return ord >> 2; }
		static std::pair<int, Vector3i> EdgeOfCell(Vector3i const &cell, int ord) { return { EdgeAxisOfCell(ord), cell + Vector3i::Unit((EdgeAxisOfCell(ord) + 1) % 3) * (ord & 1) + Vector3i::Unit((EdgeAxisOfCell(ord) + 2) % 3) * (ord >> 1 & 1) }; }

		static Vector3i NodeOfCell(Vector3i const &cell, int ord) { return cell + Vector3i(ord & 1, ord >> 1 & 1, ord >> 2 & 1); }
		static Vector3i NodeOfFace(int axis, Vector3i const &face, int ord) { return face + Vector3i::Unit((axis + 1) % 3) * (ord & 1) + Vector3i::Unit((axis + 2) % 3) * (ord >> 1 & 1); }
		static Vector3i NodeOfEdge(int axis, Vector3i const &edge, int ord) { return edge + Vector3i::Unit(axis) * (ord & 1); }
		
	private:
		int      m_BoundaryWidth;
		double   m_Spacing;
		double   m_InvSpacing;
		Vector3i m_Resolution;
		Vector3d m_Origin;

		Grid                m_NodeGrid;
		Grid                m_CellGrid;
		std::array<Grid, 3> m_FaceGrids;
	};
}
