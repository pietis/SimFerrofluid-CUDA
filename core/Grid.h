#pragma once

#include "Common.h"

namespace Pivot {
	class Grid {
	public:
		Grid(double spacing, Vector3i const &size, Vector3d const &origin) :
			m_Spacing { spacing },
			m_InvSpacing { 1 / m_Spacing },
			m_Size { size },
			m_Origin { origin } {
		}

		bool operator==(Grid const &rhs) const { return m_Spacing == rhs.m_Spacing && m_Size == rhs.m_Size && m_Origin == rhs.m_Origin; }

		double   GetSpacing   () const { return m_Spacing; }
		double   GetInvSpacing() const { return m_InvSpacing; }
		Vector3i GetSize      () const { return m_Size; }
		Vector3d GetOrigin    () const { return m_Origin; }

		int GetNumVertices() const { return m_Size.prod(); }

		bool IsInside(Vector3i const &coord, int offset = 0) const { return (coord.array() >= offset).all() && (coord.array() < m_Size.array() - offset).all(); }
		bool IsValid (Vector3i const &coord)                 const { return IsInside(coord, 0); }

		Vector3i Clamp     (Vector3i const &coord) const { return coord.cwiseMax(0).cwiseMin(m_Size - Vector3i::Ones()); }
		int      IndexOf   (Vector3i const &coord) const { return coord.z() + m_Size.z() * (coord.y() + m_Size.y() * coord.x()); }
		Vector3d PositionOf(Vector3i const &coord) const { return m_Origin + coord.cast<double>() * m_Spacing; }
		
		Vector3i CoordOf(int index) const { return { index / m_Size.z() / m_Size.y(), index / m_Size.z() % m_Size.y(), index % m_Size.z() }; }

		template <int Order> Vector3i CalcLower(Vector3d const &pos) const { return ((pos - m_Origin) * m_InvSpacing - Vector3d::Ones() * (Order - 1) / 2).array().floor().cast<int>().matrix(); }

		Array3d CalcLowerFrac(Vector3d const &pos, Vector3i const &lower) const { return (pos - m_Origin - lower.cast<double>() * m_Spacing).array() * m_InvSpacing; }

		static constexpr int GetNumNeighbors() { return 6; }
		static constexpr int NeighborAxisOf(int ord) { return ord >> 1; }
		static constexpr int NeighborSideOf(int ord) { return ord & 1 ? 1 : -1; }
		static Vector3i NeighborOf(Vector3i const &coord, int ord) { return coord + Vector3i::Unit(NeighborAxisOf(ord)) * NeighborSideOf(ord); }

	private:
		double   m_Spacing;
		double   m_InvSpacing;
		Vector3i m_Size;
		Vector3d m_Origin;
	};

	template <typename Func>
		requires std::is_convertible_v<Func, std::function<void(Vector3i const &)>>
	inline void ForEach(Grid const &grid, Func &&func) {
		for (int i = 0; i < grid.GetSize().x(); i++) {
			for (int j = 0; j < grid.GetSize().y(); j++) {
				for (int k = 0; k < grid.GetSize().z(); k++) {
					func(Vector3i(i, j, k));
				}
			}
		}
	}

	template <typename Func>
		requires std::is_convertible_v<Func, std::function<void(Vector3i const &)>>
	inline void ParallelForEach(Grid const &grid, Func &&func) {
		tbb::parallel_for(tbb::blocked_range3d<int>(0, grid.GetSize().x(), 0, grid.GetSize().y(), 0, grid.GetSize().z()), [&](tbb::blocked_range3d<int> const &r) {
			for (int i = r.pages().begin(); i != r.pages().end(); i++) {
				for (int j = r.rows().begin(); j != r.rows().end(); j++) {
					for (int k = r.cols().begin(); k != r.cols().end(); k++) {
						func(Vector3i(i, j, k));
					}
				}
			}
		});
	}

	template <typename Func>
		requires std::is_convertible_v<Func, std::function<void(int, Vector3i const &)>>
	inline void ForEach(std::array<Grid, 3> const &grids, Func &&func) {
		for (int axis = 0; axis < 3; axis++) {
			ForEach(grids[axis], [&](Vector3i const &coord) {
				func(axis, coord);
			});
		}
	}

	template <typename Func>
		requires std::is_convertible_v<Func, std::function<void(int, Vector3i const &)>>
	inline void ParallelForEach(std::array<Grid, 3> const &grids, Func &&func) {
		tbb::parallel_for(0, 3, [&](int axis) {
			ParallelForEach(grids[axis], [&](Vector3i const &coord) {
				func(axis, coord);
			});
		});
	}
}
