#pragma once

#include "SGridData.h"

namespace Pivot {
	class TriLerp {
	private:
		using WtPoint     = std::pair<Vector3i, double>;
		using GradWtPoint = std::pair<Vector3i, Vector3d>;

	public:
		static std::array<Vector3i, 8> GetPoints(Grid const &grid, Vector3d const &pos) {
			Vector3i const lower = grid.CalcLower<1>(pos);
			return {
				lower + Vector3i(0, 0, 0), lower + Vector3i(1, 0, 0),
				lower + Vector3i(0, 1, 0), lower + Vector3i(1, 1, 0),
				lower + Vector3i(0, 0, 1), lower + Vector3i(1, 0, 1),
				lower + Vector3i(0, 1, 1), lower + Vector3i(1, 1, 1),
			};
		}

		static std::array<WtPoint, 8> GetWtPoints(Grid const &grid, Vector3d const &pos) {
			Vector3i const lower = grid.CalcLower<1>(pos);
			Array3d  const frac = grid.CalcLowerFrac(pos, lower);
			std::array<Array3d, 2> const w = { 1. - frac, frac };
			return {
				WtPoint(lower + Vector3i(0, 0, 0), w[0][0] * w[0][1] * w[0][2]),
				WtPoint(lower + Vector3i(1, 0, 0), w[1][0] * w[0][1] * w[0][2]),
				WtPoint(lower + Vector3i(0, 1, 0), w[0][0] * w[1][1] * w[0][2]),
				WtPoint(lower + Vector3i(1, 1, 0), w[1][0] * w[1][1] * w[0][2]),
				WtPoint(lower + Vector3i(0, 0, 1), w[0][0] * w[0][1] * w[1][2]),
				WtPoint(lower + Vector3i(1, 0, 1), w[1][0] * w[0][1] * w[1][2]),
				WtPoint(lower + Vector3i(0, 1, 1), w[0][0] * w[1][1] * w[1][2]),
				WtPoint(lower + Vector3i(1, 1, 1), w[1][0] * w[1][1] * w[1][2]),
			};
		}

		static std::array<GradWtPoint, 8> GetGradWtPoints(Grid const &grid, Vector3d const &pos) {
			Vector3i const lower = grid.CalcLower<1>(pos);
			Array3d  const frac = grid.CalcLowerFrac(pos, lower);
			std::array<Array3d, 2> const w = { 1. - frac, frac };
			return {
				GradWtPoint(lower + Vector3i(0, 0, 0), Vector3d(-w[0][1] * w[0][2], -w[0][0] * w[0][2], -w[0][0] * w[0][1]) * grid.GetInvSpacing()),
				GradWtPoint(lower + Vector3i(1, 0, 0), Vector3d( w[0][1] * w[0][2], -w[1][0] * w[0][2], -w[1][0] * w[0][1]) * grid.GetInvSpacing()),
				GradWtPoint(lower + Vector3i(0, 1, 0), Vector3d(-w[1][1] * w[0][2],  w[0][0] * w[0][2], -w[0][0] * w[1][1]) * grid.GetInvSpacing()),
				GradWtPoint(lower + Vector3i(1, 1, 0), Vector3d( w[1][1] * w[0][2],  w[1][0] * w[0][2], -w[1][0] * w[1][1]) * grid.GetInvSpacing()),
				GradWtPoint(lower + Vector3i(0, 0, 1), Vector3d(-w[0][1] * w[1][2], -w[0][0] * w[1][2],  w[0][0] * w[0][1]) * grid.GetInvSpacing()),
				GradWtPoint(lower + Vector3i(1, 0, 1), Vector3d( w[0][1] * w[1][2], -w[1][0] * w[1][2],  w[1][0] * w[0][1]) * grid.GetInvSpacing()),
				GradWtPoint(lower + Vector3i(0, 1, 1), Vector3d(-w[1][1] * w[1][2],  w[0][0] * w[1][2],  w[0][0] * w[1][1]) * grid.GetInvSpacing()),
				GradWtPoint(lower + Vector3i(1, 1, 1), Vector3d( w[1][1] * w[1][2],  w[1][0] * w[1][2],  w[1][0] * w[1][1]) * grid.GetInvSpacing()),
			};
		}

		template <typename Type>
		static Type Interpolate(GridData<Type> const &grData, Vector3d const &pos) {
			Type val = Zero<Type>();
			for (auto const [coord, weight] : GetWtPoints(grData.GetGrid(), pos)) {
				val += grData.At(coord) * weight;
			}
			return val;
		}

		template <typename Type> 
			requires (std::is_arithmetic_v<Type>)
		static Vector<Type, 3> Interpolate(SGridData<Type> const &sgrData, Vector3d const &pos) {
			return { Interpolate(sgrData[0], pos), Interpolate(sgrData[1], pos), Interpolate(sgrData[2], pos) };
		}
	};
}
