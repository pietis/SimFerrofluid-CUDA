#pragma once

#include "SGridData.h"

namespace Pivot {
	class TriCuInterp {
	private:
		using WtPoint     = std::pair<Vector3i, double>;
		using GradWtPoint = std::pair<Vector3i, Vector3d>;

	public:
		static std::array<WtPoint, 64> GetWtPoints(Grid const &grid, Vector3d const &pos) {
			Vector3i const lower = grid.CalcLower<3>(pos);
			Array3d  const frac = grid.CalcLowerFrac(pos, lower);
            const std::array<Vector3d, 4> w = {
                ((frac.array() - 1) * (frac.array() - 2) * (frac.array() - 3) / -6).matrix(),
                (frac.array() * (frac.array() - 2) * (frac.array() - 3) / 2).matrix(),
                (frac.array() * (frac.array() - 1) * (frac.array() - 3) / -2).matrix(),
                (frac.array() * (frac.array() - 1) * (frac.array() - 2) / 6).matrix()
            };

            std::array<WtPoint, 64> wtPts;
            for (int k = 0; k < 4; k++)
                for (int j = 0; j < 4; j++)
                    for (int i = 0; i < 4; i++)
                        wtPts[k << 4 | j << 2 | i] = WtPoint(lower + Vector3i(i, j, k), w[i][0] * w[j][1] * w[k][2]);
            return wtPts;
		}

		template <typename Type>
		static Type Interpolate(GridData<Type> const &grData, Vector3d const &pos) {
			Type val = Zero<Type>();
			for (auto const [coord, weight] : GetWtPoints(grData.GetGrid(), pos)) {
				val += grData.At(coord) * weight;
			}
			return val;
		}
	};
}
