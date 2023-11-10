#pragma once

#include "Surface.h"

namespace Pivot {
	class SurfaceMesh : public Surface {
	public:
		SurfaceMesh() = default;

		virtual Vector3d ClosestNormalOf (Vector3d const &pos) const override { return Vector3d::Zero(); } // FIXME
		virtual double   SignedDistanceTo(Vector3d const &pos) const override { return 0; } // FIXME

		int size() { return Positions.size(); }

		void Clear();
		void Export(std::ostream &out) const;

		void ComputeNormals();
		void ComputeAreas();
		void ComputeMeanCurvatures();
		void SmoothCurvature(double lambda, int iteration);

	public:
		std::vector<Vector3d>      Positions;
		std::vector<Vector3d>      Normals;
		std::vector<std::uint32_t> Indices;
	
		std::vector<double>        Areas;
		std::vector<double>        MeanCurvatures;
		double TotalArea;
		double TotalVolume;
	};
}
