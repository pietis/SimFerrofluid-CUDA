#pragma once

#include "Surface.h"

namespace Pivot {
	class SurfaceMesh : public Surface {
	public:
		SurfaceMesh() = default;

		virtual Vector2d ClosestNormalOf (Vector2d const &pos) const override { return Vector2d::Zero(); } // FIXME
		virtual double   SignedDistanceTo(Vector2d const &pos) const override { return 0; } // FIXME

		std::vector<double> const &GetAreas         () const { return m_Areas; }
		std::vector<double> const &GetMeanCurvatures() const { return m_MeanCurvatures; }

		void Clear();
		void Export(std::ostream &out) const;

		void ComputeAreas();
		void ComputeMeanCurvatures();

	public:
		std::vector<Vector2d>      Positions;
		std::vector<Vector2d>      Normals;
		std::vector<std::uint32_t> Indices;
	
	private:
		std::vector<double>        m_Areas;
		std::vector<double>        m_MeanCurvatures;
	};
}
