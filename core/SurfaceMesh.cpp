#include "SurfaceMesh.h"

namespace Pivot {
	void SurfaceMesh::Clear() {
		Positions.clear();
		Normals.clear();
		Indices.clear();
	}

	void SurfaceMesh::Export(std::ostream &out) const {
		IO::Write(out, static_cast<std::uint32_t>(Positions.size()));
		for (auto const &pos : Positions) {
			IO::Write(out, pos.cast<float>().eval());
		}
		// for (auto const curv : m_MeanCurvatures) {
		// 	IO::Write(out, static_cast<float>(std::abs(curv)));
		// }
		IO::Write(out, static_cast<std::uint32_t>(Indices.size()));
		IO::Write(out, Indices);
	}

	void SurfaceMesh::ComputeAreas() {
		m_Areas.resize(Positions.size());
		std::fill(m_Areas.begin(), m_Areas.end(), 0.);
		for (std::size_t i = 0; i < Indices.size(); i += 2) {
			auto const i0 = Indices[i + 0];
			auto const i1 = Indices[i + 1];
			double const faceArea = (Positions[i1] - Positions[i0]).norm();
			m_Areas[i0] += faceArea;
			m_Areas[i1] += faceArea;
		}
		tbb::parallel_for_each(m_Areas.begin(), m_Areas.end(), [&](double &area) { area /= 2; });
	}

	void SurfaceMesh::ComputeMeanCurvatures() {
		ComputeAreas();

		std::vector<Vector2d> inEdge (Positions.size());
		std::vector<Vector2d> outEdge(Positions.size());
		for (std::size_t i = 0; i < Indices.size(); i += 2) {
			auto const i0 = Indices[i + 0];
			auto const i1 = Indices[i + 1];
			Vector2d const vec = Positions[i1] - Positions[i0];
			inEdge[i1] = vec;
			outEdge[i0] = vec;
		}
		m_MeanCurvatures.resize(Positions.size());
		tbb::parallel_for(static_cast<std::size_t>(0), Positions.size(), [&](std::size_t i) {
			double const cosTheta = inEdge[i].dot(outEdge[i]) / (inEdge[i].norm() * outEdge[i].norm());
			int const sign = inEdge[i].x() * outEdge[i].y() - inEdge[i].y() * outEdge[i].x() > 0 ? +1 : -1;
			m_MeanCurvatures[i] = std::acos(cosTheta) * sign / m_Areas[i];
		});
	}
}
