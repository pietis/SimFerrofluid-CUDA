#include "Simulation.h"

#include "Advection.h"
#include "CSG.h"
#include "Extrapolation.h"
#include "FiniteDiff.h"
#include "Reinitialization.h"
#include "SISurfaceTension.h"
#include "TriLerp.h"

namespace Pivot {
	Simulation::Simulation(StaggeredGrid const &sgrid) :
		m_SGrid { sgrid },
		m_Collider(m_SGrid),
		m_Pressure(m_SGrid),
		m_Velocity(m_SGrid.GetFaceGrids()),
		m_LevelSet(m_SGrid.GetCellGrid(), std::numeric_limits<double>::infinity()),
		m_Contour(m_SGrid.GetCellGrid()) {
	}

	void Simulation::Describe(YAML::Node &root) const {
		root["Dimension"] = 3;
		root["Radius"] = m_SGrid.GetDomainRadius() * 2;
		// { // Description of contour
		// 	YAML::Node node;
		// 	node["Name"] = "contour";
		// 	node["Animated"] = true;
		// 	node["Indexed"] = true;
		// 	// node["Shader"] = "heatmap";
		// 	node["Primitive"] = "Triangles";
		// 	node["Material"]["Albedo"] = Vector4f(0, 0, 1, 1);
		// 	root["Objects"].push_back(node);
		// }
		{ // Description of collider
			YAML::Node node;
			node["Name"] = "collider";
			node["Primitive"] = "Points";
			node["Material"]["Albedo"] = Vector4f(.5f, .5f, .5f, 1);
			root["Objects"].push_back(node);
		}
		{
			YAML::Node node;
			node["Name"] = "curv";
			node["Animated"] = true;
			node["Indexed"] = true;
			node["Shader"] = "heatmap";
			node["Primitive"] = "Triangles";
			root["Objects"].push_back(node);
		}
		if(m_MagneticEnabled){
			YAML::Node node;
			node["Name"] = "mag_pressure";
			node["Animated"] = true;
			node["Indexed"] = true;
			node["Shader"] = "heatmap";
			node["Primitive"] = "Triangles";
			root["Objects"].push_back(node);
		}
	}

	void Simulation::Export(std::filesystem::path const &dirname, bool initial) const {
		{ // Export the contour
			std::ofstream fout(dirname / "contour.out", std::ios::binary);
			m_Contour.GetMesh().Export(fout);
		}
		if (initial) { // Export the collider
			std::ofstream fout(dirname / "collider.out", std::ios::binary);
			std::uint32_t cnt = 0;
			ForEach(m_LevelSet.GetGrid(), [&](Vector3i const &cell) {
				if (m_SGrid.IsInsideCell(cell) && m_Collider.IsInside(cell)) cnt++;
			});
			IO::Write(fout, cnt);
			ForEach(m_LevelSet.GetGrid(), [&](Vector3i const &cell) {
				if (m_SGrid.IsBoundaryCell(cell) || !m_Collider.IsInside(cell)) return;
				Vector3d const pos = m_LevelSet.GetGrid().PositionOf(cell);
				IO::Write(fout, pos.cast<float>().eval());
			});
			ForEach(m_LevelSet.GetGrid(), [&](Vector3i const &cell) {
				if (m_SGrid.IsBoundaryCell(cell) || !m_Collider.IsInside(cell)) return;
				Vector3d const pos = m_LevelSet.GetGrid().PositionOf(cell);
				Vector3d const n = TriLerp::Interpolate(m_Collider.GetNormal(), pos).normalized();
				IO::Write(fout, n.cast<float>().eval());
			});
		}
		{
			std::ofstream fout(dirname / "curv.out", std::ios::binary);
			IO::Write(fout, static_cast<std::uint32_t>(m_Contour.GetMesh().Positions.size()));
			for (auto const &pos : m_Contour.GetMesh().Positions) {
				IO::Write(fout, pos.cast<float>().eval());
			}
			for (auto const &normal : m_Contour.GetMesh().Normals) {
				IO::Write(fout, normal.cast<float>().eval());
			}
			for (auto const &curv : m_Contour.GetMesh().MeanCurvatures) {
				IO::Write(fout, (float)curv);
			}
			IO::Write(fout, static_cast<std::uint32_t>(m_Contour.GetMesh().Indices.size()));
			IO::Write(fout, m_Contour.GetMesh().Indices);
		}
		if(m_MagneticEnabled){
			std::ofstream fout(dirname / "mag_pressure.out", std::ios::binary);
			IO::Write(fout, static_cast<std::uint32_t>(m_Contour.GetMesh().Positions.size()));
			for (auto const &pos : m_Contour.GetMesh().Positions) {
				IO::Write(fout, pos.cast<float>().eval());
			}
			for (auto const &normal : m_Contour.GetMesh().Normals) {
				IO::Write(fout, normal.cast<float>().eval());
			}
			for (auto const &mag_pressure : m_Magnetic.m_MagneticPressure) {
				IO::Write(fout, (float)mag_pressure);
			}
			IO::Write(fout, static_cast<std::uint32_t>(m_Contour.GetMesh().Indices.size()));
			IO::Write(fout, m_Contour.GetMesh().Indices);
		}
	}

	void Simulation::Save(std::ostream &out) const {
	}

	void Simulation::Load(std::istream &in) {
	}

	void Simulation::Initialize() {
		m_Collider.Finish(m_SGrid);
		CSG::Intersect(m_LevelSet, m_Collider.GetDomainBox());
		
		ReinitializeLevelSet(true);

		if(m_MagneticEnabled){
			m_Magnetic.Solve(m_Contour.GetMesh());
		}
	}

	void Simulation::Advance(double deltaTime) {
		AdvectFields(deltaTime);
		ApplyBodyForces(deltaTime);
		ProjectVelocity(deltaTime);
		if (m_SurfaceTensionEnabled && m_SemiImplicitSTEnabled) {
			ApplySemiImplicitST(deltaTime);
		}
	}

	void Simulation::AdvectFields(double dt) {
		ParallelForEach(m_Velocity.GetGrids(), [&](int axis, Vector3i const &face) {
			m_Velocity[axis][face] *= exp(-8 * dt);
		});
		

		Advection::Solve<2>(m_LevelSet, m_Velocity, dt);
		Advection::Solve<2>(m_Velocity, m_Velocity, dt);

		ReinitializeLevelSet();
		ComputeVolumeError(dt);
	}

	void Simulation::ApplyBodyForces(double dt) {
		if (m_GravityEnabled) {
			ParallelForEach(m_Velocity[1].GetGrid(), [&](Vector3i const &face) {
				m_Velocity[1][face] -= 9.8 * dt;
			});
		}
	}

	void Simulation::ComputeVolumeError(double dt) {
		double x = (m_CurrentVolume - m_InitVolume) / (m_InitVolume);
		m_CumulVolError += x * dt;
		double kp = 0.05 / dt;
		double ki = kp * kp / 16;
		m_VolError = 1 / (x + 1) * (-kp * x - ki * m_CumulVolError) * m_SGrid.GetSpacing();
	}

	void Simulation::ProjectVelocity(double dt) {
		if(m_MagneticEnabled){
			m_Magnetic.Solve(m_Contour.GetMesh());
		}
		m_Pressure.SetPressureJump([&](int axis, Vector3i const &face, double theta)->double {
			double pressure = 0;
			if (m_SurfaceTensionEnabled && !m_SemiImplicitSTEnabled) {
				int index =
					m_Contour.VertexIndexOf(axis, face - Vector3i::Unit(axis));
				if(index >= 0){
					pressure += m_Contour.GetMesh().MeanCurvatures[index] * m_SurfaceTensionCoeff /
								m_LiquidDensity * m_SGrid.GetInvSpacing() * dt;
				}
			}
			if (m_MagneticEnabled) {
				int index =
					m_Contour.VertexIndexOf(axis, face - Vector3i::Unit(axis));
				if(index >= 0){
					pressure += m_Magnetic.m_MagneticPressure[index] /
								m_LiquidDensity * m_SGrid.GetInvSpacing() * dt;
				}
			}
			return pressure;
		});
		m_Pressure.Project(m_Velocity, m_LevelSet, m_Collider, m_SemiImplicitSTEnabled ? 0 : m_VolError);
		Extrapolation::Solve(m_Velocity, 0., 6, [&](int axis, Vector3i const &face) {
			Vector3i const cell0 = StaggeredGrid::AdjCellOfFace(axis, face, 0);
			Vector3i const cell1 = StaggeredGrid::AdjCellOfFace(axis, face, 1);
			return m_Collider.GetFraction()[axis][face] < 1 && (m_LevelSet[cell0] <= 0 || m_LevelSet[cell1] <= 0);
		});
		m_Collider.Enforce(m_Velocity);
	}

	void Simulation::ApplySemiImplicitST(double dt) {
		m_Pressure.SetPressureJump([&](int axis, Vector3i const &face, double theta)->double {
			return 0;
		});
		SISurfaceTension::Solve(m_Velocity, m_LevelSet, m_Collider, m_SurfaceTensionCoeff / m_LiquidDensity, dt);
		m_Pressure.Reproject(m_Velocity, m_LevelSet, m_Collider, m_VolError);
		Extrapolation::Solve(m_Velocity, 0., 6, [&](int axis, Vector3i const &face) {
			Vector3i const cell0 = StaggeredGrid::AdjCellOfFace(axis, face, 0);
			Vector3i const cell1 = StaggeredGrid::AdjCellOfFace(axis, face, 1);
			return m_Collider.GetFraction()[axis][face] < 1 && (m_LevelSet[cell0] <= 0 || m_LevelSet[cell1] <= 0);
		});
		m_Collider.Enforce(m_Velocity);
	}

	void Simulation::ReinitializeLevelSet(bool initial) {
		Extrapolation::Solve(m_LevelSet, 1.5 * m_SGrid.GetSpacing(), 1, [&](Vector3i const &cell) {
			return !m_Collider.IsInside(cell);
		});
		CSG::Except(m_LevelSet, m_Collider.GetAuxLevelSet());
		// FastMarching::Solve(m_LevelSet, 5);
		Reinitialization::Solve(m_LevelSet, 10);

		// auto opLevelSet = m_LevelSet;
		// CSG::Except(opLevelSet, m_Collider.GetAuxLevelSet());
		m_Contour.Generate(m_LevelSet);
		// m_Contour.ComputeVertexInfos();

		m_Contour.ComputeVertexInfosFromLS(m_LevelSet);
		m_Contour.ComputeVolumeFromLS(m_LevelSet);
		// m_Contour.GetMesh().SmoothCurvature(0.5, 10);
		// double cmin=std::numeric_limits<double>::max();
		// double cmax=std::numeric_limits<double>::min();
		// double avg=0;
		// for(int i=0;i<m_Contour.GetMesh().size();i++){
		// 	cmin = (std::min)(cmin, m_Contour.GetMesh().MeanCurvatures[i]);
		// 	cmax = (std::max)(cmax, m_Contour.GetMesh().MeanCurvatures[i]);
		// 	avg += m_Contour.GetMesh().MeanCurvatures[i];
		// }
		// avg /= m_Contour.GetMesh().size();
		// fmt::print("curvature min {:.3e} max {:.3e} avg {:.3e} ", cmin, cmax, avg);

		m_CurrentVolume = m_Contour.GetMesh().TotalVolume;
		if (initial) {
			m_InitVolume = m_CurrentVolume;
		}
		fmt::print("volume {:.3e}/{:.3e} ", m_CurrentVolume, m_InitVolume);
	}
}
