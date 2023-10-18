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
		{ // Description of contour
			YAML::Node node;
			node["Name"] = "contour";
			node["Animated"] = true;
			node["Indexed"] = true;
			// node["Shader"] = "heatmap";
			node["Primitive"] = "Triangles";
			node["Material"]["Albedo"] = Vector4f(0, 0, 1, 1);
			root["Objects"].push_back(node);
		}
		{ // Description of collider
			YAML::Node node;
			node["Name"] = "collider";
			node["Primitive"] = "Points";
			node["Material"]["Albedo"] = Vector4f(.5f, .5f, .5f, 1);
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
	}

	void Simulation::Save(std::ostream &out) const {
	}

	void Simulation::Load(std::istream &in) {
	}

	void Simulation::Initialize() {
		m_Collider.Finish(m_SGrid);
		CSG::Intersect(m_LevelSet, m_Collider.GetDomainBox());
		
		ReinitializeLevelSet(true);
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
		double kp = 0.1 / dt;
		double ki = kp * kp / 16;
		m_VolError = 1 / (x + 1) * (-kp * x - ki * m_CumulVolError) * m_SGrid.GetSpacing();
	}

	void Simulation::ProjectVelocity(double dt) {
		if (m_SurfaceTensionEnabled && !m_SemiImplicitSTEnabled) {
			m_Pressure.SetPressureJump([&](int axis, Vector3i const &face, double theta)->double {
				Vector3i const cell0 = StaggeredGrid::AdjCellOfFace(axis, face, 0);
				Vector3i const cell1 = StaggeredGrid::AdjCellOfFace(axis, face, 1);
				double const kappa0 = FiniteDiff::CalcCurvature(m_LevelSet, cell0);
				double const kappa1 = FiniteDiff::CalcCurvature(m_LevelSet, cell1);
				double const kappa = (1 - theta) * kappa0 + theta * kappa1;
				return kappa * m_SurfaceTensionCoeff / m_LiquidDensity * m_SGrid.GetInvSpacing() * dt;
			});
		}
		m_Pressure.Project(m_Velocity, m_LevelSet, m_Collider, m_SemiImplicitSTEnabled ? 0 : m_VolError);
		Extrapolation::Solve(m_Velocity, 0., 6, [&](int axis, Vector3i const &face) {
			Vector3i const cell0 = StaggeredGrid::AdjCellOfFace(axis, face, 0);
			Vector3i const cell1 = StaggeredGrid::AdjCellOfFace(axis, face, 1);
			return m_Collider.GetFraction()[axis][face] < 1 && (m_LevelSet[cell0] <= 0 || m_LevelSet[cell1] <= 0);
		});
		m_Collider.Enforce(m_Velocity);
	}

	void Simulation::ApplySemiImplicitST(double dt) {
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
		Reinitialization::Solve(m_LevelSet, 5);

		auto opLevelSet = m_LevelSet;
		CSG::Except(opLevelSet, m_Collider.GetAuxLevelSet());
		m_Contour.Generate(opLevelSet);
		// m_Contour.ComputeVertexInfos();

		m_Contour.ComputeVertexInfosFromLS(opLevelSet);
		m_Contour.ComputeVolumeFromLS(opLevelSet);

		m_CurrentVolume = m_Contour.GetMesh().TotalVolume;
		if (initial) {
			m_InitVolume = m_CurrentVolume;
		}
		fmt::print("volume {:.3e}/{:.3e} ", m_CurrentVolume, m_InitVolume);
	}
}
