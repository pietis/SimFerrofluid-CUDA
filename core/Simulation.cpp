#include "Simulation.h"

#include "Advection.h"
#include "BiLerp.h"
#include "CSG.h"
#include "Extrapolation.h"
#include "FiniteDiff.h"
#include "Reinitialization.h"

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
		root["Dimension"] = 2;
		root["Radius"] = m_SGrid.GetDomainRadius();
		{ // Description of contour
			YAML::Node node;
			node["Name"] = "contour";
			node["Animated"] = true;
			node["Indexed"] = true;
			// node["Shader"] = "heatmap";
			node["Primitive"] = "Lines";
			node["Material"]["Albedo"] = Vector4f(1, 0, 0, 1);
			root["Objects"].push_back(node);
		}
		{ // Description of velocity
			YAML::Node node;
			node["Name"] = "velocity";
			node["Animated"] = true;
			node["Primitive"] = "Lines";
			node["Shader"] = "heatmap";
			root["Objects"].push_back(node);
		}
		{ // Description of collider
			YAML::Node node;
			node["Name"] = "collider";
			node["Primitive"] = "Points";
			node["Material"]["Albedo"] = Vector4f(.5f, .5f, .5f, 1);
			root["Objects"].push_back(node);
		}
		{ // Description of grid
			YAML::Node node;
			node["Name"] = "grid";
			node["Primitive"] = "Lines";
			node["Material"]["Albedo"] = Vector4f(.2f, .2f, .2f, 1);
			root["Objects"].push_back(node);
		}	}

	void Simulation::Export(std::filesystem::path const &dirname, bool initial) const {
		{ // Export the contour
			std::ofstream fout(dirname / "contour.out", std::ios::binary);
			m_Contour.GetMesh().Export(fout);
		}
		{ // Export the velocity
			std::ofstream fout(dirname / "velocity.out", std::ios::binary);
			IO::Write(fout, static_cast<std::uint32_t>(2 * m_SGrid.GetNumCells()));
			ForEach(m_LevelSet.GetGrid(), [&](Vector2i const &cell) {
				Vector2d const pos0 = m_LevelSet.GetGrid().PositionOf(cell);
				Vector2d const vel = BiLerp::Interpolate(m_Velocity, pos0);
				Vector2d const pos1 = pos0 + vel.normalized() * m_SGrid.GetSpacing() * std::numbers::sqrt2 * .5;
				IO::Write(fout, pos0.cast<float>().eval());
				IO::Write(fout, pos1.cast<float>().eval());
			});
			ForEach(m_LevelSet.GetGrid(), [&](Vector2i const &cell) {
				Vector2d const pos = m_LevelSet.GetGrid().PositionOf(cell);
				auto const normVel = static_cast<float>(BiLerp::Interpolate(m_Velocity, pos).norm());
				IO::Write(fout, normVel);
				IO::Write(fout, normVel);
			});
		}
		if (initial) { // Export the collider
			std::ofstream fout(dirname / "collider.out", std::ios::binary);
			std::uint32_t cnt = 0;
			ForEach(m_LevelSet.GetGrid(), [&](Vector2i const &cell) {
				if (m_Collider.IsInside(cell)) cnt++;
			});
			IO::Write(fout, cnt);
			ForEach(m_LevelSet.GetGrid(), [&](Vector2i const &cell) {
				if (!m_Collider.IsInside(cell)) return;
				Vector2d const pos = m_LevelSet.GetGrid().PositionOf(cell);
				IO::Write(fout, pos.cast<float>().eval());
			});
		}
		if (initial) { // Export the grid
			std::ofstream fout(dirname / "grid.out", std::ios::binary);
			auto const &grid = m_SGrid.GetNodeGrid();
			auto const cnt = static_cast<std::uint32_t>(grid.GetSize().sum() * 2);
			IO::Write(fout, cnt);
			for (int i = 0; i < grid.GetSize().x(); i++) {
				Vector2d const pos0 = grid.PositionOf(Vector2i(i, 0));
				Vector2d const pos1 = grid.PositionOf(Vector2i(i, grid.GetSize().y() - 1));
				IO::Write(fout, pos0.cast<float>().eval());
				IO::Write(fout, pos1.cast<float>().eval());
			}
			for (int i = 0; i < grid.GetSize().y(); i++) {
				Vector2d const pos0 = grid.PositionOf(Vector2i(0, i));
				Vector2d const pos1 = grid.PositionOf(Vector2i(grid.GetSize().x() - 1, i));
				IO::Write(fout, pos0.cast<float>().eval());
				IO::Write(fout, pos1.cast<float>().eval());
			}
		}
	}

	void Simulation::Save(std::ostream &out) const {
	}

	void Simulation::Load(std::istream &in) {
	}

	void Simulation::Initialize() {
		m_Collider.Finish(m_SGrid);
		CSG::Intersect(m_LevelSet, m_Collider.GetDomainBox());
		
		ReinitializeLevelSet();
	}

	void Simulation::Advance(double deltaTime) {
		AdvectFields(deltaTime);
		ApplyBodyForces(deltaTime);
		ProjectVelocity(deltaTime);
	}

	void Simulation::AdvectFields(double dt) {
		Advection::Solve<2>(m_LevelSet, m_Velocity, dt);
		Advection::Solve<2>(m_Velocity, m_Velocity, dt);

		ReinitializeLevelSet();
	}

	void Simulation::ApplyBodyForces(double dt) {
		if (m_GravityEnabled) {
			ParallelForEach(m_Velocity[1].GetGrid(), [&](Vector2i const &face) {
				m_Velocity[1][face] -= 9.8 * dt;
			});
		}
	}

	void Simulation::ProjectVelocity(double dt) {
		if (m_SurfaceTensionEnabled) {
			m_Pressure.SetPressureJump([&](int axis, Vector2i const &face, double theta)->double {
				Vector2i const cell0 = StaggeredGrid::AdjCellOfFace(axis, face, 0);
				Vector2i const cell1 = StaggeredGrid::AdjCellOfFace(axis, face, 1);
				double const kappa0 = FiniteDiff::CalcCurvature(m_LevelSet, cell0);
				double const kappa1 = FiniteDiff::CalcCurvature(m_LevelSet, cell1);
				double const kappa = (1 - theta) * kappa0 + theta * kappa1;
				return kappa * m_SurfaceTensionCoeff / m_LiquidDensity * m_SGrid.GetInvSpacing() * dt;
			});
		}
		m_Pressure.Project(m_Velocity, m_LevelSet, m_Collider);
		Extrapolation::Solve(m_Velocity, 0., 6, [&](int axis, Vector2i const &face) {
			Vector2i const cell0 = StaggeredGrid::AdjCellOfFace(axis, face, 0);
			Vector2i const cell1 = StaggeredGrid::AdjCellOfFace(axis, face, 1);
			return m_Collider.GetFraction()[axis][face] < 1 && (m_LevelSet[cell0] <= 0 || m_LevelSet[cell1] <= 0);
		});
		m_Collider.Enforce(m_Velocity);
	}

	void Simulation::ReinitializeLevelSet() {
		Extrapolation::Solve(m_LevelSet, 1.5 * m_SGrid.GetSpacing(), 1, [&](Vector2i const &cell) {
			return !m_Collider.IsInside(cell);
		});
		Reinitialization::Solve(m_LevelSet, 5);

		auto opLevelSet = m_LevelSet;
		CSG::Except(opLevelSet, m_Collider.GetAuxLevelSet());
		m_Contour.Generate(opLevelSet);
		// m_Contour.ComputeVertexInfos();
	}
}
