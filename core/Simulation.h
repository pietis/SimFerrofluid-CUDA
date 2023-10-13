#pragma once

#include "Collider.h"
#include "Contour.h"
#include "Pressure.h"

namespace Pivot {
	class Simulation {
	private:
		friend class SimBuilder;
	
	public:
		enum class Scene { Falling, BigBall, Slope, Droplet, DamBreak, Plane };

	public:
		explicit Simulation(StaggeredGrid const &sgrid);

		void Describe(YAML::Node &root) const;
		void Export(std::filesystem::path const &dirname, bool initial = false) const;
		void Save(std::ostream &out) const;
		void Load(std::istream &in);

		void Initialize();
		void Advance(double deltaTime);

		void AdvectFields(double dt);
		void ApplyBodyForces(double dt);
		void ApplySurfacePressure(double dt);
		void ProjectVelocity(double dt);

		void ReinitializeLevelSet();

		void SetTime(double time) { m_Time = time; }
		auto GetTime() const { return m_Time; }

		double GetCourantTimeStep() const { return m_SGrid.GetSpacing() / m_Velocity.GetMaxAbsComponent(); }
	
	private:
		double m_Time;
		Scene  m_Scene;

		StaggeredGrid          m_SGrid;
		Collider               m_Collider;
		Pressure               m_Pressure;
		SGridData<double>      m_Velocity;
		GridData<double>       m_LevelSet;
		Contour                m_Contour;

		double m_LiquidDensity       = 1e3;
		double m_SurfaceTensionCoeff = 7.28e-2;

		bool m_GravityEnabled        = true;
		bool m_SurfaceTensionEnabled = false;
	};
}
