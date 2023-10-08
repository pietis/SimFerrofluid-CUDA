#include "SimBuilder.h"

#include "CSG.h"

namespace Pivot {
	std::unique_ptr<Simulation> SimBuilder::Build(SimBuildOptions const &options) {
		std::unique_ptr<Simulation> simulation;
		switch (options.Scene) {
			case Simulation::Scene::Falling: simulation = BuildFalling(options); break;
			case Simulation::Scene::BigBall: simulation = BuildBigBall(options); break;
			case Simulation::Scene::Slope  : simulation = BuildSlope  (options); break;
			case Simulation::Scene::Droplet: simulation = BuildDroplet(options); break;
		}
		simulation->m_Scene = options.Scene;
		return simulation;
	}

	std::unique_ptr<Simulation> SimBuilder::BuildFalling(SimBuildOptions const &options) {
		constexpr double length = 1.;
		constexpr int bw = 2;
		int const scale = options.Scale < 0 ? 128 : options.Scale;
		StaggeredGrid sgrid(2, length / (scale - bw * 2), Vector2i(1, 1) * scale);
		auto sim = std::make_unique<Simulation>(sgrid);
		CSG::Union(sim->m_LevelSet, ImplicitPlane (-Vector2d::Unit(1) * length * .15, Vector2d::Unit(1)));
		CSG::Union(sim->m_LevelSet, ImplicitSphere( Vector2d::Unit(1) * length * .05, length * .1));
		return sim;
	}

	std::unique_ptr<Simulation> SimBuilder::BuildBigBall(SimBuildOptions const &options) {
		constexpr double length = 1.;
		constexpr int bw = 2;
		int const scale = options.Scale < 0 ? 128 : options.Scale;
		StaggeredGrid sgrid(2, length / (scale - bw * 2), Vector2i(1, 1) * scale);
		auto sim = std::make_unique<Simulation>(sgrid);
		CSG::Union(sim->m_LevelSet, ImplicitSphere(Vector2d::Zero(), length * .25));
		return sim;
	}

	std::unique_ptr<Simulation> SimBuilder::BuildSlope(SimBuildOptions const &options) {
		constexpr double length = 1.;
		constexpr int bw = 2;
		int const scale = options.Scale < 0 ? 128 : options.Scale;
		StaggeredGrid sgrid(2, length / (scale - bw * 2), Vector2i(1, 1) * scale);
		auto sim = std::make_unique<Simulation>(sgrid);
		CSG::Union(sim->m_LevelSet, ImplicitSphere(Vector2d::Zero(), length * .25));
		CSG::Union(sim->m_Collider.LevelSet, ImplicitPlane(Vector2d(-2, -1) * length * .25, Vector2d(1, 4).normalized()));
		return sim;
	}

	std::unique_ptr<Simulation> SimBuilder::BuildDroplet(SimBuildOptions const &options) {
		constexpr double length = .1;
		constexpr int bw = 2;
		int const scale = options.Scale < 0 ? 128 : options.Scale;
		StaggeredGrid sgrid(2, length / (scale - bw * 2), Vector2i(1, 1) * scale);
		auto sim = std::make_unique<Simulation>(sgrid);
		sim->m_GravityEnabled        = false;
		sim->m_SurfaceTensionEnabled = true;
		CSG::Union(sim->m_LevelSet, ImplicitEllipsoid(Vector2d::Zero(), Vector2d(.4, .25) * length));
		return sim;
	}
}
