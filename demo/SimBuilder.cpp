#include "SimBuilder.h"

#include "CSG.h"

namespace Pivot {
std::unique_ptr<Simulation> SimBuilder::Build(SimBuildOptions const &options) {
    std::unique_ptr<Simulation> simulation;
    switch (options.Scene) {
    case Simulation::Scene::Falling:
        simulation = BuildFalling(options);
        break;
    case Simulation::Scene::BigBall:
        simulation = BuildBigBall(options);
        break;
    case Simulation::Scene::Slope:
        simulation = BuildSlope(options);
        break;
    case Simulation::Scene::Droplet:
        simulation = BuildDroplet(options);
        break;
    case Simulation::Scene::DamBreak:
        simulation = BuildDambreak(options);
        break;
    case Simulation::Scene::Plane:
        simulation = BuildPlane(options);
        break;
    case Simulation::Scene::Dipole:
        simulation = BuildDipole(options);
        break;
    }
    simulation->m_Scene = options.Scene;
    return simulation;
}

std::unique_ptr<Simulation>
SimBuilder::BuildFalling(SimBuildOptions const &options) {
    constexpr double length = 1.;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i(1, 1, 1) * scale);
    auto sim = std::make_unique<Simulation>(sgrid);
    CSG::Union(sim->m_LevelSet, ImplicitPlane(-Vector3d::Unit(1) * length * .15,
                                              Vector3d::Unit(1)));
    CSG::Union(sim->m_LevelSet,
               ImplicitSphere(Vector3d::Unit(1) * length * .05, length * .1));
    double const vel = -10 * length;
    ParallelForEach(sim->m_Velocity[1].GetGrid(), [&](Vector3i const &face) {
        Vector3d const pos = sim->m_Velocity[1].GetGrid().PositionOf(face);
        if (pos.y() > 0.) {
            sim->m_Velocity[1][face] = vel;
        }
    });
    return sim;
}

std::unique_ptr<Simulation>
SimBuilder::BuildBigBall(SimBuildOptions const &options) {
    constexpr double length = 1.;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i(1, 1, 1) * scale);
    auto sim = std::make_unique<Simulation>(sgrid);
    CSG::Union(sim->m_LevelSet, ImplicitSphere(Vector3d::Zero(), length * .33));
    return sim;
}

std::unique_ptr<Simulation>
SimBuilder::BuildSlope(SimBuildOptions const &options) {
    constexpr double length = 1.;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i(1, 1, 1) * scale);
    auto sim = std::make_unique<Simulation>(sgrid);
    CSG::Union(sim->m_LevelSet, ImplicitSphere(Vector3d::Zero(), length * .33));
    CSG::Union(sim->m_Collider.LevelSet,
               ImplicitPlane(Vector3d(-2, -1, 0) * length * .25,
                             Vector3d(1, 4, 0).normalized()));
    return sim;
}

std::unique_ptr<Simulation>
SimBuilder::BuildDroplet(SimBuildOptions const &options) {
    constexpr double length = .1;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i(1, 1, 1) * scale);
    auto sim = std::make_unique<Simulation>(sgrid);
    sim->m_GravityEnabled = false;
    sim->m_SurfaceTensionEnabled = true;
    // sim->m_SemiImplicitSTEnabled = true;
    CSG::Union(
        sim->m_LevelSet,
        ImplicitEllipsoid(Vector3d::Zero(), Vector3d(.4, .25, .25) * length));
        // ImplicitEllipsoid(Vector3d::Zero(), Vector3d(.25, .25, .25) * length));
    return sim;
}

std::unique_ptr<Simulation>
SimBuilder::BuildDambreak(SimBuildOptions const &options) {
    constexpr double length = 1.;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i(4, 3, 4) * scale / 4);
    auto sim = std::make_unique<Simulation>(sgrid);
    CSG::Union(
        sim->m_LevelSet,
        ImplicitPlane(sgrid.GetDomainOrigin() + Vector3d::Unit(1) * length * .1,
                      Vector3d::Unit(1)));
    CSG::Union(sim->m_LevelSet, ImplicitBox(sgrid.GetDomainOrigin(),
                                            Vector3d(.3, .2, .6) * length));
    CSG::Union(sim->m_LevelSet,
               ImplicitBox(sgrid.GetDomainOrigin() + Vector3d(.7, .0, .4),
                           Vector3d(.3, .2, .6) * length));
    CSG::Union(sim->m_LevelSet,
               ImplicitBox(sgrid.GetDomainOrigin() + Vector3d(.4, .0, .0),
                           Vector3d(.6, .2, .3) * length));
    CSG::Union(sim->m_LevelSet,
               ImplicitBox(sgrid.GetDomainOrigin() + Vector3d(.0, .0, .7),
                           Vector3d(.6, .2, .3) * length));
    return sim;
}

std::unique_ptr<Simulation>
SimBuilder::BuildPlane(SimBuildOptions const &options) {
    constexpr double length = .12;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i(4, 3, 4) * scale / 4);
    auto sim = std::make_unique<Simulation>(sgrid);
    sim->m_SurfaceTensionEnabled = true;
    sim->m_MagneticEnabled = true;
    sim->m_Damping = 8;
    Vector3d Hext = Vector3d(0, 6e4, 0);
    sim->m_FieldApplied = [Hext](const Vector3d& pos) -> Vector3d{
        return Hext;
    };
    CSG::Union(sim->m_LevelSet,
               ImplicitPlane(sgrid.GetDomainOrigin() +
                                 Vector3d::Unit(1) * length * .2,
                             Vector3d::Unit(1)));
    return sim;
}

std::unique_ptr<Simulation>
SimBuilder::BuildDipole(SimBuildOptions const &options) {
    constexpr double length = .12;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i::Ones() * scale);
    auto sim = std::make_unique<Simulation>(sgrid);
    sim->m_SurfaceTensionEnabled = true;
    sim->m_MagneticEnabled = true;
    sim->m_Damping = 8;
    double HextFactor = 50;
    Vector3d DPOrient = Vector3d(0, 1, 0);
    Vector3d DPPos = Vector3d(0, - length * 1.2, 0);
    sim->m_FieldApplied = [HextFactor, DPOrient, DPPos](const Vector3d& pos) -> Vector3d{
        Vector3d r = pos - DPPos;
        double rnorm = r.norm();
        Vector3d rUnit = r.normalized();
        double cosTheta = r.dot(DPOrient) / rnorm;
        Vector3d Hr = HextFactor * rUnit * 2 * cosTheta / (rnorm * rnorm * rnorm);
        if( (1 - cosTheta) < 1e-6 ){
            return Hr;
        }else{
            double sinTheta = sqrt(1 - cosTheta * cosTheta);
            Vector3d thetaUnit = DPOrient.dot(rUnit) * rUnit - DPOrient;
            Vector3d Htheta = HextFactor * thetaUnit * sinTheta / (rnorm * rnorm * rnorm);
            return Hr + Htheta;
        }
    };
    CSG::Union(sim->m_LevelSet,
               ImplicitPlane(sgrid.GetDomainOrigin() +
                                 Vector3d::Unit(1) * length * .10,
                             Vector3d::Unit(1)));
    return sim;
}
} // namespace Pivot
