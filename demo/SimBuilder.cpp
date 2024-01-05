#include "SimBuilder.h"

#include "CSG.h"

#include "Reinitialization.h"
#include "Contour.h"

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
    case Simulation::Scene::Pattern:
        simulation = BuildPattern(options);
        break;
    case Simulation::Scene::MagSphere:
        simulation = BuildMagSphere(options);
        break;
    case Simulation::Scene::Tmp:
        simulation = BuildTmp(options);
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
    sim->m_MagneticEnabled = options.EnableMag;
    sim->m_Damping = 8;
    Vector3d Hext = Vector3d(0, 6e4, 0);
    if(options.EnableMag){
        sim->m_FieldApplied = [Hext](const Vector3d& pos, double time) -> Vector3d{
            return Hext;
        };
    }
    CSG::Union(sim->m_LevelSet,
               ImplicitPlane(sgrid.GetDomainOrigin() +
                                 Vector3d::Unit(1) * length * .2,
                             Vector3d::Unit(1)));
    return sim;
}

std::unique_ptr<Simulation>
SimBuilder::BuildDipole(SimBuildOptions const &options) {
    constexpr double length = .10;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i(4, 2, 4) * scale / 4);
    auto sim = std::make_unique<Simulation>(sgrid);
    sim->m_SurfaceTensionEnabled = true;
    sim->m_MagneticEnabled = options.EnableMag;
    sim->m_Damping = 8;
    sim->m_Magnetic.SetChi(0.45);
    double HextFactor = 14;
    Vector3d DPOrient = Vector3d(0, 1, 0);
    Vector3d DPPos = Vector3d(0, - length * 0.8, 0);
    if(options.EnableMag){
        sim->m_FieldApplied = [HextFactor, DPOrient, DPPos, length](const Vector3d& pos, double time) -> Vector3d{
            double timeFactor = 0;
            time = (std::min)(0.12, time);
            timeFactor = time / 0.12;
            Vector3d r = pos - DPPos;
            // Vector3d r = pos - dppos;
            double rnorm = r.norm();
            Vector3d rUnit = r.normalized();
            double cosTheta = r.dot(DPOrient) / rnorm;
            Vector3d Hr = timeFactor * HextFactor * rUnit * 2 * cosTheta / (rnorm * rnorm * rnorm);
            if( (1 - cosTheta) < 1e-6 ){
                return Hr;
            }else{
                double sinTheta = sqrt(1 - cosTheta * cosTheta);
                Vector3d thetaUnit = DPOrient.dot(rUnit) * rUnit - DPOrient;
                Vector3d Htheta = timeFactor * HextFactor * thetaUnit * sinTheta / (rnorm * rnorm * rnorm);
                return Hr + Htheta;
            }
        };
    }
    CSG::Union(sim->m_LevelSet, ImplicitDisk(sgrid.GetDomainOrigin() + Vector3d(1, 0, 1) * length * 0.5 + Vector3d(0, 1, 0) * length * 0.025,
                                    Vector3d(0, 1, 0), 0.4 * length, 0.025 * length));
    return sim;
}

std::unique_ptr<Simulation>
SimBuilder::BuildTmp(SimBuildOptions const &options) {
    constexpr double length = .08;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i(1, 1, 1) * scale);
    auto sim = std::make_unique<Simulation>(sgrid);
    sim->m_SurfaceTensionEnabled = true;
    sim->m_MagneticEnabled = options.EnableMag;
    sim->m_Magnetic.SetChi(0.8);
    sim->m_Damping = 16;
    Vector3d Hext = Vector3d(0, 2e4, 0);
    Vector3d DPOrient = Vector3d(0, 1, 0);
    Vector3d DPPos = Vector3d(0, - length * 0.8, 0);
    if(options.EnableMag){
        sim->m_FieldApplied = [Hext](const Vector3d& pos, double time) -> Vector3d{
            return Hext;
        };

        GridData<double> tmpLevelSet(sgrid.GetCellGrid(), std::numeric_limits<double>::infinity());
        CSG::Union(tmpLevelSet, ImplicitSphere((0.2 * length + sgrid.GetDomainOrigin()(1)) * Vector3d::Unit(1), length * .3));
        CSG::Except(tmpLevelSet, ImplicitPlane(sgrid.GetDomainOrigin()(1)*Vector3d::Unit(1), Vector3d::Unit(1)));
        FastMarching::Solve(tmpLevelSet, -1);
        Contour contour(sgrid.GetCellGrid());
        contour.Generate(tmpLevelSet);
        contour.ComputeVertexInfosFromLS(tmpLevelSet);
        sim->m_MagneticObject = std::make_shared<Magnetic>();
        sim->m_MagneticObject->SetChi(5000);
        sim->m_MagneticObject->SetIteration(100);
        sim->m_MagneticObject->AssignMesh(contour.GetMesh());
    }
    
    CSG::Union(sim->m_Collider.LevelSet, ImplicitSphere((0.2 * length + sgrid.GetDomainOrigin()(1)) * Vector3d::Unit(1), length * .3));
    CSG::Union(sim->m_LevelSet, ImplicitPlane(sgrid.GetDomainOrigin() +
                                    Vector3d::Unit(1) * length * .14,
                                    Vector3d::Unit(1)));
    return sim;
}

std::unique_ptr<Simulation>
SimBuilder::BuildMagSphere(SimBuildOptions const &options) {
    constexpr double length = .08;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i(1, 1, 1) * scale);
    auto sim = std::make_unique<Simulation>(sgrid);
    sim->m_SurfaceTensionEnabled = true;
    sim->m_MagneticEnabled = options.EnableMag;
    sim->m_Magnetic.SetChi(0.8);
    sim->m_Damping = 16;
    Vector3d Hext = Vector3d(0, 2e4, 0);
    Vector3d DPOrient = Vector3d(0, 1, 0);
    Vector3d DPPos = Vector3d(0, - length * 0.8, 0);
    if(options.EnableMag){
        sim->m_FieldApplied = [Hext](const Vector3d& pos, double time) -> Vector3d{
            return Hext;
        };

        GridData<double> tmpLevelSet(sgrid.GetCellGrid(), std::numeric_limits<double>::infinity());
        CSG::Union(tmpLevelSet, ImplicitSphere((0.2 * length + sgrid.GetDomainOrigin()(1)) * Vector3d::Unit(1), length * .3));
        CSG::Except(tmpLevelSet, ImplicitPlane(sgrid.GetDomainOrigin()(1)*Vector3d::Unit(1), Vector3d::Unit(1)));
        FastMarching::Solve(tmpLevelSet, -1);
        Contour contour(sgrid.GetCellGrid());
        contour.Generate(tmpLevelSet);
        contour.ComputeVertexInfosFromLS(tmpLevelSet);
        sim->m_MagneticObject = std::make_shared<Magnetic>();
        sim->m_MagneticObject->SetChi(5000);
        sim->m_MagneticObject->SetIteration(100);
        sim->m_MagneticObject->AssignMesh(contour.GetMesh());
    }
    
    CSG::Union(sim->m_Collider.LevelSet, ImplicitSphere((0.2 * length + sgrid.GetDomainOrigin()(1)) * Vector3d::Unit(1), length * .3));
    CSG::Union(sim->m_LevelSet, ImplicitPlane(sgrid.GetDomainOrigin() +
                                    Vector3d::Unit(1) * length * .14,
                                    Vector3d::Unit(1)));
    return sim;
}

std::unique_ptr<Simulation>
SimBuilder::BuildPattern(SimBuildOptions const &options) {
    constexpr double length = .24;
    constexpr int bw = 2;
    int const scale = options.Scale < 0 ? 64 : options.Scale;
    int thickness = ceil(0.02 * scale) + bw * 2 ;
    // int thickness = 4 + bw * 2 ;
    StaggeredGrid sgrid(2, length / (scale - bw * 2),
                        Vector3i(1, 0, 1) * scale + Vector3i(0, thickness, 0));
    auto sim = std::make_unique<Simulation>(sgrid);
    sim->m_SurfaceTensionEnabled = true;
    sim->m_GravityEnabled = false;
    sim->m_MagneticEnabled = options.EnableMag;
    sim->m_Damping = 8;
    sim->m_Magnetic.SetChi(0.8);
    Vector3d Hext0 = Vector3d(0, 4e4, 0);
    Vector3d Hext1 = Vector3d(0, 5e4, 0);

    if(options.EnableMag){
        sim->m_FieldApplied = [Hext0, Hext1](const Vector3d& pos, double time) -> Vector3d{
            double timeFactor = 0;
            time = (std::min)(0.04, time);
            timeFactor = time / 0.04;
            return Hext0 + (Hext1 - Hext0) * timeFactor;
        };
    }
    CSG::Union(sim->m_LevelSet, ImplicitDisk(sgrid.GetDomainOrigin() + Vector3d(1, 0, 1) * length * 0.5 + Vector3d(0, 1, 0) * length * 0.005,
                                    Vector3d(0, 1, 0), 0.32 * length, 0.005 * length));
    return sim;
}
} // namespace Pivot
