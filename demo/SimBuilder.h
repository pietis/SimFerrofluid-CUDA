#pragma once

#include "Simulation.h"

namespace Pivot {
struct SimBuildOptions {
    Simulation::Scene Scene;
    int Scale;
    bool EnableMag;
  Magnetic::SolverType MagSolver;
};

class SimBuilder {
  public:
    static std::unique_ptr<Simulation> Build(SimBuildOptions const &options);

  private:
    static std::unique_ptr<Simulation>
    BuildFalling(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildBigBall(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildSlope(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildDroplet(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildDambreak(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildPlane(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildDipole(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildPattern(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildPattern2(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildPattern3(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildTmp(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildMagSphere(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildLifting(SimBuildOptions const &options);
    static std::unique_ptr<Simulation>
    BuildLR(SimBuildOptions const &options);
};
} // namespace Pivot
