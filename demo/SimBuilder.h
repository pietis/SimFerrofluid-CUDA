#pragma once

#include "Simulation.h"

namespace Pivot {
struct SimBuildOptions {
    Simulation::Scene Scene;
    int Scale;
    bool EnableMag;
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
};
} // namespace Pivot
