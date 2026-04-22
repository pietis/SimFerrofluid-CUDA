#pragma once

#include "Collider.h"
#include "Contour.h"
#include "Magnetic.h"
#include "MagneticField.h"
#include "Pressure.h"

namespace Pivot {
class Simulation {
  private:
    friend class SimBuilder;

  public:
    enum class Scene { Falling, BigBall, Slope, Droplet, DamBreak, Plane, Dipole, Pattern, Pattern2, Pattern3, MagSphere, Lifting, LR, Tmp };

  public:
    explicit Simulation(StaggeredGrid const &sgrid);

    void Describe(YAML::Node &root) const;
    void Export(std::filesystem::path const &dirname,
                bool initial = false) const;
    void Save(std::ostream &out) const;
    void Load(std::istream &in);

    void Initialize();
    void Advance(double deltaTime);

    void AdvectFields(double dt);
    void ApplyBodyForces(double dt);
    void CacheMagneticObject();
    void UpdateMagneticPressure();
    void ProjectVelocity(double dt);
    void ApplySemiImplicitST(double dt);
    void ComputeVolumeError(double dt);

    void ReinitializeLevelSet(bool initial = false);


    void SetTime(double time) { m_Time = time; }
    auto GetTime() const { return m_Time; }

    double GetCourantTimeStep() const {
        return m_SGrid.GetSpacing() / m_Velocity.GetMaxAbsComponent();
    }

  private:
    double m_Time;
    Scene m_Scene;

    StaggeredGrid m_SGrid;
    Collider m_Collider;
    Pressure m_Pressure;
    SGridData<double> m_Velocity;
    GridData<double> m_LevelSet;
    Contour m_Contour;
    Magnetic m_Magnetic;
    MagneticField m_MagneticField;
    std::vector<double> m_MagneticPressures;
    std::shared_ptr<Magnetic> m_MagneticObject;
    std::function<Vector3d(const Vector3d&, double time)> m_FieldApplied;

    double m_InitVolume;
    double m_CurrentVolume;
    double m_CumulVolError = 0;
    double m_VolError = 0;
    double m_Damping = 0;

    double m_LiquidDensity = 1e3;
    double m_SurfaceTensionCoeff = 7.28e-2;

    bool m_GravityEnabled = true;
    bool m_SurfaceTensionEnabled = false;
    bool m_SemiImplicitSTEnabled = false;
    bool m_MagneticEnabled = false;

    int m_MaxMeshSize = 0;
};
} // namespace Pivot
