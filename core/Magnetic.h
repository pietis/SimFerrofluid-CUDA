#pragma once

#include "Collider.h"
#include "StopWatch.h"

#include "omp.h"

#include "MagneticFMM.h"

namespace Pivot {
class Magnetic {
  private:
    friend class Simulation;

  public:
    enum class SolverType {
        IoB,
        FDM,
    };

    void
    Solve(SurfaceMesh &mesh,
          const std::function<Vector3d(const Vector3d &, double)> &fieldApplied,
          double time, double h,
          std::shared_ptr<Magnetic> magneticObject = nullptr) {
        m_Mesh = mesh;
        m_Eps = m_EpsFactor * h;

        fmt::print("meshsize {} ", mesh.size());
        auto sw = StopWatch("mag.");
        InitSolver(fieldApplied, time, magneticObject);
        SolveMagneticFMM(m_Mesh.Positions.data(), m_Mesh.Normals.data(),
                 m_Mesh.Areas.data(), m_Hext.data(),
                 m_MagneticPressure.data(), m_Mesh.size(),
                 m_NumIteration, m_Lambda, m_Chi, m_Eps, m_Trunc);
        fmt::print("{:.3f}s ", sw.Stop());
    }
    void
    Cache(const std::function<Vector3d(const Vector3d &, double)> &fieldApplied,
          double time, double h) {
        m_Eps = m_EpsFactor * h;

        fmt::print(fmt::fg(fmt::color::yellow_green), "[Cache] ");
        fmt::print("meshsize {} ", m_Mesh.size());
        auto sw = StopWatch("mcache.");
        InitCache(fieldApplied, time);
        CacheMagneticFMM(m_Mesh.Positions.data(), m_Mesh.Normals.data(),
                         m_Mesh.Areas.data(), m_Hext.data(), m_Charges.data(),
                         m_Mesh.size(), m_NumIteration, m_Lambda, m_Chi, m_Eps);
        fmt::print("{:.3f}s\n", sw.Stop());
    }
    void SetChi(double chi) {
        m_Chi = chi;
        m_Lambda = (-m_Chi) / (2 + m_Chi);
    }
    void SetIteration(int iter) { m_NumIteration = iter; }
    void SetEpsFactor(double epsFactor) { m_EpsFactor = epsFactor; }
    void SetTrunc() { m_Trunc = true; }
    void AssignMesh(SurfaceMesh &mesh) { m_Mesh = mesh; }
    void SetSolverType(SolverType solverType) { m_SolverType = solverType; }
    SolverType GetSolverType() const { return m_SolverType; }
    double GetChi() const { return m_Chi; }

  private:
    void InitSolver(
        const std::function<Vector3d(const Vector3d &, double)> &fieldApplied,
        double time, std::shared_ptr<Magnetic> magneticObject) {
        m_MagneticPressure.resize(m_Mesh.size());
        m_Hext.resize(m_Mesh.size());

        if (magneticObject != nullptr) {
            magneticObject->ApplyCache(m_Mesh.Positions, m_Hext);
        } else {
            for (int i = 0; i < m_Mesh.size(); i++) {
                m_Hext[i] = fieldApplied(m_Mesh.Positions[i], time);
            }
        }
    }
    void ApplyCache(std::vector<Vector3d> &targets,
                    std::vector<Vector3d> &Hind) {
        Hind.resize(targets.size());
        fmt::print("[cache app] ");
        auto sw = StopWatch("mcacheapp.");
        ApplyCacheFMM(m_Mesh.Positions.data(), targets.data(), m_Mesh.size(),
                      targets.size(), m_Eps, m_Charges, Hind.data());
        fmt::print("{:.3f}s ", sw.Stop());
    }
    void InitCache(
        const std::function<Vector3d(const Vector3d &, double)> &fieldApplied,
        double time) {
        m_Charges.resize(m_Mesh.size());
        m_Hext.resize(m_Mesh.size());
        for (int i = 0; i < m_Mesh.size(); i++) {
            m_Hext[i] = fieldApplied(m_Mesh.Positions[i], time);
        }
    }

  private:
    SurfaceMesh m_Mesh;
    std::vector<double> m_MagneticPressure;
    std::vector<double> m_Charges;

    double m_Chi = 0.33;
    double m_Lambda = (-m_Chi) / (2 + m_Chi);
    std::vector<Vector3d> m_Hext;

    int m_NumIteration = 10;
    double m_EpsFactor = 1.0;
    double m_Eps;
    double m_StopThres = 1e-6;
    bool m_Trunc = false;
    SolverType m_SolverType = SolverType::IoB;
};
} // namespace Pivot
