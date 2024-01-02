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
    void Solve(SurfaceMesh &mesh, const std::function<Vector3d(const Vector3d&, double)>& fieldApplied, double time, double h) {
        m_Mesh = &mesh;
        double eps = m_EpsFactor * h;

        fmt::print("meshsize {} ", mesh.size());
        auto sw = StopWatch("mag.");
        InitSolver(fieldApplied, time);
        SolveMagneticFMM(m_Mesh->Positions.data(), m_Mesh->Normals.data(),
                         m_Mesh->Areas.data(), m_Hext.data(),
                         m_MagneticPressure.data(), m_Mesh->size(),
                         m_NumIteration, m_Lambda, m_Chi, eps);
        fmt::print("{:>8.3f}s ", sw.Stop());
    }
    void SetChi(double chi){
      m_Chi = chi;
      m_Lambda = (-m_Chi) / (2 + m_Chi);
    }

  private:
    void InitSolver(const std::function<Vector3d(const Vector3d&, double)>& fieldApplied, double time) {
        m_MagneticPressure.resize(m_Mesh->size());
        m_Hext.resize(m_Mesh->size());
        for (int i = 0; i < m_Mesh->size(); i++) {
            m_Hext[i] = fieldApplied(m_Mesh->Positions[i], time);
        }
    }
  private:
    SurfaceMesh *m_Mesh;
    std::vector<double> m_MagneticPressure;

    double m_Chi = 0.33;
    double m_Lambda = (-m_Chi) / (2 + m_Chi);
    std::vector<Vector3d> m_Hext;

    int m_NumIteration = 10;
    double m_EpsFactor = 1.0;
    double m_StopThres = 1e-6;
};
} // namespace Pivot
