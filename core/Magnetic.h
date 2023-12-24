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
        // SolveMagnetic();
        SolveMagneticFMM(m_Mesh->Positions.data(), m_Mesh->Normals.data(),
                         m_Mesh->Areas.data(), m_Hext.data(),
                         m_MagneticPressure.data(), m_Mesh->size(),
                         m_NumIteration, m_Lambda, m_Chi, eps);
        fmt::print("{:>8.3f}s ", sw.Stop());
    }

  private:
    void InitSolver(const std::function<Vector3d(const Vector3d&, double)>& fieldApplied, double time) {
        m_MagneticPressure.resize(m_Mesh->size());
        m_Hext.resize(m_Mesh->size());
        for (int i = 0; i < m_Mesh->size(); i++) {
            m_Hext[i] = fieldApplied(m_Mesh->Positions[i], time);
        }
    }
    //     void SolveMagnetic() {
    //         static std::vector<double> buffer[2];
    //         int size = m_Mesh->size();
    //         buffer[0].resize(size);
    //         buffer[1].resize(size);
    //         int u, utmp;
    //         u = 0;
    //         utmp = 1;
    //         double sum_res = 0;
    // #pragma omp parallel shared(sum_res)
    //         {
    // #pragma omp for
    //             for (int i = 0; i < size; i++) {
    //                 buffer[u][i] = -2 * m_Lambda *
    //                 m_Hext.dot(m_Mesh->Normals[i]);
    //             }
    //             for (int iter = 0; iter < m_NumIteration; iter++) {
    // #pragma omp single
    //                 { std::swap(u, utmp); }
    // #pragma omp for collapse(1)
    //                 for (int i = 0; i < size; i++) {
    //                     buffer[u][i] =
    //                         -2 * m_Lambda * m_Hext.dot(m_Mesh->Normals[i]);
    //                     double sum = 0;
    //                     for (int j = 0; j < size; j++) {
    //                         if (i == j) {
    //                             continue;
    //                         }
    //                         sum += dGdxd(m_Mesh->Positions[i],
    //                         m_Mesh->Positions[j],
    //                                      m_Mesh->Normals[i], m_EpsFPI) *
    //                                m_Mesh->Areas[j] * buffer[utmp][j];
    //                     }
    //                     buffer[u][i] += 2 * m_Lambda * sum;
    //                 }
    // #pragma omp barrier
    //             }
    // #pragma omp for reduction(+ : sum_res)
    //             for (int i = 0; i < size; i++) {
    //                 sum_res += abs(buffer[u][i] - buffer[utmp][i]);
    //             }
    //             if (omp_get_thread_num() == 0) {
    //                 fmt::print("residual {:.3e} ", sum_res / size);
    //             }
    // #pragma omp for
    //             for (int i = 0; i < size; i++) {
    //                 double Hn = 1 / (m_Chi)*buffer[u][i];
    //                 double Hn_ = Hn * (1 + m_Chi);

    //                 Vector3d nx = m_Mesh->Normals[i];
    //                 Vector3d tx1, tx2;
    //                 if (abs(nx(0)) > 0.1) {
    //                     tx1 = nx.cross(Vector3d::Unit(1)).normalized();
    //                 } else {
    //                     tx1 = nx.cross(Vector3d::Unit(0)).normalized();
    //                 }
    //                 tx2 = nx.cross(tx1).normalized();
    //                 double Ht1 = m_Hext.dot(tx1);
    //                 double Ht2 = m_Hext.dot(tx2);
    //                 for (int j = 0; j < size; j++) {
    //                     if (i == j) {
    //                         continue;
    //                     }
    //                     Ht1 += -dGdxd(m_Mesh->Positions[i],
    //                     m_Mesh->Positions[j],
    //                                   tx1, m_EpsFPI) *
    //                            m_Mesh->Areas[j] * buffer[u][j];
    //                     Ht2 += -dGdxd(m_Mesh->Positions[i],
    //                     m_Mesh->Positions[j],
    //                                   tx2, m_EpsFPI) *
    //                            m_Mesh->Areas[j] * buffer[u][j];
    //                 }
    //                 double HtSquared = Ht1 * Ht1 + Ht2 * Ht2;

    //                 double pressure = 0;

    //                 pressure += m_MU * (1 + m_Chi) * 0.5 * (Hn * Hn -
    //                 HtSquared); pressure -= m_MU * 0.5 * (Hn_ * Hn_ -
    //                 HtSquared); m_MagneticPressure[i] = pressure;
    //             }
    //         }
    //     }
    double dGdxd(const Vector3d &x, const Vector3d &y, const Vector3d &xd,
                 double eps) {
        Vector3d r = y - x;
        return 1.0 / (4.0 * m_PI) * r.dot(xd) /
               (std::max)(r.squaredNorm() * r.norm(), eps);
    }

  private:
    inline static const double m_PI = 3.141592653589793;
    inline static const double m_MU = 4e-7 * m_PI;

    SurfaceMesh *m_Mesh;
    std::vector<double> m_MagneticPressure;

    double m_Chi = 0.33;
    double m_Lambda = (-m_Chi) / (2 + m_Chi);
    // Vector3d m_Hext = Vector3d(0, 6e4, 0);
    std::vector<Vector3d> m_Hext;

    int m_NumIteration = 10;
    double m_EpsFactor = 1.0;
    double m_StopThres = 1e-6;
};
} // namespace Pivot
