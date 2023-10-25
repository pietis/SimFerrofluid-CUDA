#include "MagneticFMM.h"

#include <cstdio>

#include "LaplaceSpherical.hpp"
#include "fmmtl/KernelMatrix.hpp"
#include "fmmtl/numeric/random.hpp"

typedef fmmtl::LaplaceSpherical kernel_type;
typedef kernel_type::source_type source_type;
typedef kernel_type::target_type target_type;
typedef kernel_type::charge_type charge_type;
typedef kernel_type::result_type result_type;

constexpr double m_PI = 3.141592653589793;
constexpr double m_MU = 4e-7 * m_PI;

void SolveMagneticFMM(const void *positions, const void *normals,
                      const void *areas, const void *Hext, void *pressures,
                      const int size, const int num_iter, const double lambda,
                      const double chi, const double epsilon) {
    const double *positions_ = (const double *)positions;
    const double *normals_ = (const double *)normals;
    const double *areas_ = (const double *)areas;
    const double *Hext_ = (const double *)Hext;
    double *pressures_ = (double *)pressures;

    // std::random_device rdevice;
    // std::default_random_engine rand_engine(rdevice());
    // std::uniform_real_distribution<double> noise(-epsilon, epsilon);

    std::vector<source_type> points(size);

    for (int i = 0; i < size; i++) {
        points[i][0] = positions_[i * 3 + 0];
        points[i][1] = positions_[i * 3 + 1];
        points[i][2] = positions_[i * 3 + 2];
    }

    kernel_type K(10);
    K.epsilon = epsilon;
    fmmtl::kernel_matrix<kernel_type> A{K, points, points};
    FMMOptions opts;
    A.set_options(opts);

    std::vector<double> b(size);
    std::vector<double> u(size);
    std::vector<result_type> force(size);
    std::vector<charge_type> charges(size);

    for (int i = 0; i < size; i++) {
        b[i] =
            -2 * lambda *
            (Hext_[0] * normals_[i * 3 + 0] + Hext_[1] * normals_[i * 3 + 1] +
             Hext_[2] * normals_[i * 3 + 2]);
        u[i] = b[i];
    }

    for (int i = 0; i < size; i++) {
        charges[i] = areas_[i] * u[i];
    }

    double residual;
    for (int iter = 0; iter < num_iter; iter++) {
        force = A * charges;

        residual = 0;
        for (int i = 0; i < size; i++) {
            double tmp = b[i] + 2 * lambda / (4 * m_PI) *
                                    (force[i][1] * normals_[i * 3 + 0] +
                                     force[i][2] * normals_[i * 3 + 1] +
                                     force[i][3] * normals_[i * 3 + 2]);
            residual += abs(u[i] - tmp);
            u[i] = tmp;
            charges[i] = areas_[i] * u[i];
        }
        residual /= size;
    }

    printf("residual %.3e ", residual);

    force = A * charges;
    for (int i = 0; i < size; i++) {
        double Hn = 1 / chi * u[i];
        double Hn_ = Hn * (1 + chi);

        double nx[3];
        nx[0] = normals_[i * 3 + 0];
        nx[1] = normals_[i * 3 + 1];
        nx[2] = normals_[i * 3 + 2];
        double tx1[3];
        double tx2[3];

        double x, y, z, rn;
        if (abs(nx[0]) > 0.1) {
            x = -nx[2];
            y = 0;
            z = nx[0];
        } else {
            x = 0;
            y = nx[2];
            z = -nx[1];
        }
        rn = sqrt(x * x + y * y + z * z);
        tx1[0] = x / rn;
        tx1[1] = y / rn;
        tx1[2] = z / rn;

        x = nx[1] * tx1[2] - nx[2] * tx1[1];
        y = nx[2] * tx1[0] - nx[0] * tx1[2];
        z = nx[0] * tx1[1] - nx[1] * tx1[0];

        rn = sqrt(x * x + y * y + z * z);
        tx2[0] = x / rn;
        tx2[1] = y / rn;
        tx2[2] = z / rn;

        double Ht1 = Hext_[0] * tx1[0] + Hext_[1] * tx1[1] + Hext_[2] * tx1[2];
        double Ht2 = Hext_[0] * tx2[0] + Hext_[1] * tx2[1] + Hext_[2] * tx2[2];

        Ht1 += -1 / (4 * m_PI) *
               (force[i][1] * tx1[0] + force[i][2] * tx1[1] +
                force[i][3] * tx1[2]);
        Ht2 += -1 / (4 * m_PI) *
               (force[i][1] * tx2[0] + force[i][2] * tx2[1] +
                force[i][3] * tx2[2]);

        double HtSquared = Ht1 * Ht1 + Ht2 * Ht2;

        double pressure = 0;

        pressure += m_MU * (1 + chi) * 0.5 * (Hn * Hn - HtSquared);
        pressure -= m_MU * 0.5 * (Hn_ * Hn_ - HtSquared);
        pressures_[i] = pressure;
    }
};