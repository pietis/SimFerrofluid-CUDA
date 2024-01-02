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

    std::vector<source_type> points(size);

    for (int i = 0; i < size; i++) {
        points[i][0] = positions_[i * 3 + 0];
        points[i][1] = positions_[i * 3 + 1];
        points[i][2] = positions_[i * 3 + 2];
    }

    kernel_type K(6);
    K.epsilon = epsilon * epsilon;
    fmmtl::kernel_matrix<kernel_type> A{K, points, points};
    FMMOptions opts;
    A.set_options(opts);

    std::vector<double> b(size);
    std::vector<double> u(size);
    std::vector<result_type> force(size);
    std::vector<charge_type> charges(size);

    for (int i = 0; i < size; i++) {
        b[i] = -2 * lambda *
               (Hext_[i * 3 + 0] * normals_[i * 3 + 0] +
                Hext_[i * 3 + 1] * normals_[i * 3 + 1] +
                Hext_[i * 3 + 2] * normals_[i * 3 + 2]);
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

        double nx[3];
        nx[0] = normals_[i * 3 + 0];
        nx[1] = normals_[i * 3 + 1];
        nx[2] = normals_[i * 3 + 2];
        double Ht[3];
        double Hextn = Hext_[i * 3 + 0] * nx[0] + Hext_[i * 3 + 1] * nx[1] +
                       Hext_[i * 3 + 2] * nx[2];
        Ht[0] = Hext_[i * 3 + 0] - Hextn * nx[0];
        Ht[1] = Hext_[i * 3 + 1] - Hextn * nx[1];
        Ht[2] = Hext_[i * 3 + 2] - Hextn * nx[2];
        double coef = 1.0 / (4 * m_PI);
        double Dn =
            force[i][1] * nx[0] + force[i][2] * nx[1] + force[i][3] * nx[2];
        
        Ht[0] += -coef * (force[i][1] - nx[0] * Dn);
        Ht[1] += -coef * (force[i][2] - nx[1] * Dn);
        Ht[2] += -coef * (force[i][3] - nx[2] * Dn);

        double HtSquared = Ht[0] * Ht[0] + Ht[1] * Ht[1] + Ht[2] * Ht[2];
        double H2 = Hn * Hn + HtSquared;

        double pressure;

        pressure = 0.5 * m_MU * (chi * H2 + Hn * Hn * chi * chi);
        pressures_[i] = pressure;
    }
};