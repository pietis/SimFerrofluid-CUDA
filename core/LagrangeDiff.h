#pragma once

#include "GridData.h"

namespace Pivot {
class LagrangeDiff {
  public:
    template <int N>
    static double CalcCurvature(GridData<double> const &data,
                                const Vector3d &pos) {
        Vector3i lower = data.GetGrid().CalcLower<1>(pos);
        Array3d frac = data.GetGrid().CalcLowerFrac(pos, lower);
        double h = data.GetGrid().GetSpacing();
        double b[3][N];
        double phi[N][N][N];
        double beta[3][3][N];
        double A[N][N];
        const int bias = (N - 1) / 2;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    phi[i][j][k] =
                        data.At(lower + Vector3i(i - bias, j - bias, k - bias));
                }
            }
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < N; j++) {
                b[i][j] = ((j - bias) - frac(i)) * h;
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double tmp = 1;
                for (int k = 0; k < N; k++) {
                    if (k == j)
                        continue;
                    tmp *= (b[i][k] - b[i][j]);
                }
                A[i][j] = tmp;
            }
        }

        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < N; j++) {
                double tmp = 1;
                for (int k = 0; k < N; k++) {
                    if (k == j)
                        continue;
                    tmp *= b[i][k];
                }
                beta[0][i][j] = tmp / A[i][j];
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < N; j++) {
                double tmp0 = 0;
                for (int k = 0; k < N; k++) {
                    if (k == j)
                        continue;
                    double tmp1 = 1;
                    for (int l = 0; l < N; l++) {
                        if (l == j || l == k)
                            continue;
                        tmp1 *= b[i][l];
                    }
                    tmp0 += tmp1;
                }
                beta[1][i][j] = -tmp0 / A[i][j];
            }
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < N; j++) {
                double tmp0 = 0;
                for (int k = 0; k < N; k++) {
                    if (k == j)
                        continue;
                    double tmp1 = 0;
                    for (int l = 0; l < N; l++) {
                        if (l == j || l == k)
                            continue;
                        double tmp2 = 1;
                        for (int m = 0; m < N; m++) {
                            if (m == j || m == k || m == l)
                                continue;
                            tmp2 *= b[i][m];
                        }
                        tmp1 += tmp2;
                    }
                    tmp0 += tmp1;
                }
                beta[2][i][j] = tmp0 / A[i][j];
            }
        }
        double phi_x = 0;
        double phi_y = 0;
        double phi_z = 0;
        double phi_xx = 0;
        double phi_yy = 0;
        double phi_zz = 0;
        double phi_xy = 0;
        double phi_xz = 0;
        double phi_yz = 0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    phi_x += beta[1][0][i] * beta[0][1][j] * beta[0][2][k] *
                             phi[i][j][k];
                    phi_y += beta[0][0][i] * beta[1][1][j] * beta[0][2][k] *
                             phi[i][j][k];
                    phi_z += beta[0][0][i] * beta[0][1][j] * beta[1][2][k] *
                             phi[i][j][k];
                    phi_xx += beta[2][0][i] * beta[0][1][j] * beta[0][2][k] *
                              phi[i][j][k];
                    phi_yy += beta[0][0][i] * beta[2][1][j] * beta[0][2][k] *
                              phi[i][j][k];
                    phi_zz += beta[0][0][i] * beta[0][1][j] * beta[2][2][k] *
                              phi[i][j][k];
                    phi_xy += beta[1][0][i] * beta[1][1][j] * beta[0][2][k] *
                              phi[i][j][k];
                    phi_xz += beta[1][0][i] * beta[0][1][j] * beta[1][2][k] *
                              phi[i][j][k];
                    phi_yz += beta[0][0][i] * beta[1][1][j] * beta[1][2][k] *
                              phi[i][j][k];
                }
            }
        }
        double const gradNorm2 = phi_x * phi_x + phi_y * phi_y + phi_z * phi_z;

        double kappa = phi_x * phi_x * (phi_yy + phi_zz) +
                       phi_y * phi_y * (phi_xx + phi_zz) +
                       phi_z * phi_z * (phi_xx + phi_yy);
        kappa -= (phi_x * phi_y * phi_xy + phi_x * phi_z * phi_xz +
                  phi_y * phi_z * phi_yz) *
                 2;
        kappa /= gradNorm2 * std::sqrt(gradNorm2);
        double const invDx = data.GetGrid().GetInvSpacing();

        // printf("%.3e %.3e %.3e %.3e %.3e %.3e\n\n", phi_x, phi_y, phi_z,
        // phi_xx,
        //        phi_xy, kappa);
        return std::abs(kappa) < invDx ? kappa : (kappa < 0 ? -1 : +1) * invDx;
        // return kappa;
    }
};
} // namespace Pivot
