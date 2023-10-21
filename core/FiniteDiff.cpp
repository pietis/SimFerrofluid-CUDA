#include "FiniteDiff.h"

namespace Pivot {
	Vector3d FiniteDiff::CalcGradient(GridData<double> const &grData, Vector3i const &coord) {
		return {
			CalcFirstDrv(grData, coord, 0), CalcFirstDrv(grData, coord, 1), CalcFirstDrv(grData, coord, 2)
		};
	}

	Matrix3d FiniteDiff::CalcHessian(GridData<double> const &grData, Vector3i const &coord) {
		Matrix3d hessian;
		hessian << 
			CalcSecondDrv(grData, coord, 0, 0), CalcSecondDrv(grData, coord, 0, 1), CalcSecondDrv(grData, coord, 0, 2),
			CalcSecondDrv(grData, coord, 1, 0), CalcSecondDrv(grData, coord, 1, 1), CalcSecondDrv(grData, coord, 1, 2),
			CalcSecondDrv(grData, coord, 2, 0), CalcSecondDrv(grData, coord, 2, 1), CalcSecondDrv(grData, coord, 2, 2);
		return hessian;
	}

	double FiniteDiff::CalcCurvature(GridData<double> const &grData, Vector3i const &coord) {
		double const phi_x = CalcFirstDrv(grData, coord, 0);
		double const phi_y = CalcFirstDrv(grData, coord, 1);
		double const phi_z = CalcFirstDrv(grData, coord, 2);
		double const phi_xx = CalcSecondDrv(grData, coord, 0, 0);
		double const phi_yy = CalcSecondDrv(grData, coord, 1, 1);
		double const phi_zz = CalcSecondDrv(grData, coord, 2, 2);
		double const phi_xy = CalcSecondDrv(grData, coord, 0, 1);
		double const phi_xz = CalcSecondDrv(grData, coord, 0, 2);
		double const phi_yz = CalcSecondDrv(grData, coord, 1, 2);
		double const gradNorm2 = phi_x * phi_x + phi_y * phi_y + phi_z * phi_z;

		double kappa = phi_x * phi_x * (phi_yy + phi_zz) + phi_y * phi_y * (phi_xx + phi_zz) + phi_z * phi_z * (phi_xx + phi_yy);
		kappa -= (phi_x * phi_y * phi_xy + phi_x * phi_z * phi_xz + phi_y * phi_z * phi_yz) * 2;
		kappa /= gradNorm2 * std::sqrt(gradNorm2);
		double const invDx = grData.GetGrid().GetInvSpacing();

		// return std::abs(kappa) < invDx ? kappa: (kappa < 0 ? -1 : +1) * invDx;
		return kappa;
	}
}
