#include "Reinitialization.h"

namespace Pivot {
	static inline std::pair<double, double> GetDrv(std::array<double, 7> const &v, double invDx) {
		constexpr auto Square = [](double x) { return x * x; };

		std::pair<double, double> drv;
		for (int i = 0; i < 2; i++) {
			std::array<double, 5> dev = i ?
				std::array { v[6] - v[5], v[5] - v[4], v[4] - v[3], v[3] - v[2], v[2] - v[1] } :
				std::array { v[1] - v[0], v[2] - v[1], v[3] - v[2], v[4] - v[3], v[5] - v[4] };
			double eps = 0;
			for (int j = 0; j < 5; j++) eps = std::max(eps, dev[j] * dev[j]);
			eps = eps * 1e-6 + 1e-99;
			double const phix1 = dev[0] / 3 - dev[1] * 7 / 6 + dev[2] * 11 / 6;
			double const phix2 = -dev[1] / 6 + dev[2] * 5 / 6 + dev[3] / 3;
			double const phix3 = dev[2] / 3 + dev[3] * 5 / 6 - dev[4] / 6;
			double const s1 = Square(dev[0] - 2 * dev[1] + dev[2]) * 13 / 12 + Square(dev[0] - 4 * dev[1] + 3 * dev[2]) / 4;
			double const s2 = Square(dev[1] - 2 * dev[2] + dev[3]) * 13 / 12 + Square(dev[1] - dev[3]) / 4;
			double const s3 = Square(dev[2] - 2 * dev[3] + dev[4]) * 13 / 12 + Square(3 * dev[2] - 4 * dev[3] + dev[4]) / 4;
			double const alpha1 = .1 / Square(s1 + eps);
			double const alpha2 = .6 / Square(s2 + eps);
			double const alpha3 = .3 / Square(s3 + eps);
			double const sum = alpha1 + alpha2 + alpha3;
			(i ? drv.second : drv.first) = (alpha1 * phix1 + alpha2 * phix2 + alpha3 * phix3) / sum * invDx;
		}
		return drv;
	};

	static std::array<double, 7> GetStencil(GridData<double> const &field, Vector3i const &coord, int axis) {
		std::array<double, 7> vals;
		for (int i = 0; i <= 6; i++)
			vals[i] = field.At(coord + Vector3i::Unit(axis) * i);
		return vals;
	}

	void Reinitialization::Solve(GridData<double> &phi, int maxSteps) {
		double const dt = phi.GetGrid().GetSpacing() * .5;
		for (int iter = 0; iter < maxSteps; iter++) {
			GridData<double> newPhi1(phi.GetGrid());
			GridData<double> newPhi2(phi.GetGrid());
			Advance(phi, newPhi1, dt);
			Advance(newPhi1, newPhi2, dt);
			ParallelForEach(phi.GetGrid(), [&](Vector3i const &coord) { newPhi1[coord] = (phi[coord] * 3 + newPhi2[coord]) / 4; });
			Advance(newPhi1, newPhi2, dt);
			ParallelForEach(phi.GetGrid(), [&](Vector3i const &coord) { phi[coord] = (phi[coord] + newPhi2[coord] * 2) / 3; });
		}
	}

	void Reinitialization::Advance(GridData<double> const &field, GridData<double> &newField, double dt) {
		constexpr auto Square = [](double x) { return x * x; };
		auto const Sign = [](double val, double sp) { return val / std::sqrt(val * val + sp * sp); };

		ParallelForEach(field.GetGrid(), [&](Vector3i const &coord) {
			std::array<std::pair<double, double>, 3> drv = {
				GetDrv(GetStencil(field, coord - Vector3i(3, 0, 0), 0), field.GetGrid().GetInvSpacing()),
				GetDrv(GetStencil(field, coord - Vector3i(0, 3, 0), 1), field.GetGrid().GetInvSpacing()),
				GetDrv(GetStencil(field, coord - Vector3i(0, 0, 3), 2), field.GetGrid().GetInvSpacing())
			};
			double sum = 0;
			if (field[coord] < 0) {
				for (int axis = 0; axis < 3; axis++) {
					sum += Square(std::min(drv[axis].first , 0.));
					sum += Square(std::max(drv[axis].second, 0.));
				}
			}
			else {
				for (int axis = 0; axis < 3; axis++) {
					sum += Square(std::max(drv[axis].first , 0.));
					sum += Square(std::min(drv[axis].second, 0.));
				}
			}
			sum = std::sqrt(sum);
			newField[coord] = field[coord] + dt * Sign(field[coord], field.GetGrid().GetSpacing() * sum) * (1. - sum);
		});
	}
}
