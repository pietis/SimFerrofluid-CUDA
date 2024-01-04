#include "SISurfaceTension.h"

#include "FiniteDiff.h"
#include "TriLerp.h"

namespace Pivot {
	void SISurfaceTension::Solve(
		SGridData<double>       &velocity,
		GridData<double>  const &levelSet,
		Collider          const &collider,
		double                   coeff,
		double                   dt) {
		double const dx = levelSet.GetGrid().GetSpacing();
		double const invDx = levelSet.GetGrid().GetInvSpacing();
		double const eps = 2. * dx;
		int n = 0;

		SGridData<int> grid2mat(velocity.GetGrids(), -Vector3i::Ones());
		std::vector<std::pair<int, int>> mat2grid;
		std::vector<Triplet<double>> elements;

		auto const GetFaceLevelSet = [&](int axis, Vector3i const &face)->double {
			return (levelSet.At(StaggeredGrid::AdjCellOfFace(axis, face, 0)) + levelSet.At(StaggeredGrid::AdjCellOfFace(axis, face, 1))) * .5;
		};

		auto const GetFaceNormal = [&](int axis, Vector3i const &face)->Vector3d {
			return (FiniteDiff::CalcGradient(levelSet, StaggeredGrid::AdjCellOfFace(axis, face, 0)) + FiniteDiff::CalcGradient(levelSet, StaggeredGrid::AdjCellOfFace(axis, face, 1))).normalized();
		};

		auto const GetFaceCurvature = [&](int axis, Vector3i const &face)->double {
			return (FiniteDiff::CalcCurvature(levelSet, StaggeredGrid::AdjCellOfFace(axis, face, 0)) + FiniteDiff::CalcCurvature(levelSet, StaggeredGrid::AdjCellOfFace(axis, face, 1))) * .5;
		};

		auto const Dirac = [&](double phi)->double {
			if (phi < -eps) {
				return 0.;
			} else if (phi > eps) {
				return 0.;
			} else {
				return .5 * (1. + std::cos(std::numbers::pi * phi / eps)) / eps;
			}
		};

		// Find interface faces.
		ForEach(velocity.GetGrids(), [&](int axis, Vector3i const &face) {
			Vector3d const pos = velocity[axis].GetGrid().PositionOf(face);
			double const phi = GetFaceLevelSet(axis, face);
			if (-eps < phi && phi < eps && collider.GetFraction()[axis][face] < 1) {
				grid2mat[axis][face] = n++;
				mat2grid.push_back(std::pair(axis, velocity[axis].GetGrid().IndexOf(face)));
			}
		});

		SparseMatrix<double> matLaplacian(n, n);
		VectorXd vel(n);
		VectorXd rhs(n);

		for (int r = 0; r < n; r++) {
			int      const axis   = mat2grid[r].first;
			Vector3i const face   = velocity[axis].GetGrid().CoordOf(mat2grid[r].second);
			double   const phi    = GetFaceLevelSet(axis, face);
			Vector3d const normal = GetFaceNormal(axis, face);
			double   const kappa  = GetFaceCurvature(axis, face);
			double   const delta  = std::max(Dirac(phi), 1e-7 * invDx);
			Vector3d const jacobian = FiniteDiff::CalcGradient(velocity[axis], face);
			Matrix3d const hessian = FiniteDiff::CalcHessian(velocity[axis], face);
			double diaCoef = 1. / delta;
			// vel[r] = rhs[r] = velocity[axis][face] / delta - coeff * dt * (kappa * normal[axis]);
			vel[r] = rhs[r] = velocity[axis][face] / delta - coeff * dt * (kappa * normal[axis] + dt * ((normal.transpose() * hessian * normal).value() + kappa * jacobian.dot(normal)));
			for (int i = 0; i < Grid::GetNumNeighbors(); i++) {
				Vector3i const nbFace = Grid::NeighborOf(face, i);
				if (!velocity[axis].GetGrid().IsValid(nbFace) || collider.GetFraction()[axis][nbFace] == 1) continue;
				double const weight = coeff * dt * dt / (dx * dx);
				diaCoef += weight;
				if (int c = grid2mat[axis][nbFace]; c != -1) {
					elements.push_back(Triplet<double>(r, c, -weight));
				} else {
					rhs[r] += weight * velocity[axis][nbFace];
				}
			}
			elements.push_back(Triplet<double>(r, r, diaCoef));
		}
		matLaplacian.setFromTriplets(elements.begin(), elements.end());

		{ // Solve the linear system
			ConjugateGradient<SparseMatrix<double>, Eigen::Lower | Eigen::Upper, IncompleteCholesky<double>> solver(matLaplacian);
			solver.setTolerance(std::max(1e-5, std::numeric_limits<double>::epsilon() / rhs.norm()));
			vel = solver.solveWithGuess(rhs, vel);
			std::cout << fmt::format("{:>6} iters", solver.iterations());
		}
		
		for (int r = 0; r < n; r++) {
			int const axis = mat2grid[r].first;
			Vector3i const face = velocity[axis].GetGrid().CoordOf(mat2grid[r].second);
			velocity[axis][face] = vel[r];
		}
	}
}
