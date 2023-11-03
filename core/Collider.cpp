#include "Collider.h"

#include "FiniteDiff.h"
#include "Reinitialization.h"
#include "TriLerp.h"

namespace Pivot {
	Collider::Collider(StaggeredGrid const &sgrid) :
		LevelSet(sgrid.GetNodeGrid()),
		Velocity(sgrid.GetFaceGrids()),
		m_DomainBox(sgrid.GetDomainOrigin(), sgrid.GetDomainLengths()),
		m_Fraction(sgrid.GetFaceGrids()),
		m_Normal(sgrid.GetFaceGrids()),
		m_AuxLevelSet(sgrid.GetCellGrid()) {
		ParallelForEach(LevelSet.GetGrid(), [&](Vector3i const &node) {
			Vector3d const pos = LevelSet.GetGrid().PositionOf(node);
			LevelSet[node] = -.01 * sgrid.GetSpacing() - m_DomainBox.SignedDistanceTo(pos);
		});
	}

	void Collider::Finish(StaggeredGrid const &sgrid) {
		FastMarching::Solve(LevelSet, -1);
		// Reinitialization::Solve(LevelSet, sgrid.GetResolution().maxCoeff() * 2);
		ParallelForEach(m_AuxLevelSet.GetGrid(), [&](Vector3i const &cell) {
			for (int i = 0; i < StaggeredGrid::GetNumNodesPerCell(); i++) {
				m_AuxLevelSet[cell] += LevelSet[StaggeredGrid::NodeOfCell(cell, i)];
			}
			m_AuxLevelSet[cell] /= StaggeredGrid::GetNumNodesPerCell();
		});
		auto levelSetDrvs = std::to_array({ GridData<double>(sgrid.GetNodeGrid()), GridData<double>(sgrid.GetNodeGrid()), GridData<double>(sgrid.GetNodeGrid()) });
		tbb::parallel_for(0, 3, [&](int axis) {
			ParallelForEach(sgrid.GetNodeGrid(), [&](Vector3i const &node) {
				levelSetDrvs[axis][node] = FiniteDiff::CalcFirstDrv(LevelSet, node, axis);
			});
		});

		ParallelForEach(m_Fraction.GetGrids(), [&](int axis, Vector3i const &face) {
			m_Fraction[axis][face] = CalcFaceFraction(axis, face);
			double sum = 0;
			for (int i = 0; i < StaggeredGrid::GetNumNodesPerFace(); i++) {
				sum += levelSetDrvs[axis][StaggeredGrid::NodeOfFace(axis, face, i)];
			}
			m_Normal[axis][face] = sum / StaggeredGrid::GetNumNodesPerFace();
		});
	}

	double Collider::CalcFaceFraction(int axis, Vector3i const &face) const {
		constexpr auto theta = [](double phi0, double phi1) { return phi0 / (phi0 - phi1); };

		constexpr auto fraction = [&theta](double phi0, double phi1, double phi2) {
			if (phi0 > phi1) std::swap(phi0, phi1);
			if (phi1 > phi2) std::swap(phi1, phi2);
			if (phi0 > phi1) std::swap(phi0, phi1);
			if (phi2 <= 0) {
				return 1.;
			} else if (phi1 <= 0) {
				return 1. - theta(phi2, phi0) * theta(phi2, phi1);
			} else if (phi0 <= 0) {
				return theta(phi0, phi1) * theta(phi0, phi2);
			} else {
				return 0.;
			}
		};

		double const phi0 = LevelSet[StaggeredGrid::NodeOfFace(axis, face, 0)];
		double const phi1 = LevelSet[StaggeredGrid::NodeOfFace(axis, face, 1)];
		double const phi2 = LevelSet[StaggeredGrid::NodeOfFace(axis, face, 2)];
		double const phi3 = LevelSet[StaggeredGrid::NodeOfFace(axis, face, 3)];
		double const centerPhi = (phi0 + phi1 + phi2 + phi3) * .25;
		double faceFraction = (fraction(centerPhi, phi0, phi1) + fraction(centerPhi, phi1, phi3) + fraction(centerPhi, phi3, phi2) + fraction(centerPhi, phi2, phi0)) * .25;

		return faceFraction > .9 ? 1. : faceFraction;
	}

	void Collider::Enforce(SGridData<double> &fluidVelocity) const {
		auto const oldFluidVelocity = fluidVelocity;
		ParallelForEach(fluidVelocity.GetGrids(), [&](int axis, Vector3i const &face) {
			if (m_Fraction[axis][face] == 1.) {
				Vector3d const pos = fluidVelocity[axis].GetGrid().PositionOf(face);
				Vector3d const vel = TriLerp::Interpolate(oldFluidVelocity, pos);
				Vector3d const vel0 = TriLerp::Interpolate(Velocity, pos);
				Vector3d const n = TriLerp::Interpolate(m_Normal, pos).normalized();
				fluidVelocity[axis][face] = (vel - (vel - vel0).dot(n) * n)[axis];
			}
		});
	}
}
