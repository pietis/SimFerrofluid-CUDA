#include "SurfaceMesh.h"

namespace Pivot {
void SurfaceMesh::Clear() {
    Positions.clear();
    Normals.clear();
    Indices.clear();
}

void SurfaceMesh::Export(std::ostream &out) const {
    IO::Write(out, static_cast<std::uint32_t>(Positions.size()));
    for (auto const &pos : Positions) {
        IO::Write(out, pos.cast<float>().eval());
    }
    for (auto const &normal : Normals) {
        IO::Write(out, normal.cast<float>().eval());
    }
    IO::Write(out, static_cast<std::uint32_t>(Indices.size()));
    IO::Write(out, Indices);
}

void SurfaceMesh::ComputeNormals() {
    Normals.resize(Positions.size());
    std::fill(Normals.begin(), Normals.end(), Vector3d::Zero());

    for (std::size_t i = 0; i < Indices.size(); i += 3) {
        auto const i0 = Indices[i + 0];
        auto const i1 = Indices[i + 1];
        auto const i2 = Indices[i + 2];
        auto const v0 = (Positions[i2] - Positions[i1]).normalized();
        auto const v1 = (Positions[i0] - Positions[i2]).normalized();
        auto const v2 = (Positions[i1] - Positions[i0]).normalized();
        Vector3d const fn = (Positions[i1] - Positions[i0])
                                .cross(Positions[i2] - Positions[i0])
                                .normalized();
        double const a0 = std::acos(-v1.dot(v2));
        double const a1 = std::acos(-v0.dot(v2));
        double const a2 = std::numbers::pi - a0 - a1;
        Normals[i0] += fn * a0;
        Normals[i1] += fn * a1;
        Normals[i2] += fn * a2;
    }

    tbb::parallel_for_each(Normals.begin(), Normals.end(),
                           [&](Vector3d &n) { n.normalize(); });
}

void SurfaceMesh::ComputeAreas() {
    Areas.resize(Positions.size());
    TotalArea = 0.0;
    std::fill(Areas.begin(), Areas.end(), 0.);
    for (std::size_t i = 0; i < Indices.size(); i += 3) {
        auto const i0 = Indices[i + 0];
        auto const i1 = Indices[i + 1];
        auto const i2 = Indices[i + 2];
        auto const v0 = (Positions[i2] - Positions[i1]).normalized();
        auto const v1 = (Positions[i0] - Positions[i2]).normalized();
        auto const v2 = (Positions[i1] - Positions[i0]).normalized();
        double fa = (Positions[i1] - Positions[i0])
                        .cross(Positions[i2] - Positions[i0])
                        .norm() /
                    2;
        TotalArea += fa;
        // double const a0 = std::acos(-v1.dot(v2)) / std::numbers::pi;
        // double const a1 = std::acos(-v0.dot(v2)) / std::numbers::pi;
        // double const a2 = 1 - a0 - a1;
        Areas[i0] += fa / 3;
        Areas[i1] += fa / 3;
        Areas[i2] += fa / 3;
    }
}

void SurfaceMesh::ComputeMeanCurvatures() {
    // ComputeAreas();
    MeanCurvatures.resize(Positions.size());
    std::vector<Vector3d> sum(Positions.size(), Vector3d::Zero());
    for (std::size_t i = 0; i < Indices.size(); i += 3) {
        auto const i0 = Indices[i + 0];
        auto const i1 = Indices[i + 1];
        auto const i2 = Indices[i + 2];
        auto const v0 = (Positions[i2] - Positions[i1]).normalized();
        auto const v1 = (Positions[i0] - Positions[i2]).normalized();
        auto const v2 = (Positions[i1] - Positions[i0]).normalized();
        Vector3d const l0 = Positions[i2] - Positions[i1];
        Vector3d const l1 = Positions[i0] - Positions[i2];
        Vector3d const l2 = Positions[i1] - Positions[i0];
        double cot0 = 1.0 / tan(acos(-v1.dot(v2)));
        double cot1 = 1.0 / tan(acos(-v2.dot(v0)));
        double cot2 = 1.0 / tan(acos(-v0.dot(v1)));
        sum[i0] += l2 * cot2 - l1 * cot1;
        sum[i1] += l0 * cot0 - l2 * cot2;
        sum[i2] += l1 * cot1 - l0 * cot0;
    }

    for (int i = 0; i < Positions.size(); i += 1) {
        MeanCurvatures[i] = sum[i].norm() / (4 * Areas[i]);
    }
}

void SurfaceMesh::ComputeQuadricCurvatures(double lowerThreshold, double upperThreshold) {
    // require compute normal first
    int size = Positions.size();
    MeanCurvatures.resize(size);

    std::vector<std::set<std::uint32_t>> neighbors;
    std::vector<std::set<std::uint32_t>> calNeighbors;
    neighbors.resize(size);
    calNeighbors.resize(size);
    
    for (int i = 0; i < Indices.size(); i += 3) {
        auto const i0 = Indices[i + 0];
        auto const i1 = Indices[i + 1];
        auto const i2 = Indices[i + 2];
        neighbors[i0].insert(i1);
        neighbors[i0].insert(i2);
        neighbors[i1].insert(i0);
        neighbors[i1].insert(i2);
        neighbors[i2].insert(i0);
        neighbors[i2].insert(i1);
    }
    for(int i = 0; i < size; i++){
        for(auto &index: neighbors[i]){
            if((Positions[index] - Positions[i]).norm() < upperThreshold && (Positions[index] - Positions[i]).norm() > lowerThreshold){
                calNeighbors[i].insert(index);
            }
            for(auto &index2: neighbors[index]){
                if((Positions[index2] - Positions[i]).norm() < upperThreshold && (Positions[index2] - Positions[i]).norm() > lowerThreshold){
                    calNeighbors[i].insert(index2);
                }
            }
        }
    }
    for(int i = 0; i < size; i++){
		Matrix<double, 3, 5> QuadricCoeff; // { u, v, u^2-v^2, 2uv, u^2+v^2 }
        // Matrix<double, Dynamic, 5> quadricMat(neighbors[i].size(), 5);
		// Matrix<double, Dynamic, 3> rhs(neighbors[i].size(), 3);
        Matrix<double, Dynamic, 5> quadricMat(calNeighbors[i].size(), 5);
		Matrix<double, Dynamic, 3> rhs(calNeighbors[i].size(), 3);
		CompleteOrthogonalDecomposition<Matrix<double, Dynamic, 5>> QuadricMatDecomp;
        auto const normal = Normals[i];
        Vector3d tx, ty;
        if(normal.x() > 0.5){
            tx = Vector3d::Unit(1).cross(normal).normalized();
            ty = normal.cross(tx).normalized();
        }else{
            tx = Vector3d::Unit(0).cross(normal).normalized();
            ty = normal.cross(tx).normalized();
        }
        int j = 0;
        // for(auto &index: neighbors[i]){
        for(auto &index: calNeighbors[i]){
            double u, v;
            Vector3d dpos = (Positions[index] - Positions[i]);
            u = dpos.dot(tx);
            v = dpos.dot(ty);
            quadricMat.row(j) << u, v, u * u - v * v, 2 * u * v, u * u + v * v;
            rhs.row(j) = dpos.transpose();
            j++;
        }
        QuadricCoeff = QuadricMatDecomp.compute(quadricMat).solve(rhs).transpose();

        double const E = QuadricCoeff.col(0).dot(QuadricCoeff.col(0));
		double const F = QuadricCoeff.col(0).dot(QuadricCoeff.col(1));
		double const G = QuadricCoeff.col(1).dot(QuadricCoeff.col(1));
		Vector3d const n = QuadricCoeff.col(0).cross(QuadricCoeff.col(1)).normalized();
		double const L = (QuadricCoeff.col(4) + QuadricCoeff.col(2)).dot(n) * 2;
		double const M = QuadricCoeff.col(3).dot(n) * 2;
		double const N = (QuadricCoeff.col(4) - QuadricCoeff.col(2)).dot(n) * 2;
		MeanCurvatures[i] = 2 * (2 * M * F - L * G - N * E) / (2 * (E * G - F * F));
    }
}

void SurfaceMesh::SmoothCurvature(double lambda, int iteration) {
    std::vector<double> newCurvatures(size());
    std::vector<double> weightSum(size());

    for (int iter = 0; iter < iteration; iter++) {
        newCurvatures.assign(size(), 0);
        weightSum.assign(size(), 0);
        for (std::size_t i = 0; i < Indices.size(); i += 3) {
            auto const i0 = Indices[i + 0];
            auto const i1 = Indices[i + 1];
            auto const i2 = Indices[i + 2];
            auto const v0 = (Positions[i2] - Positions[i1]).normalized();
            auto const v1 = (Positions[i0] - Positions[i2]).normalized();
            auto const v2 = (Positions[i1] - Positions[i0]).normalized();
            double const a0 = std::acos(-v1.dot(v2));
            double const a1 = std::acos(-v2.dot(v0));
            double const a2 = std::acos(-v0.dot(v1));
            newCurvatures[i0] += MeanCurvatures[i1] * a2;
            newCurvatures[i0] += MeanCurvatures[i2] * a1;
            newCurvatures[i1] += MeanCurvatures[i0] * a2;
            newCurvatures[i1] += MeanCurvatures[i2] * a0;
            newCurvatures[i2] += MeanCurvatures[i0] * a1;
            newCurvatures[i2] += MeanCurvatures[i1] * a0;
            weightSum[i0] += a1 + a2;
            weightSum[i1] += a0 + a2;
            weightSum[i2] += a0 + a1;
        }
        for (int i = 0; i < size(); i++) {
            MeanCurvatures[i] = lambda * (newCurvatures[i] / weightSum[i]) +
                                (1 - lambda) * MeanCurvatures[i];
        }
    }
}

} // namespace Pivot
