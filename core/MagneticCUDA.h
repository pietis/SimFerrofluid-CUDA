#pragma once

namespace Pivot {
void SolveMagneticCUDA(const void *positions, const void *normals,
                       const void *areas, const void *Hext, void *pressures,
                       const int size, const int num_iter, const double lambda,
                       const double chi, const double epsilon);
} // namespace Pivot