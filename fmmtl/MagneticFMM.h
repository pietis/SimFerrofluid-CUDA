#pragma once

#include <vector>

void SolveMagneticFMM(const void *positions, const void *normals,
                      const void *areas, const void *Hext, void *pressures,
                      const int size, const int num_iter, const double lambda,
                      const double chi, const double epsilon, bool trunc);

void CacheMagneticFMM(const void *positions, const void *normals,
                      const void *areas, const void *Hext, const void *charges,
                      const int size, const int num_iter, const double lambda,
                      const double chi, const double epsilon);
void ApplyCacheFMM(const void *sources, const void *targets,
                   const int source_size, const int target_size,
                   const double epsilon, std::vector<double> &charges,
                   const void *Hind);