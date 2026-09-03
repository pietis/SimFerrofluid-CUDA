#pragma once

#include "../grid/GridDesc.h"

namespace Pivot::Cuda {

struct PlaneScene {
    int scale;
    int boundary_width;
    GridDesc cell_grid;
    GridDesc face_grids[3];
    Double3 allocated_min;
    Double3 allocated_max;
    Double3 interior_min;
    Double3 interior_max;
    double liquid_surface_y;
    double initial_time;
    Double3 initial_velocity;
    double density;
    double surface_tension;
    Double3 gravity;
    double damping;
    Double3 applied_field;
    double susceptibility;
    double lambda;
    double iob_epsilon_factor;
    int iob_iterations;
};

PlaneScene MakePlaneScene(int scale);

}  // namespace Pivot::Cuda
