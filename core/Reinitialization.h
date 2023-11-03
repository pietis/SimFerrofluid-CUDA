#pragma once

#include "FastMarching.h"

namespace Pivot {
	class Reinitialization {
	public:
		static void Solve(GridData<double> &phi, int maxSteps);
	
	private:
		static void Advance(GridData<double> const &field, GridData<double> &newField, double dt);
	};
}
