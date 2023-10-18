#pragma once

#include "Collider.h"

namespace Pivot {
	class SISurfaceTension {
	public:
		static void Solve(
			SGridData<double>       &velocity,
			GridData<double>  const &levelSet,
			Collider          const &collider,
			double                   coeff,
			double                   dt);
	};
}
