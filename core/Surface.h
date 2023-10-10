#pragma once

#include "Common.h"

namespace Pivot {
	class Surface {
	public:
		virtual ~Surface() = default;

		virtual Vector3d ClosestPositionOf(Vector3d const &pos) const { return pos - SignedDistanceTo(pos) * ClosestNormalOf(pos); }
		virtual Vector3d ClosestNormalOf  (Vector3d const &pos) const = 0;
		virtual double   DistanceTo       (Vector3d const &pos) const { return std::abs(SignedDistanceTo(pos)); }
		virtual double   SignedDistanceTo (Vector3d const &pos) const = 0;
		virtual bool     Surrounds        (Vector3d const &pos) const { return SignedDistanceTo(pos) <= 0; }
	};
}
