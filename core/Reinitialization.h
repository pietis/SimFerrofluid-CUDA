#pragma once

#include "GridData.h"

namespace Pivot {
	class Reinitialization {
	private:
		using HeapElement = std::pair<double, int>;
		using Heap = std::priority_queue<HeapElement, std::vector<HeapElement>, std::greater<HeapElement>>;

	public:
		static void Solve(GridData<double> &phi, int maxSteps);
	
	private:
		static void   UpdateNeighbors     (Vector3i const &coord, GridData<std::int8_t> const &visited, GridData<double>       &tent, Heap &heap);
		static double SolveEikonalEquation(Vector3i const &coord, GridData<std::int8_t> const &visited, GridData<double> const &tent);
	};
}
