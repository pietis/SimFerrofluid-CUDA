#include "StopWatch.h"

namespace Pivot {
	void StopWatch::PrintStats(ofstream &fout) {
		fmt::print(fmt::fg(fmt::color::yellow_green), "[Statistics]\n");
		fout << fmt::format("[Statistics]") << std::endl;
		for (std::size_t i = 0; i < s_Counts.size(); i++) {
			fmt::print("{:>3}: {:>15}, {:>12} times,     average = {:.3f}s\n", i + 1, s_Names[i], s_Counts[i], s_AvgTime[i]);
			fout << fmt::format("{:>3}: {:>15}, {:>12} times,     average = {:.3f}s", i + 1, s_Names[i], s_Counts[i], s_AvgTime[i]) << std::endl;
		}
	}
}
