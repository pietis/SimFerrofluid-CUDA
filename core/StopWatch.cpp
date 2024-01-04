#include "StopWatch.h"

namespace Pivot {
	void StopWatch::PrintStats() {
		fmt::print(fmt::fg(fmt::color::yellow_green), "[Statistics]\n");
		for (std::size_t i = 0; i < s_Counts.size(); i++) {
			fmt::print("{:>3}: {:>15}, {:>12} times,     average = {:.3f}s,     maximum = {:.3f}s\n", i + 1, s_Names[i], s_Counts[i], s_AvgTime[i], s_MaxTime[i]);
		}
	}

	void StopWatch::PrintStats(std::filesystem::path const &filename) {
		std::ofstream fout(filename);
		for (std::size_t i = 0; i < s_Counts.size(); i++) {
			fout << fmt::format("{:>3}: {:>15}, {:>12} times,     average = {:.3f}s,     maximum = {:.3f}s", i + 1, s_Names[i], s_Counts[i], s_AvgTime[i], s_MaxTime[i]) << std::endl;
		}
	}
}
