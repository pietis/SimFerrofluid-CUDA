#pragma once

#include <iostream>

//#include "fmmtl/dispatch/S2T/S2T_Compressed.hpp"

template <typename _Kernel_Ty_>
S2T_Compressed<_Kernel_Ty_>::S2T_Compressed() {
}

template <typename _Kernel_Ty_>
S2T_Compressed<_Kernel_Ty_>::S2T_Compressed(
    std::vector<std::pair<unsigned,unsigned> >&,
    std::vector<unsigned>&,
    std::vector<std::pair<unsigned,unsigned> >&,
    const std::vector<source_type>&,
    const std::vector<target_type>&) {
}

template <typename _Kernel_Ty_>
S2T_Compressed<_Kernel_Ty_>::~S2T_Compressed() {
  // TODO: delete
}

template <typename _Kernel_Ty_>
void S2T_Compressed<_Kernel_Ty_>::execute(
    const _Kernel_Ty_&,
    const std::vector<charge_type>&,
    std::vector<result_type>&) {
  std::cerr << "ERROR: Calling unimplemented S2T_compressed CPU" << std::endl;
}

template <typename _Kernel_Ty_>
void S2T_Compressed<_Kernel_Ty_>::execute(
    const _Kernel_Ty_&,
    const std::vector<source_type>&,
    const std::vector<charge_type>&,
    const std::vector<target_type>&,
    std::vector<result_type>&) {
  std::cerr << "ERROR: Calling unimplemented S2T_compressed CPU" << std::endl;
}
