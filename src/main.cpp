#include <iostream>

#include "FIRFilter.h"
#include "benchmark.h"
#include "test.h"

using namespace fir;

void test() {
  std::cout << "#------------- FIR filter single --------------------#"
            << std::endl;
  testFirFilterBigRandomVectors<alignof(float)>(applyFirFilterInnerLoopVectorization);

  std::cout
      << "#------------- FIR filter AVX --------------------#" << std::endl
      << "#------------- Inner Loop Vectorization --------------------#"
      << std::endl;
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterInnerLoopVectorization);
  testFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterAVX_innerLoopVectorization);
}

void benchmark() {
  std::cout << "#------------- FIR filter single --------------------#"
            << std::endl;
  benchmarkFirFilterBigRandomVectors<alignof(float)>(applyFirFilterSingle);

  std::cout
      << "#------------- FIR filter AVX --------------------#" << std::endl
      << "#------------- Inner Loop Vectorization --------------------#"
      << std::endl;
  benchmarkFirFilterBigRandomVectors<alignof(float)>(
      applyFirFilterAVX_innerLoopVectorization);
}

int main() {
  test();
  benchmark();
  std::cout << "Success!" << std::endl;
}
