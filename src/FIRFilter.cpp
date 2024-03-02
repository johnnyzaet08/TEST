#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <new>
#include <numeric>
#include <vector>

#ifdef __AVX__
#include <immintrin.h>
#endif

#include "FIRFilter.h"

namespace {
bool is_aligned(const void* p, std::size_t n) {
  std::cout << reinterpret_cast<std::uintptr_t>(p) % n << std::endl;
}
}  // namespace

namespace fir {
std::vector<float> applyFirFilterSingle(FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;
  auto* y = input.y;

  for (auto i = 0u; i < input.outputLength; ++i) {
    y[i] = x[i] * c[0];
    for (auto j = 1u; j < input.filterLength; ++j) {
      y[i] += x[i + j] * c[j];
    }
  }
  return input.output();
}

std::vector<float> applyFirFilterInnerLoopVectorization(
    FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;
  auto* y = input.y;

  for (auto i = 0u; i < input.outputLength; ++i) {
    y[i] = 0.f;
    for (auto j = 0u; j < input.filterLength; j += 4) {
      y[i] += x[i + j] * c[j] + x[i + j + 1] * c[j + 1] +
              x[i + j + 2] * c[j + 2] + x[i + j + 3] * c[j + 3];
    }
  }
  return input.output();
}


#ifdef __AVX__
std::vector<float> applyFirFilterAVX_innerLoopVectorization(
    FilterInput<float>& input) {
  const auto* x = input.x;
  const auto* c = input.c;

  std::array<float, AVX_FLOAT_COUNT> outStore;

  for (auto i = 0u; i < input.outputLength; ++i) {
    auto outChunk = _mm256_setzero_ps();

    for (auto j = 0u; j < input.filterLength; j += AVX_FLOAT_COUNT) {
      auto xChunk = _mm256_loadu_ps(x + i + j);
      auto cChunk = _mm256_loadu_ps(c + j);

      auto temp = _mm256_mul_ps(xChunk, cChunk);

      outChunk = _mm256_add_ps(outChunk, temp);
    }

    _mm256_storeu_ps(outStore.data(), outChunk);

    input.y[i] = std::accumulate(outStore.begin(), outStore.end(), 0.f);
  }

  return input.output();
}
#endif

std::vector<float> applyFirFilter(FilterInput<float>& input) {
#ifdef __AVX__
  std::cout << "Using AVX instructions." << std::endl;
  return applyFirFilterAVX_innerLoopVectorization(input);
#else
  std::cout << "Using single instructions." << std::endl;
  return applyFirFilterSingle(input);
#endif
}
}  // namespace fir
