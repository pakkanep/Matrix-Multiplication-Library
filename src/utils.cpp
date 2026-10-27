#include <cstdint>
#include <iostream>

uint32_t pcg_hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

double random_double(uint32_t seed)
{
    return static_cast<double>(pcg_hash(seed)) / static_cast<double>(UINT32_MAX);
}

void detectSimd()
{
    #if defined(__x86_64__) || defined(__i386__)
        __builtin_cpu_init();

        if (__builtin_cpu_supports("avx512f")) {
            std::cout << "AVX-512: 8 doubles per vector\n";
        } else if (__builtin_cpu_supports("avx2")) {
            std::cout << "AVX2: 4 doubles per vector\n";
        } else if (__builtin_cpu_supports("sse2")) {
            std::cout << "SSE2: 2 doubles per vector\n";
        } else {
            std::cout << "Scalar implementation\n";
        }
    #endif
}