#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <type_traits>
#include <vector>

namespace ap {
// Large limb arrays and transform buffers use the same Linux huge-page hint.
inline constexpr size_t large_page_bytes = 2u << 20;
inline constexpr size_t arena_unit_bytes = 64u << 10;
void* allocate_large_buffer(size_t bytes);
// Optional process-lifetime cache. Reserve before starting the worker team.
void reserve_buffer_arena(size_t bytes);
void* acquire_work_buffer(size_t bytes);
void release_work_buffer(void* p) noexcept;
// Tag used solely when a multiplication kernel initializes every output limb.
struct OverwriteLimb {};
template <class T> struct LimbAllocator {
    using value_type = T;
    LimbAllocator() = default;
    template <class U> LimbAllocator(const LimbAllocator<U>&) noexcept {}
    [[nodiscard]] T* allocate(size_t n) {
        if (n > SIZE_MAX / sizeof(T))
            throw std::bad_array_new_length();
        size_t bytes = n * sizeof(T);
        if (bytes >= arena_unit_bytes)
            return static_cast<T*>(acquire_work_buffer(bytes));
        void* p = std::malloc(bytes);
        if (!p && bytes)
            throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void construct(T* p, const OverwriteLimb&) const noexcept {
        static_assert(std::is_trivially_default_constructible_v<T>);
        ::new (static_cast<void*>(p)) T;
    }
    void deallocate(T* p, size_t n) noexcept {
        if (n * sizeof(T) >= arena_unit_bytes)
            release_work_buffer(p);
        else
            std::free(p);
    }
    template <class U> bool operator==(const LimbAllocator<U>&) const noexcept {
        return true;
    }
};
using Nat =
    std::vector<uint32_t, LimbAllocator<uint32_t>>; // Little endian, base 10^8. Empty means zero.
constexpr uint32_t base = 100000000;
void trim(Nat& a);
Nat number(uint64_t x);
// Copy a suffix and prepend low zero limbs. An empty suffix represents zero.
Nat copy_limbs(const Nat&, size_t skip, size_t zeros, int threads);
Nat add(Nat a, const Nat& b, int threads = 1);
Nat sub(Nat a, const Nat& b, int threads = 1); // requires a >= b
// Exact a*a*b. Identical input objects share the forward transform for a cube.
Nat mul_square_times(const Nat& a, const Nat& b, int threads);
Nat mul(const Nat& a, const Nat& b, int threads = 1);
// Exact floor of a*b divided by base raised to discard.
Nat mul_high(const Nat& a, const Nat& b, size_t discard, int threads = 1);
// A nonzero period denotes a product modulo base^period - 1.
struct WrappedProduct {
    Nat value;
    size_t period;
};
WrappedProduct wrapped_mul(const Nat& a, const Nat& b, size_t minimum_period, int threads);
// On success each output is its exact product with common, divided by the
// corresponding discard power of base and rounded down.
// A false result lets the caller select independent multiplication plans.
bool shared_products(const Nat& common, const Nat& a, const Nat& b, Nat& x, Nat& y, int threads,
                     size_t first_discard = 0, size_t second_discard = 0);
} // namespace ap
