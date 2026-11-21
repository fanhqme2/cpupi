#include "integer.hpp"
#include "ntt.hpp"
#include "parallel.hpp"
#include <bit>
#include <numbers>
#include <cmath>
#include <charconv>
#include <condition_variable>
#include <cstdlib>
#include <immintrin.h>
#include <iterator>
#include <new>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <span>
#include <cstring>
#include <unistd.h>
#include <sys/mman.h>

namespace ap {
using U = uint64_t;
using Wide = __uint128_t;
void* allocate_large_buffer(size_t bytes) {
    constexpr size_t page = large_page_bytes;
    if (bytes > SIZE_MAX - (page - 1))
        throw std::bad_alloc();
    void* data = std::aligned_alloc(page, (bytes + page - 1) / page * page);
    if (!data) {
        data = std::aligned_alloc(64, (bytes + 63) / 64 * 64);
        if (!data)
            throw std::bad_alloc();
    } else {
        madvise(data, bytes, MADV_HUGEPAGE);
    }
    return data;
}
namespace {
// The two boundary entries of a run hold its length in 64 KiB units.
// Positive runs are free. Negative runs belong to a limb array or transform.
// This lets release coalesce adjacent runs without allocating metadata.
class BufferArena {
    static constexpr int max_pages = (32ULL << 30) / arena_unit_bytes;
    std::array<int, max_pages> lengths{};
    char* data = nullptr;
    int pages = 0;
    std::mutex lock;

    void mark(int first, int size, bool free) {
        lengths[first] = lengths[first + size - 1] = free ? size : -size;
    }

  public:
    void reserve(size_t bytes) {
        const int count = std::min<size_t>(max_pages, bytes / arena_unit_bytes);
        if (!count || data)
            return;
        const size_t size = size_t(count) * arena_unit_bytes;
        const size_t mapping_size = size + large_page_bytes;
        auto* mapping = static_cast<char*>(mmap(nullptr, mapping_size, PROT_READ | PROT_WRITE,
                                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (mapping == MAP_FAILED)
            return;
        size_t prefix = -reinterpret_cast<uintptr_t>(mapping) % large_page_bytes;
        data = mapping + prefix;
        if (prefix)
            munmap(mapping, prefix);
        munmap(data + size, large_page_bytes - prefix);
        madvise(data, size, MADV_HUGEPAGE);
        pages = count;
        mark(0, pages, true);
    }

    void* take(size_t bytes) {
        if (!data || bytes < arena_unit_bytes || bytes > size_t(pages) * arena_unit_bytes)
            return nullptr;
        const int count = 1 + (bytes - 1) / arena_unit_bytes;
        std::lock_guard guard(lock);
        for (int i = 0; i < pages; i += std::abs(lengths[i])) {
            int size = lengths[i];
            if (size < count)
                continue;
            if (size > count)
                mark(i + count, size - count, true);
            mark(i, count, false);
            return data + size_t(i) * arena_unit_bytes;
        }
        return nullptr;
    }

    bool release(void* p) {
        uintptr_t offset = reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(data);
        if (!data || offset >= size_t(pages) * arena_unit_bytes)
            return false;
        std::lock_guard guard(lock);
        int begin = offset / arena_unit_bytes;
        int end = begin - lengths[begin];
        if (begin && lengths[begin - 1] > 0)
            begin -= lengths[begin - 1];
        if (end < pages && lengths[end] > 0)
            end += lengths[end];
        mark(begin, end - begin, true);
        return true;
    }

    void reclaim() {
        // Under memory pressure, return all inactive pages to Linux before
        // the workspace planner decides whether it needs a tiled product.
        if (!data)
            return;
        std::lock_guard guard(lock);
        for (int i = 0; i < pages; i += std::abs(lengths[i])) {
            if (lengths[i] > 0)
                madvise(data + size_t(i) * arena_unit_bytes, size_t(lengths[i]) * arena_unit_bytes,
                        MADV_DONTNEED);
        }
    }
};
BufferArena buffer_arena;
} // namespace

void reserve_buffer_arena(size_t bytes) {
    buffer_arena.reserve(bytes);
}
void* acquire_work_buffer(size_t bytes) {
    if (void* p = buffer_arena.take(bytes))
        return p;
    if (bytes >= large_page_bytes)
        return allocate_large_buffer(bytes);
    void* data = std::malloc(bytes);
    if (!data && bytes)
        throw std::bad_alloc();
    return data;
}
void release_work_buffer(void* p) noexcept {
    if (!buffer_arena.release(p))
        std::free(p);
}
void trim(Nat& a) {
    while (!a.empty() && !a.back())
        a.pop_back();
}
Nat number(U n) {
    Nat a;
    while (n) {
        a.push_back(n % base);
        n /= base;
    }
    return a;
}
namespace {
// A forward range lets vector allocate exactly once. Constructing a limb
// from the tag starts its lifetime without first writing a zero to its page.
struct OverwriteIterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = OverwriteLimb;
    using difference_type = std::ptrdiff_t;
    using pointer = const OverwriteLimb*;
    using reference = const OverwriteLimb&;
    static constexpr OverwriteLimb tag{};
    size_t index;
    reference operator*() const { return tag; }
    pointer operator->() const { return &tag; }
    OverwriteIterator& operator++() {
        ++index;
        return *this;
    }
    OverwriteIterator operator++(int) {
        auto previous = *this;
        ++*this;
        return previous;
    }
    bool operator==(const OverwriteIterator&) const = default;
};
Nat overwrite_limbs(size_t size) {
    return Nat(OverwriteIterator{0}, OverwriteIterator{size});
}
// A generate bit starts a carry/borrow. Consecutive propagate bits carry it
// across adjacent lanes. One scalar mask addition resolves the whole vector.
template <bool subtract> Nat combine(Nat a, const Nat& b, int threads) {
    if (a.size() < b.size()) {
        if constexpr (subtract)
            throw std::logic_error("negative difference");
        a.resize(b.size());
    }
    auto range = [&](size_t begin, size_t end) {
        unsigned carry = 0;
        size_t i = begin;
        const auto radix = _mm512_set1_epi32(base);
        const auto one = _mm512_set1_epi32(1);
        for (; i + 16 <= end; i += 16) {
            auto x = _mm512_loadu_si512(a.data() + i), y = _mm512_loadu_si512(b.data() + i);
            auto v = subtract ? _mm512_sub_epi32(x, y) : _mm512_add_epi32(x, y);
            unsigned generate =
                subtract ? _mm512_cmplt_epi32_mask(x, y) : _mm512_cmpge_epi32_mask(v, radix);
            unsigned propagate = subtract
                                     ? _mm512_cmpeq_epi32_mask(x, y)
                                     : _mm512_cmpeq_epi32_mask(v, _mm512_sub_epi32(radix, one));
            unsigned bits = (propagate + 2 * generate + carry) ^ propagate;
            carry = bits >> 16;
            if constexpr (subtract) {
                v = _mm512_mask_sub_epi32(v, bits, v, one);
                v = _mm512_mask_add_epi32(v, bits >> 1, v, radix);
            } else {
                v = _mm512_mask_add_epi32(v, bits, v, one);
                v = _mm512_mask_sub_epi32(v, bits >> 1, v, radix);
            }
            _mm512_storeu_si512(a.data() + i, v);
        }
        for (; i < end; ++i) {
            unsigned rhs = carry + b[i];
            if constexpr (subtract) {
                carry = a[i] < rhs;
                a[i] = a[i] + carry * base - rhs;
            } else {
                unsigned sum = a[i] + rhs;
                carry = sum >= base;
                a[i] = sum - carry * base;
            }
        }
        return carry;
    };
    auto ripple = [&](unsigned carry, size_t begin, size_t end) {
        for (size_t i = begin; carry && i < end; ++i) {
            if constexpr (subtract) {
                carry = a[i] == 0;
                a[i] = carry ? base - 1 : a[i] - 1;
            } else {
                carry = a[i] == base - 1;
                a[i] = carry ? 0 : a[i] + 1;
            }
        }
        return carry;
    };
    unsigned carry = 0;
    constexpr size_t block = 1 << 18;
    if (threads > 1 && b.size() > block && Team::active()) {
        size_t count = (b.size() + block - 1) / block;
        std::vector<unsigned> carries(count);
        blocks(count, threads, [&](size_t j) {
            carries[j] = range(j * block, std::min((j + 1) * block, b.size()));
        });
        // Each block assumed a zero incoming carry. Apply the carry from
        // its left neighbor, including propagation through an entire block.
        for (size_t j = 0; j < count; ++j)
            carry = carries[j] | ripple(carry, j * block, std::min((j + 1) * block, b.size()));
    } else
        carry = range(0, b.size());
    carry = ripple(carry, b.size(), a.size());
    if constexpr (subtract) {
        if (carry)
            throw std::logic_error("negative difference");
        trim(a);
    } else if (carry)
        a.push_back(1);
    return a;
}
} // namespace
Nat copy_limbs(const Nat& a, size_t skip, size_t zeros, int threads) {
    if (skip >= a.size())
        return {};
    size_t n = a.size() - skip;
    if (zeros > a.max_size() - n)
        throw std::length_error("limb shift exceeds allocation limit");
    Nat out = overwrite_limbs(n + zeros);
    constexpr size_t chunk = 1 << 18;
    auto copy = [&](size_t i) {
        size_t start = i * chunk, end = std::min(out.size(), start + chunk);
        if (start < zeros)
            std::fill(out.begin() + start, out.begin() + std::min(end, zeros), 0);
        start = std::max(start, zeros);
        if (start < end)
            std::copy_n(a.data() + skip + (start - zeros), end - start, out.data() + start);
    };
    blocks((out.size() + chunk - 1) / chunk, Team::active() ? threads : 1, copy);
    return out;
}
Nat add(Nat a, const Nat& b, int threads) { return combine<false>(std::move(a), b, threads); }
Nat sub(Nat a, const Nat& b, int threads) { return combine<true>(std::move(a), b, threads); }

namespace {
constexpr U prime0 = 2246998040248321, prime1 = 2246482644172801, prime2 = 2242939296153601,
            R = U(1) << 52;
using ntt::Field;
struct Buffer {
    U* data;
    explicit Buffer(size_t n) {
        size_t bytes = n * sizeof(U);
        data = bytes >= arena_unit_bytes ? static_cast<U*>(acquire_work_buffer(bytes))
                                         : static_cast<U*>(std::malloc(bytes));
        if (!data && n)
            throw std::bad_alloc();
    }
    ~Buffer() {
        release();
    }
    void release() {
        release_work_buffer(data);
        data = nullptr;
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
#include "short_fft.hpp"
// Normalize eight independent convolution coefficients. A rounded floating
// quotient gives a bounded remainder even for the largest Karatsuba leaf.
// The second quotient and the final one-bit carry are exact integer steps.
struct CoefficientCarry {
    __m512i high{}, extra{};
    unsigned carry = 0;
    template <bool signed_input = false> __m256i take(__m512i v) {
        auto splat = [](U x) { return _mm512_set1_epi64(x); };
        constexpr U mask = (U(1) << 52) - 1;
        const auto value = signed_input ? _mm512_cvtepi64_pd(v) : _mm512_cvtepu64_pd(v);
        const auto estimate = _mm512_mul_pd(value, _mm512_set1_pd(1.0 / base));
        // Deliberately subtract one. Even a rounding error at a half
        // integer leaves r between 0.49B and 1.51B. Masking off the
        // high bits therefore recovers its exact positive value.
        auto q = _mm512_cvt_roundpd_epi64(estimate, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        q -= splat(1);
        auto r = _mm512_madd52lo_epu64(v, q, splat((U(1) << 52) - base)) & splat(mask);
        r += _mm512_alignr_epi64(q, high, 7);
        high = q;
        // r < 2^38. Division by B = 256 * 390625 needs one IFMA.
        constexpr U reciprocal = ((U(1) << 52) + 390624) / 390625;
        auto q2 = _mm512_madd52hi_epu64(_mm512_setzero_si512(), _mm512_srli_epi64(r, 8),
                                       splat(reciprocal));
        r -= _mm512_mul_epu32(q2, splat(base));
        r += _mm512_alignr_epi64(q2, extra, 7);
        extra = q2;
        unsigned generates = _mm512_cmpge_epu64_mask(r, splat(base));
        unsigned propagates = _mm512_cmpeq_epi64_mask(r, splat(base - 1));
        unsigned carries = (propagates + (2 * generates + carry)) ^ propagates;
        r = _mm512_mask_add_epi64(r, carries, r, splat(1));
        r = _mm512_mask_sub_epi64(r, carries >> 1, r, splat(base));
        carry = carries >> 8;
        return _mm512_cvtepi64_epi32(r);
    }
};
Nat school(const Nat& a, const Nat& b) {
    if (b.size() <= 64 && a.size() >= 32) {
        Nat out = overwrite_limbs(a.size() + b.size());
        CoefficientCarry carry;
        for (size_t j = 0; j < out.size(); j += 32) {
            alignas(64) uint32_t edge[128];
            const uint32_t* row;
            size_t first = j >= a.size() ? j + 1 - a.size() : 0;
            size_t end = std::min(b.size(), j + 32);
            if (j >= end - 1 && j + 32 <= a.size())
                row = a.data() + j;
            else {
                std::fill_n(edge, 128, 0);
                size_t lo = j >= end ? j - end + 1 : 0;
                size_t hi = std::min(j + 32, a.size());
                std::copy(a.begin() + lo, a.begin() + hi, edge + (64 + lo - j));
                row = edge + 64;
            }
            __m512i sums[4]{};
            for (size_t i = first; i < end; ++i) {
                auto x = _mm512_set1_epi64(b[i]);
                for (unsigned t = 0; t < 4; ++t) {
                    auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row - i + 8 * t));
                    sums[t] += _mm512_mul_epu32(x, _mm512_cvtepu32_epi64(v));
                }
            }
            for (unsigned t = 0; t < 4 && j + 8 * t < out.size(); ++t) {
                size_t count = std::min<size_t>(8, out.size() - j - 8 * t);
                _mm256_mask_storeu_epi32(out.data() + j + 8 * t, (1u << count) - 1,
                                         carry.take(sums[t]));
            }
        }
        trim(out);
        return out;
    }
    std::vector<U> sums(a.size() + b.size());
    for (size_t i = 0; i < b.size(); ++i)
        for (size_t j = 0; j < a.size(); ++j)
            sums[i + j] += U(b[i]) * a[j];
    Nat c(sums.size());
    U carry = 0;
    for (size_t i = 0; i < c.size(); ++i) {
        carry += sums[i];
        c[i] = carry % base;
        carry /= base;
    }
    trim(c);
    return c;
}
// Coefficients are accumulated modulo 2^64. Only the final convolution
// needs to fit in a word; intermediate Karatsuba sums may wrap.
void karatsuba(const uint32_t* a, const uint32_t* b, U* c, size_t n, unsigned depth, uint32_t* sums,
               U* middle) {
    if (!depth) {
        // Each SIMD lane accumulates one output diagonal. Padding lets the
        // same inner loop cover the rising and falling ends of the product.
        alignas(64) uint32_t padded[160]{};
        constexpr size_t padding = 32;
        std::copy_n(b, n, padded + padding);
        for (size_t j = 0; j < 2 * n; j += 32) {
            __m512i v0{}, v1{}, v2{}, v3{};
            size_t first = j >= n ? j - n + 1 : 0;
            size_t end = std::min(n, j + 32);
            for (size_t i = first; i < end; ++i) {
                auto x = _mm512_set1_epi64(a[i]);
                const uint32_t* y = padded + (padding + j - i);
                auto term = [&](size_t k) {
                    auto z = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(y + k));
                    return _mm512_mul_epu32(x, _mm512_cvtepu32_epi64(z));
                };
                v0 += term(0);
                v1 += term(8);
                v2 += term(16);
                v3 += term(24);
            }
            auto save = [&](size_t k, __m512i v) {
                if (j + k < 2 * n) {
                    size_t count = std::min<size_t>(8, 2 * n - j - k);
                    _mm512_mask_storeu_epi64(c + j + k, (1u << count) - 1, v);
                }
            };
            save(0, v0);
            save(8, v1);
            save(16, v2);
            save(24, v3);
        }
        return;
    }
    size_t h = n / 2;
    for (size_t i = 0; i < h; ++i) {
        sums[i] = a[i] + a[h + i];
        sums[h + i] = b[i] + b[h + i];
    }
    karatsuba(a, b, c, h, depth - 1, sums + n, middle + n);
    karatsuba(a + h, b + h, c + n, h, depth - 1, sums + n, middle + n);
    karatsuba(sums, sums + h, middle, h, depth - 1, sums + n, middle + n);
    for (size_t i = 0; i < n; ++i)
        middle[i] -= c[i] + c[n + i];
    for (size_t i = 0; i < n; ++i)
        c[h + i] += middle[i];
}
Nat medium(const Nat& a, const Nat& b) {
    unsigned depth = std::bit_width(a.size() / 32) - 1;
    size_t block = size_t(1) << depth;
    size_t n = (a.size() + block - 1) / block * block;
    thread_local std::vector<uint32_t> inputs;
    thread_local std::vector<U> scratch;
    inputs.resize(4 * n);
    scratch.resize(4 * n);
    auto x = inputs.data(), y = x + n;
    std::copy(a.begin(), a.end(), x);
    std::fill(x + a.size(), x + n, 0);
    std::copy(b.begin(), b.end(), y);
    std::fill(y + b.size(), y + n, 0);
    karatsuba(x, y, scratch.data(), n, depth, y + n, scratch.data() + 2 * n);
    Nat out = overwrite_limbs(a.size() + b.size());
    CoefficientCarry carry;
    for (size_t i = 0; i < out.size(); i += 8) {
        unsigned mask = (1u << std::min<size_t>(8, out.size() - i)) - 1;
        auto words = _mm512_maskz_loadu_epi64(mask, scratch.data() + i);
        _mm256_mask_storeu_epi32(out.data() + i, mask, carry.take(words));
    }
    trim(out);
    return out;
}
template <int digits> size_t extent(size_t limbs) {
    return (limbs * 8 + digits - 1) / digits;
}
template <int digits> U coefficient(const Nat& a, size_t j) {
    if constexpr (digits == 5) {
        size_t i = j * 5 / 8;
        U next = i + 1 < a.size() ? a[i + 1] : 0;
        switch (j % 8) {
        case 0:
            return a[i] % 100000;
        case 1:
            return a[i] / 100000 + (next % 100) * 1000;
        case 2:
            return a[i] / 100 % 100000;
        case 3:
            return a[i] / 10000000 + (next % 10000) * 10;
        case 4:
            return a[i] / 10000 + (next % 10) * 10000;
        case 5:
            return a[i] / 10 % 100000;
        case 6:
            return a[i] / 1000000 + (next % 1000) * 100;
        case 7:
            return a[i] / 1000;
        }
    }
    if constexpr (digits == 4)
        return j % 2 ? a[j / 2] / 10000 : a[j / 2] % 10000;
    if constexpr (digits == 6) {
        size_t i = j * 3 / 4;
        U next = i + 1 < a.size() ? a[i + 1] : 0;
        switch (j % 4) {
        case 0:
            return a[i] % 1000000;
        case 1:
            return a[i] / 1000000 + (next % 10000) * 100;
        case 2:
            return a[i] / 10000 + (next % 100) * 10000;
        default:
            return a[i] / 100;
        }
    }
    if constexpr (digits == 10) {
        size_t i = j / 4 * 5 + j % 4;
        U next = i + 1 < a.size() ? a[i + 1] : 0;
        switch (j % 4) {
        case 0:
            return a[i] + (next % 100) * base;
        case 1:
            return a[i] / 100 + (next % 10000) * 1000000;
        case 2:
            return a[i] / 10000 + (next % 1000000) * 10000;
        default:
            return a[i] / 1000000 + next * 100;
        }
    }
    if constexpr (digits == 16)
        return a[2 * j] + (2 * j + 1 < a.size() ? U(a[2 * j + 1]) * base : 0);
    // Three 8-digit words occupy two 12-digit coefficients.
    size_t i = (j / 2) * 3;
    U mid = i + 1 < a.size() ? a[i + 1] : 0;
    if (j % 2 == 0)
        return a[i] + (mid % 10000) * base;
    return mid / 10000 + (i + 2 < a.size() ? U(a[i + 2]) * 10000 : 0);
}
template <int digits, U prime, bool pair = false>
void residues(U* x, U* y, size_t n, const Nat& a, const Nat& b, int workers,
              const Nat* other = nullptr, U* z = nullptr) {
    using F = Field<prime>;
    bool square = !pair && &a == &b;
    const auto& plan = F::plan_for(n);
    size_t nx = extent<digits>(a.size()), ny = extent<digits>(b.size());
    size_t nz = pair ? extent<digits>(other->size()) : 0;
    auto unpack = [&](U* dest, size_t pos, size_t count, int input) {
        const Nat& src = input == 0 ? a : input == 1 ? b : *other;
        size_t length = input == 0 ? nx : input == 1 ? ny : nz;
        size_t valid = pos < length ? std::min(count, length - pos) : 0;
        size_t j = 0;
        // Each source limb is below 10^8, so this rounded-up reciprocal divides exactly.
        constexpr U div10000 = ((U(1) << 45) + 9999) / 10000;
        if constexpr (digits == 12) {
            auto middle = _mm256_setr_epi32(1, 1, 4, 4, 7, 7, 10, 10);
            auto ends = _mm256_setr_epi32(0, 2, 3, 5, 6, 8, 9, 11);
            auto scale = _mm512_setr_epi64(base, 10000, base, 10000, base, 10000, base, 10000);
            // Calls start on stripe boundaries, hence on an even coefficient.
            for (; j + 8 <= valid && ((pos + j) / 2) * 3 + 12 <= src.size(); j += 8) {
                auto words = _mm512_maskz_loadu_epi32(0xfff, src.data() + (pos + j) / 2 * 3);
                auto mid = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(
                    _mm512_permutexvar_epi32(_mm512_castsi256_si512(middle), words)));
                auto edge = _mm512_cvtepu32_epi64(_mm512_castsi512_si256(
                    _mm512_permutexvar_epi32(_mm512_castsi256_si512(ends), words)));
                auto q = _mm512_srli_epi64(_mm512_mul_epu32(mid, F::splat(div10000)), 45);
                auto r = mid - _mm512_mul_epu32(q, F::splat(10000));
                F::save(dest + j,
                        _mm512_madd52lo_epu64(_mm512_mask_blend_epi64(0xaa, edge, q),
                                              _mm512_mask_blend_epi64(0xaa, r, edge), scale));
            }
        } else if constexpr (digits == 4) {
            for (; j + 16 <= valid; j += 16) {
                auto v = _mm512_cvtepu32_epi64(_mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(src.data() + (pos + j) / 2)));
                auto hi = _mm512_srli_epi64(_mm512_mul_epu32(v, F::splat(div10000)), 45);
                auto lo = v - _mm512_mul_epu32(hi, F::splat(10000));
                F::save(dest + j, _mm512_permutex2var_epi64(
                                      lo, _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), hi));
                F::save(dest + j + 8, _mm512_permutex2var_epi64(
                                          lo, _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15), hi));
            }
        }
        for (; j < valid; ++j) {
            U v = coefficient<digits>(src, pos + j);
            dest[j] = digits == 16 ? v % prime : v;
        }
        std::fill(dest + valid, dest + count, 0);
    };
    U scale = plan.inverse_scale;
    F::template convolution<pair>(x, y, z, plan, workers, square, nx, ny, nz, scale, unpack);
}

size_t transform_length(size_t needed) {
    size_t n = std::bit_ceil(needed);
    if (n >= 1024 && needed <= 5 * (n / 8))
        return 5 * (n / 8);
    return needed <= 3 * (n / 4) && n >= 256 ? 3 * (n / 4) : n;
}
// Reserve the largest temporary arrays before first-touch. If concurrent
// transforms would exhaust available RAM, split a product along its long
// operand and accumulate the overlapping partial products in memory.
class Workspace {
    static inline std::mutex lock;
    static inline std::condition_variable released;
    static inline size_t outstanding = 0;
    size_t held = 0;
    static size_t limit() {
        const char* setting = std::getenv("PI_TRANSFORM_MIB");
        if (!setting)
            return SIZE_MAX;
        size_t mib = 0;
        std::string_view value(setting);
        auto [end, ec] = std::from_chars(value.begin(), value.end(), mib);
        if (ec != std::errc{} || end != value.end() || !mib || mib > (SIZE_MAX >> 20))
            throw std::invalid_argument("PI_TRANSFORM_MIB must be a positive integer");
        return mib << 20;
    }

  public:
    size_t tile_budget = 256ULL << 20;
    bool acquire(size_t bytes, int workers, bool wait = true) {
        static const size_t configured = limit();
        // Concurrent subtrees own disjoint worker ranges. Assign them
        // corresponding shares of an explicit buffer budget.
        if (configured != SIZE_MAX) {
            size_t count = Team::active() ? Team::active()->size() : 1;
            size_t share = Wide(configured) * std::min<size_t>(workers, count) / count;
            if (bytes > share) {
                tile_budget = std::max<size_t>(4096, share - share / 8);
                return false;
            }
        }
        if (bytes < (128ULL << 20))
            return true;
        std::unique_lock guard(lock);
        for (;;) {
            size_t free = available_memory_bytes(), budget = free - free / 16;
            if (bytes > budget - std::min(budget, outstanding)) {
                buffer_arena.reclaim();
                free = available_memory_bytes();
                budget = free - free / 16;
            }
            if (bytes <= budget - std::min(budget, outstanding)) {
                outstanding += bytes;
                held = bytes;
                return true;
            }
            if (!outstanding) {
                tile_budget = std::max<size_t>(4096, budget - budget / 4);
                return false;
            }
            if (!wait)
                return false;
            released.wait(guard);
        }
    }
    ~Workspace() {
        if (held) {
            std::lock_guard guard(lock);
            outstanding -= held;
            released.notify_all();
        }
    }
};
int packing(size_t large, size_t small);
size_t scratch_bytes(size_t a, size_t b) {
    int digits = packing(std::max(a, b), std::min(a, b));
    size_t coefficients = (a * 8 + digits - 1) / digits + (b * 8 + digits - 1) / digits;
    size_t buffers = digits >= 24 ? size_t(digits / 8 + 2) : digits <= 6 ? 2 : digits == 16 ? 4 : 3;
    return transform_length(coefficients) * sizeof(U) * buffers;
}
void accumulate(Nat& out, std::span<const uint32_t> part, size_t offset, int workers) {
    constexpr size_t block = 8192;
    size_t groups = (part.size() + block - 1) / block;
    std::vector<uint8_t> carries(groups);
    blocks(groups, workers, [&](size_t k) {
        unsigned carry = 0;
        for (size_t i = k * block; i < std::min((k + 1) * block, part.size()); ++i) {
            unsigned sum = out[offset + i] + part[i] + carry;
            carry = sum >= base;
            out[offset + i] = sum - carry * base;
        }
        carries[k] = carry;
    });
    for (size_t k = 0; k < groups; ++k) {
        size_t i = offset + std::min((k + 1) * block, part.size());
        for (unsigned carry = carries[k]; carry; ++i) {
            unsigned sum = out[i] + carry;
            carry = sum >= base;
            out[i] = sum - carry * base;
        }
    }
}
Nat discard_low(Nat&& a, size_t places) {
    if (places >= a.size())
        return {};
    if (!places)
        return std::move(a);
    // Keep a multi-gigabyte product in its original allocation while taking
    // its high words. Release whole unused pages inside the allocation.
    if (a.size() < 262144)
        return {a.begin() + places, a.end()};
    size_t length = a.size() - places;
    std::memmove(a.data(), a.data() + places, length * sizeof(a[0]));
    a.resize(length);
    static const size_t page = sysconf(_SC_PAGESIZE);
    auto first = (reinterpret_cast<uintptr_t>(a.data() + a.size()) + page - 1) / page * page;
    auto end = reinterpret_cast<uintptr_t>(a.data() + a.capacity()) / page * page;
    if (end > first)
        madvise(reinterpret_cast<void*>(first), end - first, MADV_DONTNEED);
    return std::move(a);
}
Nat tiled_product(const Nat& a, const Nat& b, int workers, size_t budget, size_t discard = 0) {
    // Accumulate into the final destination. Only one tile product is live,
    // even when both operands are much larger than the scratch allowance.
    size_t width = a.size(), height = b.size();
    do {
        if (width >= height)
            width = (width + 1) / 2;
        else
            height = (height + 1) / 2;
    } while (scratch_bytes(width, height) > budget && std::min(width, height) > 24);
    const bool square = &a == &b;
    if (square)
        width = height = std::min(width, height);
    const size_t guard = std::min<size_t>(discard, 4);
    const size_t origin = discard - guard;
    Nat result(a.size() + b.size() - origin);
    // Each partial product below origin contributes less than one unit
    // there. Counting clipped tiles bounds the carry into the guard words.
    Wide omitted = 0;
    for (size_t j = 0; j < b.size(); j += height) {
        Nat short_b;
        if (height < b.size()) {
            short_b.assign(b.begin() + j, b.begin() + std::min(j + height, b.size()));
            trim(short_b);
        }
        const Nat& bv = height < b.size() ? short_b : b;
        const size_t end_b = std::min(j + height, b.size());
        for (size_t i = square ? j : 0; i < a.size(); i += width) {
            size_t start = i + j;
            // Symmetric off-diagonal tiles occur twice in a square.
            unsigned copies = square && i != j ? 2 : 1;
            if (start < origin)
                omitted += copies;
            const size_t end_a = std::min(i + width, a.size());
            if (end_a + end_b <= origin)
                continue;
            Nat part;
            if (square && i == j) {
                part = mul(bv, bv, workers);
            } else if (width < a.size()) {
                Nat short_a(a.begin() + i, a.begin() + end_a);
                trim(short_a);
                part = mul(short_a, bv, workers);
            } else {
                part = mul(a, bv, workers);
            }
            size_t skip = origin - std::min(origin, start);
            if (skip >= part.size())
                continue;
            auto retained = std::span<const uint32_t>(part).subspan(skip);
            for (unsigned copy = 0; copy < copies; ++copy)
                accumulate(result, retained, start + skip - origin, workers);
        }
    }
    if (discard) {
        // Every clipped tile contributes a fraction smaller than one at
        // origin. Certify that their sum cannot carry into the retained words.
        Wide guard_value = 0, power = 1;
        for (size_t i = 0; i < guard; ++i) {
            guard_value += power * result[i];
            power *= base;
        }
        if (omitted && omitted >= power - guard_value) {
            Nat{}.swap(result);
            Nat exact = tiled_product(a, b, workers, budget);
            return discard_low(std::move(exact), discard);
        }
        result.erase(result.begin(), result.begin() + guard);
    }
    trim(result);
    return result;
}
template <int digits>
Nat recover(Buffer& x, Buffer& z, Buffer& third, size_t used, size_t n, int workers,
            size_t period = 0) {
    constexpr bool crt = digits > 6;
    constexpr U radix = digits == 4    ? 10000
                        : digits == 5  ? 100000
                        : digits == 6  ? 1000000
                        : digits == 10 ? 10000000000ULL
                        : digits == 12 ? 1000000000000ULL
                                       : 10000000000000000ULL;
    using F1 = Field<prime1>;
    using F2 = Field<prime2>;
    static const U inv1 = F1::power(prime0 % prime1, prime1 - 2);
    constexpr Wide first_two = Wide(prime0) * prime1;
    static const U inv2 = F2::power(first_two % prime2, prime2 - 2);
    auto value = [&](size_t j) {
        if constexpr (crt)
            return Wide(z.data[j]) * prime0 + x.data[j];
        else
            return x.data[j];
    };
    const ntt::Layout layout(n);
    // Align block boundaries with both the transform radix and limb radix.
    constexpr size_t chunk = 2048;
    size_t groups = (used + chunk - 1) / chunk;
    using Carry = decltype(value(0));
    std::vector<Carry> carries(groups);
    Nat result = overwrite_limbs((used * digits + 7) / 8 + 2);
    // Partial final coefficients and the last block carry can use this tail.
    std::fill_n(result.rbegin(), std::min<size_t>(result.size(), 4), 0);
    blocks(groups, workers, [&](size_t group) {
        size_t start = group * chunk, end = std::min(used, start + chunk);
        // All 32 residues of a stripe are physically contiguous.
        auto stripes = [&](auto visit) {
            for (size_t j = start; j < end; j += 32) {
                size_t pos = layout.tiled ? layout.index(j / layout.cols, j % layout.cols) : j;
                visit(j, pos, std::min<size_t>(32, end - j));
            }
        };
        if constexpr (crt) {
            const auto w1 = F1::splat(inv1), q1 = F1::splat((Wide(inv1) << 52) / prime1);
            stripes([&](size_t, size_t first, size_t count) {
                for (size_t k = 0; k < count; k += 8) {
                    size_t pos = first + k;
                    auto lo = F1::load(x.data + pos);
                    auto diff = F1::load(z.data + pos) + F1::splat(prime1) - F1::reduce(lo);
                    auto k1 = F1::reduce(F1::shoup(diff, w1, q1));
                    if constexpr (digits == 12) {
                        // Split k1*prime0 at the transform radix before carrying.
                        // The reciprocal underestimates k1*c/radix by less than one.
                        // The redundant remainder is below 2*radix and fits in 52 bits.
                        constexpr U c = prime0 % radix;
                        constexpr U reciprocal = (Wide(c) << 52) / radix;
                        auto q = F1::splat(reciprocal);
                        auto quotient = _mm512_madd52hi_epu64(_mm512_setzero_si512(), k1, q);
                        auto product =
                            _mm512_madd52lo_epu64(_mm512_setzero_si512(), k1, F1::splat(c));
                        auto remainder =
                            _mm512_madd52lo_epu64(product, quotient, F1::splat(R - radix)) &
                            F1::splat(R - 1);
                        F1::save(x.data + pos, lo + remainder);
                        F1::save(z.data + pos,
                                 quotient + _mm512_mullo_epi64(k1, F1::splat(prime0 / radix)));
                    } else
                        F1::save(z.data + pos, k1);
                    if constexpr (digits == 16) {
                        constexpr U a = prime0 % prime2;
                        auto m = F2::shoup(k1, F2::splat(a), F2::splat((Wide(a) << 52) / prime2));
                        auto d = F2::load(third.data + pos) + F2::splat(2 * prime2) -
                                 F2::reduce(lo) - F2::reduce(m);
                        auto k2 = F2::shoup(F2::twice_reduce(d), F2::splat(inv2),
                                            F2::splat((Wide(inv2) << 52) / prime2));
                        F2::save(third.data + pos, F2::reduce(k2));
                    }
                }
            });
        }
        Carry carry = 0;
        stripes([&](size_t logical, size_t first, size_t count) {
            for (size_t k = 0; k < count; ++k) {
                size_t pos = first + k, j = logical + k;
                U digit;
                if constexpr (digits == 12) {
                    // prime0*prime1/radix is below 2^63, leaving room for the low part.
                    U v = U(carry) + x.data[pos];
                    digit = v % radix;
                    carry = z.data[pos] + v / radix;
                } else {
                    auto partial = value(pos);
                    carry += partial;
                    Wide carry_high = 0;
                    if constexpr (digits == 16) {
                        // Split the 153-bit CRT solution at a decimal word so both
                        // partial products and the propagated carry fit in 128 bits.
                        U k = third.data[pos];
                        carry += Wide(k) * U(first_two % radix);
                        carry_high = Wide(k) * U(first_two / radix);
                    }
                    digit = carry % radix;
                    carry = carry / radix + carry_high;
                }
                if constexpr (digits == 5) {
                    size_t i = j * 5 / 8;
                    switch (j % 8) {
                    case 0:
                        result[i] = digit;
                        break;
                    case 1:
                        result[i] += digit % 1000 * 100000;
                        result[i + 1] = digit / 1000;
                        break;
                    case 2:
                        result[i] += digit * 100;
                        break;
                    case 3:
                        result[i] += digit % 10 * 10000000;
                        result[i + 1] = digit / 10;
                        break;
                    case 4:
                        result[i] += digit % 10000 * 10000;
                        result[i + 1] = digit / 10000;
                        break;
                    case 5:
                        result[i] += digit * 10;
                        break;
                    case 6:
                        result[i] += digit % 100 * 1000000;
                        result[i + 1] = digit / 100;
                        break;
                    case 7:
                        result[i] += digit * 1000;
                        break;
                    }
                } else if constexpr (digits == 4) {
                    if (j % 2 == 0)
                        result[j / 2] = digit;
                    else
                        result[j / 2] += digit * 10000;
                } else if constexpr (digits == 6) {
                    size_t i = j * 3 / 4;
                    switch (j % 4) {
                    case 0:
                        result[i] = digit;
                        break;
                    case 1:
                        result[i] += digit % 100 * 1000000;
                        result[i + 1] = digit / 100;
                        break;
                    case 2:
                        result[i] += digit % 10000 * 10000;
                        result[i + 1] = digit / 10000;
                        break;
                    default:
                        result[i] += digit * 100;
                        break;
                    }
                } else if constexpr (digits == 10) {
                    size_t i = j / 4 * 5 + j % 4;
                    switch (j % 4) {
                    case 0:
                        result[i] = digit % base;
                        result[i + 1] = digit / base;
                        break;
                    case 1:
                        result[i] += digit % 1000000 * 100;
                        result[i + 1] = digit / 1000000;
                        break;
                    case 2:
                        result[i] += digit % 10000 * 10000;
                        result[i + 1] = digit / 10000;
                        break;
                    default:
                        result[i] += digit % 100 * 1000000;
                        result[i + 1] = digit / 100;
                        break;
                    }
                } else if constexpr (digits == 16) {
                    result[2 * j] = digit % base;
                    result[2 * j + 1] = digit / base;
                } else {
                    size_t i = j / 2 * 3;
                    if (j % 2 == 0) {
                        result[i] = digit % base;
                        result[i + 1] = digit / base;
                    } else {
                        result[i + 1] += digit % 10000 * 10000;
                        result[i + 2] = digit / 10000;
                    }
                }
            }
        });
        carries[group] = carry;
    });
    for (size_t k = 1; k < groups; ++k) {
        Carry carry = carries[k - 1];
        for (size_t i = k * chunk * digits / 8; carry; ++i) {
            carry += result[i];
            result[i] = carry % base;
            carry /= base;
        }
    }
    if (period) {
        Carry carry = carries.back() + result[period] + Carry(result[period + 1]) * base;
        result.resize(period);
        for (size_t i = 0; carry; i = (i + 1 == period ? 0 : i + 1)) {
            carry += result[i];
            result[i] = carry % base;
            carry /= base;
        }
        if (result.back() == base - 1 &&
            std::all_of(result.begin(), result.end(), [](auto d) { return d == base - 1; }))
            return {};
    }
    trim(result);
    return result;
}
template <int digits>
Nat transform_product(const Nat& a, const Nat& b, int threads, size_t* requested = nullptr,
                      size_t discard = 0) {
    constexpr bool crt = digits > 6;
    size_t used = extent<digits>(a.size()) + extent<digits>(b.size());
    size_t n = transform_length(used);
    size_t period = 0;
    if (requested) {
        size_t short_n =
            transform_length(extent<digits>(std::max({*requested, a.size(), b.size()})));
        if (short_n < n) {
            n = short_n;
            used = n;
            period = n * digits / 8;
        }
        *requested = 0;
    }
    Workspace reservation;
    size_t buffers = 1 + size_t(&a != &b) + size_t(crt) + size_t(digits == 16);
    if (n > (U(1) << 32) || !reservation.acquire(n * sizeof(U) * buffers, threads))
        return tiled_product(a, b, threads, reservation.tile_budget, discard);
    int workers = std::min<size_t>(threads, 1 + n / 16384);
    Buffer x(n), y(&a == &b ? 0 : n), z(crt ? n : 0), third(digits == 16 ? n : 0);
    if constexpr (!crt) {
        constexpr U small_prime = 1125818302464001, max_digit = digits == 4   ? 9999
                                                                : digits == 5 ? 99999
                                                                              : 999999;
        if (extent<digits>(std::min(a.size(), b.size())) <=
            (small_prime - 1) / (max_digit * max_digit))
            residues<digits, small_prime>(x.data, y.data, n, a, b, workers);
        else
            residues<digits, prime0>(x.data, y.data, n, a, b, workers);
    } else
        residues<digits, prime0>(x.data, y.data, n, a, b, workers);
    if constexpr (crt)
        residues<digits, prime1>(z.data, y.data, n, a, b, workers);
    if constexpr (digits == 16)
        residues<digits, prime2>(third.data, y.data, n, a, b, workers);
    y.release();
    if (requested)
        *requested = period;
    return discard_low(recover<digits>(x, z, third, used, n, workers, period), discard);
}
#include "wide.hpp"

int packing(size_t large, size_t small) {

    size_t total = large + small;
    if (extent<6>(small) <= (prime0 - 1) / (999999ULL * 999999) &&
        transform_length(extent<6>(small) + extent<6>(large)) < transform_length(total * 2))
        return 6;
    const bool use_five =
        extent<5>(small) <= (prime0 - 1) / (99999ULL * 99999) &&
        transform_length(extent<5>(large) + extent<5>(small)) < transform_length(total * 2);
    if (total < 32768)
        return use_five ? 5 : 4;
    constexpr Wide modulus = Wide(prime0) * prime1;
    constexpr U max12 = 999999999999ULL;
    size_t points = transform_length(2 * total);
    int best = 4;
    if (use_five) {
        points = transform_length(extent<5>(large) + extent<5>(small));
        best = 5;
    }
    if (small > (prime0 - 1) / (2ULL * 9999 * 9999))
        points = SIZE_MAX;
    auto consider = [&](int digits, size_t cost) {
        if (cost < points) {
            points = cost;
            best = digits;
        }
    };
    if (extent<12>(small) <= modulus / (Wide(max12) * max12))
        consider(12, 2 * transform_length(extent<12>(large) + extent<12>(small)));
    consider(16, 3 * transform_length(extent<16>(large) + extent<16>(small)));
    consider(10, 2 * transform_length(extent<10>(large) + extent<10>(small)));
    // Wide fields operate on shorter rows. Once the inputs are large enough,
    // preparing all fields from each stripe pays even with more modular points.
    if (total >= 100000) {
        size_t threshold = points + points / 2;
        for (size_t width = 5; width <= 8; ++width) {
            size_t n = transform_length((large + width - 1) / width + (small + width - 1) / width);
            size_t cost = wide_plan_cost(width, n);
            if (linear_width_allowed(width, n) && cost <= threshold) {
                threshold = cost;
                best = 8 * width;
            }
        }
    }
    return best;
}
template <int digits>
bool shared_transform(const Nat& common, const Nat& a, const Nat& b, Nat& x, Nat& y, int threads) {
    constexpr bool crt = digits > 6;
    size_t size_a = extent<digits>(common.size()) + extent<digits>(a.size());
    size_t size_b = extent<digits>(common.size()) + extent<digits>(b.size());
    size_t n = transform_length(size_a);
    if (n != transform_length(size_b) || n > (1ULL << 32))
        return false;
    Workspace reservation;
    if (!reservation.acquire(n * sizeof(U) * (crt ? 5 : 3), threads))
        return false;
    int workers = std::min<size_t>(threads, 1 + n / 16384);
    Buffer a0(n), b0(n), a1(crt ? n : 0), b1(crt ? n : 0), shared(n), empty(0);
    residues<digits, prime0, true>(a0.data, shared.data, n, a, common, workers, &b, b0.data);
    if constexpr (crt)
        residues<digits, prime1, true>(a1.data, shared.data, n, a, common, workers, &b, b1.data);
    shared.release();
    x = recover<digits>(a0, a1, empty, size_a, n, workers);
    a0.release();
    a1.release();
    y = recover<digits>(b0, b1, empty, size_b, n, workers);
    return true;
}
// The arithmetic dispatcher receives operands in descending size order.
Nat product_window(const Nat& a, const Nat& b, size_t discard, int threads) {
    if (b.size() <= 2) {
        Nat out = overwrite_limbs(a.size() + b.size());
        auto multiply_block = [&](size_t first, size_t end) {
            U carry = 0;
            size_t i = first;
            if (end - first >= 64) {
                CoefficientCarry normalization;
                auto previous = _mm512_set1_epi64(first ? a[first - 1] : 0);
                const auto low = _mm512_set1_epi64(b[0]);
                const auto high = _mm512_set1_epi64(b.size() == 2 ? b[1] : 0);
                for (; i + 8 <= std::min(end, a.size()); i += 8) {
                    auto x = _mm512_cvtepu32_epi64(_mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(a.data() + i)));
                    auto sums = _mm512_mul_epu32(x, low);
                    if (b.size() == 2)
                        sums += _mm512_mul_epu32(_mm512_alignr_epi64(x, previous, 7), high);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out.data() + i),
                                        normalization.take(sums));
                    previous = x;
                }
                auto tail = normalization.take(_mm512_setzero_si512());
                carry = uint32_t(_mm256_extract_epi32(tail, 0)) +
                        U(uint32_t(_mm256_extract_epi32(tail, 1))) * base;
            }
            for (; i < end; ++i) {
                if (i < a.size())
                    carry += U(a[i]) * b[0];
                if (b.size() == 2 && i && i - 1 < a.size())
                    carry += U(a[i - 1]) * b[1];
                out[i] = carry % base;
                carry /= base;
            }
            return carry;
        };
        constexpr size_t block = 32768;
        if (threads == 1 || out.size() <= block)
            multiply_block(0, out.size());
        else {
            std::vector<U> carries((out.size() + block - 1) / block);
            blocks(carries.size(), threads, [&](size_t i) {
                carries[i] = multiply_block(i * block, std::min((i + 1) * block, out.size()));
            });
            for (size_t i = 1; i < carries.size(); ++i) {
                U carry = carries[i - 1];
                for (size_t j = i * block; carry; ++j) {
                    carry += out[j];
                    out[j] = carry % base;
                    carry /= base;
                }
            }
        }
        trim(out);
        return discard_low(std::move(out), discard);
    }
    if (b.size() <= 24 || a.size() <= 64)
        return discard_low(school(a, b), discard);
    if (b.size() >= 48 && b.size() <= 16384 && a.size() + b.size() >= 400 &&
        (threads == 1 || a.size() <= 65536)) {
        size_t n = std::min(std::bit_ceil(a.size() + b.size()),
                            std::min<size_t>(32768, std::bit_ceil(4 * b.size())));
        // The three- and five-way splits fill the gaps between binary plans.
        if (n >= 1024 && a.size() + b.size() <= 5 * n / 8)
            n = 5 * n / 8;
        else if (n >= 512 && a.size() + b.size() <= 3 * n / 4)
            n = 3 * n / 4;
        Workspace ws;
        if (ws.acquire(n * sizeof(double) * (&a == &b ? 2 : 4), threads))
            return discard_low(short_fft::product(a, b, n), discard);
    }
    if (a.size() <= 768 && 2 * b.size() >= a.size())
        return discard_low(medium(a, b), discard);
    if (a.size() <= 512 && a.size() * b.size() <= 32768)
        return discard_low(school(a, b), discard);
    switch (packing(a.size(), b.size())) {
    case 64:
        return wide_product<8>(a, b, threads, nullptr, discard);
    case 40:
        return wide_product<5>(a, b, threads, nullptr, discard);
    case 48:
        return wide_product<6>(a, b, threads, nullptr, discard);
    case 56:
        return wide_product<7>(a, b, threads, nullptr, discard);
    case 5:
        return transform_product<5>(a, b, threads, nullptr, discard);
    case 6:
        return transform_product<6>(a, b, threads, nullptr, discard);
    case 10:
        return transform_product<10>(a, b, threads, nullptr, discard);
    case 12:
        return transform_product<12>(a, b, threads, nullptr, discard);
    case 16:
        return transform_product<16>(a, b, threads, nullptr, discard);
    default:
        return transform_product<4>(a, b, threads, nullptr, discard);
    }
}
} // namespace
Nat mul_square_times(const Nat& a, const Nat& b, int threads) {
    if (a.empty() || b.empty())
        return {};
    if (a.size() < 32768 || b.size() < 32768)
        return mul(mul(a, a, threads), b, threads);
    if (threads > 1 && !Team::active()) {
        Team team(threads);
        return mul_square_times(a, b, threads);
    }
    constexpr size_t width = 3, primes = 6;
    size_t na = (a.size() + width - 1) / width, nb = (b.size() + width - 1) / width;
    size_t used = 2 * na + nb, n = transform_length(used);
    Workspace space;
    size_t words = n;
    bool packed = ntt::have_packing && ntt::Layout(n).tiled;
    if (packed)
        words = n / 16 * 13;
    // At n<=2^30 every coefficient is <2^60 * 10^72, below P/4.
    if (n > (1UL << 30) ||
        !space.acquire((primes + (&a != &b)) * words * sizeof(U), threads, false))
        return mul(mul(a, a, threads), b, threads);
    int workers = std::min<size_t>(threads, 1 + n / 16384);
    std::array<std::unique_ptr<Buffer>, primes> residues;
    Buffer work(&a == &b ? 0 : words);
    auto each = [&]<size_t I>() {
        residues[I] = std::make_unique<Buffer>(words);
        wide_residues<I, ntt::have_packing, false, width, true>(*residues[I], work, n, a, b,
                                                                workers, crt_tables[0]);
    };
    [&]<size_t... I>(std::index_sequence<I...>) {
        (each.template operator()<I>(), ...);
    }(std::make_index_sequence<primes>{});
    return wide_recover<width>(residues, n, used, workers, packed, 0, 0);
}
Nat mul(const Nat& a, const Nat& b, int threads) {
    return mul_high(a, b, 0, threads);
}
Nat mul_high(const Nat& a, const Nat& b, size_t discard, int threads) {
    if (threads > 1 && !Team::active()) {
        Team team(threads);
        return mul_high(a, b, discard, threads);
    }
    if (a.empty() || b.empty() || discard >= a.size() + b.size())
        return {};
    if (a.size() < b.size())
        return mul_high(b, a, discard, threads);
    // If only a short leading slice survives, discard low input words before
    // transforming. Four extra words bound the omitted contribution. Keep
    // three of them in the product to detect a carry into the requested slice.
    constexpr size_t guard = 3;
    const size_t keep = a.size() + b.size() - discard + guard + 1;
    if (a.size() > keep && a.size() - keep >= 4096) {
        const size_t skip_a = a.size() - keep;
        const size_t skip_b = b.size() - std::min(b.size(), keep);
        Nat high_a(a.begin() + skip_a, a.end()), high_b;
        if (skip_b && &a != &b)
            high_b.assign(b.begin() + skip_b, b.end());
        const Nat& right = &a == &b ? high_a : skip_b ? high_b : b;
        const size_t cut = discard - skip_a - skip_b;
        auto leading = product_window(high_a, right, cut - guard, threads);
        // In units of the shortened operands' radix, the omitted part is
        // below right if only a was clipped, or high_a + right + 1 otherwise.
        // Either bound fits below B^(cut-guard). It can increase leading
        // by at most one, and only three all-nines words can propagate it.
        if (leading.size() < guard ||
            std::any_of(leading.begin(), leading.begin() + guard,
                        [](uint32_t digit) { return digit != base - 1; }))
            return discard_low(std::move(leading), guard);
    }
    return product_window(a, b, discard, threads);
}
WrappedProduct wrapped_mul(const Nat& a, const Nat& b, size_t minimum, int threads) {
    if (std::min(a.size(), b.size()) < 1024)
        return {mul(a, b, threads), 0};
    if (threads > 1 && !Team::active()) {
        Team team(threads);
        return wrapped_mul(a, b, minimum, threads);
    }
    size_t period = minimum;
    Nat value;
    int digits = packing(std::max(a.size(), b.size()), std::min(a.size(), b.size()));
    if (digits >= 40) {
        // Newton residuals admit a shorter cyclic transform. Compare the
        // lengths that will actually run for each eligible coefficient width.
        size_t best = SIZE_MAX;
        const int linear_digits = digits;
        for (size_t w = 5; w <= 8; ++w) {
            auto count = [w](size_t limbs) { return (limbs + w - 1) / w; };
            size_t needed = std::min(count(a.size()) + count(b.size()),
                                     count(std::max({minimum, a.size(), b.size()})));
            size_t n = transform_length(needed);
            if (w == 8 && n > (size_t(1) << 30))
                continue;
            size_t score = wide_plan_cost(w, n);
            if (score < best || (score == best && linear_digits == int(8 * w))) {
                best = score;
                digits = 8 * w;
            }
        }
    }
    switch (digits) {
    case 64:
        value = wide_product<8>(a, b, threads, &period, 0);
        break;
    case 40:
        value = wide_product<5>(a, b, threads, &period, 0);
        break;
    case 48:
        value = wide_product<6>(a, b, threads, &period, 0);
        break;
    case 56:
        value = wide_product<7>(a, b, threads, &period, 0);
        break;
    case 5:
        value = transform_product<5>(a, b, threads, &period);
        break;
    case 6:
        value = transform_product<6>(a, b, threads, &period);
        break;
    case 10:
        value = transform_product<10>(a, b, threads, &period);
        break;
    case 12:
        value = transform_product<12>(a, b, threads, &period);
        break;
    case 16:
        value = transform_product<16>(a, b, threads, &period);
        break;
    default:
        value = transform_product<4>(a, b, threads, &period);
    }
    return {std::move(value), period};
}
bool shared_products(const Nat& common, const Nat& a, const Nat& b, Nat& x, Nat& y, int threads,
                     size_t first_discard, size_t second_discard) {
    if (std::min({a.size(), b.size(), common.size()}) < 512)
        return false;
    if (threads > 1 && !Team::active()) {
        Team team(threads);
        return shared_products(common, a, b, x, y, threads, first_discard, second_discard);
    }
    auto choice = [&](const Nat& v) {
        return packing(std::max(common.size(), v.size()), std::min(common.size(), v.size()));
    };
    int digits = choice(a);
    int second = choice(b);
    if (digits >= 40 && second >= 40) {
        size_t score = SIZE_MAX;
        int preferred = 0;
        for (int width : {5, 6, 7, 8}) {
            auto count = [width](size_t n) { return (n + width - 1) / width; };
            size_t c = count(common.size());
            size_t n = transform_length(c + count(std::max(a.size(), b.size())));
            size_t shorter = transform_length(c + count(std::min(a.size(), b.size())));
            const size_t cost = wide_plan_cost(width, n);
            if (linear_width_allowed(width, n) && n <= (U(1) << 32) &&
                n <= shorter + shorter / 4 && cost < score) {
                score = cost;
                preferred = 8 * width;
            }
        }
        if (!preferred)
            return false;
        if (preferred == 64)
            return wide_pair<8>(common, a, b, x, y, threads, first_discard, second_discard);
        if (preferred == 40)
            return wide_pair<5>(common, a, b, x, y, threads, first_discard, second_discard);
        if (preferred == 48)
            return wide_pair<6>(common, a, b, x, y, threads, first_discard, second_discard);
        return wide_pair<7>(common, a, b, x, y, threads, first_discard, second_discard);
    }
    if (first_discard || second_discard || digits != second)
        return false;
    if (digits == 12)
        return shared_transform<12>(common, a, b, x, y, threads);
    if (digits == 4)
        return shared_transform<4>(common, a, b, x, y, threads);
    return false;
}
} // namespace ap
