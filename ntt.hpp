#pragma once
#include <array>
#include "parallel.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>
#include <memory>
#include <mutex>
#include <vector>

namespace ap::ntt {
using U = uint64_t;
using Wide = __uint128_t;
// Both passes overwrite their scratch before reading it. Keep the largest
// allocation on each worker and align the 512-bit loads and stores to cache lines.
class Scratch {
    std::unique_ptr<U, decltype(&std::free)> words{nullptr, std::free};
    size_t capacity = 0;

public:
    void resize(size_t n) {
        if (n <= capacity)
            return;
        size_t rounded = (n + 7) / 8 * 8;
        auto* data = static_cast<U*>(std::aligned_alloc(64, rounded * sizeof(U)));
        if (!data)
            throw std::bad_alloc();
        words.reset(data);
        capacity = rounded;
    }
    U* data() { return words.get(); }
};
inline thread_local Scratch column_scratch;
inline thread_local Scratch row_scratch;

// A tile contains sixteen 32-word slices of consecutive rows. A column
// worker consumes a whole tile before advancing to another group of rows.
struct Layout {
    size_t cols;
    bool tiled;
    explicit Layout(size_t n, bool prepared_columns = false)
        : cols(n < 16384 ? n : n >> std::min(13, std::countr_zero(n) / 2)),
          tiled(n >= (1u << 18) || (prepared_columns && n >= (1u << 16))) {
    }
    size_t index(size_t row, size_t col) const {
        return tiled ? (row / 16 * 16) * cols + (col / 32) * 512 + (row % 16) * 32 + col % 32
                     : row * cols + col;
    }
};
// Eight residues occupy 52 bytes. The low three 16-bit words of each
// residue precede a 32-bit word containing the eight high nibbles.
#if defined(__AVX512VBMI2__) && defined(__BMI2__)
inline constexpr bool have_packing = true;
inline __m512i load_packed(const U* buffer, size_t index) {
    auto* src = reinterpret_cast<const unsigned char*>(buffer) + (index / 8) * 52;
    auto lo = _mm512_maskz_expandloadu_epi16(0x77777777u, src);
    uint32_t high;
    std::memcpy(&high, src + 48, sizeof(high));
    auto bytes = _mm_cvtsi64_si128(_pdep_u64(high, 0x0f0f0f0f0f0f0f0fULL));
    return lo | _mm512_slli_epi64(_mm512_cvtepu8_epi64(bytes), 48);
}
inline void save_packed(U* buffer, size_t index, __m512i v) {
    auto* dst = reinterpret_cast<unsigned char*>(buffer) + (index / 8) * 52;
    _mm512_mask_compressstoreu_epi16(dst, 0x77777777u, v);
    auto bytes = _mm512_cvtepi64_epi8(_mm512_srli_epi64(v, 48));
    uint32_t high = _pext_u64(_mm_cvtsi128_si64(bytes), 0x0f0f0f0f0f0f0f0fULL);
    std::memcpy(dst + 48, &high, sizeof(high));
}
#else
inline constexpr bool have_packing = false;
// Only instantiated when the compiler target supplies the packing instructions.
__m512i load_packed(const U*, size_t);
void save_packed(U*, size_t, __m512i);
#endif
constexpr U R = U(1) << 52;
// Wrapping arithmetic uses R=2^52 and p=k*2^32+1 with generator 7 for the supported orders.
// The forward transform stores frequencies in bit-reversed order within
// each radix-2 dimension. No global bit-reversal permutation is needed.
// Butterflies use Shoup weights, while point products and the rotations
// between dimensions use Montgomery weights.
template <U p> struct Field {
    static U product(U a, U b) {
        return Wide(a) * b % p;
    }
    static U power(U a, U n) {
        U r = 1;
        for (; n; n >>= 1, a = product(a, a))
            if (n & 1)
                r = product(r, a);
        return r;
    }
    static constexpr bool decimal_prime = p == 1536000000000001ULL;
    static constexpr U generator = decimal_prime ? 23 : 7;
    static constexpr U mont_inverse = decimal_prime ? 1459069674061825ULL : R + 2 - p;
    static_assert(((p * mont_inverse) & (R - 1)) == 1);
    using V = __m512i;
    static V splat(U n) {
        return _mm512_set1_epi64(n);
    }
    static V load(const U* a) {
        return _mm512_loadu_si512(a);
    }
    static void save(U* a, V x) {
        _mm512_storeu_si512(a, x);
    }
    static V reduce(V x) {
        return _mm512_min_epu64(x, x - splat(p));
    }
    // Before the final reduction this is in [0, p + a*b/R].
    static V mont_lazy(V a, V b) {
        V zero = _mm512_setzero_si512();
        V lo = _mm512_madd52lo_epu64(zero, a, b);
        V hi = _mm512_madd52hi_epu64(zero, a, b);
        // mont_inverse is the inverse of p modulo R. The low halves
        // of a*b and m*p match and their subtraction needs no borrow.
        V m = _mm512_madd52lo_epu64(zero, lo, splat(mont_inverse));
        return hi + splat(p) - _mm512_madd52hi_epu64(zero, m, splat(p));
    }
    static V mont(V a, V b) { return reduce(mont_lazy(a, b)); }

    // Shoup products return residues in [0,2p). Since 2p < R,
    // a second lazy residue is a valid input to the next butterfly.
    static V twice_reduce(V x) {
        return _mm512_min_epu64(x, x - splat(2 * p));
    }
    static V shoup(V x, V w, V reciprocal) {
        V zero = _mm512_setzero_si512();
        V q = _mm512_madd52hi_epu64(zero, x, reciprocal);
        V lo = _mm512_madd52lo_epu64(zero, x, w);
        return _mm512_and_si512(_mm512_madd52lo_epu64(lo, q, splat(R - p)), splat(R - 1));
    }
    struct Weights {
        std::vector<U> w{1}, q{R / p};
    };
    struct Roots {
        Weights f, b;
        explicit Roots(size_t count) {
            auto build = [count](Weights& table, bool inverse) {
                table.w.reserve(count + 8);
                table.q.reserve(count + 8);
                for (size_t k = 1; k < count; k *= 2) {
                    U w = power(generator, (p - 1) / (4 * k));
                    if (inverse)
                        w = power(w, p - 2);
                    for (size_t i = 0; i < k; ++i) {
                        U v = product(table.w[i], w);
                        table.w.push_back(v);
                        table.q.push_back((Wide(v) << 52) / p);
                    }
                }
                table.w.resize(count + 8);
                table.q.resize(count + 8);
            };
            build(f, false);
            build(b, true);
        }
    };
    static std::shared_ptr<const Roots> roots(size_t count) {
        static std::mutex lock;
        static std::shared_ptr<const Roots> current;
        std::lock_guard guard(lock);
        if (!current || current->f.w.size() < count + 8)
            current = std::make_shared<Roots>(count);
        return current;
    }

    template <bool inverse, bool unit = false> static inline void butterfly(V& a, V& b, V w, V q) {
        if constexpr (4 * p < R && !inverse) {
            // With a 50-bit prime the unsigned forward residues fit in
            // four multiples of p, including the inputs to Shoup reduction.
            a = twice_reduce(a);
            if constexpr (unit)
                b = twice_reduce(b);
            else
                b = shoup(b, w, q);
            V d = a + splat(2 * p) - b;
            a += b;
            b = d;
        } else if constexpr (4 * p < R && inverse && !unit) {
            V d = a + splat(2 * p) - b;
            a = twice_reduce(a + b);
            b = shoup(d, w, q);
        } else {
            if constexpr (!inverse && !unit)
                b = shoup(b, w, q);
            V diff = twice_reduce(a + splat(2 * p) - b);
            a = twice_reduce(a + b);
            b = inverse && !unit ? shoup(diff, w, q) : diff;
        }
    }
    // Map a natural lane in a 16-point block to the two vectors holding
    // the left and right butterfly inputs at a given half-length.
    template <int half> static constexpr unsigned split_lane(unsigned i) {
        if constexpr (!half)
            return i;
        else
            return (i / half % 2) * 8 + i / (2 * half) * half + i % half;
    }
    template <int from, int to> static inline void reorder(V& x, V& y) {
        alignas(64) static constexpr auto indices = [] {
            std::array<U, 16> indices{};
            for (unsigned i = 0; i < 16; ++i)
                indices[split_lane<to>(i)] = split_lane<from>(i);
            return indices;
        }();
        auto a = _mm512_permutex2var_epi64(x, load(indices.data()), y);
        y = _mm512_permutex2var_epi64(x, load(indices.data() + 8), y);
        x = a;
    }
    template <bool inverse, int half>
    static inline void short_stage(V& x, V& y, size_t off, const Weights& tw) {
        alignas(64) U indices[8];
        for (unsigned i = 0; i < 8; ++i)
            indices[i] = i / half;
        size_t g = off / (2 * half);
        V w = _mm512_permutexvar_epi64(load(indices), load(tw.w.data() + g));
        V q = _mm512_permutexvar_epi64(load(indices), load(tw.q.data() + g));
        butterfly<inverse>(x, y, w, q);
    }
    // Fuse two levels so each group crosses the register/cache boundary
    // once. Width is one for contiguous rows and a full stripe for columns.
    template <bool inverse>
    static void pass4(U* data, size_t n, size_t h, size_t width, const Weights& tw) {
        size_t stride = h * width;
        for (size_t g = 0; g < n / (4 * h); ++g) {
            V w = splat(tw.w[g]), q = splat(tw.q[g]);
            V v = splat(tw.w[2 * g]), r = splat(tw.q[2 * g]);
            V z = splat(tw.w[2 * g + 1]), s = splat(tw.q[2 * g + 1]);
            U* block = data + 4 * g * stride;
            auto group = [&]<bool unit>() {
                for (size_t j = 0; j < stride; j += 8) {
                    V a = load(block + j), b = load(block + j + stride);
                    V c = load(block + j + 2 * stride), d = load(block + j + 3 * stride);
                    if constexpr (!inverse) {
                        butterfly<false, unit>(a, c, w, q);
                        butterfly<false, unit>(b, d, w, q);
                        butterfly<false, unit>(a, b, v, r);
                        butterfly<false>(c, d, z, s);
                    } else {
                        butterfly<true, unit>(a, b, v, r);
                        butterfly<true>(c, d, z, s);
                        butterfly<true, unit>(a, c, w, q);
                        butterfly<true, unit>(b, d, w, q);
                    }
                    save(block + j, a);
                    save(block + j + stride, b);
                    save(block + j + 2 * stride, c);
                    save(block + j + 3 * stride, d);
                }
            };
            if (g == 0)
                group.template operator()<true>();
            else
                group.template operator()<false>();
        }
    }
    template <bool inverse>
    static void levels(U* data, size_t n, size_t length, size_t minimum, size_t width,
                       const Weights& tw) {
        size_t h = inverse ? minimum : length / 2;
        if constexpr (inverse) {
            while (2 * h < length) {
                pass4<true>(data, n, h, width, tw);
                h *= 4;
            }
            if (h >= length)
                return;
        } else {
            while (h >= 2 * minimum) {
                pass4<false>(data, n, h / 2, width, tw);
                h /= 4;
            }
            if (h < minimum)
                return;
        }
        size_t stride = h * width;
        for (size_t g = 0; g < n / (2 * h); ++g) {
            V w = splat(tw.w[g]), q = splat(tw.q[g]);
            U* block = data + 2 * g * stride;
            for (size_t j = 0; j < stride; j += 8) {
                V a = load(block + j), b = load(block + j + stride);
                butterfly<inverse>(a, b, w, q);
                save(block + j, a);
                save(block + j + stride, b);
            }
        }
    }
    template <bool inverse> static void short_transform(U* data, size_t n, const Weights& tw) {
        auto small = [&] {
            for (size_t off = 0; off < n; off += 16) {
                V x = load(data + off), y = load(data + off + 8);
                // Leave the terminal butterflies in even/odd lane order. Pointwise
                // multiplication preserves that order for the first inverse stage.
                if constexpr (!inverse)
                    reorder<0, 4>(x, y);
                short_stage<inverse, inverse ? 1 : 4>(x, y, off, tw);
                reorder<inverse ? 1 : 4, 2>(x, y);
                short_stage<inverse, 2>(x, y, off, tw);
                reorder<2, inverse ? 4 : 1>(x, y);
                short_stage<inverse, inverse ? 4 : 1>(x, y, off, tw);
                if constexpr (inverse)
                    reorder<4, 0>(x, y);
                save(data + off, x);
                save(data + off + 8, y);
            }
        };
        if constexpr (inverse)
            small();
        levels<inverse>(data, n, n, 8, 1, tw);
        if constexpr (!inverse)
            small();
    }

    // Bailey decomposition. Process a stripe of columns in private cache,
    // then transform contiguous rows. The two dimensions stay bit reversed.
    static constexpr size_t stripe = 32;
    template <bool inverse, bool packed, class Reader>
    static void columns(U* data, size_t rows, size_t cols, size_t col, const Weights& tw,
                        size_t extent, U scale, U scale_q, Reader read, const Layout& layout,
                        size_t width, bool compact = false, size_t skip_rows = 0) {
        auto& scratch = column_scratch;
        scratch.resize(rows * width);
        size_t first_step = 1;
        if constexpr (inverse) {
            for (size_t i = 0; i < rows; ++i) {
                auto pos = layout.index(i, col);
                if constexpr (packed) {
                    for (size_t j = 0; j < width; j += 8)
                        save(scratch.data() + i * width + j, load_packed(data, pos + j));
                } else {
                    for (size_t j = 0; j < width; j += 8)
                        save(scratch.data() + i * width + j, load(data + pos + j));
                }
            }
        } else {
            // All-zero upper halves merely duplicate the occupied lower half
            // through the first butterfly levels.
            size_t occupied = std::max<size_t>(1, (extent + cols - 1) / cols);
            size_t initial = std::bit_ceil(occupied);
            for (size_t i = 0; i < occupied; ++i)
                read(scratch.data() + i * width, i * cols + col, width);
            std::fill(scratch.data() + occupied * width, scratch.data() + initial * width, 0);
            first_step = rows / initial;
            if (first_step > 1 && initial > 1) {
                // The zero upper rows duplicate the occupied block. Apply its
                // first nontrivial butterfly as we write the different copies.
                // Process group zero last so the input stays live for every group.
                size_t half = initial * width / 2;
                for (size_t g = first_step; g-- > 0;) {
                    U* dest = scratch.data() + g * initial * width;
                    auto emit = [&]<bool unit>() {
                        auto w = splat(tw.w[g]), q = splat(tw.q[g]);
                        for (size_t j = 0; j < half; j += 8) {
                            auto a = load(scratch.data() + j);
                            auto b = load(scratch.data() + half + j);
                            butterfly<false, unit>(a, b, w, q);
                            save(dest + j, a);
                            save(dest + half + j, b);
                        }
                    };
                    if (g)
                        emit.template operator()<false>();
                    else
                        emit.template operator()<true>();
                }
                first_step *= 2;
            } else if (initial == 1) {
                for (size_t i = initial; i < rows; i *= 2)
                    std::copy_n(scratch.data(), i * width, scratch.data() + i * width);
            }
        }
        levels<inverse>(scratch.data(), rows, rows / first_step, 1, width, tw);
        bool streaming = !compact && !packed && rows * cols >= (1u << 21);
        for (size_t i = inverse ? skip_rows : 0; i < rows; ++i) {
            size_t pos = layout.index(i, col);
            if constexpr (inverse)
                if (i * cols + col >= extent)
                    break;
            for (size_t j = 0; j < width; j += 8) {
                V v = load(scratch.data() + i * width + j);
                if constexpr (inverse)
                    v = reduce(shoup(v, splat(scale), splat(scale_q)));
                // Each worker owns all 512 words of each tile it compacts.
                // Input was gathered before these overlapping stores begin.
                if constexpr (inverse && !packed && have_packing) {
                    if (compact) {
                        save_packed(data + (pos & ~size_t(511)), (pos & 511) + j, v);
                        continue;
                    }
                }
                if constexpr (packed)
                    save_packed(data, pos + j, v);
                else if (streaming)
                    _mm512_stream_si512(reinterpret_cast<V*>(data + pos + j), v);
                else
                    save(data + pos + j, v);
            }
        }
        if (streaming)
            _mm_sfence();
    }

    struct RowPhase {
        alignas(64) U initial[8];
        U step, step32, quot32;
    };
    struct Phase3 {
        Weights one, two;
        U root, quot;
        void init(size_t n, bool inverse) {
            U gen = power(generator, (p - 1) / n);
            if (inverse)
                gen = power(gen, p - 2);
            root = power(gen, n / 3);
            quot = (Wide(root) << 52) / p;
            auto build = [&](Weights& t, U step) {
                U v = 1;
                t.w.resize(n / 3);
                t.q.resize(n / 3);
                for (size_t j = 0; j < n / 3; ++j) {
                    t.w[j] = v;
                    t.q[j] = (Wide(v) << 52) / p;
                    v = product(v, step);
                }
            };
            build(one, gen);
            build(two, product(gen, gen));
        }
    };
    struct Phase5 {
        std::array<Weights, 4> rotations;
        std::array<U, 4> weights, reciprocals;
        void init(size_t cols, bool inverse) {
            const size_t n = cols / 5;
            U gen = power(generator, (p - 1) / cols);
            if (inverse)
                gen = power(gen, p - 2);
            U w = power(gen, n), w2 = product(w, w), w3 = product(w, w2), w4 = product(w, w3);
            constexpr U half = (p + 1) / 2;
            weights = {product(w + w4, half), product(w2 + w3, half), product(p + w - w4, half),
                       product(p + w2 - w3, half)};
            for (unsigned i = 0; i < 4; ++i) {
                reciprocals[i] = (Wide(weights[i]) << 52) / p;
                auto& table = rotations[i];
                table.w.resize(n);
                table.q.resize(n);
                U step = power(gen, i + 1), v = 1;
                for (size_t j = 0; j < n; ++j) {
                    table.w[j] = v;
                    table.q[j] = (Wide(v) << 52) / p;
                    v = product(v, step);
                }
            }
        }
    };
    struct Plan {
        size_t rows, cols;
        U inverse_scale; // The point product contributes a factor of 1/R.
        std::shared_ptr<const Roots> table;
        std::vector<RowPhase> f, b;
        Phase3 extra[2];
        Phase5 fifth[2];
        explicit Plan(size_t n) : inverse_scale(product(R % p, power(n, p - 2))) {
            cols = Layout(n).cols;
            rows = n / cols;
            table = roots(std::max(rows, cols) / 2);
            if (cols % 3 == 0) {
                extra[0].init(cols, false);
                extra[1].init(cols, true);
            }
            if (cols % 5 == 0) {
                fifth[0].init(cols, false);
                fifth[1].init(cols, true);
            }
            if (rows == 1)
                return;
            auto phases = [&](std::vector<RowPhase>& phase, U root) {
                phase.resize(rows);
                U w = 1;
                size_t rev = 0;
                for (size_t i = 0; i < rows; ++i) {
                    auto& r = phase[rev];
                    r.initial[0] = R % p;
                    for (int j = 1; j < 8; ++j)
                        r.initial[j] = product(r.initial[j - 1], w);
                    r.step = product(r.initial[7], w);
                    r.step32 = power(w, 32);
                    r.quot32 = (Wide(r.step32) << 52) / p;
                    w = product(w, root);
                    size_t bit = rows / 2;
                    while (bit && (rev & bit)) {
                        rev ^= bit;
                        bit /= 2;
                    }
                    rev ^= bit;
                }
            };
            U root = power(generator, (p - 1) / n);
            phases(f, root);
            phases(b, power(root, p - 2));
        }
    };
    static const Plan& plan_for(size_t n) {
        static std::unique_ptr<Plan> plans[102];
        static std::once_flag flags[102];
        auto k = 3 * std::countr_zero(n) + (n % 3 == 0) + 2 * (n % 5 == 0);
        std::call_once(flags[k], [=] { plans[k] = std::make_unique<Plan>(n); });
        return *plans[k];
    }
    template <bool paired = false>
    static void rotate_row(U* row, size_t cols, const RowPhase& phase, U* other = nullptr) {
        V step = splat(phase.step), stride = splat(phase.step32), quot = splat(phase.quot32);
        V weights[4];
        weights[0] = load(phase.initial);
        for (unsigned i = 1; i < 4; ++i)
            weights[i] = mont(weights[i - 1], step);
        for (size_t j = 0; j < cols; j += 32) {
            for (unsigned i = 0; i < 4; ++i) {
                auto offset = j + 8 * i;
                // A canonical Montgomery weight times a residue below R gives
                // a lazy result below 2p, ready for the next butterfly.
                save(row + offset, mont_lazy(load(row + offset), weights[i]));
                if constexpr (paired)
                    save(other + offset, mont_lazy(load(other + offset), weights[i]));
                weights[i] = reduce(shoup(weights[i], stride, quot));
            }
        }
    }
    template <bool inverse> static void radix_five(U* row, const Plan& plan, const Weights& tw) {
        const size_t n = plan.cols / 5;
        const auto& factor = plan.fifth[inverse];
        if constexpr (inverse)
            for (unsigned k = 0; k < 5; ++k)
                short_transform<true>(row + k * n, n, tw);
        auto add = [](V a, V b) { return twice_reduce(a + b); };
        auto sub = [](V a, V b) { return twice_reduce(a + splat(2 * p) - b); };
        auto scale = [&](V x, unsigned k) {
            return shoup(x, splat(factor.weights[k]), splat(factor.reciprocals[k]));
        };
        for (size_t j = 0; j < n; j += 8) {
            V v[5];
            auto rotate = [&](V x, unsigned k) {
                const auto& r = factor.rotations[k - 1];
                return shoup(x, load(r.w.data() + j), load(r.q.data() + j));
            };
            for (unsigned k = 0; k < 5; ++k) {
                v[k] = load(row + k * n + j);
                if constexpr (inverse)
                    if (k)
                        v[k] = rotate(v[k], k);
            }
            auto t1 = add(v[1], v[4]), t2 = add(v[2], v[3]);
            auto d1 = sub(v[1], v[4]), d2 = sub(v[2], v[3]);
            auto u = add(v[0], add(scale(t1, 0), scale(t2, 1)));
            auto w = add(v[0], add(scale(t2, 0), scale(t1, 1)));
            auto a = add(scale(d1, 2), scale(d2, 3));
            auto b = sub(scale(d1, 3), scale(d2, 2));
            v[0] = add(v[0], add(t1, t2));
            v[1] = add(u, a);
            v[4] = sub(u, a);
            v[2] = add(w, b);
            v[3] = sub(w, b);
            for (unsigned k = 0; k < 5; ++k) {
                if constexpr (!inverse)
                    if (k)
                        v[k] = rotate(v[k], k);
                save(row + k * n + j, v[k]);
            }
        }
        if constexpr (!inverse)
            for (unsigned k = 0; k < 5; ++k)
                short_transform<false>(row + k * n, n, tw);
    }
    template <bool inverse> static void row_transform(U* row, const Plan& plan, const Weights& tw) {
        if (plan.cols % 5 == 0) {
            radix_five<inverse>(row, plan, tw);
            return;
        }
        if (plan.cols % 3) {
            short_transform<inverse>(row, plan.cols, tw);
            return;
        }
        const size_t n = plan.cols / 3;
        const auto &factor = plan.extra[inverse];
        if constexpr (inverse)
            for (size_t i = 0; i < 3; ++i)
                short_transform<true>(row + i * n, n, tw);
        for (size_t j = 0; j < n; j += 8) {
            V x = load(row + j), y = load(row + n + j), z = load(row + 2 * n + j);
            V w1 = load(factor.one.w.data() + j), q1 = load(factor.one.q.data() + j);
            V w2 = load(factor.two.w.data() + j), q2 = load(factor.two.q.data() + j);
            if constexpr (inverse) {
                y = shoup(y, w1, q1);
                z = shoup(z, w2, q2);
            }
            V d = shoup(twice_reduce(y + splat(2 * p) - z), splat(factor.root), splat(factor.quot));
            V a = twice_reduce(x + twice_reduce(y + z));
            V b = twice_reduce(twice_reduce(x + splat(2 * p) - z) + d);
            V c = twice_reduce(twice_reduce(x + splat(2 * p) - y) + splat(2 * p) - d);
            if constexpr (!inverse) {
                b = shoup(b, w1, q1);
                c = shoup(c, w2, q2);
            }
            save(row + j, a);
            save(row + n + j, b);
            save(row + 2 * n + j, c);
        }
        if constexpr (!inverse)
            for (size_t i = 0; i < 3; ++i)
                short_transform<false>(row + i * n, n, tw);
    }
    template <bool pair, bool packed = false, bool triple = false, class Reader>
    static void convolution(U* x, U* y, U* z, const Plan& plan, int workers, bool square, size_t nx,
                            size_t ny, size_t nz, U scale, Reader read, bool columns_ready = false, bool compact = false, size_t skip = 0) {
        const size_t rows = plan.rows, cols = plan.cols;
        const auto& f = plan.table->f;
        const auto& b = plan.table->b;
        U scale_q = (Wide(scale) << 52) / p;
        const Layout layout(rows * cols, columns_ready);
        auto do_row = [&](size_t i) {
            auto forward = [&](U* row) {
                if (rows > 1)
                    rotate_row(row, cols, plan.f[i]);
                row_transform<false>(row, plan, f);
            };
            auto multiply = [&](U* row, U* common) {
                for (size_t j = 0; j < cols; j += 8) {
                    auto x = load(row + j), y = load(common + j);
                    if constexpr (triple) {
                        // The second Montgomery product is undone by the extra R in the inverse
                        // scale.
                        x = reduce(twice_reduce(x));
                        x = mont(x, x);
                    }
                    // For the smaller field both inputs below 2p give ab/R < p.
                    // In the larger field ab/R < 2p, so one subtraction of 2p suffices.
                    if constexpr (4 * p < R)
                        save(row + j, mont_lazy(twice_reduce(x), twice_reduce(y)));
                    else
                        save(row + j, twice_reduce(mont_lazy(x, y)));
                }
                row_transform<true>(row, plan, b);
                if (rows > 1)
                    rotate_row(row, cols, plan.b[i]);
            };
            auto get = [&](U* array, size_t slot) {
                if (!layout.tiled)
                    return array + i * cols;
                U* row = row_scratch.data() + slot * cols;
                for (size_t col = 0; col < cols; col += stripe) {
                    auto pos = layout.index(i, col);
                    if constexpr (packed) {
                        for (size_t j = 0; j < stripe; j += 8)
                            save(row + col + j, load_packed(array, pos + j));
                    } else
                        std::copy_n(array + pos, stripe, row + col);
                }
                return row;
            };
            auto put = [&](U* array, U* row) {
                if (layout.tiled)
                    for (size_t col = 0; col < cols; col += stripe) {
                        auto pos = layout.index(i, col);
                        if constexpr (packed) {
                            for (size_t j = 0; j < stripe; j += 8)
                                save_packed(array, pos + j, load(row + col + j));
                        } else
                            std::copy_n(row + col, stripe, array + pos);
                    }
            };
            if (layout.tiled)
                row_scratch.resize(cols * (square ? 1 : 2));
            U* row = get(x, 0);
            U* common = square ? row : get(y, 1);
            if (!square && rows > 1) {
                rotate_row<true>(row, cols, plan.f[i], common);
                row_transform<false>(row, plan, f);
                row_transform<false>(common, plan, f);
            } else {
                forward(row);
                if (!square)
                    forward(common);
            }
            multiply(row, common);
            put(x, row);
            if constexpr (pair) {
                U* second = get(z, 0);
                forward(second);
                multiply(second, common);
                put(z, second);
            }
        };
        auto reader = [&](int input) {
            return [&, input](U* dest, size_t pos, size_t count) { read(dest, pos, count, input); };
        };
        if (rows == 1) {
            read(x, 0, cols, 0);
            if (!square)
                read(y, 0, cols, 1);
            if constexpr (pair)
                read(z, 0, cols, 2);
            do_row(0);
            auto rescale = [&](U* data, size_t size) {
                for (size_t j = 0; j < size; j += 8)
                    save(data + j, reduce(shoup(load(data + j), splat(scale), splat(scale_q))));
            };
            rescale(x, std::min(cols, nx + ny + (triple ? nx : 0)));
            if constexpr (pair)
                rescale(z, nz + ny);
            return;
        }
        const size_t width = packed ? stripe : rows >= 8192 ? 8 : rows >= 2048 ? 16 : stripe;
        if (!columns_ready) {
            blocks(cols / width, workers, [&](size_t i) {
                columns<false, packed>(x, rows, cols, i * width, f, nx, scale, scale_q, reader(0),
                                       layout, width);
                if (!square)
                    columns<false, packed>(y, rows, cols, i * width, f, ny, scale, scale_q, reader(1),
                                           layout, width);
                if constexpr (pair)
                    columns<false, packed>(z, rows, cols, i * width, f, nz, scale, scale_q, reader(2),
                                           layout, width);
            });
        }
        blocks(rows, workers, do_row);
        const size_t output_width = compact ? stripe : width;
        blocks(cols / output_width, workers, [&](size_t i) {
            columns<true, packed>(x, rows, cols, i * output_width, b, nx + ny + (triple ? nx : 0), scale,
                                  scale_q, reader(0), layout, output_width, compact, skip / cols);
            if constexpr (pair)
                columns<true, packed>(z, rows, cols, i * output_width, b, nz + ny, scale, scale_q,
                                      reader(2), layout, output_width, compact, skip / cols);
        });
    }
};
} // namespace ap::ntt
