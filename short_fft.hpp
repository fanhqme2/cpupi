// Multiplication of short base-10^4 polynomials modulo X^n-i.
// The imaginary component holds the upper half of the linear convolution.
// With at most 32768 points, the product of the two input 2-norms
// is at most 32768 * 9999^2. A radix-2 level has unitary rotations and
// sums/differences. Allowing 16 binary64 unit roundoffs per level for
// rounded weights, products and additions in the two forward transforms,
// point product and inverse gives an absolute error below 0.27 per output
// component. The optional three-way plans are at most 24576 points. The
// radix-three sum, difference, rounded sin(2*pi/3), and final division by
// three fit within two additional binary levels at that shorter length.
// Five-way plans stop at 20480. Allowing four additional binary levels
// for the five-point butterfly, rotation and scale keeps the bound below 0.25.
namespace short_fft {
using Vec = __m512d;
struct Complex {
    Vec re, im;
};
struct Plan {
    std::vector<double> re, im, re3, im3;
    explicit Plan(size_t n) : re(n + 8), im(n + 8), re3(n / 2 + 8), im3(n / 2 + 8) {
        for (size_t groups = 1; groups < n; groups *= 2) {
            size_t reverse = 0;
            for (size_t g = 0; g < groups; ++g) {
                // The first split uses sqrt(i). Each child selects one
                // of its parent's two square roots.
                long double angle = std::numbers::pi_v<long double> * (reverse + 0.25L) / groups;
                re[groups + g] = std::cos(angle);
                im[groups + g] = std::sin(angle);
                if (groups < n / 2) {
                    re3[groups + g] = std::cos(3 * angle);
                    im3[groups + g] = std::sin(3 * angle);
                }
                size_t bit = groups / 2;
                while (bit && (reverse & bit)) {
                    reverse ^= bit;
                    bit /= 2;
                }
                reverse ^= bit;
            }
        }
    }
};
inline const Plan &get_plan(size_t n) {
    static std::once_flag flags[16];
    static std::unique_ptr<Plan> plans[16];
    auto k = std::countr_zero(n);
    std::call_once(flags[k], [=] { plans[k] = std::make_unique<Plan>(n); });
    return *plans[k];
}
inline Complex times(Complex a, Complex b) {
    return {_mm512_fmsub_pd(a.re, b.re, a.im * b.im), _mm512_fmadd_pd(a.re, b.im, a.im * b.re)};
}
template <bool inverse> inline void butterfly(Complex &a, Complex &b, Complex w) {
    if constexpr (!inverse)
        b = times(b, w);
    Complex d{a.re - b.re, a.im - b.im};
    a = {a.re + b.re, a.im + b.im};
    if constexpr (inverse)
        w.im = -w.im;
    b = inverse ? times(d, w) : d;
}
template <int from, int to> inline void reorder(Complex &a, Complex &b) {
    using F = Field<prime0>;
    auto part = [](Vec &x, Vec &y) {
        auto u = _mm512_castpd_si512(x), v = _mm512_castpd_si512(y);
        F::template reorder<from, to>(u, v);
        x = _mm512_castsi512_pd(u);
        y = _mm512_castsi512_pd(v);
    };
    part(a.re, b.re);
    part(a.im, b.im);
}
template <int half, bool inverse>
inline void last(Complex &a, Complex &b, const Plan &plan, size_t offset) {
    const auto indices =
        _mm512_setr_epi64(0, 1 / half, 2 / half, 3 / half, 4 / half, 5 / half, 6 / half, 7 / half);
    Complex w{_mm512_permutexvar_pd(indices, _mm512_loadu_pd(plan.re.data() + offset)),
              _mm512_permutexvar_pd(indices, _mm512_loadu_pd(plan.im.data() + offset))};
    butterfly<inverse>(a, b, w);
}
template <bool inverse> void transform(Complex *data, size_t n, const Plan &plan) {
    auto short_levels = [&](size_t begin, size_t end) {
        for (size_t i = begin; i < end; i += 2) {
            auto a = data[i], b = data[i + 1];
            if constexpr (!inverse) {
                reorder<0, 4>(a, b);
                last<4, false>(a, b, plan, n / 8 + i);
                reorder<4, 2>(a, b);
                last<2, false>(a, b, plan, n / 4 + 2 * i);
                reorder<2, 1>(a, b);
                last<1, false>(a, b, plan, n / 2 + 4 * i);
            } else {
                last<1, true>(a, b, plan, n / 2 + 4 * i);
                reorder<1, 2>(a, b);
                last<2, true>(a, b, plan, n / 4 + 2 * i);
                reorder<2, 4>(a, b);
                last<4, true>(a, b, plan, n / 8 + i);
                reorder<4, 0>(a, b);
            }
            data[i] = a;
            data[i + 1] = b;
        }
    };
    constexpr auto plus = [](Complex a, Complex b) { return Complex{a.re + b.re, a.im + b.im}; };
    constexpr auto minus = [](Complex a, Complex b) { return Complex{a.re - b.re, a.im - b.im}; };
    auto rotate = [](Complex x, Complex w) {
        if constexpr (inverse)
            w.im = -w.im;
        return times(x, w);
    };
    auto levels = [&](size_t h, size_t limit, size_t begin, size_t end) {
        while (inverse ? h < limit : h > limit) {
            if (inverse ? h * 4 > n / 8 : h == 1) {
                size_t groups = n / (16 * h);
                for (size_t g = begin / (2 * h); g < end / (2 * h); ++g) {
                    Complex w{_mm512_set1_pd(plan.re[groups + g]), _mm512_set1_pd(plan.im[groups + g])};
                    auto block = data + 2 * h * g;
                    for (size_t j = 0; j < h; ++j)
                        butterfly<inverse>(block[j], block[j + h], w);
                }
                break;
            }
            size_t stride = inverse ? h : h / 2, groups = n / (32 * stride);
            auto broadcast = [](const auto &r, const auto &i, size_t idx) {
                return Complex{_mm512_set1_pd(r[idx]), _mm512_set1_pd(i[idx])};
            };
            for (size_t g = begin / (4 * stride); g < end / (4 * stride); ++g) {
                auto w1 = broadcast(plan.re, plan.im, 2 * (groups + g));
                auto w2 = broadcast(plan.re, plan.im, groups + g);
                auto w3 = broadcast(plan.re3, plan.im3, 2 * (groups + g));
                auto block = data + 4 * stride * g;
                for (size_t j = 0; j < stride; ++j) {
                    auto a = block[j], b = block[j + stride], c = block[j + 2 * stride],
                         d = block[j + 3 * stride];
                    if constexpr (inverse) {
                        std::swap(b, c);
                    } else {
                        b = rotate(b, w1);
                        c = rotate(c, w2);
                        d = rotate(d, w3);
                    }
                    auto ac = plus(a, c), bd = plus(b, d), delta = minus(b, d);
                    auto difference = minus(a, c);
                    Complex imag =
                        inverse ? Complex{delta.im, -delta.re} : Complex{-delta.im, delta.re};
                    a = plus(ac, bd);
                    c = minus(ac, bd);
                    b = plus(difference, imag);
                    d = minus(difference, imag);
                    if constexpr (inverse) {
                        b = rotate(b, w1);
                        c = rotate(c, w2);
                        d = rotate(d, w3);
                    } else {
                        std::swap(b, c);
                    }
                    block[j] = a;
                    block[j + stride] = b;
                    block[j + 2 * stride] = c;
                    block[j + 3 * stride] = d;
                }
            }
            h = inverse ? h * 4 : h / 4;
        }
    };
    const size_t extent = n / 8;
    // Finish the short stages while both halves of a tile are in L1.
    constexpr size_t tile = 64;
    if constexpr (inverse) {
        if (extent <= tile) {
            short_levels(0, extent);
            levels(1, extent, 0, extent);
        } else {
            for (size_t i = 0; i < extent; i += tile) {
                short_levels(i, i + tile);
                levels(1, tile, i, i + tile);
            }
            levels(tile, extent, 0, extent);
        }
    } else {
        size_t chunk = extent;
        while (chunk > tile)
            chunk /= 4;
        levels(extent / 2, chunk / 2, 0, extent);
        for (size_t i = 0; i < extent; i += chunk) {
            levels(chunk / 2, 0, i, i + chunk);
            short_levels(i, i + chunk);
        }
    }
}
// Conjugate the coefficients to map X^(3m)=i to X^(3m)=-i. With
// Y^m=i and Z^3=1 we can now substitute X=YZ. The three blocks at
// offsets 0,m,2m acquire factors 1,i,-1. Group their lanes by the
// exponent of Z, then evaluate at the three cube roots of unity.
// All three branches share the same binary right-angle plan.
template <bool inverse> void transform_three(Complex* data, size_t n, const Plan& plan) {
    const size_t m = n / 3, length = m / 8;
    auto subtransforms = [&] {
        for (size_t branch = 0; branch < 3; ++branch)
            transform<inverse>(data + branch * length, m, plan);
    };
    if constexpr (inverse)
        subtransforms();
    const auto half = _mm512_set1_pd(-0.5);
    const auto sine = _mm512_set1_pd(inverse ? -0.86602540378443864676 : 0.86602540378443864676);
    // Move each term into the slot given by its exponent modulo three.
    // A mask chooses one cyclic permutation independently in each lane.
    constexpr unsigned masks[] = {0x49, 0x92, 0x24};
    auto permute = [](Complex& a, Complex& b, Complex& c, unsigned one, unsigned two) {
        auto choose = [&](Vec x, Vec y, Vec z) {
            return _mm512_mask_mov_pd(_mm512_mask_mov_pd(x, one, z), two, y);
        };
        Complex u{choose(a.re, b.re, c.re), choose(a.im, b.im, c.im)};
        Complex v{choose(b.re, c.re, a.re), choose(b.im, c.im, a.im)};
        c = {choose(c.re, a.re, b.re), choose(c.im, a.im, b.im)};
        a = u;
        b = v;
    };
    for (size_t j = 0; j < length; ++j) {
        Complex a = data[j], b = data[j + length], c = data[j + 2 * length];
        unsigned phase = (8 * j) % 3;
        unsigned one = masks[(1 + 3 - phase) % 3], two = masks[(2 + 3 - phase) % 3];
        // X = Y*w^(t/m mod 3) labels the branch by X^m=i*w^t.
        if (m % 3 == 2)
            std::swap(one, two);
        if constexpr (!inverse) {
            a.im = -a.im;
            b = {b.im, b.re}; // i times the conjugate
            c.re = -c.re;
            permute(a, b, c, one, two);
        }
        Complex sum{b.re + c.re, b.im + c.im};
        Complex mean{_mm512_fmadd_pd(half, sum.re, a.re), _mm512_fmadd_pd(half, sum.im, a.im)};
        Complex diff{c.im - b.im, b.re - c.re};
        a = {a.re + sum.re, a.im + sum.im};
        b = {_mm512_fmadd_pd(sine, diff.re, mean.re), _mm512_fmadd_pd(sine, diff.im, mean.im)};
        c = {_mm512_fnmadd_pd(sine, diff.re, mean.re), _mm512_fnmadd_pd(sine, diff.im, mean.im)};
        if constexpr (inverse) {
            permute(a, b, c, two, one);
            a.im = -a.im;
            b = {b.im, b.re};
            c.re = -c.re;
        }
        data[j] = a;
        data[j + length] = b;
        data[j + 2 * length] = c;
    }
    if constexpr (!inverse)
        subtransforms();
}
inline Complex sum(Complex a, Complex b) { return {a.re + b.re, a.im + b.im}; }
inline Complex difference(Complex a, Complex b) { return {a.re - b.re, a.im - b.im}; }
inline Complex quarter_turn(Complex a) { return {-a.im, a.re}; }
inline Complex negate(Complex a) { return {-a.re, -a.im}; }
inline Complex add_scaled(Complex a, double weight, Complex b) {
    auto w = _mm512_set1_pd(weight);
    return {_mm512_fmadd_pd(w, b.re, a.re), _mm512_fmadd_pd(w, b.im, a.im)};
}

// For n=5m, substitute X=YZ with Y^m=i and Z^5=1. A block at bm
// contributes i^b. In branch k, Z^m is the kth fifth root of unity,
// so lane j needs the fifth-root phase k*j*m^(-1) modulo five. All branches reuse the same binary
// plan.
template <bool inverse> void transform_five(Complex* data, size_t n, const Plan& plan) {
    size_t m = n / 5, span = m / 8;
    auto binary = [&] {
        for (size_t k = 0; k < 5; ++k)
            transform<inverse>(data + k * span, m, plan);
    };
    if constexpr (inverse)
        binary();
    static const auto weights = [] {
        std::array<std::array<Complex, 5>, 5> table{};
        for (unsigned start = 0; start < 5; ++start)
            for (unsigned k = 0; k < 5; ++k) {
                alignas(64) double re[8], im[8];
                for (unsigned lane = 0; lane < 8; ++lane) {
                    auto phase = ((start + lane) * k) % 5;
                    auto angle = (inverse ? -2 : 2) * std::numbers::pi_v<long double> * phase / 5;
                    re[lane] = std::cos(angle);
                    im[lane] = std::sin(angle);
                }
                table[start][k] = {_mm512_load_pd(re), _mm512_load_pd(im)};
            }
        return table;
    }();
    size_t index = m % 5, inv = (index * index * index) % 5;
    for (size_t j = 0; j < span; ++j) {
        auto a = data[j], b = data[span + j], c = data[2 * span + j], d = data[3 * span + j],
             e = data[4 * span + j];
        const auto& w = weights[(8 * j) % 5];
        auto rotate = [&] {
            b = times(b, w[inv]);
            c = times(c, w[2 * inv % 5]);
            d = times(d, w[3 * inv % 5]);
            e = times(e, w[4 * inv % 5]);
        };
        if constexpr (inverse)
            rotate();
        else {
            b = quarter_turn(b);
            c = negate(c);
            d = negate(quarter_turn(d));
        }
        auto t1 = sum(b, e), t2 = sum(c, d), t = sum(t1, t2), v = difference(t1, t2);
        auto mean = add_scaled(a, -0.25, t);
        auto m1 = add_scaled(mean, 0.5590169943749474241, v),
             m2 = add_scaled(mean, -0.5590169943749474241, v);
        constexpr double s1 = inverse ? -0.9510565162951535721 : 0.9510565162951535721;
        constexpr double s2 = inverse ? -0.5877852522924731292 : 0.5877852522924731292;
        auto v1 = quarter_turn(difference(b, e)), v2 = quarter_turn(difference(c, d));
        auto z1 = add_scaled(add_scaled(Complex{_mm512_setzero_pd(), _mm512_setzero_pd()}, s1, v1),
                             s2, v2),
             z2 = add_scaled(add_scaled(Complex{_mm512_setzero_pd(), _mm512_setzero_pd()}, s2, v1),
                             -s1, v2);
        a = sum(a, t);
        b = sum(m1, z1);
        e = difference(m1, z1);
        c = sum(m2, z2);
        d = difference(m2, z2);
        if constexpr (!inverse)
            rotate();
        if constexpr (inverse) {
            b = negate(quarter_turn(b));
            c = negate(c);
            d = quarter_turn(d);
        }
        data[j] = a;
        data[span + j] = b;
        data[2 * span + j] = c;
        data[3 * span + j] = d;
        data[4 * span + j] = e;
    }
    if constexpr (!inverse)
        binary();
}
inline Vec unpack(std::span<const uint32_t> a, size_t j) {
    if (j / 2 + 4 <= a.size()) {
        // Interleave the low and high base-10^4 digits of four limbs.
        auto x = _mm256_cvtepu32_epi64(
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(a.data() + j / 2)));
        auto q = _mm256_srli_epi64(_mm256_mul_epu32(x, _mm256_set1_epi64x(3518437209ULL)), 45);
        auto r = x - _mm256_mul_epu32(q, _mm256_set1_epi64x(10000));
        return _mm512_cvtepu32_pd(r | _mm256_slli_epi64(q, 32));
    }
    alignas(64) double v[8]{};
    for (unsigned k = 0; k < 8 && (j + k) / 2 < a.size(); k += 2) {
        v[k] = a[(j + k) / 2] % 10000;
        v[k + 1] = a[(j + k) / 2] / 10000;
    }
    return _mm512_load_pd(v);
}
// Normalize eight decimal limbs at once. Each rounded coefficient is below
// 32768 * 9999^2. Combining a pair and an existing output word stays below
// 2^55. The two quotient estimates below are exact over these ranges.
inline void recover(const Complex* x, size_t n, std::span<uint32_t> output) {
    const auto scale = _mm512_set1_pd(1. / n);
    const auto radix = _mm512_set1_epi64(base);
    const auto even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const auto odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    auto coefficients = [&](size_t i) {
        Vec v = i < n ? x[i / 8].re : x[(i - n) / 8].im;
        return _mm512_cvt_roundpd_epi64(v * scale, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    };
    __m512i previous_high{}, previous_extra{};
    unsigned carry_bit = 0;
    size_t j = 0;
    for (; j + 8 <= output.size(); j += 8) {
        auto a = coefficients(2 * j), b = coefficients(2 * j + 8);
        auto v = _mm512_permutex2var_epi64(a, even, b);
        v += _mm512_mullo_epi64(_mm512_permutex2var_epi64(a, odd, b), _mm512_set1_epi64(10000));
        v += _mm512_cvtepu32_epi64(_mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(output.data() + j)));

        // base = 256 * 390625. After the first shift, rounding the
        // reciprocal upward introduces less than 1/390625 of a quotient.
        constexpr U reciprocal = ((Wide(1) << 70) + 390624) / 390625;
        auto high = _mm512_srli_epi64(_mm512_madd52hi_epu64(
            _mm512_setzero_si512(), _mm512_srli_epi64(v, 8), _mm512_set1_epi64(reciprocal)), 18);
        v -= _mm512_mul_epu32(high, radix);
        v += _mm512_alignr_epi64(high, previous_high, 7);
        previous_high = high;

        // v is below 430 million, so a 32-by-32 product divides it by base.
        constexpr U reciprocal32 = ((U(1) << 58) + base - 1) / base;
        auto extra = _mm512_srli_epi64(_mm512_mul_epu32(v, _mm512_set1_epi64(reciprocal32)), 58);
        v -= _mm512_mul_epu32(extra, radix);
        v += _mm512_alignr_epi64(extra, previous_extra, 7);
        previous_extra = extra;
        unsigned generates = _mm512_cmpge_epu64_mask(v, radix);
        unsigned propagates = _mm512_cmpeq_epi64_mask(v, _mm512_set1_epi64(base - 1));
        unsigned carries = (propagates + (2 * generates + carry_bit)) ^ propagates;
        v = _mm512_mask_add_epi64(v, carries, v, _mm512_set1_epi64(1));
        v = _mm512_mask_sub_epi64(v, carries >> 1, v, radix);
        carry_bit = carries >> 8;
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output.data() + j), _mm512_cvtepi64_epi32(v));
    }
    if (j == output.size())
        return;
    auto last = previous_high + previous_extra;
    U carry = U(_mm_extract_epi64(_mm512_extracti64x2_epi64(last, 3), 1)) + carry_bit;
    for (; j < output.size(); j += 4) {
        alignas(64) U c[8];
        _mm512_store_si512(c, coefficients(2 * j));
        for (size_t k = 0; k < std::min<size_t>(4, output.size() - j); ++k) {
            carry += c[2 * k] + 10000 * c[2 * k + 1] + output[j + k];
            output[j + k] = carry % base;
            carry /= base;
        }
    }
}
// One forward transform of the shorter input can serve several disjoint
// blocks of the longer input. Their products overlap by b.size() limbs.
Nat product(const Nat &a, const Nat &b, size_t n) {
    const bool mixed = !std::has_single_bit(n);
    const auto& plan = get_plan(n % 5 == 0 ? n / 5 : mixed ? n / 3 : n);
    // Each block has eight real lanes followed by the eight imaginary lanes.
    thread_local std::vector<Complex> storage;
    const bool square = &a == &b;
    const size_t needed = n / 8 * (square ? 1 : 2);
    if (storage.size() < needed)
        storage.resize(needed);
    auto x = storage.data(), y = square ? x : x + n / 8;
    auto forward = [&](auto z, const auto &src) {
        for (size_t i = 0; i < n; i += 8) {
            z[i / 8].re = i < 2 * src.size() ? unpack(src, i) : _mm512_setzero_pd();
            z[i / 8].im = i + n < 2 * src.size() ? unpack(src, i + n) : _mm512_setzero_pd();
        }
        if (n % 5 == 0)
            transform_five<false>(z, n, plan);
        else if (mixed)
            transform_three<false>(z, n, plan);
        else
            transform<false>(z, n, plan);
    };
    if (!square)
        forward(y, b);
    Nat out(a.size() + b.size());
    const size_t block = n - b.size();
    for (size_t start = 0; start < a.size(); start += block) {
        auto part = std::span<const uint32_t>(a).subspan(start, std::min(block, a.size() - start));
        forward(x, part);
        for (size_t i = 0; i < n / 8; ++i)
            x[i] = times(x[i], y[i]);
        if (n % 5 == 0)
            transform_five<true>(x, n, plan);
        else if (mixed)
            transform_three<true>(x, n, plan);
        else
            transform<true>(x, n, plan);
        recover(x, n, std::span(out).subspan(start, part.size() + b.size()));
        // Prefix products cannot reach the limb above start+used-1.
    }
    trim(out);
    return out;
}
} // namespace short_fft
