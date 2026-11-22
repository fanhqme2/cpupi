#include "integer.hpp"
#include "leaf.hpp"
#include "parallel.hpp"
#include <array>
#include <bit>
#include <mutex>
#include <map>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <sched.h>
#include <openssl/sha.h>
#include <limits>

namespace {
using namespace ap;
using U = uint64_t;
using W = __uint128_t;
Nat wide(W x) {
    Nat a;
    while (x) {
        a.push_back(x % base);
        x /= base;
    }
    return a;
}
Nat shifted(Nat a, size_t places, int threads = 1);
struct Series {
    Nat p, q, t;
    bool exact = true;
    size_t q_shift = 0; // The denominator is q * base^q_shift.
};
Nat drop(const Nat& a, size_t places, int threads = 1);

// Allow a few units of absolute error below the guard words. If only K
// high words survive, words below the top K+1 of either input contribute
// less than 1/base to the scaled result.
Nat high_product(const Nat& a, const Nat& b, size_t shift, size_t cut, int threads) {
    if (a.empty() || b.empty())
        return {};
    if (shift >= cut)
        return shifted(mul(a, b, threads), shift - cut, threads);
    cut -= shift;
    if (cut >= a.size() + b.size())
        return {};
    size_t keep = a.size() + b.size() - cut + 1;
    size_t skip_a = a.size() - std::min(keep, a.size());
    size_t skip_b = b.size() - std::min(keep, b.size());
    Nat short_a, short_b;
    if (skip_a)
        short_a = drop(a, skip_a, threads);
    if (skip_b)
        short_b = drop(b, skip_b, threads);
    return mul_high(skip_a ? short_a : a, skip_b ? short_b : b, cut - skip_a - skip_b, threads);
}
// Each numerator arithmetic progression covers every residue class modulo
// any prime >= 5. Thus h consecutive leaves contribute at least 3*v_p(h!)
// copies of p to P. At p=3 only the (2k-1) progression contributes.
// The denominator's h consecutive cubes supply all of these factors too.
struct WordDivisor {
    U divisor, reciprocal;
};
using CancellationTable = std::array<std::vector<WordDivisor>, 10>;
const CancellationTable& cancellations() {
    static const auto table = [] {
        CancellationTable entries;
        for (int p = 3; p <= 256; p += 2) {
            bool prime = true;
            for (int q = 3; q * q <= p; q += 2)
                if (p % q == 0)
                    prime = false;
            if (!prime)
                continue;
            int removed = 0;
            for (unsigned depth = 3; depth <= 8; ++depth) {
                int count = 0;
                for (int v = (1 << depth) / p; v; v /= p)
                    count += v;
                count *= p == 3 ? 1 : 3;
                auto& factors = entries[depth];
                for (int i = removed; i < count; ++i) {
                    if (factors.empty() || factors.back().divisor > (UINT64_MAX / base) / p)
                        factors.push_back({1, 0});
                    factors.back().divisor *= p;
                }
                removed += count;
            }
        }
        for (auto& factors : entries)
            for (auto& f : factors)
                f.reciprocal = (W(1) << 64) / f.divisor;
        return entries;
    }();
    return table;
}
void divide_exact(Nat& a, const WordDivisor& factor) {
    U rem = 0;
    for (size_t i = a.size(); i--;) {
        U value = rem * base + a[i];
        U q = (W(value) * factor.reciprocal) >> 64;
        rem = value - q * factor.divisor;
        if (rem >= factor.divisor) {
            ++q;
            rem -= factor.divisor;
        }
        a[i] = q;
    }
    if (rem)
        throw std::logic_error("inexact common-factor removal");
    trim(a);
}
Nat reciprocal(const Nat& d, size_t precision, int threads);
Nat expand_factors(const std::vector<U>& words, size_t lo, size_t hi) {
    if (hi - lo == 1)
        return number(words[lo]);
    auto mid = lo + (hi - lo) / 2;
    return mul(expand_factors(words, lo, mid), expand_factors(words, mid, hi));
}
bool wide_level(size_t h) {
    return h == 4096 || h == 65536 || h == 1048576;
}
Nat common_divisor(unsigned h) {
    std::vector<bool> composite(h + 1);
    std::vector<U> groups;
    U word = 1;
    for (unsigned p = 3; p <= h; p += 2) {
        if (composite[p])
            continue;
        if (p <= h / p)
            for (size_t i = p * p; i <= h; i += 2 * p)
                composite[i] = true;
        unsigned removed = 0;
        for (unsigned size = 8;; size *= 2) {
            unsigned count = 0;
            for (auto v = size / p; v; v /= p)
                count += v;
            count *= p == 3 ? 1 : 3;
            if (size == h) {
                for (unsigned i = removed; i < count; ++i) {
                    if (word > UINT64_MAX / p) {
                        groups.push_back(word);
                        word = 1;
                    }
                    word *= p;
                }
                break;
            }
            if (size <= 256 || wide_level(size))
                removed += count;
            else
                removed *= 2;
        }
    }
    groups.push_back(word);
    return expand_factors(groups, 0, groups.size());
}
struct WideDivisor {
    const Nat d;
    std::mutex lock;
    std::map<size_t, Nat> inverses;
    explicit WideDivisor(size_t h) : d(common_divisor(h)) {}
    const Nat& inverse(size_t k) {
        std::lock_guard guard(lock);
        auto it = inverses.find(k);
        if (it != inverses.end())
            return it->second;
        return inverses.emplace(k, reciprocal(d, k, 1)).first->second;
    }
    void divide(Nat& a, int workers) {
        size_t k = a.size() - d.size() + 6;
        // The exact quotient is an integer, so a few high guard words of
        // the dividend and a precomputed reciprocal suffice for recovery.
        size_t table_k = std::bit_ceil(k);
        const Nat& inv = inverse(table_k);
        Nat x = drop(inv, table_k - k);
        size_t skip = d.size() > 3 ? d.size() - 3 : 0;
        Nat product = mul(drop(a, skip), x, workers);
        size_t cutoff = k + d.size() - skip;
        bool up = product.size() >= cutoff && product[cutoff - 1] >= base / 2;
        a = drop(product, cutoff);
        if (up)
            a = add(std::move(a), {1});
    }
};
WideDivisor& wide_divisor(size_t h) {
    static std::once_flag flags[21];
    static std::unique_ptr<WideDivisor> tables[21];
    auto i = std::countr_zero(h);
    std::call_once(flags[i], [=] { tables[i] = std::make_unique<WideDivisor>(h); });
    return *tables[i];
}
// Large exact subtrees can hold both children's P/Q/T, partial products
// and transform buffers at the same time. Keep the largest merges serial
// across siblings, while giving each sibling the entire worker team.
size_t simultaneous_terms() {
    // Allow 384 bytes per term for simultaneously live siblings and products.
    static const size_t terms = available_memory_bytes() / 384;
    return terms;
}
// t is the magnitude of the local alternating sum, with denominator q.
Series split(U lo, U hi, bool need_p, int threads, size_t precision) {
    if (hi - lo <= (hi <= U(UINT32_MAX) + 1 ? 8u : 4u)) {
        LeafSum leaf;
        leaf_sum(leaf, lo, hi, need_p);
        return {need_p ? leaf.p.expand() : Nat{}, leaf.q.expand(), leaf.t.expand(), true,
                leaf.shift};
    }
    U mid = lo + (hi - lo) / 2;
    if (hi - lo > 128)
        mid = lo + ((mid - lo + 32) / 64) * 64;
    Series left, right;
    if (threads > 1 && hi - lo > simultaneous_terms()) {
        right = split(mid, hi, need_p, threads, precision);
        left = split(lo, mid, true, threads, precision);
    } else if (threads > 1 && hi - lo >= 512) {
        int left_workers =
            std::clamp<int>((threads * (mid - lo) + (hi - lo) / 2) / (hi - lo), 1, threads - 1);
        Team::active()->fork(
            left_workers, [&] { left = split(lo, mid, true, left_workers, precision); },
            [&] { right = split(mid, hi, need_p, threads - left_workers, precision); });
    } else {
        left = split(lo, mid, true, 1, precision);
        right = split(mid, hi, need_p, 1, precision);
    }
    U h = mid - lo;
    if (lo && h == hi - mid && h <= 256 && std::has_single_bit(h) && left.exact && right.exact) {
        for (auto factor : cancellations()[std::countr_zero(h)]) {
            divide_exact(left.p, factor);
            divide_exact(right.q, factor);
        }
    }
    if (lo && h == hi - mid && wide_level(h) && left.exact && right.exact) {
        auto& divisor = wide_divisor(h);
        divisor.divide(left.p, threads);
        divisor.divide(right.q, threads);
    }
    // The original prefix ratio at lo is bounded by 10^(-14.18*(lo-1)).
    // Cap all three entries at a common scale. Q's implicit shift is exact.
    size_t saved = lo ? size_t((lo - 1) * 14.18L / 8) : 0;
    size_t cap = precision - std::min(precision - 8, saved);
    size_t shift = left.q_shift + right.q_shift;
    size_t length = left.q.size() + right.q.size() + shift;
    size_t cut = length - std::min(cap, length);
    Series result;
    result.exact = left.exact && right.exact && cut == 0;
    result.q_shift = shift - std::min(shift, cut);
    Nat a, b;
    bool have_q = !cut && shared_products(right.q, left.t, left.q, a, result.q, threads);
    if (have_q)
        a = shifted(std::move(a), right.q_shift);
    else
        a = high_product(left.t, right.q, right.q_shift, cut, threads);
    Nat{}.swap(left.t);
    bool have_p = !cut && need_p && shared_products(left.p, right.t, right.p, b, result.p, threads);
    if (!have_p)
        b = high_product(left.p, right.t, 0, cut, threads);
    Nat{}.swap(right.t);
    if (!need_p)
        Nat{}.swap(left.p);
    result.t = (mid - lo) % 2 ? sub(std::move(a), b) : add(std::move(a), b);
    Nat{}.swap(b);
    if (need_p && !have_p)
        result.p = high_product(left.p, right.p, 0, cut, threads);
    Nat{}.swap(left.p);
    Nat{}.swap(right.p);
    if (!have_q)
        result.q = high_product(left.q, right.q, 0, cut - std::min(shift, cut), threads);
    return result;
}

Nat shifted(Nat a, size_t places, int threads) {
    if (places && !a.empty())
        return copy_limbs(a, 0, places, threads);
    return a;
}
#include "factored.hpp"

Nat drop(const Nat& a, size_t places, int threads) {
    if (threads > 1 && a.size() > places + 262144)
        return copy_limbs(a, places, 0, threads);
    if (a.size() <= places)
        return {};
    return {a.begin() + places, a.end()};
}
Nat unit(size_t places) {
    Nat a(places + 1);
    a.back() = 1;
    return a;
}
bool less(const Nat& a, const Nat& b) {
    if (a.size() != b.size())
        return a.size() < b.size();
    return std::lexicographical_compare(a.rbegin(), a.rend(), b.rbegin(), b.rend());
}
Nat normalized(const Nat& a, size_t original, size_t precision, int threads) {
    return precision >= original ? copy_limbs(a, 0, precision - original, threads)
                                 : drop(a, original - precision, threads);
}
// Newton residuals have a known small magnitude. A cyclic convolution
// therefore recovers them even when the two unreduced products are longer.
struct Difference {
    Nat magnitude;
    bool negative;
};
Difference difference(Nat target, Nat value, size_t period, int threads) {
    auto fold = [period, threads](Nat& a) {
        while (period && a.size() > period) {
            Nat upper = drop(a, period, threads);
            a.resize(period);
            trim(a);
            a = add(std::move(a), upper, threads);
        }
    };
    fold(target);
    fold(value);
    bool negative = less(target, value);
    Nat delta = negative ? sub(std::move(value), target, threads) : sub(std::move(target), value, threads);
    if (period && delta.size() == period && delta.back() >= base / 2) {
        constexpr size_t chunk = 262144;
        blocks((period + chunk - 1) / chunk, threads, [&](size_t block) {
            size_t end = std::min(period, (block + 1) * chunk);
            for (size_t i = block * chunk; i < end; ++i)
                delta[i] = base - 1 - delta[i];
        });
        trim(delta);
        negative = !negative;
    }
    return {std::move(delta), negative};
}
// A Newton residual compares a power of the limb radix with a canonical
// product. Subtracting that power touches only its limb and a borrow chain.
Difference power_difference(size_t exponent, Nat value, size_t period, int threads) {
    if (period && value.size() > period)
        return difference(unit(exponent), std::move(value), period, threads);
    bool over = value.size() > exponent;
    if (over) {
        size_t i = exponent;
        while (value[i] == 0) {
            value[i++] = base - 1;
        }
        --value[i];
    } else if (value.empty()) {
        return {unit(exponent), false};
    } else {
        value.resize(exponent, 0);
        size_t i = 0;
        while (!value[i])
            ++i;
        value[i] = base - value[i];
        size_t start = i + 1;
        constexpr size_t chunk = 262144;
        blocks((exponent - start + chunk - 1) / chunk, threads, [&](size_t j) {
            for (size_t k = start + j * chunk; k < std::min(exponent, start + (j + 1) * chunk); ++k)
                value[k] = base - 1 - value[k];
        });
    }
    trim(value);
    if (period && value.size() == period && value.back() >= base / 2) {
        constexpr size_t chunk = 262144;
        blocks((period + chunk - 1) / chunk, threads, [&](size_t j) {
            for (size_t k = j * chunk; k < std::min(period, (j + 1) * chunk); ++k)
                value[k] = base - 1 - value[k];
        });
        over = !over;
        trim(value);
    }
    bool negative = over && !value.empty();
    return {std::move(value), negative};
}
// Choose precisions backwards so the penultimate iterate is half-sized.
std::vector<size_t> precisions(size_t m) {
    std::vector<size_t> steps;
    while (m > 10) {
        steps.push_back(m);
        m = (m + 1) / 2 + 2;
    }
    steps.push_back(m);
    if (m > 6)
        steps.push_back(6);
    steps.push_back(4);
    std::reverse(steps.begin(), steps.end());
    return steps;
}
Nat reciprocal(const Nat& d, size_t m, int workers) {
    long double lead = 0;
    for (size_t i = std::min<size_t>(4, d.size()); i > 0; --i)
        lead = (lead + d[d.size() - i]) / base;
    size_t k = 2;
    Nat x = wide(W(1.0e16L / lead));
    for (size_t s : precisions(m)) {
        Nat v = normalized(d, d.size(), s, workers);
        auto product = wrapped_mul(v, x, s + 5, workers);
        size_t exponent = product.period ? (s + k) % product.period : s + k;
        auto [err, high] =
            power_difference(exponent, std::move(product.value), product.period, workers);
        Nat delta = mul_high(x, err, 2 * k, workers);
        Nat old = shifted(std::move(x), s - k, workers);
        x = high ? sub(std::move(old), delta, workers) : add(std::move(old), delta, workers);
        // Before rounding the new error in ulps is err^2 / (v*B^(2k)).
        if (s == m && 2 * err.size() + 1 > s + 2 * k) {
            // A small leading limb can cost a Newton step when the input
            // precision increases. Refine against the fully normalized divisor.
            Nat target = unit(2 * s);
            for (;;) {
                Nat check = mul(v, x, workers);
                bool over = less(target, check);
                Nat residual = over ? sub(std::move(check), target) : sub(target, check);
                if (!less(add(v, v), residual))
                    break;
                Nat correction = mul_high(x, residual, 2 * s, workers);
                if (correction.empty())
                    throw std::runtime_error("reciprocal correction underflow");
                x = over ? sub(std::move(x), correction) : add(std::move(x), correction);
            }
        }
        k = s;
    }
    return x;
}
// Karp-Markstein division: use a half-precision reciprocal for both the
// initial quotient and the residual correction.
Nat quotient(Nat q, size_t shift, Nat d, size_t m, int workers) {
    size_t k = (m + 1) / 2 + 3;
    size_t scale = d.size();
    // The factored root commonly supplies a few extra guard limbs at the
    // same scale. Clear their discarded limbs in place and keep the common
    // radix factor through the residual. At 5B this avoids sliding two
    // 625-million-limb arrays to remove eleven words from each.
    size_t padding = !shift && scale > m && scale - m <= 64 && m >= 1024 ? scale - m : 0;
    if (padding) {
        std::fill_n(d.begin(), padding, 0);
        std::fill_n(q.begin(), std::min(padding, q.size()), 0);
        trim(q);
    } else {
        auto normalize = [&](Nat a, size_t offset) {
            if (scale <= m + offset)
                return shifted(std::move(a), m + offset - scale, workers);
            size_t cut = scale - m - offset;
            if (cut >= a.size())
                return Nat{};
            if (cut >= a.size() / 4)
                return drop(a, cut, workers);
            a.erase(a.begin(), a.begin() + cut);
            return a;
        };
        d = normalize(std::move(d), 0);
        q = normalize(std::move(q), shift);
    }
    const size_t input_precision = m + padding;
    Nat x = reciprocal(d, k, workers);
    Nat y = mul_high(drop(q, input_precision - k, workers), x, k, workers);
    auto product = wrapped_mul(d, y, input_precision + 5, workers);
    Nat{}.swap(d);
    Nat target;
    if (product.period > k && q.size() <= product.period && q.size() > product.period - k) {
        // A left shift modulo B^period-1 rotates the limbs. Construct
        // the rotation directly, without allocating the unfolded value.
        size_t split = product.period - k;
        target.reserve(product.period);
        target.insert(target.end(), q.begin() + split, q.end());
        target.resize(k);
        target.insert(target.end(), q.begin(), q.begin() + split);
        trim(target);
        Nat{}.swap(q);
    } else {
        target = shifted(std::move(q), k);
    }
    auto [err, high] = difference(std::move(target), std::move(product.value), product.period, workers);
    // Both normalized inputs carry B^padding, so the signed residual does
    // too. Drop those words along with the usual low correction window.
    size_t skip = padding + (m > k + 4 ? m - k - 4 : 0);
    Nat delta = mul_high(drop(err, skip, workers), x, 2 * k + padding - skip, workers);
    Nat old = shifted(std::move(y), m - k, workers);
    return high ? sub(std::move(old), delta, workers) : add(std::move(old), delta, workers);
}
void halve(Nat &dx, int workers) {
    // In an even radix the incoming remainder is just the low bit
    // of the next original limb. Preserve it at worker boundaries.
    constexpr size_t block = 32768;
    size_t count = (dx.size() + block - 1) / block;
    std::vector<uint32_t> upper(count);
    for (size_t i = 1; i < count; ++i)
        upper[i - 1] = dx[i * block] & 1;
    blocks(count, workers, [&](size_t i) {
        size_t end = std::min(dx.size(), (i + 1) * block);
        for (size_t j = i * block; j + 1 < end; ++j)
            dx[j] = dx[j] / 2 + (dx[j + 1] & 1) * (base / 2);
        dx[end - 1] = dx[end - 1] / 2 + upper[i] * (base / 2);
    });
    trim(dx);
}
Difference root_residual(const Nat &x, size_t k, int workers) {
    auto square = wrapped_mul(x, x, k + 5, workers);
    size_t exponent = square.period ? (2 * k) % square.period : 2 * k;
    return power_difference(exponent, mul(square.value, number(10005), workers), square.period,
                            workers);
}
Nat invsqrt(size_t m, int workers) {
    size_t k = 2;
    Nat x = number(U(1.0e16L / std::sqrt(10005.L)));
    for (size_t s : precisions(m)) {
        auto [error, negative] = root_residual(x, k, workers);
        if (negative && !error.empty())
            throw std::runtime_error("invalid inverse-square-root residual");
        // Flooring commutes with a subsequent integer division by two.
        // Drop the unused product limbs before carrying and halving.
        Nat dx = mul_high(x, error, 3 * k - s, workers);
        Nat{}.swap(error);
        // x <- x + x*(1 - 10005*x*x)/2, rounded down.
        halve(dx, workers);
        x = add(shifted(std::move(x), s - k, workers), dx, workers);
        k = s;
    }
    return x;
}
struct FreeText {
    void operator()(char* p) const noexcept { release_work_buffer(p); }
};
using DecimalText = std::unique_ptr<char[], FreeText>;

// Format eight base-10^8 limbs in most-significant-first order.
void print_eight_limbs(char* dest, const uint32_t* src) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
    auto words = _mm512_cvtepu32_epi64(
        _mm256_permutevar8x32_epi32(x, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0)));
    // At most eight decimal digits per word, four per group and two per pair.
    // The upward-rounded reciprocals are exact over each successive range.
    auto hi = _mm512_srli_epi64(_mm512_mul_epu32(words, _mm512_set1_epi64(3518437209ULL)), 45);
    auto lo = words - _mm512_mul_epu32(hi, _mm512_set1_epi64(10000));
    auto groups = hi | _mm512_slli_epi64(lo, 32);
    auto hundreds = _mm512_srli_epi16(_mm512_mulhi_epu16(groups, _mm512_set1_epi16(5243)), 3);
    auto remainder = _mm512_sub_epi32(groups, _mm512_mullo_epi16(hundreds, _mm512_set1_epi16(100)));
    auto pairs = hundreds | _mm512_slli_epi32(remainder, 16);
    auto tens = _mm512_mulhi_epu16(pairs, _mm512_set1_epi16(6554));
    auto ones = _mm512_sub_epi16(pairs, _mm512_mullo_epi16(tens, _mm512_set1_epi16(10)));
    auto ascii = _mm512_add_epi8(tens | _mm512_slli_epi16(ones, 8), _mm512_set1_epi8('0'));
    _mm512_storeu_si512(dest, ascii);
}
DecimalText compute_at(U digits, int workers, size_t extra) {
    const size_t limbs = digits / 8 + (digits % 8 != 0);
    if (extra > (std::numeric_limits<ptrdiff_t>::max() - 2) / 8 - limbs)
        throw std::length_error("decimal result exceeds addressable memory");
    size_t m = limbs + extra;
    U terms = U((8 * m) / 14.181647462725477L) + 3;
    // Factor indices fit in 32 bits. Keep the low-memory splitting
    // schedule for larger computations and machines with less free RAM.
    bool use_factors =
        terms >= 512 && terms <= UINT32_MAX / 6 && terms <= available_memory_bytes() / 192;
    const size_t k = std::min(m, (m + 1) / 2 + 6);
    Nat root, residual;
    auto prepare_root = [&](int count) {
        root = invsqrt(k, count);
        auto error = root_residual(root, k, count);
        if (error.negative || error.magnitude.size() > k + 2)
            throw std::runtime_error("inverse-square-root error exceeds precision bound");
        residual = std::move(error.magnitude);
        root = mul(root, number(426880ULL * 10005), count);
    };
    int preparation_group = 0;
    if (workers >= 8 && digits >= 10000000 && digits <= available_memory_bytes() / 20 &&
        !std::getenv("PI_TRANSFORM_MIB")) {
        preparation_group = workers / 2;
    }
    auto tree = use_factors
        ? SeriesFactors(terms, workers, m, preparation_group, prepare_root).run(terms, workers)
        : split(0, terms, false, workers, m);
    Nat ratio = quotient(std::move(tree.q), tree.q_shift, std::move(tree.t), m, workers);
    tree = {};
    // Apply the last inverse-square-root correction to the scaled quotient.
    // For e = 1 - 10005*(x/B^k)^2 the scale factor is 1+e/2+O(e^2).
    // The measured residual bounds e by B^(2-k). At k >= ceil(m/2)+6
    // the quadratic remainder is below one working ulp. Small m already
    // has a full-precision root and the CLI retains six guard words.
    if (root.empty())
        prepare_root(workers);
    Nat scaled = mul_high(ratio, root, k, workers);
    Nat{}.swap(root);
    Nat{}.swap(ratio);
    Nat correction = high_product(scaled, residual, 0, 2 * k, workers);
    Nat{}.swap(residual);
    halve(correction, workers);
    scaled = add(std::move(scaled), correction, workers);
    Nat{}.swap(correction);
    if (scaled.size() != m + 1 || scaled.back() != 3)
        throw std::runtime_error("invalid scale");
    static constexpr auto pairs = [] {
        std::array<std::array<char, 2>, 100> t{};
        for (int i = 0; i < 100; ++i)
            t[i] = {char('0' + i / 10), char('0' + i % 10)};
        return t;
    }();
    DecimalText out(static_cast<char*>(acquire_work_buffer(8 * m + 2)));
    out[0] = '3';
    out[1] = '.';
    blocks((m + 4095) / 4096, std::min<size_t>(workers, 1 + m / 32768), [&](size_t block) {
        size_t i = 4096 * block, end = std::min(m, i + 4096);
        for (; i + 8 <= end; i += 8)
            print_eight_limbs(out.get() + 2 + 8 * i, scaled.data() + m - i - 8);
        for (; i < end; ++i) {
            char* dest = out.get() + 2 + i * 8;
            unsigned v = scaled[m - 1 - i];
            for (int j = 6; j >= 0; j -= 2) {
                std::memcpy(dest + j, pairs[v % 100].data(), 2);
                v /= 100;
            }
        }
    });
    // Keep the last six words out of the boundary decision. If the
    // remaining guard digits straddle the cutoff, repeat at higher precision.
    std::string_view guard(out.get() + digits + 2, 8 * m - digits - 48);
    if (guard.find_first_not_of('0') == guard.npos || guard.find_first_not_of('9') == guard.npos)
        return {};
    return out;
}
DecimalText compute(U digits, int workers) {
    for (size_t extra = 8;; extra += 4) {
        auto result = compute_at(digits, workers, extra);
        if (result)
            return result;
    }
}
U parse_number(std::string_view s) {
    U n;
    auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), n);
    if (ec != std::errc{} || end != s.data() + s.size())
        throw std::invalid_argument("expected a nonnegative decimal integer");
    return n;
}
unsigned default_workers() {
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0)
        return std::clamp(CPU_COUNT(&allowed), 1, 32);
    return std::clamp(std::thread::hardware_concurrency(), 1u, 32u);
}
constexpr std::string_view usage =
    "Usage: pi_cpu DIGITS [--benchmark | --stdout | --output FILE] [--threads N]\n";
enum class Output { standard, file, benchmark };
struct Options {
    U digits;
    Output output = Output::standard;
    std::string_view file;
    unsigned threads = default_workers();
};
Options parse_options(int argc, char** argv) {
    if (argc < 2)
        throw std::invalid_argument("missing digit count");
    Options options{parse_number(argv[1]), Output::standard, {}};
    bool mode_set = false, threads_set = false;
    for (int i = 2; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--threads" || arg == "-t" || (!arg.empty() && arg.front() != '-')) {
            if (threads_set)
                throw std::invalid_argument("thread count specified twice");
            if (arg == "--threads" || arg == "-t") {
                if (++i == argc)
                    throw std::invalid_argument("missing thread count");
                arg = argv[i];
            }
            U n = parse_number(arg);
            if (!n || n > 256)
                throw std::invalid_argument("thread count must be between 1 and 256");
            options.threads = n;
            threads_set = true;
        } else {
            if (mode_set)
                throw std::invalid_argument("choose one output mode");
            if (arg == "--benchmark" || arg == "-b")
                options.output = Output::benchmark;
            else if (arg == "--output" || arg == "-o") {
                if (++i == argc || !argv[i][0])
                    throw std::invalid_argument("missing output path");
                options.file = argv[i];
                options.output = options.file == "-" ? Output::standard : Output::file;
            } else if (arg != "--stdout")
                throw std::invalid_argument("unknown option: " + std::string(arg));
            mode_set = true;
        }
    }
    return options;
}
CrtSchedule schedule_for_precision(U digits) {
    // Use the nearest measured precision at each schedule crossover.
    if (digits <= 1'500'000'000ULL)
        return CrtSchedule::decimal;
    if (digits <= 2'500'000'000ULL)
        return CrtSchedule::standard;
    if (digits <= 4'500'000'000ULL)
        return CrtSchedule::decimal;
    return CrtSchedule::standard;
}
void print_benchmark(std::string_view result, double seconds) {
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if (!SHA256(reinterpret_cast<const unsigned char*>(result.data()), result.size(), digest))
        throw std::runtime_error("SHA-256 failed");
    const auto tail = result.size() > 2 ? result.substr(2) : result;
    const auto last = tail.substr(tail.size() - std::min<size_t>(32, tail.size()));
    std::printf("Last digits: %.*s\nSHA-256: ", int(last.size()), last.data());
    for (auto byte : digest)
        std::printf("%02x", static_cast<unsigned>(byte));
    std::printf("\nComputation time: %.6f s\n", seconds);
    if (std::fflush(stdout) || std::ferror(stdout))
        throw std::runtime_error("could not write benchmark output");
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
            std::cout << usage;
            return 0;
        }
        const Options options = parse_options(argc, argv);
        const U digits = options.digits;
        const auto start = std::chrono::steady_clock::now();
        const size_t available = available_memory_bytes();
        if (digits > std::numeric_limits<ptrdiff_t>::max() - 128 || digits > available)
            throw std::length_error("decimal output alone exceeds available memory");
        if (available != SIZE_MAX && digits <= available / 16 && !std::getenv("PI_TRANSFORM_MIB"))
            reserve_buffer_arena(std::min<size_t>(12 * digits, available / 4));
        Team team(options.threads, schedule_for_precision(digits));
        DecimalText storage;
        std::string_view result = "3";
        if (digits) {
            storage = compute(digits, options.threads);
            result = {storage.get(), digits + 2};
        }
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (options.output == Output::benchmark) {
            print_benchmark(result, seconds);
        } else if (options.output == Output::file) {
            std::ofstream file;
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file.open(std::string(options.file), std::ios::binary);
            file << result << '\n';
            file.close();
        } else {
            std::cout << result << '\n' << std::flush;
            if (!std::cout)
                throw std::runtime_error("could not write decimal output");
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "CpuPi: " << e.what() << '\n' << usage;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "CpuPi: " << e.what() << '\n';
        return 1;
    }
}
