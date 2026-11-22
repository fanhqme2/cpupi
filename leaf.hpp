#pragma once
#include "integer.hpp"
#include <array>

namespace ap {
// Eight terms fit through a 32-bit term index. The general splitting tree
// uses at most four terms per leaf above that range.
struct LeafInt {
    std::array<uint32_t, 56> data;
    size_t size;
    void set(__uint128_t n) {
        size = 0;
        // Extract two decimal limbs per hardware divide. Reducing the high
        // half first guarantees that divq's quotient fits in 64 bits.
        constexpr uint64_t pair_base = uint64_t(base) * base;
        while (n >> 64) {
            uint64_t high = n >> 64, low, rem;
            uint64_t upper = high / pair_base;
            high -= upper * pair_base;
            asm("divq %[divisor]"
                : "=a"(low), "=d"(rem)
                : "a"(uint64_t(n)), "d"(high), [divisor] "r"(pair_base)
                : "cc");
            data[size++] = rem % base;
            data[size++] = rem / base;
            n = (__uint128_t(upper) << 64) | low;
        }
        for (uint64_t word = n; word; word /= base)
            data[size++] = word % base;
    }
    Nat expand() const {
        return {data.begin(), data.begin() + size};
    }
};
inline void leaf_product(LeafInt& out, const LeafInt& a, const LeafInt& b, size_t shift = 0) {
    out.size = a.size + b.size + shift;
    std::fill_n(out.data.begin(), shift, 0);
    uint64_t carry = 0;
    for (size_t i = 0; i < a.size + b.size; ++i) {
        size_t first = i >= b.size ? i - b.size + 1 : 0;
        size_t end = std::min(a.size, i + 1);
        for (size_t j = first; j < end; ++j)
            carry += uint64_t(a.data[j]) * b.data[i - j];
        out.data[i + shift] = carry % base;
        carry /= base;
    }
    while (out.size && !out.data[out.size - 1])
        --out.size;
}
struct LeafSum {
    LeafInt p, q, t;
    size_t shift;
};
template <bool prefix_supplies_p = false>
inline void leaf_term_scaled(LeafSum& out, uint64_t k, __uint128_t numerator, uint64_t denominator,
                             uint64_t correction) {
    using W = __uint128_t;
    if (!k) {
        out.p.set(prefix_supplies_p ? numerator : 1);
        out.q.set(1);
        out.t.set(13591409);
        out.shift = 0;
        return;
    }
    constexpr uint64_t decimal[] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000};
    out.p.set(numerator);
    LeafInt scratch;
    LeafInt& linear = prefix_supplies_p ? out.t : scratch;
    const W linear_weight = W(13591409 + 545140134 * k) * correction;
    W scaled_weight;
    auto scale = decimal[(8 - 3 * k % 8) % 8];
    if (!__builtin_mul_overflow(linear_weight, W(scale), &scaled_weight))
        linear.set(scaled_weight);
    else {
        LeafInt a, b;
        a.set(linear_weight);
        b.set(scale);
        leaf_product(linear, a, b);
    }
    if constexpr (!prefix_supplies_p)
        leaf_product(out.t, out.p, linear);
    W cube = W(denominator) * denominator * denominator;
    constexpr uint64_t c = 10939058860032ULL;
    if (cube <= ~W(0) / c)
        out.q.set(cube * c);
    else {
        LeafInt v, w;
        v.set(cube);
        w.set(c);
        leaf_product(out.q, v, w);
    }
    out.shift = (3 * k + 7) / 8 - (3 * (k - 1) + 7) / 8;
}
inline void leaf_term(LeafSum& out, uint64_t k) {
    if (k <= UINT32_MAX) {
        leaf_term_scaled(out, k, __uint128_t(6 * k - 5) * (2 * k - 1) * (6 * k - 1), k, 1);
        return;
    }
    // The three-factor products can exceed 128 bits at large indices.
    // Split them into two factors before converting to decimal limbs.
    LeafInt a, b, cube;
    a.set(__uint128_t(6 * k - 5) * (2 * k - 1));
    b.set(6 * k - 1);
    leaf_product(out.p, a, b);
    a.set(__uint128_t(k) * k);
    b.set(k);
    leaf_product(cube, a, b);
    b.set(10939058860032ULL);
    leaf_product(out.q, cube, b);
    __uint128_t linear = 13591409 + __uint128_t(545140134) * k;
    for (unsigned i = 0; i < (8 - 3 * k % 8) % 8; ++i)
        linear *= 10;
    a.set(linear);
    leaf_product(out.t, out.p, a);
    out.shift = (3 * k + 7) / 8 - (3 * (k - 1) + 7) / 8;
}
template <class Term>
inline void leaf_sum_with(LeafSum& out, uint64_t lo, uint64_t hi, const Term& term,
                          bool need_p = true, bool need_q = true) {
    if (hi - lo == 1) {
        term(out, lo);
        return;
    }
    auto mid = lo + (hi - lo) / 2;
    LeafSum l, r;
    leaf_sum_with(l, lo, mid, term, true, need_q);
    leaf_sum_with(r, mid, hi, term, need_p, true);
    if (need_p)
        leaf_product(out.p, l.p, r.p);
    if (need_q)
        leaf_product(out.q, l.q, r.q);
    out.shift = l.shift + r.shift;
    // Accumulate both products before normalizing. With at most eight terms,
    // each diagonal has fewer than 56 products of digits below base.
    // Adding bias*(base-1) per limb and bias at the start adds bias*base^len.
    // Subtract the bias from the final carry to recover the signed sum.
    size_t len = std::max(l.t.size + r.q.size + r.shift, l.p.size + r.t.size);
    uint64_t sums[56]{};
    auto accumulate = [&]<bool subtract>(const LeafInt& a, const LeafInt& b, size_t shift) {
        for (size_t i = 0; i < a.size; ++i) {
            uint64_t x = a.data[i];
            for (size_t j = 0; j < b.size; ++j) {
                uint64_t v = x * b.data[j];
                if constexpr (subtract)
                    sums[i + j + shift] -= v;
                else
                    sums[i + j + shift] += v;
            }
        }
    };
    accumulate.template operator()<false>(l.t, r.q, r.shift);
    if ((mid - lo) % 2)
        accumulate.template operator()<true>(l.p, r.t, 0);
    else
        accumulate.template operator()<false>(l.p, r.t, 0);
    constexpr uint64_t bias = 64ULL * base;
    uint64_t carry = bias;
    for (size_t i = 0; i < len; ++i) {
        uint64_t v = sums[i] + bias * (base - 1) + carry;
        out.t.data[i] = v % base;
        carry = v / base;
    }
    out.t.size = len;
    if (carry > bias)
        out.t.data[out.t.size++] = carry - bias;
    while (out.t.size && !out.t.data[out.t.size - 1])
        --out.t.size;
}
inline void leaf_sum(LeafSum& out, uint64_t lo, uint64_t hi, bool need_p = true) {
    leaf_sum_with(out, lo, hi, leaf_term, need_p, true);
}
} // namespace ap
