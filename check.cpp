#include "integer.hpp"
#include "leaf.hpp"
#include "parallel.hpp"
#include <random>
#include <cstdio>
#include <stdexcept>
#include <string_view>

namespace {
void check_series_leaves() {
    using namespace ap;
    // At these indices the term itself no longer fits in a 128-bit product.
    for (uint64_t k : {(1ULL << 43) + 7, (1ULL << 57) + 3}) {
        LeafSum term;
        leaf_term(term, k);
        Nat p = mul(mul(number(6 * k - 5), number(2 * k - 1)), number(6 * k - 1));
        Nat q = mul(mul(number(k), number(k)), mul(number(k), number(10939058860032ULL)));
        Nat t = mul(p, add(mul(number(545140134), number(k)), number(13591409)));
        for (unsigned j = 0; j < (8 - 3 * k % 8) % 8; ++j)
            t = mul(t, number(10));
        if (term.p.expand() != p || term.q.expand() != q || term.t.expand() != t)
            throw std::runtime_error("wide-index Chudnovsky term mismatch");
    }
    for (uint64_t lo : {0ULL, 1ULL, 1000000ULL, 350000000ULL, (1ULL << 32) - 8,
                        7050000000ULL, (1ULL << 57) + 3}) {
        Nat p{1}, q{1}, t;
        size_t shift = 0;
        const unsigned count = lo <= UINT32_MAX - 7 ? 8 : 4;
        for (uint64_t k = lo; k < lo + count; ++k) {
            LeafSum term;
            leaf_term(term, k);
            auto left = mul(t, term.q.expand());
            if (!left.empty())
                left.insert(left.begin(), term.shift, 0);
            auto right = mul(p, term.t.expand());
            t = (k - lo) % 2 ? sub(std::move(left), right) : add(std::move(left), right);
            p = mul(p, term.p.expand());
            q = mul(q, term.q.expand());
            shift += term.shift;
            for (unsigned mask = 0; mask < 4; ++mask) {
                LeafSum leaf;
                leaf_sum_with(leaf, lo, k + 1, leaf_term, mask & 1, mask & 2);
                if (leaf.t.expand() != t || leaf.shift != shift ||
                    ((mask & 1) && leaf.p.expand() != p) || ((mask & 2) && leaf.q.expand() != q))
                    throw std::runtime_error("fixed-storage series merge differs from separate products");
            }
        }
    }
}
void check_product(size_t m, size_t n, bool square) {
    using namespace ap;
    Nat a(m, base - 1), b(n, base - 1);
    Nat product = square ? mul(a, a, 16) : mul(a, b, 16);
    if (product.size() != m + n)
        throw std::runtime_error("wrong product length");
    // (B^m-1)(B^n-1) = B^(m+n) - B^m - B^n + 1.
    for (size_t i = 0; i < product.size(); ++i) {
        unsigned expected = i == 0 ? 1 : i < n ? 0 : (i == m ? base - 2 : base - 1);
        if (product[i] != expected)
            throw std::runtime_error("carry propagation mismatch");
    }
}
// Check FFT packing, terminal lane order and overlapping blocks against
// an independent exact convolution. These shapes straddle dispatch boundaries.
void check_short_products() {
    using namespace ap;
    std::mt19937 rng(47281);
    for (auto [m, n] : {std::pair{63, 61},
                        {313, 23},
                        {32767, 1},
                        {65537, 2},
                        {198, 193},
                        {257, 150},
                        {619, 411},
                        {1025, 513},
                        {360, 408},
                        {3121, 2511},
                        {10000, 9000},
                        {12800, 7680},
                        {2990, 2090},
                        {320, 320},
                        {7001, 157},
                        {8191, 8192},
                        {16383, 16384},
                        {16037, 311}}) {
        Nat a(m), b(n);
        for (auto *v : {&a, &b})
            for (auto &x : *v)
                x = rng() % base;
        std::vector<__uint128_t> sums(m + n);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                sums[i + j] += uint64_t(a[i]) * b[j];
        Nat expected(m + n);
        __uint128_t carry = 0;
        for (int i = 0; i < m + n; ++i) {
            carry += sums[i];
            expected[i] = carry % base;
            carry /= base;
        }
        trim(expected);
        if (mul(a, b) != expected)
            throw std::runtime_error("short transform differs from exact convolution");
    }
}
void check_transform_variants() {
    using namespace ap;
    std::mt19937 rng(1729);
    for (size_t n : {4096, 16384, 65536}) {
        Nat a(n), b(n - 7), c(n - 11);
        for (auto* v : {&a, &b, &c})
            for (auto& x : *v)
                x = rng() % base;
        a.back() = b.back() = c.back() = base - 1;
        Nat ab = mul(a, b, 16), ac = mul(a, c, 16), x, y;
        for (size_t cut : {size_t(1), n / 2, n + 3, ab.size() - 1, ab.size()}) {
            Nat expected(ab.begin() + cut, ab.end());
            if (mul_high(a, b, cut, 16) != expected)
                throw std::runtime_error("high product disagrees with full product");
        }
        if (shared_products(a, b, c, x, y, 16) && (ab != x || ac != y))
            throw std::runtime_error("shared-transform mismatch");
        auto wrapped = wrapped_mul(a, b, n + 5, 16);
        if (wrapped.period) {
            while (ab.size() > wrapped.period) {
                Nat high(ab.begin() + wrapped.period, ab.end());
                ab.resize(wrapped.period);
                trim(ab);
                ab = add(std::move(ab), high);
            }
            if (ab.size() == wrapped.period &&
                std::all_of(ab.begin(), ab.end(), [](auto v) { return v == base - 1; }))
                ab.clear();
        }
        if (ab != wrapped.value)
            throw std::runtime_error("wrapped-product mismatch");
    }
    // The square has a long string of zero low words. This catches a
    // carry across a clipped tile boundary in a low-workspace run.
    Nat all_nines(131072, base - 1);
    Nat upper(all_nines);
    --upper.front();
    if (mul_high(all_nines, all_nines, all_nines.size(), 16) != upper)
        throw std::runtime_error("high square carry mismatch");
    // The retained operand times three ends in five all-nines words.
    // The discarded tail contributes two more units at that position.
    // The high product must retry when the guard cannot certify the carry.
    Nat edged(8192, base - 1);
    constexpr size_t retained = 16;
    std::fill_n(edged.end() - retained, 5, (base - 1) / 3);
    auto entire = mul(edged, {3});
    size_t cut = edged.size() + 1 - (retained - 4);
    if (mul_high(edged, {3}, cut, 16) != Nat(entire.begin() + cut, entire.end()))
        throw std::runtime_error("discarded operand tail lost a carry");
    Nat nines(64, base - 1), unit(65);
    unit.back() = 1;
    if (add(nines, {1}) != unit || sub(unit, {1}) != nines)
        throw std::runtime_error("SIMD carry chain mismatch");
}
} // namespace
int main(int argc, char** argv) {
    try {
        // A small arena exercises splitting, coalescing and conventional
        // allocations once the simultaneous products outgrow the region.
        if (!std::getenv("PI_TRANSFORM_MIB"))
            ap::reserve_buffer_arena(64ULL << 20);
        ap::Team team(16);
        check_series_leaves();
        check_short_products();
        check_transform_variants();
        if (!ap::mul({}, ap::number(1)).empty() || !ap::sub({1}, {1}).empty())
            throw std::runtime_error("zero mismatch");
        for (size_t n : {1, 24, 64, 129, 767, 768, 769, 8193, 16387, 131073}) {
            check_product(n, n, true);
            check_product(n + 19, n, false);
            check_product(9 * n, n, false);
        }
        check_product(100003, 2, false);
        if (argc == 2 && std::string_view(argv[1]) == "--large") {
            // Cross the one-prime coefficient bound above eleven million limbs.
            check_product(12000000, 12000000, true);
            check_product(16000000, 12000000, false);
            // These sizes exercise wide linear plans through nine moduli. All-nines
            // operands exercise CRT quotient estimation and cross-block carries.
            for (size_t n : {14690001, 11260001, 12590001, 32000003})
                check_product(n + 1, n, false);
            // The first pair is a square by alias, but its forward
            // transform must survive for the second product.
            for (size_t n : {12590001, 32000003}) {
                ap::Nat common(n, ap::base - 1), other(common), x, y;
                --other[0];
                if (ap::shared_products(common, common, other, x, y, 16)) {
                    ap::Nat expected(2 * n, ap::base - 1);
                    std::fill_n(expected.begin(), n, 0);
                    expected[0] = 1;
                    --expected[n];
                    if (x != expected || y != ap::sub(expected, common))
                        throw std::runtime_error("wide shared products disagree");
                    for (size_t cut : {n, 2 * n + 1}) {
                        ap::Nat upper(expected.begin() + std::min(cut, expected.size()), expected.end());
                        if (!ap::shared_products(common, common, other, x, y, 16, cut) ||
                            x != upper || y != ap::sub(expected, common))
                            throw std::runtime_error("truncated shared products disagree");
                        auto second = ap::sub(expected, common);
                        ap::Nat high(second.begin() + std::min(cut, second.size()), second.end());
                        if (!ap::shared_products(common, common, other, x, y, 16, 0, cut) ||
                            x != expected || y != high)
                            throw std::runtime_error("truncated second shared product disagrees");
                    }
                }
            }
            // These shared plans pack six and seven distinct decimal limbs per
            // coefficient. Uniform inputs can conceal a wrong column stride.
            for (size_t n : {574000, 520000}) {
                std::mt19937 rng(53);
                ap::Nat common(n), left(n), right(n), x, y;
                for (auto* operand : {&common, &left, &right})
                    for (auto& digit : *operand)
                        digit = rng() % ap::base;
                bool shared = ap::shared_products(common, left, right, x, y, 16);
                if (!shared && !std::getenv("PI_TRANSFORM_MIB"))
                    throw std::runtime_error("expected a wide shared plan");
                if (shared && (x != ap::mul(common, left, 16) || y != ap::mul(common, right, 16)))
                    throw std::runtime_error("wide shared digit columns disagree");
            }
            // An all-nines guard can propagate a carry from a discarded
            // inverse row. The exact retry must reproduce (B^m-1)(B^m+1).
            {
                constexpr size_t m = 2000000, cut = m / 2;
                ap::Nat left(m, ap::base - 1), right(m + 1);
                right.front() = right.back() = 1;
                if (ap::mul_high(left, right, cut, 16) != ap::Nat(2 * m - cut, ap::base - 1))
                    throw std::runtime_error("ambiguous high-product guard mismatch");
            }
            // Verify an upper slice around the long carry chain in a wide CRT square.
            const size_t n = 16000000;
            ap::Nat nines(n, ap::base - 1);
            for (size_t cut : {n, n + 4096}) {
                ap::Nat expected(2 * n - cut, ap::base - 1);
                if (cut == n)
                    --expected[0];
                if (ap::mul_high(nines, nines, cut, 16) != expected)
                    throw std::runtime_error("wide upper square mismatch");
            }
            // At 918000 limbs a packed full transform can shorten to an untiled
            // cyclic transform. Check both storage layouts against the closed form.
            for (size_t limbs : {918000, 2929688, 12590001}) {
                ap::Nat a(limbs, ap::base - 1);
                auto wrapped = ap::wrapped_mul(a, a, a.size() + 3, 16);
                // (B^n-1)^2 = B^(2n) - 2 B^n + 1 modulo B^period-1.
                size_t length = wrapped.period ? wrapped.period : 2 * a.size();
                ap::Nat expected(length);
                expected[2 * a.size() - length] = 1;
                expected[a.size()] = ap::base - 2;
                std::fill(expected.begin() + a.size() + 1, expected.end(), ap::base - 1);
                if (wrapped.value != expected)
                    throw std::runtime_error("wide cyclic square mismatch");
            }
        }
        {
            // The incoming carry and borrow cross multiple worker blocks.
            constexpr size_t n = 3 * 262144 + 11;
            ap::Nat nines(n, ap::base - 1), one(n, 0), power(n + 1);
            one[0] = 1;
            one.back() = 1;
            power.back() = 1;
            power[n - 1] = 1;
            if (ap::add(nines, one, 4) != power || ap::sub(power, one, 4) != nines)
                throw std::runtime_error("parallel carry chain mismatch");
        }
        {
            // Closed form (B^n-1)^3, including both long borrow chains.
            size_t n = 32769;
            ap::Nat a(n, ap::base - 1), expected(3 * n, ap::base - 1);
            expected[n] = 2;
            std::fill(expected.begin() + n + 1, expected.begin() + 2 * n, 0);
            expected[2 * n] = ap::base - 3;
            if (ap::mul_square_times(a, a, 16) != expected)
                throw std::runtime_error("cubic convolution carry mismatch");
            ap::Nat b(n + 17);
            std::mt19937 rng(71);
            for (auto& digit : b)
                digit = rng() % ap::base;
            if (ap::mul_square_times(a, b, 16) != ap::mul(ap::mul(a, a, 16), b, 16))
                throw std::runtime_error("squared operand times third operand mismatch");
        }
        check_product(10240, 10240, true);
        check_product(12288, 12288, true);
        puts("Integer carry and multiplication checks passed.");
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
