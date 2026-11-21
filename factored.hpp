// Factored binary splitting for the Chudnovsky series.
struct Power {
    uint32_t prime;
    uint32_t exponent;
};
using Factors = std::vector<Power, ap::LimbAllocator<Power>>;

class SeriesFactors {
    std::vector<uint16_t, ap::LimbAllocator<uint16_t>> smallest;
    const size_t precision;
    const int preparation_group;
    mutable std::atomic_flag preparing = ATOMIC_FLAG_INIT;
    const std::function<void(int)> prepare_root;
    // P is odd and every exponent of two in Q is divisible by three.
    // Store 8 as a factor. Its maximum count is 4*N + v2(N!) < 5*N,
    // so both keys and counts fit in a word at the sieve index limit.
    // Frequent small factors share counters throughout a leaf. Sort the
    // remaining primes as 32-bit keys before constructing the factor list.
    struct LeafFactors {
        std::array<unsigned, 64> small{};
        std::vector<uint32_t> large;
        void sort_primes() {
            if (large.size() < 1024) {
                std::sort(large.begin(), large.end());
                return;
            }
            std::vector<uint32_t> scratch(large.size());
            for (unsigned shift = 0; shift < 32; shift += 8) {
                std::array<unsigned, 256> offsets{};
                for (auto prime : large)
                    ++offsets[(prime >> shift) & 255];
                unsigned total = 0;
                for (auto& offset : offsets) {
                    unsigned count = offset;
                    offset = total;
                    total += count;
                }
                for (auto prime : large)
                    scratch[offsets[(prime >> shift) & 255]++] = prime;
                large.swap(scratch);
            }
        }
        Factors finish(unsigned multiplicity, unsigned terms = 0) {
            small[8] += 4 * terms;
            small[23] += terms;
            small[29] += terms;
            Factors result;
            result.reserve(large.size() + 16);
            for (unsigned p = 2; p < small.size(); ++p) {
                auto exponent = small[p] * (p == 8 ? 1 : multiplicity) + (p == 3 ? 2 * terms : 0);
                if (exponent)
                    result.push_back({p, exponent});
            }
            sort_primes();
            for (auto p : large) {
                if (!result.empty() && result.back().prime == p)
                    result.back().exponent += multiplicity;
                else
                    result.push_back({p, multiplicity});
            }
            return result;
        }
    };
    // Factor a block of odd integers one sieve lookup at a time. Each
    // round packs the surviving cofactors at the front of the input.
    void factor_block(LeafFactors& factors, std::vector<uint32_t>& values) const {
        static const auto inverse = [] {
            std::array<uint32_t, 32768> table{};
            for (uint32_t i = 0; i < table.size(); ++i) {
                uint32_t p = 2 * i + 1, x = p;
                for (unsigned step = 0; step < 4; ++step)
                    x *= 2 - p * x;
                table[i] = x;
            }
            return table;
        }();
        size_t count = values.size(), emitted = 0;
        factors.large.resize(2 * count + 16);
        while (count) {
            size_t next = 0;
            for (size_t i = 0; i < count; i += 16) {
                __mmask16 active = (1u << std::min<size_t>(16, count - i)) - 1;
                auto x = _mm512_maskz_loadu_epi32(active, values.data() + i);
                active &= _mm512_cmpgt_epu32_mask(x, _mm512_set1_epi32(1));
                // Gather aligned pairs of 16-bit sieve entries. Bit one of
                // the odd input selects the high or low entry in the pair.
                auto pair = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), active,
                                                        _mm512_srli_epi32(x, 2), smallest.data(), 4);
                auto shift = _mm512_slli_epi32(x & _mm512_set1_epi32(2), 3);
                auto p = _mm512_srlv_epi32(pair, shift) & _mm512_set1_epi32(65535);
                __mmask16 composite = active & _mm512_cmpneq_epi32_mask(p, _mm512_setzero_si512());
                auto factor = _mm512_mask_blend_epi32(composite, x, p);
                __mmask16 big = active & _mm512_cmpge_epu32_mask(factor, _mm512_set1_epi32(64));
                if (emitted + 16 > factors.large.size())
                    factors.large.resize(2 * factors.large.size());
                _mm512_mask_compressstoreu_epi32(factors.large.data() + emitted, big, factor);
                emitted += std::popcount(big);

                alignas(64) uint32_t small[16];
                __mmask16 small_mask = active & ~big;
                _mm512_mask_compressstoreu_epi32(small, small_mask, factor);
                for (int j = 0; j < std::popcount(small_mask); ++j)
                    ++factors.small[small[j]];

                // p divides x exactly. Multiplication by its inverse mod
                // 2^32 therefore returns the ordinary quotient in one word.
                auto inv = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), composite,
                                                       _mm512_srli_epi32(p, 1), inverse.data(), 4);
                _mm512_mask_compressstoreu_epi32(values.data() + next, composite,
                                                 _mm512_mullo_epi32(x, inv));
                next += std::popcount(composite);
            }
            count = next;
        }
        factors.large.resize(emitted);
    }
    // Split at prime boundaries. Each worker cancels and compacts a disjoint
    // range, then the surviving exponents are moved together in key order.
    static void cancel(Factors& p, Factors& q, int threads) {
        const size_t tasks = std::min<size_t>(threads, 1 + p.size() / 32768);
        struct Slice {
            size_t p0, q0, np, nq;
        };
        auto compact = [&](size_t p0, size_t p1, size_t q0, size_t q1) {
            size_t i = p0, j = q0, wp = p0, wq = q0;
            while (i < p1 && j < q1) {
                auto a = p[i], b = q[j];
                if (a.prime < b.prime) {
                    p[wp++] = a;
                    ++i;
                } else if (b.prime < a.prime) {
                    q[wq++] = b;
                    ++j;
                } else {
                    if (a.exponent > b.exponent)
                        p[wp++] = {a.prime, a.exponent - b.exponent};
                    if (b.exponent > a.exponent)
                        q[wq++] = {b.prime, b.exponent - a.exponent};
                    ++i;
                    ++j;
                }
            }
            if (wp != i)
                std::move(p.begin() + i, p.begin() + p1, p.begin() + wp);
            if (wq != j)
                std::move(q.begin() + j, q.begin() + q1, q.begin() + wq);
            return Slice{p0, q0, wp - p0 + p1 - i, wq - q0 + q1 - j};
        };
        if (tasks == 1) {
            auto s = compact(0, p.size(), 0, q.size());
            p.resize(s.np);
            q.resize(s.nq);
            return;
        }
        std::vector<Slice> slices(tasks + 1);
        for (size_t i = 1; i < tasks; ++i) {
            auto pi = p.size() * i / tasks;
            auto qi = std::lower_bound(q.begin(), q.end(), p[pi].prime,
                                       [](Power v, uint32_t key) { return v.prime < key; }) -
                      q.begin();
            slices[i] = {pi, size_t(qi), 0, 0};
        }
        slices.back() = {p.size(), q.size(), 0, 0};
        blocks(tasks, threads, [&](size_t i) {
            auto v = compact(slices[i].p0, slices[i + 1].p0, slices[i].q0, slices[i + 1].q0);
            slices[i].np = v.np;
            slices[i].nq = v.nq;
        });
        size_t np = slices[0].np, nq = slices[0].nq;
        for (size_t i = 1; i < tasks; ++i) {
            auto v = slices[i];
            if (np != v.p0)
                std::move(p.begin() + v.p0, p.begin() + v.p0 + v.np, p.begin() + np);
            if (nq != v.q0)
                std::move(q.begin() + v.q0, q.begin() + v.q0 + v.nq, q.begin() + nq);
            np += v.np;
            nq += v.nq;
        }
        p.resize(np);
        q.resize(nq);
    }
    static Factors join(const Factors& a, const Factors& b) {
        Factors out;
        out.reserve(a.size() + b.size());
        size_t i = 0, j = 0;
        while (i < a.size() || j < b.size()) {
            Power v;
            if (j == b.size() || (i < a.size() && a[i].prime < b[j].prime))
                v = a[i++];
            else if (i == a.size() || b[j].prime < a[i].prime)
                v = b[j++];
            else {
                v = a[i++];
                v.exponent += b[j++].exponent;
            }
            if (v.exponent)
                out.push_back(v);
        }
        return out;
    }
    static Nat words_product(const std::vector<U, ap::LimbAllocator<U>>& v, size_t a, size_t b,
                             int threads) {
        if (b - a <= 12) {
            Nat out(3 * (b - a) + 1);
            out[0] = 1;
            size_t used = 1;
            for (size_t i = a; i < b; ++i) {
                U w = v[i], low = w % base, mid = w / base % base, high = w / base / base;
                U x1 = 0, x2 = 0, carry = 0;
                for (size_t j = 0; j < used; ++j) {
                    U x = out[j];
                    U t = x * low + x1 * mid + x2 * high + carry;
                    out[j] = t % base;
                    carry = t / base;
                    x2 = x1;
                    x1 = x;
                }
                carry += x1 * mid + x2 * high;
                out[used++] = carry % base;
                carry = carry / base + x1 * high;
                while (carry) {
                    out[used++] = carry % base;
                    carry /= base;
                }
                while (!out[used - 1])
                    --used;
            }
            out.resize(used);
            return out;
        }
        auto mid = a + (b - a) / 2;
        Nat x, y;
        if (threads > 1 && b - a >= 4096) {
            int w = threads / 2;
            Team::active()->fork(
                w, [&] { x = words_product(v, a, mid, w); },
                [&] { y = words_product(v, mid, b, threads - w); });
        } else {
            x = words_product(v, a, mid, 1);
            y = words_product(v, mid, b, 1);
        }
        return mul(x, y, threads);
    }
    // Pack small factors into machine words. For high exponents, form
    // the product of the half powers once and square it.
    static Nat expand(const Factors& factors, int threads) {
        Factors powers;
        std::vector<U, ap::LimbAllocator<U>> words;
        U word = 1;
        for (auto [p, e] : factors) {
            if (e > 4) {
                powers.push_back({p, e / 2});
                e %= 2;
            }
            while (e--) {
                U next;
                if (__builtin_mul_overflow(word, U(p), &next)) {
                    words.push_back(word);
                    word = p;
                } else
                    word = next;
            }
        }
        if (word != 1 || words.empty())
            words.push_back(word);
        auto value = words_product(words, 0, words.size(), threads);
        if (!powers.empty()) {
            auto root = expand(powers, threads);
            auto squared = mul(root, root, threads);
            value = mul(value, squared, threads);
        }
        return value;
    }
    // Every denominator has large powers of the same four primes. Cache
    // powers of their product shared by disjoint subtrees.
    static const Nat& denominator_power(unsigned mantissa, unsigned level, int threads) {
        struct Entry {
            std::once_flag ready;
            Nat value;
        };
        static Entry powers[16][32];
        Entry& entry = powers[mantissa - 16][level];
        std::call_once(entry.ready, [&] {
            if (level == 0) {
                Nat base_power = number(24 * 10939058860032ULL);
                entry.value = number(1);
                for (unsigned e = mantissa; e; e >>= 1) {
                    if (e & 1)
                        entry.value = mul(entry.value, base_power);
                    if (e > 1)
                        base_power = mul(base_power, base_power);
                }
            } else {
                const Nat& half = denominator_power(mantissa, level - 1, threads);
                entry.value = mul(half, half, threads);
            }
        });
        return entry.value;
    }
    static Nat expand_denominator(const Factors& factors, int threads) {
        if (factors.size() < 16)
            return expand(factors, threads);
        Factors roots, remainder;
        roots.reserve(factors.size());
        remainder.reserve(factors.size() / 2);
        for (auto [p, e] : factors) {
            if (e >= 3)
                roots.push_back({p, e / 3});
            if (e % 3)
                remainder.push_back({p, e % 3});
        }
        // Each uncancelled k^3 contributes three copies of every factor.
        // Build their product once, then cube it after the product tree.
        if (roots.size() < factors.size() / 4)
            return expand(factors, threads);
        auto root = expand(roots, threads);
        if (remainder.empty())
            return mul_square_times(root, root, threads);
        auto third = mul(root, expand(remainder, threads), threads);
        return mul_square_times(root, third, threads);
    }
    static Nat denominator(const Factors& factors, int threads) {
        constexpr uint32_t primes[] = {8, 3, 23, 29};
        constexpr unsigned powers[] = {5, 3, 3, 3};
        size_t offsets[4];
        U count = UINT64_MAX;
        for (unsigned i = 0; i < 4; ++i) {
            auto it = std::lower_bound(factors.begin(), factors.end(), primes[i],
                                       [](const Power& a, unsigned p) { return a.prime < p; });
            if (it == factors.end() || it->prime != primes[i])
                return expand_denominator(factors, threads);
            offsets[i] = it - factors.begin();
            count = std::min<U>(count, it->exponent / powers[i]);
        }
        if (count < 16)
            return expand_denominator(factors, threads);
        unsigned level = std::bit_width(count) - 5;
        unsigned mantissa = count >> level;
        Factors rest = factors;
        for (unsigned i = 0; i < 4; ++i)
            rest[offsets[i]].exponent -= U(powers[i]) * (U(mantissa) << level);
        auto residual = expand_denominator(rest, threads);
        return mul(residual, denominator_power(mantissa, level, threads), threads);
    }
    struct Node {
        Nat t;
        Factors p, q;
        Nat prefix_q;
        size_t low = 0; // t represents t * base^low after rounding.
    };
    // For p coprime to 6, the first layer of p-adic valuation of
    // (6k)!/((3k)! k!^3) depends only on k modulo p. Remove that
    // layer from the recurrence and put its prefix product into L(k).
    struct Wheel {
        uint32_t inv[4]{1, 1, 1, 1};
        U correction = 1;
    };
    template <unsigned period, unsigned... primes> static const Wheel &wheel(U k) {
        static const auto entries = [] {
            std::array<Wheel, period> table{};
            for (unsigned k = 0; k < period; ++k) {
                auto &e = table[k];
                // Start each residue class with the identity transformation.
                std::fill_n(e.inv, 4, 1u);
                e.correction = 1;
                for (unsigned p : {primes...}) {
                    uint32_t inv = p;
                    for (unsigned j = 0; j < 4; ++j)
                        inv *= 2 - p * inv;
                    unsigned rem = k % p;
                    unsigned residues[] = {(6 * rem + 5 * p - 5) % p, (2 * rem + p - 1) % p,
                                           (6 * rem + p - 1) % p, rem};
                    for (unsigned j = 0; j < 4; ++j)
                        if (!residues[j])
                            e.inv[j] *= inv;
                    for (unsigned j = 0; j < 6 * rem / p - 3 * rem / p; ++j)
                        e.correction *= p;
                }
            }
            return table;
        }();
        return entries[k % period];
    }
    struct TermFactors {
        uint32_t a, b, c, d;
        U correction;
    };
    static TermFactors term_factors(U k) {
        // Separate coprime periods keep both tables small. Each prime
        // contributes at most its cube to the linear correction.
        auto &w = wheel<5005, 5, 7, 11, 13>(k);
        auto &v = wheel<323, 17, 19>(k);
        return {uint32_t(6 * k - 5) * w.inv[0] * v.inv[0],
                uint32_t(2 * k - 1) * w.inv[1] * v.inv[1],
                uint32_t(6 * k - 1) * w.inv[2] * v.inv[2], uint32_t(k) * w.inv[3] * v.inv[3],
                w.correction * v.correction};
    }
    static constexpr U raw_leaf_terms = 8;
    static constexpr U factor_leaf_terms = 2048;

    template <class Visit>
    static Series raw(U lo, U hi, bool need_p, bool need_q, const Visit& visit) {
        if (hi - lo <= raw_leaf_terms) {
            LeafSum leaf;
            leaf_sum_with(leaf, lo, hi, visit, need_p, need_q);
            return {need_p ? leaf.p.expand() : Nat{}, need_q ? leaf.q.expand() : Nat{},
                    leaf.t.expand(), true, leaf.shift};
        }
        U half = (hi - lo) / 2;
        if (hi - lo >= 2 * raw_leaf_terms)
            half = (half + raw_leaf_terms / 2) / raw_leaf_terms * raw_leaf_terms;
        U mid = lo + half;
        auto l = raw(lo, mid, true, need_q, visit), r = raw(mid, hi, need_p, true, visit);
        auto a = shifted(mul(l.t, r.q), r.q_shift);
        auto b = mul(l.p, r.t);
        return {need_p ? mul(l.p, r.p) : Nat{}, need_q ? mul(l.q, r.q) : Nat{},
                (mid - lo) % 2 ? sub(std::move(a), b) : add(std::move(a), b), true,
                l.q_shift + r.q_shift};
    }
    Node tree(U lo, U hi, bool need_p, int threads) const {
        Node result = subtree(lo, hi, need_p, threads);
        // The first completed half of the tree can use its own workers
        // for Newton preparation while the other half is still running.
        if (threads == preparation_group && !preparing.test_and_set(std::memory_order_relaxed))
            prepare_root(threads);
        return result;
    }
    Node subtree(U lo, U hi, bool need_p, int threads) const {
        if (hi - lo <= factor_leaf_terms) {
            Node node;
            LeafFactors p, q;
            std::vector<uint32_t> numerators, denominators;
            if (need_p)
                numerators.reserve(3 * (hi - lo));
            denominators.reserve(hi - lo);
            U count = hi - std::max<U>(lo, 1);
            auto term = [&](LeafSum& out, U k) {
                if (!k) {
                    leaf_term(out, k);
                    return;
                }
                auto f = term_factors(k);
                if (need_p) {
                    numerators.push_back(f.a);
                    numerators.push_back(f.b);
                    numerators.push_back(f.c);
                }
                unsigned twos = std::countr_zero(f.d);
                q.small[8] += twos;
                denominators.push_back(f.d >> twos);
                leaf_term_scaled(out, k, W(f.a) * f.b * f.c, f.d, f.correction);
            };
            node.t = raw(lo, hi, false, false, term).t;
            if (need_p)
                factor_block(p, numerators);
            factor_block(q, denominators);
            node.p = p.finish(1);
            node.q = q.finish(3, count);
            if (!lo) {
                node.prefix_q = denominator(node.q, 1);
                node.q = {};
            }
            return node;
        }
        // Round the left interval to a whole number of factor leaves so the
        // interior subtrees also use whole eight-term arithmetic leaves.
        U half = (hi - lo) / 2;
        U mid = lo + (half + factor_leaf_terms / 2) / factor_leaf_terms * factor_leaf_terms;
        Node l, r;
        if (threads > 1 && hi - lo >= 512) {
            int w = threads / 2;
            Team::active()->fork(
                w, [&] { l = tree(lo, mid, true, w); },
                [&] { r = tree(mid, hi, need_p, threads - w); });
        } else {
            l = tree(lo, mid, true, 1);
            r = tree(mid, hi, need_p, 1);
        }
        // Both terms of the sum contain gcd(P_left, Q_right). Removing
        // it here scales T, Q and the retained P by the same divisor.
        cancel(l.p, r.q, threads);
        Nat q, p;
        if (threads > 1) {
            int left = threads / 2;
            Team::active()->fork(
                left, [&] { q = denominator(r.q, left); },
                [&] { p = expand(l.p, threads - left); });
        } else {
            q = denominator(r.q, 1);
            p = expand(l.p, 1);
        }
        // Prefix weight at lo is at most 10^(-14.18*(lo-1)). Retain the
        // corresponding precision in T; P and Q remain exact. Twelve
        // additional base words cover the linear factor, wheel correction,
        // decimal shift rounding and error accumulation over the tree.
        size_t sa = l.low + (3 * (hi - 1) + 7) / 8 - (3 * (mid - 1) + 7) / 8, sb = r.low;
        size_t highest = std::max(l.t.size() + q.size() + sa, r.t.size() + p.size() + sb);
        size_t saved = lo ? size_t((lo - 1) * 14.18L / 8) : 0;
        size_t keep = precision - std::min(precision - 8, saved) + 12;
        size_t low = std::max(std::min(sa, sb), highest - std::min(keep, highest));
        Nat a, b, prefix_q;
        // A prefix needs its numeric denominator at the root. Its left sum
        // and denominator both multiply by the same cancelled right Q.
        // At the root the denominator and T must have the same scale.
        // Its implicit power of ten already accounts for part of T's shift.
        const size_t q_shift = (3 * (hi - 1) + 7) / 8;
        const size_t q_cut = !lo && !need_p && low > q_shift ? low - q_shift : 0;
        auto first = [&](int workers) {
            bool shared = !lo && shared_products(q, l.t, l.prefix_q, a, prefix_q, workers,
                                            low > sa ? low - sa : 0, q_cut);
            if (shared) {
                if (sa > low)
                    a = shifted(std::move(a), sa - low, workers);
            } else {
                a = high_product(l.t, q, sa, low, workers);
                if (!lo)
                    prefix_q = mul_high(l.prefix_q, q, q_cut, workers);
            }
        };
        auto second = [&](int workers) { b = high_product(r.t, p, sb, low, workers); };
        if (threads > 1) {
            // The first product is longer and prefixes also need the denominator.
            int left = std::clamp(lo ? 2 * threads / 3 : 3 * threads / 4, 1, threads - 1);
            Team::active()->fork(left, [&] { first(left); }, [&] { second(threads - left); });
        } else {
            first(1);
            second(1);
        }
        return {(mid - lo) % 2 ? sub(std::move(a), b, threads) : add(std::move(a), b, threads),
                need_p ? join(l.p, r.p) : Factors{}, lo ? join(l.q, r.q) : Factors{},
                std::move(prefix_q), low};
    }

  public:
    SeriesFactors(size_t max, int threads, size_t prec, int group, std::function<void(int)> prepare)
        : smallest(3 * max + 2), precision(prec), preparation_group(group),
          prepare_root(std::move(prepare)) {
        const size_t limit = 6 * max;
        std::vector<uint32_t> primes;
        for (size_t p = 3; p * p <= limit; p += 2) {
            bool prime = true;
            for (auto q : primes) {
                if (U(q) * q > p)
                    break;
                if (p % q == 0) {
                    prime = false;
                    break;
                }
            }
            if (prime)
                primes.push_back(p);
        }
        // Index i represents 2*i+1. Only composite indices need a
        // divisor and the smallest one fits in 16 bits.
        constexpr size_t block = 1 << 17;
        blocks((smallest.size() + block - 1) / block, threads, [&](size_t chunk) {
            auto lo = chunk * block, hi = std::min(smallest.size(), lo + block);
            for (auto p : primes) {
                size_t start = U(p) * p / 2;
                if (start >= hi)
                    break;
                if (start < lo)
                    start += ((lo - start + p - 1) / p) * p;
                for (auto i = start; i < hi; i += p)
                    if (!smallest[i])
                        smallest[i] = p;
            }
        });
    }
    Series run(U terms, int threads) const {
        auto v = tree(0, terms, false, threads);
        auto q = std::move(v.prefix_q);
        size_t shift = (3 * (terms - 1) + 7) / 8;
        return {{}, std::move(q), std::move(v.t), false, shift - std::min(shift, v.low)};
    }
};
