// All primes admit roots of order 15*2^32 generated from 7. A group
// of m primes represents coefficients of m-1 decimal limbs.
static constexpr U moduli[] = {2246998040248321, 2246482644172801, 2244743182417921,
                               2242939296153601, 2241070985379841, 2238429580492801,
                               2237463212851201, 2236561269719041, 2233468893265921};
using CrtDigits = std::array<uint32_t, 18>;
struct CrtWeights {
    CrtDigits modulus{};
    std::array<CrtDigits, 9> cofactors{};
    std::array<U, 9> inverses{};
};
constexpr CrtWeights make_crt_weights(size_t count) {
    CrtWeights table;
    auto product_except = [count](size_t skip) {
        CrtDigits value{1};
        for (size_t i = 0; i < count; ++i) {
            if (i == skip)
                continue;
            U carry = 0;
            for (auto& word : value) {
                Wide v = Wide(word) * moduli[i] + carry;
                word = v % base;
                carry = v / base;
            }
        }
        return value;
    };
    table.modulus = product_except(count);
    for (size_t i = 0; i < count; ++i) {
        auto& cofactor = table.cofactors[i] = product_except(i);
        U v = 0, p = moduli[i];
        for (size_t k = cofactor.size(); k--;)
            v = (Wide(v) * base + cofactor[k]) % p;
        U inv = 1;
        for (U e = p - 2; e; e >>= 1) {
            if (e & 1)
                inv = Wide(inv) * v % p;
            v = Wide(v) * v % p;
        }
        table.inverses[i] = inv;
    }
    return table;
}
constexpr std::array crt_tables{make_crt_weights(6), make_crt_weights(7), make_crt_weights(8), make_crt_weights(9)};

template <size_t I> const auto& wide_weights() {
    constexpr U prime = moduli[I];
    using F = Field<prime>;
    static const auto weights = [] {
        std::array<U, 8> w{};
        w[0] = R % prime;
        for (size_t k = 1; k < 8; ++k)
            w[k] = F::product(w[k - 1], base);
        return w;
    }();
    return weights;
}
// weights[k] contains base^k * R modulo the prime. The high/low
// accumulators contain their dot product with eight decimal coefficients.
template <size_t I> __m512i reduce_input(__m512i lo, __m512i hi) {
    constexpr U prime = moduli[I];
    using F = Field<prime>;
    hi += _mm512_srli_epi64(lo, 52);
    auto m = _mm512_madd52lo_epu64(_mm512_setzero_si512(), lo, F::splat(R + 2 - prime));
    return F::reduce(hi + F::splat(prime) -
                     _mm512_madd52hi_epu64(_mm512_setzero_si512(), m, F::splat(prime)));
}

// Read eight consecutive coefficients. Each permutation produces two columns
// of eight base-10^8 digits, one in each half of the register. Widths above
// four span two pairs of input registers. The final load stops at the last limb.
template <size_t width, class Emit> inline void limb_columns(const uint32_t* src, Emit emit) {
    alignas(64) static constexpr auto index = [] {
        std::array<std::array<uint32_t, 16>, (width + 1) / 2> values{};
        for (size_t k = 0; k < width; k += 2)
            for (size_t j = 0; j < 16; ++j)
                values[k / 2][j] = (j % 8) * width + k + j / 8;
        return values;
    }();
    static_assert(width >= 1 && width <= 8);
    constexpr size_t words = 8 * width;
    auto load = [&]<size_t offset>() {
        if constexpr (offset >= words)
            return _mm512_setzero_si512();
        else if constexpr (words - offset >= 16)
            return _mm512_loadu_si512(src + offset);
        else
            return _mm512_maskz_loadu_epi32((1u << (words - offset)) - 1, src + offset);
    };
    auto a = load.template operator()<0>(), b = load.template operator()<16>();
    auto c = load.template operator()<32>(), d = load.template operator()<48>();
    for (size_t k = 0; k < width; k += 2) {
        auto i = _mm512_load_si512(index[k / 2].data());
        auto v = _mm512_permutex2var_epi32(a, i, b);
        if constexpr (words > 32) {
            auto high = _mm512_permutex2var_epi32(c, i, d);
            v = _mm512_mask_blend_epi32(_mm512_cmpge_epi32_mask(i, _mm512_set1_epi32(32)), v, high);
        }
        emit(k, _mm512_cvtepu32_epi64(_mm512_castsi512_si256(v)));
        if (k + 1 < width)
            emit(k + 1, _mm512_cvtepu32_epi64(_mm512_extracti64x4_epi64(v, 1)));
    }
}
// Convert a stripe of eight coefficients into the requested prime fields while its decimal
// limbs are in registers. The first NTT pass consumes those residues on the
// same worker, so no full-size table of untransformed inputs is needed.
inline thread_local ntt::Scratch input_stripe;
template <size_t width, size_t primes, bool packed, size_t begin, size_t field_count>
void forward_columns(const Nat& input, std::array<std::unique_ptr<Buffer>, primes>& planes,
                     size_t n, size_t words, int workers) {
    for (size_t i = begin; i < begin + field_count; ++i)
        planes[i] = std::make_unique<Buffer>(words);
    const ntt::Layout layout(n);
    const size_t count = (input.size() + width - 1) / width;
    const size_t occupied = (count + layout.cols - 1) / layout.cols;
    const size_t stride = occupied * 8;
    blocks(layout.cols / 8, workers, [&](size_t stripe) {
        input_stripe.resize(field_count * stride);
        const size_t col = stripe * 8;
        for (size_t r = 0; r < occupied; ++r) {
            size_t offset = (r * layout.cols + col) * width;
            alignas(64) uint32_t tail[8 * width];
            const uint32_t* src;
            if (offset + 8 * width <= input.size())
                src = input.data() + offset;
            else {
                std::fill_n(tail, 8 * width, 0);
                if (offset < input.size())
                    std::copy(input.begin() + offset, input.end(), tail);
                src = tail;
            }
            __m512i digits[width];
            limb_columns<width>(src, [&](size_t k, __m512i x) { digits[k] = x; });
            auto convert = [&]<size_t I>() {
                const auto& weights = wide_weights<I>();
                __m512i lo{}, hi{};
                for (size_t k = 0; k < width; ++k) {
                    auto w = _mm512_set1_epi64(weights[k]);
                    lo = _mm512_madd52lo_epu64(lo, digits[k], w);
                    hi = _mm512_madd52hi_epu64(hi, digits[k], w);
                }
                _mm512_store_si512(input_stripe.data() + (I - begin) * stride + r * 8, reduce_input<I>(lo, hi));
            };
            [&]<size_t... I>(std::index_sequence<I...>) {
                (convert.template operator()<begin + I>(), ...);
            }(std::make_index_sequence<field_count>{});
        }
        auto forward = [&]<size_t I>() {
            using F = Field<moduli[I]>;
            const auto& plan = F::plan_for(n);
            const U* next = input_stripe.data() + (I - begin) * stride;
            auto read = [&](U* dest, size_t, size_t) {
                F::save(dest, F::load(next));
                next += 8;
            };
            F::template columns<false, packed>(planes[I]->data, plan.rows, plan.cols, col,
                                               plan.table->f, count, 0, 0, read, layout, 8);
        };
        [&]<size_t... I>(std::index_sequence<I...>) {
            (forward.template operator()<begin + I>(), ...);
        }(std::make_index_sequence<field_count>{});
    });
}

template <size_t I, bool packed, bool pair, size_t width, bool triple = false>
void wide_residues(Buffer& out, Buffer& work, size_t n, const Nat& a, const Nat& b, int workers,
                   const CrtWeights& crt_weights, const Nat* other = nullptr,
                   Buffer* second = nullptr, bool columns_ready = false) {
    constexpr U prime = moduli[I];
    using F = Field<prime>;
    const auto& weights = wide_weights<I>();
    size_t na = (a.size() + width - 1) / width, nb = (b.size() + width - 1) / width;
    auto unpack = [&](U* dest, size_t pos, size_t count, int input) {
        const Nat& src = input == 0 ? a : input == 1 ? b : *other;
        size_t j = 0;
        for (; j + 8 <= count && (pos + j + 8) * width <= src.size(); j += 8) {
            __m512i lo{}, hi{};
            limb_columns<width>(src.data() + (pos + j) * width, [&](size_t k, __m512i x) {
                auto w = F::splat(weights[k]);
                lo = _mm512_madd52lo_epu64(lo, x, w);
                hi = _mm512_madd52hi_epu64(hi, x, w);
            });
            F::save(dest + j, reduce_input<I>(lo, hi));
        }
        for (; j < count && (pos + j) * width < src.size(); ++j) {
            size_t begin = (pos + j) * width;
            U v = 0;
            for (size_t k = std::min(begin + width, src.size()); k > begin; --k)
                v = (Wide(v) * base + src[k - 1]) % prime;
            dest[j] = v;
        }
        std::fill(dest + j, dest + count, 0);
    };
    // Fold each inverse CRT cofactor into the inverse transform scale.
    const auto& plan = F::plan_for(n);
    U scale = F::product(plan.inverse_scale, crt_weights.inverses[I]);
    // A cubic point product introduces two Montgomery divisions by R.
    if constexpr (triple)
        scale = F::product(scale, R % prime);
    size_t nz = pair ? (other->size() + width - 1) / width : 0;
    bool square = !pair && &a == &b;
    U* extra = pair ? second->data : nullptr;
    if constexpr (packed) {
        if (ntt::Layout(n).tiled) {
            F::template convolution<pair, true, triple>(out.data, work.data, extra, plan, workers,
                                                        square, na, nb, nz, scale, unpack, columns_ready);
            return;
        }
    }
    F::template convolution<pair, false, triple>(out.data, work.data, extra, plan, workers, square,
                                                 na, nb, nz, scale, unpack, columns_ready);
}
template <size_t width, size_t primes>
Nat wide_recover(std::array<std::unique_ptr<Buffer>, primes>& residues, size_t n, size_t used,
                 int workers, bool packed, size_t discard, size_t period) {
    constexpr size_t ndigits = primes * 2;
    constexpr auto& crt_weights = crt_tables[primes - 6];
    constexpr bool can_pack = ntt::have_packing && primes == 6;
    // Overlap the redundant CRT words locally before carrying through the output.
    constexpr size_t chunk = 256;
    Nat prefix;
    auto reconstruct = [&](size_t first, size_t drop) {
        const size_t digits = width * (used - first);
        const size_t stride = width * chunk;
        // Keep the complete block containing the carry guard in scratch.
        // Every other worker can normalize straight into its final position.
        const size_t boundary = std::min(digits, (drop + stride - 1) / stride * stride);
        prefix.resize(boundary);
        std::vector<std::array<uint32_t, 16>> tails((used - first + chunk - 1) / chunk);
        Nat out = overwrite_limbs(digits - drop + 16);
        // Workers write the body. The final overlap is added into this tail.
        std::fill_n(out.end() - 16, 16, 0);
        const ntt::Layout layout(n);
        blocks(tails.size(), workers, [&](size_t block) {
            alignas(64) int64_t sums[chunk * width + 16]{};
            size_t start = first + block * chunk, count = std::min(chunk, used - start);
            for (size_t j = 0; j < count; j += 8) {
                size_t logical = start + j,
                       pos = layout.index(logical / layout.cols, logical % layout.cols);
                __m512i source[primes];
                __m512d quotient{};
                for (unsigned i = 0; i < primes; ++i) {
                    auto u = [&] {
                        if constexpr (can_pack)
                            if (packed)
                                return ntt::load_packed(residues[i]->data, pos);
                        return _mm512_loadu_si512(residues[i]->data + pos);
                    }();
                    quotient = _mm512_fmadd_pd(_mm512_cvtepu64_pd(u),
                                               _mm512_set1_pd(1. / moduli[i]), quotient);
                    source[i] = u;
                }
                // All supported coefficients are below P/4. Rounding the sum
                // of the normalized residues recovers the quotient exactly.
                auto q = _mm512_cvt_roundpd_epi64(quotient,
                                                  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                __m512i upper{};
                for (unsigned k = 0; k < ndigits; ++k) {
                    __m512i low{}, high{};
                    for (unsigned i = 0; i < primes; ++i) {
                        auto weight = _mm512_set1_epi64(crt_weights.cofactors[i][k]);
                        low = _mm512_madd52lo_epu64(low, source[i], weight);
                        high = _mm512_madd52hi_epu64(high, source[i], weight);
                    }
                    // 2^52 = (2^52 / B)*B + (2^52 % B). Convert both
                    // accumulators after all primes have contributed.
                    // Each residue is below 2^51 and each weight below B,
                    // so nine high halves sum to less than 4.5B and fit in 32 bits.
                    auto v = low + upper +
                        _mm512_mul_epu32(high, _mm512_set1_epi64(R % base)) -
                        _mm512_mul_epu32(q, _mm512_set1_epi64(crt_weights.modulus[k]));
                    upper = _mm512_mul_epu32(high, _mm512_set1_epi64(R / base));
                    alignas(64) int64_t lanes[8];
                    _mm512_store_si512(lanes, v);
                    for (unsigned l = 0; l < std::min<size_t>(8, count - j); ++l)
                        sums[(j + l) * width + k] += lanes[l];
                }
            }
            const size_t offset = (start - first) * width;
            auto* dest = offset < boundary ? prefix.data() + offset : out.data() + (offset - drop);
            // The cofactor subtraction can leave a negative digit sum.
            // At most four coefficients overlap, each subtracting less
            // than 9*base. The first quotient is therefore >= -37, which
            // keeps the biased remainder positive in CoefficientCarry.
            if (count * width % 8 == 0) {
                CoefficientCarry normalization;
                for (size_t k = 0; k < count * width; k += 8) {
                    auto digits = normalization.take<true>(_mm512_loadu_si512(sums + k));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dest + k), digits);
                }
                for (size_t k = 0; k < 16; k += 8) {
                    auto digits = normalization.take<true>(_mm512_loadu_si512(sums + count * width + k));
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tails[block].data() + k), digits);
                }
                return;
            }
            int64_t carry = 0;
            for (size_t k = 0; k < count * width + 16; ++k) {
                int64_t v = sums[k] + carry;
                carry = v / int64_t(base);
                int64_t word = v - carry * base;
                if (word < 0) {
                    word += base;
                    --carry;
                }
                if (k < count * width)
                    dest[k] = word;
                else
                    tails[block][k - count * width] = word;
            }
        });
        for (size_t i = 0; i < tails.size(); ++i) {
            size_t k = width * std::min((i + 1) * chunk, used - first);
            unsigned carry = 0;
            for (size_t j = 0; j < 16 || carry; ++j) {
                const size_t index = k + j;
                auto& digit = index < boundary ? prefix[index] : out[index - drop];
                U v = digit + carry + (j < 16 ? tails[i][j] : 0);
                carry = v >= base;
                digit = v - carry * base;
            }
        }
        if (boundary > drop)
            std::copy(prefix.begin() + drop, prefix.end(), out.begin());
        return out;
    };
    // Dropping all coefficients before first omits less than
    // 2^32*B^(2w)/(B^w-1) < B^(w+2) units at that position.
    // The guard digits certify that this omitted carry dies below discard.
    const size_t guard = width + 4;
    size_t first = discard > guard ? (discard - guard) / (width * chunk) * chunk : 0;
    const size_t cut = discard - first * width;
    auto out = reconstruct(first, cut);
    if (first) {
        if (std::any_of(prefix.begin() + width + 2, prefix.begin() + cut,
                        [](uint32_t x) { return x != base - 1; })) {
            trim(out);
            return out;
        }
        Nat{}.swap(out);
        out = reconstruct(0, 0);
        trim(out);
        return discard_low(std::move(out), discard);
    }
    if (period) {
        for (size_t k = period, j = 0; k < out.size(); ++k, ++j) {
            unsigned carry = out[k];
            for (size_t i = j; carry; i = i + 1 == period ? 0 : i + 1) {
                unsigned v = out[i] + carry;
                carry = v >= base;
                out[i] = v - carry * base;
            }
        }
        out.resize(period);
        if (std::all_of(out.begin(), out.end(), [](auto d) { return d == base - 1; }))
            out.clear();
    }
    trim(out);
    return out;
}
// Each prime owns its transform buffers. Give independent fields disjoint
// worker ranges when a single transform is too short to occupy the team.
template <size_t first, size_t end, class Transform>
void transform_fields(int workers, int groups, const Transform& transform) {
    if constexpr (end == first + 1) {
        transform.template operator()<first>(workers);
    } else if (groups == 1) {
        [&]<size_t... I>(std::index_sequence<I...>) {
            (transform.template operator()<first + I>(workers), ...);
        }(std::make_index_sequence<end - first>{});
    } else {
        constexpr size_t middle = (first + end) / 2;
        const int left = std::clamp<int>(
            (workers * (middle - first) + (end - first) / 2) / (end - first), 1, workers - 1);
        const int left_groups = groups / 2;
        Team::active()->fork(
            left, [&] { transform_fields<first, middle>(left, std::min(left, left_groups), transform); },
            [&] { transform_fields<middle, end>(workers - left,
                        std::min(workers - left, groups - left_groups), transform); });
    }
}

// Only the transient operand is batched. The result fields must all survive
// until CRT reconstruction, so they can be prepared from one pass over the input.
constexpr size_t field_batch_size = 4;
constexpr size_t parallel_field_limit = size_t(1) << 21;
constexpr size_t large_field_transform = size_t(1) << 24;

template <size_t primes, class Batch>
void field_batches(bool split, bool five_fields, const Batch& batch) {
    if constexpr (primes == 9) {
        if (five_fields) {
            batch.template operator()<0, 5>();
            batch.template operator()<5, 4>();
            return;
        }
    }
    static_assert(primes > field_batch_size && primes <= 2 * field_batch_size + 1);
    if (split) {
        batch.template operator()<0, field_batch_size>();
        if constexpr (primes > 2 * field_batch_size) {
            batch.template operator()<field_batch_size, field_batch_size>();
            batch.template operator()<2 * field_batch_size, primes - 2 * field_batch_size>();
        } else
            batch.template operator()<field_batch_size, primes - field_batch_size>();
    } else {
        batch.template operator()<0, primes>();
    }
}
// A six-field plan packs and unpacks 52-bit residues between transform
// passes. On large plans its measured time per modular point is higher.
size_t wide_plan_cost(size_t width, size_t n) {
    const size_t points = (width + 1) * n;
    if (ntt::have_packing && width == 5 && n >= (size_t(1) << 23))
        return points + points / 4;
    return points;
}
bool linear_width_allowed(size_t width, size_t n) {
    return width != 8 || (n >= (size_t(1) << 23) && n <= (size_t(1) << 30));
}

template <size_t width>
Nat wide_product(const Nat& a, const Nat& b, int threads, size_t* requested, size_t discard) {
    constexpr size_t primes = width + 1, ndigits = primes * 2;
    constexpr const auto& crt_weights = crt_tables[primes - 6];
    // With n <= 2^32, each coefficient is below 2^32 * base^(2*width).
    // Reconstruction requires the product of the primes to exceed four
    // times that bound. The nine-prime planner caps its length at 2^30.
    static_assert(crt_weights.modulus[ndigits - 1] > (primes == 9 ? 44 : 200));
    size_t used = (a.size() + width - 1) / width + (b.size() + width - 1) / width,
           n = transform_length(used);
    size_t period = 0;
    if (requested) {
        auto short_n =
            transform_length((std::max({*requested, a.size(), b.size()}) + width - 1) / width);
        if (short_n < n) {
            used = n = short_n;
            period = n * width;
        }
        *requested = 0;
    }
    constexpr bool can_pack = ntt::have_packing && primes == 6;
    // A wrapped product may choose a shorter, untiled plan.
    const bool packed = can_pack && ntt::Layout(n).tiled;
    const size_t buffer_words = packed ? (n / 16) * 13 : n;
    Workspace reservation;
    const size_t base_bytes = buffer_words * sizeof(U) * (primes + (&a != &b));
    const ntt::Layout layout(n);
    const bool square = &a == &b;
    const bool tiled = n <= (U(1) << 32) && layout.tiled;
    // For a large nonsquare, prepare one batch of the right operand at a
    // time. Nine primes fit in two passes with space for five temporaries.
    bool split = !square && n >= large_field_transform;
    bool prepared = tiled && !split &&
        reservation.acquire((square ? primes : 2 * primes) * buffer_words * sizeof(U), threads, false);
    bool five_fields = false;
    if (!prepared && !square && tiled) {
        split = true;
        if constexpr (primes == 9)
            five_fields = reservation.acquire((primes + 5) * buffer_words * sizeof(U), threads, false);
        prepared = five_fields ||
            reservation.acquire((primes + field_batch_size) * buffer_words * sizeof(U), threads, false);
    }
    if (!prepared && (n > (U(1) << 32) || !reservation.acquire(base_bytes, threads)))
        return tiled_product(a, b, threads, reservation.tile_budget, discard);
    int workers = std::min<size_t>(threads, 1 + n / 16384);
    std::array<std::unique_ptr<Buffer>, primes> residues;
    std::array<std::unique_ptr<Buffer>, primes> right;
    Buffer work(prepared || square ? 0 : buffer_words);
    auto transform = [&]<size_t I>(int field_workers) {
        if (!prepared)
            residues[I] = std::make_unique<Buffer>(buffer_words);
        Buffer& second = right[I] ? *right[I] : work;
        wide_residues<I, can_pack, false, width>(*residues[I], second, n, a, b, field_workers,
                                                 crt_weights, nullptr, nullptr, prepared);
        right[I].reset();
    };
    if (prepared)
        forward_columns<width, primes, can_pack, 0, primes>(a, residues, n, buffer_words, workers);
    // A sibling series subtree may already be using the other cores.
    // Split across fields only when the caller owns the entire team.
    const bool parallel_fields = prepared && threads >= 8 && n <= parallel_field_limit &&
                                 size_t(threads) == Team::active()->size();
    field_batches<primes>(prepared && split, five_fields, [&]<size_t first, size_t count>() {
        if (prepared && !square)
            forward_columns<width, primes, can_pack, first, count>(b, right, n, buffer_words, workers);
        transform_fields<first, first + count>(workers, parallel_fields ? 4 : 1, transform);
    });
    work.release();
    if (requested)
        *requested = period;
    return wide_recover<width>(residues, n, used, workers, packed, discard, period);
}
template <size_t width>
bool wide_pair(const Nat& common, const Nat& a, const Nat& b, Nat& x, Nat& y, int threads,
               size_t first_discard, size_t second_discard) {
    constexpr size_t primes = width + 1;
    constexpr bool can_pack = ntt::have_packing && primes == 6;
    auto groups = [](size_t limbs) { return (limbs + width - 1) / width; };
    size_t used_x = groups(a.size()) + groups(common.size());
    size_t used_y = groups(b.size()) + groups(common.size());
    size_t n = transform_length(std::max(used_x, used_y));
    size_t shorter = transform_length(std::min(used_x, used_y));
    // One forward transform is shared. Limit padding of the shorter product.
    if (n > (U(1) << 32) || n > shorter + shorter / 4)
        return false;
    const bool tiled = ntt::Layout(n).tiled;
    bool packed = can_pack && tiled;
    size_t words = packed ? n / 16 * 13 : n;
    Workspace reservation;
    bool split = n >= large_field_transform;
    bool prepared = tiled && !split &&
        reservation.acquire(3 * primes * words * sizeof(U), threads, false);
    bool five_fields = false;
    if (!prepared && tiled) {
        split = true;
        if constexpr (primes == 9)
            five_fields = reservation.acquire((2 * primes + 5) * words * sizeof(U), threads, false);
        prepared = five_fields ||
            reservation.acquire((2 * primes + field_batch_size) * words * sizeof(U), threads, false);
    }
    if (!prepared && !reservation.acquire((2 * primes + 1) * words * sizeof(U), threads))
        return false;
    int workers = std::min<size_t>(threads, 1 + n / 16384);
    std::array<std::unique_ptr<Buffer>, primes> left, right;
    std::array<std::unique_ptr<Buffer>, primes> shared;
    Buffer work(prepared ? 0 : words);
    auto run_prime = [&]<size_t i>(int field_workers) {
        if (!prepared) {
            left[i] = std::make_unique<Buffer>(words);
            right[i] = std::make_unique<Buffer>(words);
        }
        Buffer& second = prepared ? *shared[i] : work;
        wide_residues<i, can_pack, true, width>(*left[i], second, n, a, common, field_workers,
                                                crt_tables[primes - 6], &b, right[i].get(), prepared);
        shared[i].reset();
    };
    if (prepared) {
        forward_columns<width, primes, can_pack, 0, primes>(a, left, n, words, workers);
        forward_columns<width, primes, can_pack, 0, primes>(b, right, n, words, workers);
    }
    field_batches<primes>(prepared && split, five_fields, [&]<size_t first, size_t count>() {
        if (prepared)
            forward_columns<width, primes, can_pack, first, count>(common, shared, n, words, workers);
        transform_fields<first, first + count>(workers, 1, run_prime);
    });
    work.release();
    Nat first;
    if (first_discard < common.size() + a.size())
        first = wide_recover<width>(left, n, used_x, workers, packed, first_discard, 0);
    left = {};
    y = second_discard < common.size() + b.size()
        ? wide_recover<width>(right, n, used_y, workers, packed, second_discard, 0) : Nat{};
    x = std::move(first);
    return true;
}
