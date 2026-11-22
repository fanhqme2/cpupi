#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>
#include <immintrin.h>

namespace ap {
size_t available_memory_bytes();

enum class CrtSchedule { standard, decimal };

// Tasks own contiguous, disjoint ranges of worker slots. A worker can fork
// recursively within its range without creating another OS thread.
class Team {
    struct alignas(64) Slot {
        std::atomic<int> state{0};
        void (*run)(void*, size_t) = nullptr;
        void* context = nullptr;
        size_t index = 0;
        std::exception_ptr error;
    };
    std::unique_ptr<Slot[]> slots;
    std::vector<std::thread> workers;
    const CrtSchedule crt_schedule;
    static inline thread_local Team* current = nullptr;
    static inline thread_local size_t rank = 0;
    void worker(size_t id);
    void shutdown();
    void finish(size_t id) {
        auto& s = slots[id];
        for (int spin = 0; spin < 1500 && s.state.load(std::memory_order_acquire); ++spin)
            _mm_pause();
        while (s.state.load(std::memory_order_acquire))
            s.state.wait(1, std::memory_order_relaxed);
    }
    template <class F> void start(size_t id, F& f, size_t index) {
        auto& s = slots[id];
        s.context = &f;
        s.run = [](void* ctx, size_t item) { (*static_cast<F*>(ctx))(item); };
        s.index = index;
        s.error = nullptr;
        s.state.store(1, std::memory_order_release);
        s.state.notify_one();
    }

  public:
    explicit Team(size_t n, CrtSchedule crt = CrtSchedule::decimal);
    static bool use_decimal_crt() {
        return !current || current->crt_schedule == CrtSchedule::decimal;
    }
    ~Team();
    Team(const Team&) = delete;
    Team& operator=(const Team&) = delete;
    size_t size() const {
        return workers.size() + 1;
    }
    static Team* active() {
        return current;
    }

    template <class Left, class Right> void fork(int left_workers, Left left, Right right) {
        auto other = rank + left_workers;
        auto task = [&](size_t) { right(); };
        start(other, task, 0);
        std::exception_ptr error;
        try {
            left();
        } catch (...) {
            error = std::current_exception();
        }
        finish(other);
        if (error)
            std::rethrow_exception(error);
        if (slots[other].error)
            std::rethrow_exception(slots[other].error);
    }
    template <class F> void range(size_t count, int tasks, const F& work) {
        auto chunk = [&](size_t i) {
            size_t start = count * i / tasks, end = count * (i + 1) / tasks;
            for (size_t j = start; j < end; ++j)
                work(j);
        };
        for (int i = 1; i < tasks; ++i)
            start(rank + i, chunk, i);
        std::exception_ptr error;
        try {
            chunk(0);
        } catch (...) {
            error = std::current_exception();
        }
        for (int i = 1; i < tasks; ++i) {
            finish(rank + i);
            if (slots[rank + i].error)
                error = slots[rank + i].error;
        }
        if (error)
            std::rethrow_exception(error);
    }
};

template <class F> void blocks(size_t count, int workers, const F& work) {
    int tasks = std::min<size_t>(count, workers);
    if (tasks <= 1) {
        for (size_t i = 0; i < count; ++i)
            work(i);
    } else {
        Team::active()->range(count, tasks, work);
    }
}
} // namespace ap
