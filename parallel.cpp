#include "parallel.hpp"
#include <cstdio>
#include <cstdint>

namespace ap {
size_t available_memory_bytes() {
    auto* file = std::fopen("/proc/meminfo", "r");
    if (!file)
        return SIZE_MAX;
    char line[128];
    size_t kib = 0;
    while (std::fgets(line, sizeof(line), file))
        if (std::sscanf(line, "MemAvailable: %zu kB", &kib) == 1)
            break;
    std::fclose(file);
    return kib ? kib * 1024 : SIZE_MAX;
}

void Team::worker(size_t id) {
    current = this;
    rank = id;
    auto& s = slots[id];
    for (;;) {
        for (int spin = 0; spin < 1500 && !s.state.load(std::memory_order_acquire); ++spin)
            _mm_pause();
        int state;
        while (!(state = s.state.load(std::memory_order_acquire)))
            s.state.wait(0, std::memory_order_relaxed);
        if (state == 2)
            break;
        try {
            s.run(s.context, s.index);
        } catch (...) {
            s.error = std::current_exception();
        }
        s.state.store(0, std::memory_order_release);
        s.state.notify_one();
    }
}
Team::Team(size_t n, CrtSchedule crt) : slots(std::make_unique<Slot[]>(n)), crt_schedule(crt) {
    if (current || !n)
        throw std::logic_error("invalid worker team");
    try {
        for (size_t i = 1; i < n; ++i)
            workers.emplace_back(&Team::worker, this, i);
    } catch (...) {
        shutdown();
        throw;
    }
    current = this;
    rank = 0;
}
void Team::shutdown() {
    for (size_t i = 0; i < workers.size(); ++i) {
        slots[i + 1].state.store(2, std::memory_order_release);
        slots[i + 1].state.notify_one();
    }
    for (auto& w : workers)
        w.join();
}
Team::~Team() {
    shutdown();
    current = nullptr;
}
} // namespace ap
