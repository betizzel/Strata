// metal/tests/runtime_purge.cpp - purge_device_allocations gives the memory back.
//   Device allocations sit in the runtime's residency set. setPurgeableState(Empty) alone frees nothing for such a buffer
//   (phys_footprint and resident size stay), so the purge releases them. Checked here: three 64 MB device buffers and one
//   8 MB buffer made inside weight_scope(), all written, are purged; the process footprint drops by about their size;
//   a buffer made outside weight_scope below the threshold survives, as does a host allocation.
#include "strata/gpu/runtime.hpp"

#include <mach/mach.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

namespace strata::gpu {
uint64_t purge_device_allocations(size_t min_bytes);   // metal/src/gpu/runtime.cpp
void weight_scope(bool on);
}
namespace gpu = strata::gpu;

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } } while (0)

static double footprint_mb() {
    task_vm_info_data_t vi{};
    mach_msg_type_number_t n = TASK_VM_INFO_COUNT;
    task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &vi, &n);
    return vi.phys_footprint / 1048576.0;
}

static void* device_written(size_t bytes) {
    void* p = nullptr;
    CHECK(gpu::malloc_device_raw(&p, bytes) == gpu::kSuccess, "malloc_device %zu", bytes);
    std::memset(gpu::host_view(p), 0x5a, bytes);   // resident pages, as an uploaded weight copy has
    return p;
}

int main() {
    const size_t big = 64u << 20, small = 8u << 20;
    void* keep_small = device_written(small);   // below the threshold, outside weight_scope: stays
    void* host = nullptr;
    CHECK(gpu::host_alloc_raw(&host, big, 0) == gpu::kSuccess, "host_alloc");
    std::memset(host, 0x33, big);
    void* b0 = device_written(big);
    void* b1 = device_written(big);
    void* b2 = device_written(big);
    gpu::weight_scope(true);
    void* w = device_written(small);            // a weight copy below the threshold: purged
    gpu::weight_scope(false);
    const double before = footprint_mb();

    const uint64_t given = gpu::purge_device_allocations(32u << 20);
    CHECK(given == 3 * big + small, "purged %llu bytes, want %zu", (unsigned long long) given, 3 * big + small);
    double after = footprint_mb();
    for (int i = 0; i < 40 && after > before - 0.9 * given / 1048576.0; ++i) {   // the release lands asynchronously
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        after = footprint_mb();
    }
    std::printf("footprint %.0f -> %.0f MB after purging %.0f MB\n", before, after, given / 1048576.0);
    CHECK(before - after >= 0.9 * given / 1048576.0, "the footprint dropped by only %.0f MB", before - after);

    CHECK(gpu::host_view(b0) == nullptr && gpu::host_view(w) == nullptr, "a purged allocation still has a CPU view");
    CHECK(gpu::host_view(keep_small) != nullptr, "the small device buffer was purged");
    CHECK(static_cast<unsigned char*>(gpu::host_view(keep_small))[small - 1] == 0x5a, "the small device buffer changed");
    CHECK(static_cast<unsigned char*>(host)[big - 1] == 0x33, "the host allocation changed");

    // the released ranges can be reused: new allocations work and are found as themselves, and freeing a purged
    // allocation (a host-code destructor at exit) is harmless
    void* again = device_written(big);
    CHECK(gpu::host_view(again) != nullptr, "a new allocation after the purge has no CPU view");
    CHECK(gpu::free_device(b1) == gpu::kSuccess, "free of a purged allocation");
    CHECK(gpu::free_device(again) == gpu::kSuccess && gpu::free_device(keep_small) == gpu::kSuccess, "free");
    (void) b2;
    std::printf(g_fail ? "runtime_purge: %d FAILED\n" : "runtime_purge: ok\n", g_fail);
    return g_fail ? 1 : 0;
}
