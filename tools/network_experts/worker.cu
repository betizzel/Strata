// Isolated, loopback-only expert service. Weights never cross the inference socket.
#include "strata/core/network_expert_protocol.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include <cuda_runtime.h>
#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>

namespace net = strata::core::network;
namespace kernels = strata::kernels;
namespace {
constexpr uint64_t kMiB = 1ull << 20, kHeadroom = 8ull << 30;
volatile sig_atomic_t stopping = 0;
volatile sig_atomic_t active_socket = -1;

void stop_worker(int) {
    stopping = 1;
    // Wake the shared exact-IO helpers even when interrupted in their EINTR loop.
    if (active_socket >= 0) ::shutdown(static_cast<int>(active_socket), SHUT_RDWR);
}
[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }
void syscheck(bool ok, const char* what) { if (!ok) fail(net::system_error(what)); }
void cucheck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}
using Clock = std::chrono::steady_clock;
uint64_t elapsed_ns(Clock::time_point begin, Clock::time_point end) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count());
}
struct Fd {
    int value = -1;
    explicit Fd(int fd = -1) : value(fd) {}
    ~Fd() { if (value >= 0) ::close(value); }
    Fd(const Fd&) = delete;
    Fd& operator=(const Fd&) = delete;
};
struct DeviceMemory {
    void* data = nullptr;
    ~DeviceMemory() { if (data) cudaFree(data); }
    void allocate(size_t bytes) { cucheck(cudaMalloc(&data, bytes), "cudaMalloc"); }
    template<class T> T* as() { return static_cast<T*>(data); }
};
struct Stream {
    cudaStream_t value = nullptr;
    ~Stream() { if (value) cudaStreamDestroy(value); }
};
struct RunEvents {
    enum Stage : size_t { H2DBegin, KernelBegin, D2HBegin, End, Count };
    std::array<cudaEvent_t, Count> events{};
    ~RunEvents() { for (auto event : events) if (event) cudaEventDestroy(event); }
    void initialize() {
        for (auto& event : events) cucheck(cudaEventCreate(&event), "create expert timing event");
    }
    void record(Stage stage, cudaStream_t stream) {
        cucheck(cudaEventRecord(events[stage], stream), "record expert timing event");
    }
    uint64_t duration_ns(Stage begin, Stage end) const {
        float milliseconds = 0;
        cucheck(cudaEventElapsedTime(&milliseconds, events[begin], events[end]), "read expert timing event");
        return static_cast<uint64_t>(std::llround(static_cast<double>(milliseconds) * 1000000.0));
    }
};
struct AnonymousWeights {
    uint8_t* data = nullptr;
    size_t bytes = 0;
    ~AnonymousWeights() { if (data) ::munmap(data, bytes); }
    void allocate(size_t n) {
        void* p = ::mmap(nullptr, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p == MAP_FAILED) fail(net::system_error("anonymous weight mmap"));
        data = static_cast<uint8_t*>(p); bytes = n;
#ifdef MADV_DONTDUMP
        ::madvise(data, bytes, MADV_DONTDUMP);
#endif
    }
};
uint64_t proc_kib(const char* path, const char* wanted) {
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, std::strlen(wanted), wanted) != 0) continue;
        const char* p = line.data() + std::strlen(wanted);
        const char* end = line.data() + line.size();
        while (p != end && (*p == ' ' || *p == '\t')) ++p;
        uint64_t kib = 0;
        const auto result = std::from_chars(p, end, kib);
        if (result.ec == std::errc{} && kib <= std::numeric_limits<uint64_t>::max() / 1024) return kib * 1024;
        break;
    }
    fail(std::string("cannot read ") + wanted + " from " + path);
}
uint64_t available_ram() { return proc_kib("/proc/meminfo", "MemAvailable:"); }
void require_headroom(uint64_t additional) {
    const uint64_t available = available_ram();
    if (available < kHeadroom || additional > available - kHeadroom)
        fail("insufficient MemAvailable: available=" + std::to_string(available) +
             " additional=" + std::to_string(additional) + " required_headroom=" + std::to_string(kHeadroom));
}
void file_read(int fd, void* destination, size_t bytes) {
    auto* p = static_cast<uint8_t*>(destination);
    while (bytes) {
        if (stopping) fail("shutdown during weight loading");
        const ssize_t n = ::read(fd, p, bytes);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) fail(n == 0 ? "truncated expert weight file" : net::system_error("weight read"));
        p += n; bytes -= static_cast<size_t>(n);
    }
}
struct Options {
    std::string weights;
    uint16_t port = 0;
    uint64_t ram = 0, vram = 0;
};
uint64_t positive_number(const char* value, const char* name) {
    uint64_t n = 0;
    const char* end = value + std::strlen(value);
    const auto parsed = std::from_chars(value, end, n);
    if (parsed.ec != std::errc{} || parsed.ptr != end || n == 0) fail(std::string("invalid ") + name);
    return n;
}
Options options(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::puts("strata-network-experts --weights PATH --port N --ram-mib N --vram-mib N\n"
                      "Binds 127.0.0.1 only. RAM cap applies to anonymous expert weights; VRAM cap applies to the fixed-slot cache.\n"
                      "Requires 8 GiB MemAvailable headroom. Buffers and CUDA context are additional to the cache.");
            std::exit(0);
        }
        if (i + 1 == argc) fail("missing value for " + arg);
        const char* value = argv[++i];
        if (arg == "--weights") {
            if (!o.weights.empty()) fail("duplicate --weights");
            o.weights = value;
        } else if (arg == "--port") {
            if (o.port) fail("duplicate --port");
            const uint64_t n = positive_number(value, "--port");
            if (n > 65535) fail("--port must be in 1..65535");
            o.port = static_cast<uint16_t>(n);
        } else if (arg == "--ram-mib" || arg == "--vram-mib") {
            uint64_t& budget = arg == "--ram-mib" ? o.ram : o.vram;
            if (budget) fail("duplicate " + arg);
            const uint64_t n = positive_number(value, arg.c_str());
            if (n > std::numeric_limits<size_t>::max() / kMiB) fail("oversized " + arg);
            budget = n * kMiB;
        } else fail("unknown option: " + arg);
    }
    if (o.weights.empty() || !o.port || !o.ram || !o.vram)
        fail("required: --weights PATH --port N --ram-mib N --vram-mib N");
    return o;
}

struct Slot {
    int key = -1;
    uint64_t used = 0;
    bool pinned = false;
};
struct HostMetadata {
    unsigned long long pointers[net::kMaxJobs]{};
    int32_t starts[net::kMaxJobs + 1]{};
    int32_t destinations[net::kMaxJobs]{}, tokens[net::kMaxJobs]{};
    int32_t groups = 0;
};

class Worker {
public:
    void load(const Options& o) {
        Fd file(::open(o.weights.c_str(), O_RDONLY | O_CLOEXEC));
        syscheck(file.value >= 0, "open weights");
        struct stat st{};
        syscheck(::fstat(file.value, &st) == 0, "stat weights");
        if (!S_ISREG(st.st_mode) || st.st_size < 12) fail("weights must be a regular STRNEXP1 file");
        char magic[8]; uint32_t manifest_bytes = 0;
        file_read(file.value, magic, sizeof(magic));
        file_read(file.value, &manifest_bytes, sizeof(manifest_bytes));
        if (std::memcmp(magic, net::kFileMagic, sizeof(magic)) || manifest_bytes > 65536 || manifest_bytes == 0)
            fail("invalid expert file magic or manifest length");
        std::vector<uint8_t> encoded(manifest_bytes);
        file_read(file.value, encoded.data(), encoded.size());
        std::string error;
        if (!net::decode_manifest(encoded.data(), encoded.size(), manifest_, error)) fail(error);
        const uint64_t header_bytes = 12ull + manifest_bytes;
        if (manifest_.weight_bytes > std::numeric_limits<size_t>::max() ||
            manifest_.weight_bytes > std::numeric_limits<uint64_t>::max() - header_bytes ||
            static_cast<uint64_t>(st.st_size) != header_bytes + manifest_.weight_bytes)
            fail("expert file length differs from exact manifest payload (truncation or trailing data)");
        if (manifest_.weight_bytes > o.ram)
            fail("expert payload exceeds explicit --ram-mib cap: payload=" + std::to_string(manifest_.weight_bytes));
        uint64_t offset = 0;
        for (uint32_t i = 0; i < manifest_.layers; ++i) {
            const auto& l = manifest_.layout[i];
            // This wire/file version represents raw, plain-basis GGUF rows only.
            // Do not accept trellis/Hadamard, nibble transcodes or NVFP4 tails.
            const bool gu = l.gu_type == 18 || l.gu_type == 21 || l.gu_type == 22 || l.gu_type == 23;
            const bool down = l.down_type == 20 || l.down_type == 42;
            if (!gu || !down || !kernels::native_expert_supported(l.gu_type, l.down_type, manifest_.embedding, manifest_.feed_forward))
                fail("unsupported expert type/basis at layer " + std::to_string(i));
            layouts_[i] = kernels::native_expert_layout(l.gu_type, l.down_type, manifest_.embedding, manifest_.feed_forward);
            const auto& native = layouts_[i];
            if (native.gu_row != l.gu_row || native.d_row != l.down_row || native.bytes != l.blob_bytes ||
                native.up_off != l.gu_row * manifest_.feed_forward || native.down_off != 2 * native.up_off ||
                native.had_block || native.d_signs || native.gu_nib || native.tail_off)
                fail("native expert row/blob/basis mismatch at layer " + std::to_string(i));
            offsets_[i] = offset;
            offset += l.blob_bytes * manifest_.owned;
            stride_ = std::max(stride_, static_cast<size_t>(l.blob_bytes));
        }
        require_headroom(manifest_.weight_bytes);
        weights_.allocate(static_cast<size_t>(manifest_.weight_bytes));
        // read() writes directly into the only anonymous weight allocation. Drop
        // completed clean file pages in bounded chunks instead of keeping two copies.
        int advice = ::posix_fadvise(file.value, 0, 0, POSIX_FADV_SEQUENTIAL);
        if (advice) fail("weight sequential fadvise: " + std::string(std::strerror(advice)));
        constexpr size_t chunk = 8u << 20;
        for (size_t done = 0; done < weights_.bytes;) {
            const size_t n = std::min(chunk, weights_.bytes - done);
            file_read(file.value, weights_.data + done, n);
            const long page = ::sysconf(_SC_PAGESIZE);
            if (page <= 0) fail("cannot determine page size");
            const uint64_t end = header_bytes + done + n;
            const uint64_t start = done == 0 ? 0 : (header_bytes + done) / page * page;
            const uint64_t dropped_end = end / page * page;
            advice = ::posix_fadvise(file.value, static_cast<off_t>(start), static_cast<off_t>(dropped_end - start), POSIX_FADV_DONTNEED);
            if (advice) fail("weight cache fadvise: " + std::string(std::strerror(advice)));
            done += n;
            require_headroom(0);
        }
        advice = ::posix_fadvise(file.value, 0, 0, POSIX_FADV_DONTNEED);
        if (advice) fail("weight final fadvise: " + std::string(std::strerror(advice)));
        reply_manifest_ = net::encode_manifest(manifest_);
        probe_input_.resize(net::kMaxProbe);
        probe_zero_.resize(net::kMaxProbe, 0);
        stats_.host_bytes = weights_.bytes;
        stats_.pid = static_cast<uint64_t>(::getpid());
        initialize_cuda(o.vram);
    }

    const char* device_name() const { return device_name_.c_str(); }
    size_t stride() const { return stride_; }
    size_t slots() const { return slots_.size(); }
    net::Stats snapshot() {
        stats_.resident_bytes = proc_kib("/proc/self/status", "VmRSS:");
        return stats_;
    }
    void report(const char* event) {
        const auto s = snapshot();
        std::fprintf(stderr,
            "%s host_bytes=%llu vram_cache_bytes=%llu slots=%zu slot_stride=%zu requests=%llu jobs=%llu "
            "cache_hits=%llu cache_misses=%llu upload_bytes=%llu activation_bytes=%llu result_bytes=%llu pid=%llu VmRSS=%llu\n",
            event, static_cast<unsigned long long>(s.host_bytes), static_cast<unsigned long long>(s.vram_bytes), slots_.size(), stride_,
            static_cast<unsigned long long>(s.requests), static_cast<unsigned long long>(s.jobs),
            static_cast<unsigned long long>(s.cache_hits), static_cast<unsigned long long>(s.cache_misses),
            static_cast<unsigned long long>(s.upload_bytes), static_cast<unsigned long long>(s.activation_bytes),
            static_cast<unsigned long long>(s.result_bytes), static_cast<unsigned long long>(s.pid),
            static_cast<unsigned long long>(s.resident_bytes));
        std::fflush(stderr);
    }

    void session(int fd) {
        bool hello = false;
        uint64_t sequence = 0;
        while (!stopping) {
            net::Header header;
            std::string error;
            if (!net::read_header(fd, header, error)) {
                if (error == "invalid expert frame header") net::send_error(fd, header.sequence, error);
                if (!stopping) std::fprintf(stderr, "client IO: %s\n", error.c_str());
                return;
            }
            const auto received = Clock::now();
            try {
                if (header.sequence == 0 || (sequence != 0 && header.sequence != sequence + 1))
                    fail("invalid expert request sequence");
                sequence = header.sequence;
                if (!hello) {
                    if (header.op != static_cast<uint32_t>(net::Op::Hello) || header.bytes != 68)
                        fail("HELLO must be first, with version and 64-byte source identity");
                    std::array<uint8_t, 68> body;
                    if (!net::read_exact(fd, body.data(), body.size(), error)) fail(error);
                    uint32_t version; std::memcpy(&version, body.data(), sizeof(version));
                    if (version != net::kVersion) fail("expert protocol version mismatch");
                    if (std::memcmp(body.data() + 4, manifest_.model_id.data(), 64)) fail("expert model identity mismatch");
                    if (!net::send_frame(fd, header.op | net::kReply, sequence, reply_manifest_.data(), reply_manifest_.size(), error)) fail(error);
                    hello = true;
                } else if (header.op == static_cast<uint32_t>(net::Op::Stats)) {
                    if (header.bytes != 0) fail("STATS request must be empty");
                    const auto s = snapshot();
                    if (!net::send_frame(fd, header.op | net::kReply, sequence, &s, sizeof(s), error)) fail(error);
                } else if (header.op == static_cast<uint32_t>(net::Op::Run)) {
                    receive_run(fd, header);
                    const auto body_ready = Clock::now();
                    net::RunTiming timing;
                    timing.receive_ns = elapsed_ns(received, body_ready);
                    execute(body_ready, timing);
                    const size_t bytes = static_cast<size_t>(run_.jobs) * manifest_.embedding * sizeof(float);
                    const net::Part parts[]{{&timing, sizeof(timing)}, {output_.data(), bytes}};
                    if (!net::send_parts(fd, header.op | net::kReply, sequence, parts, 2, error)) fail(error);
                    stats_.result_bytes += bytes;
                } else if (header.op == static_cast<uint32_t>(net::Op::Probe)) {
                    probe(fd, header, received);
                } else fail("unexpected expert operation (HELLO cannot repeat)");
            } catch (const std::exception& e) {
                net::send_error(fd, header.sequence, e.what());
                std::fprintf(stderr, "client error: %s\n", e.what());
                return; // no ambiguous stream recovery, retries or zero contributions
            }
        }
    }

private:
    void initialize_cuda(uint64_t budget) {
        cucheck(cudaSetDevice(0), "select CUDA device 0");
        cudaDeviceProp prop{};
        cucheck(cudaGetDeviceProperties(&prop, 0), "CUDA device properties");
        if (prop.major * 10 + prop.minor < 75) fail("expert worker requires compute capability >= 7.5");
        device_name_ = prop.name;
        // Keep the full gate/up -> SwiGLU -> down operation even if benchmark
        // phase knobs exist in the environment. No CPU or fake fallback.
        kernels::native_expert_set_mode(0, 0);
        kernels::native_grouped_set_v1(false);
        size_t free = 0, total = 0;
        cucheck(cudaMemGetInfo(&free, &total), "CUDA free memory");
        const size_t count = static_cast<size_t>(budget / stride_);
        if (count == 0 || count > std::numeric_limits<int32_t>::max()) fail("--vram-mib yields invalid cache slot count");
        const size_t capacity = count * stride_;
        const size_t scratch = kernels::native_expert_scratch_bytes(net::kMaxJobs, net::kFeedForward);
        const size_t activation = static_cast<size_t>(net::kMaxTokens) * net::kEmbedding * sizeof(float);
        const size_t quantized = static_cast<size_t>(net::kMaxTokens) * (net::kEmbedding / QK8_1) * sizeof(block_q8_1);
        const size_t result = static_cast<size_t>(net::kMaxJobs) * net::kEmbedding * sizeof(float);
        const size_t overhead = scratch + activation + quantized + result + sizeof(HostMetadata);
        if (free < overhead || capacity > free - overhead) fail("insufficient free CUDA memory for cache and fixed buffers");
        slots_.resize(count);
        index_.assign(static_cast<size_t>(manifest_.layers) * manifest_.owned, -1);
        cache_.allocate(capacity);
        x_.allocate(activation); xq_.allocate(quantized); scratch_.allocate(scratch); result_.allocate(result);
        metadata_.allocate(sizeof(HostMetadata));
        cucheck(cudaStreamCreateWithFlags(&stream_.value, cudaStreamNonBlocking), "create expert CUDA stream");
        run_events_.initialize();
        stats_.vram_bytes = capacity;
    }
    void receive_run(int fd, const net::Header& h) {
        std::string error;
        if (h.bytes < sizeof(net::Run)) fail("RUN body is shorter than its header");
        if (!net::read_exact(fd, &run_, sizeof(run_), error)) fail(error);
        if (run_.flags & ~net::kProfileRun) fail("RUN contains unknown flags");
        if (run_.layer >= manifest_.layers || run_.tokens == 0 || run_.tokens > net::kMaxTokens ||
            run_.jobs == 0 || run_.jobs > net::kMaxJobs || run_.jobs > run_.tokens * 10)
            fail("invalid RUN layer/token/job counts");
        const size_t job_bytes = static_cast<size_t>(run_.jobs) * sizeof(net::Job);
        const size_t activation_bytes = static_cast<size_t>(run_.tokens) * manifest_.embedding * sizeof(float);
        if (h.bytes != sizeof(net::Run) + job_bytes + activation_bytes) fail("RUN body length mismatch");
        if (!net::read_exact(fd, jobs_.data(), job_bytes, error)) fail(error);
        std::array<uint32_t, net::kMaxTokens> per_token{};
        for (uint32_t i = 0; i < run_.jobs; ++i) {
            const auto& j = jobs_[i];
            // row is opaque caller metadata, never used as an output array index.
            if (j.expert >= manifest_.owned || j.token >= run_.tokens) fail("RUN expert/token outside ownership or batch");
            if (++per_token[j.token] > 10) fail("RUN has more than ten routed jobs for a token");
            for (uint32_t k = 0; k < i; ++k)
                if (jobs_[k].expert == j.expert && jobs_[k].token == j.token) fail("duplicate RUN expert/token job");
        }
        if (!net::read_exact(fd, input_.data(), activation_bytes, error)) fail(error);
        for (size_t i = 0; i < activation_bytes / sizeof(float); ++i)
            if (!std::isfinite(input_[i])) fail("RUN contains non-finite activation");
    }
    void probe(int fd, const net::Header& h, Clock::time_point received) {
        if (h.bytes < sizeof(net::Probe) || h.bytes - sizeof(net::Probe) > net::kMaxProbe)
            fail("PROBE request padding exceeds its bounded size");
        std::string error;
        net::Probe request;
        if (!net::read_exact(fd, &request, sizeof(request), error)) fail(error);
        if (request.reserved != 0 || request.response_bytes > net::kMaxProbe)
            fail("invalid PROBE reserved field or response size");
        const size_t padding = static_cast<size_t>(h.bytes - sizeof(request));
        if (!net::read_exact(fd, probe_input_.data(), padding, error)) fail(error);
        const net::ProbeReply reply{elapsed_ns(received, Clock::now())};
        const net::Part parts[]{{&reply, sizeof(reply)}, {probe_zero_.data(), request.response_bytes}};
        if (!net::send_parts(fd, h.op | net::kReply, h.sequence, parts, 2, error)) fail(error);
    }
    void execute(Clock::time_point body_ready, net::RunTiming& timing) {
        const bool profile = (run_.flags & net::kProfileRun) != 0;
        const uint64_t hits = stats_.cache_hits, misses = stats_.cache_misses, uploaded = stats_.upload_bytes;
        std::array<uint32_t, net::kMaxJobs> order;
        for (uint32_t i = 0; i < run_.jobs; ++i) order[i] = i;
        std::sort(order.begin(), order.begin() + run_.jobs, [&](uint32_t a, uint32_t b) {
            return jobs_[a].expert < jobs_[b].expert || (jobs_[a].expert == jobs_[b].expert && a < b);
        });
        std::array<uint32_t, net::kMaxJobs> experts{};
        uint32_t groups = 0;
        for (uint32_t e = 0; e < run_.jobs; ++e) {
            const auto& job = jobs_[order[e]];
            if (e == 0 || job.expert != experts[groups - 1]) {
                experts[groups] = job.expert;
                host_metadata_.starts[groups++] = static_cast<int32_t>(e);
            }
            host_metadata_.destinations[e] = static_cast<int32_t>(order[e]);
            host_metadata_.tokens[e] = static_cast<int32_t>(job.token);
        }
        host_metadata_.groups = static_cast<int32_t>(groups);
        host_metadata_.starts[groups] = static_cast<int32_t>(run_.jobs);
        if (groups > slots_.size()) fail("RUN distinct expert batch exceeds VRAM cache capacity");
        for (auto& slot : slots_) slot.pinned = false;
        // First pin EVERY hit. Inserting an early miss must never evict a later
        // already-cached expert belonging to this same active RUN.
        for (uint32_t g = 0; g < groups; ++g) {
            const size_t key = static_cast<size_t>(run_.layer) * manifest_.owned + experts[g];
            const int32_t slot = index_[key];
            if (slot >= 0) slots_[slot].pinned = true;
        }
        // This interval includes host enqueue/pageable-staging gaps, not just DMA.
        if (profile) run_events_.record(RunEvents::H2DBegin, stream_.value);
        for (uint32_t g = 0; g < groups; ++g) {
            const size_t key = static_cast<size_t>(run_.layer) * manifest_.owned + experts[g];
            int32_t selected = index_[key];
            if (selected >= 0) ++stats_.cache_hits;
            else {
                ++stats_.cache_misses;
                size_t victim = slots_.size();
                for (size_t s = 0; s < slots_.size(); ++s) {
                    if (slots_[s].pinned) continue;
                    if (victim == slots_.size() || slots_[s].key < 0 || slots_[s].used < slots_[victim].used) victim = s;
                    if (slots_[s].key < 0) break;
                }
                if (victim == slots_.size()) fail("internal cache pinning capacity violation");
                auto& slot = slots_[victim];
                if (slot.key >= 0) index_[slot.key] = -1;
                slot.key = -1;
                const size_t bytes = layouts_[run_.layer].bytes;
                const uint8_t* source = weights_.data + offsets_[run_.layer] + static_cast<size_t>(experts[g]) * bytes;
                // Upload on the expert stream so kernels cannot overtake a
                // pageable-host copy staged asynchronously by the CUDA driver.
                cucheck(cudaMemcpyAsync(cache_.as<uint8_t>() + victim * stride_, source, bytes,
                                       cudaMemcpyHostToDevice, stream_.value), "expert PC-RAM to GPU upload");
                stats_.upload_bytes += bytes;
                selected = static_cast<int32_t>(victim);
                slot.key = static_cast<int>(key);
                index_[key] = selected;
            }
            slots_[selected].pinned = true;
            slots_[selected].used = ++clock_;
            host_metadata_.pointers[g] = reinterpret_cast<unsigned long long>(cache_.as<uint8_t>() + static_cast<size_t>(selected) * stride_);
        }
        const size_t activation_bytes = static_cast<size_t>(run_.tokens) * manifest_.embedding * sizeof(float);
        const size_t result_bytes = static_cast<size_t>(run_.jobs) * manifest_.embedding * sizeof(float);
        cucheck(cudaMemcpyAsync(x_.data, input_.data(), activation_bytes, cudaMemcpyHostToDevice, stream_.value), "upload activations");
        cucheck(cudaMemcpyAsync(metadata_.data, &host_metadata_, sizeof(host_metadata_), cudaMemcpyHostToDevice, stream_.value), "upload grouped metadata");
        if (profile) run_events_.record(RunEvents::KernelBegin, stream_.value);
        // CPU preparation overlaps the same-stream H2D stage; these are not additive.
        timing.prepare_ns = elapsed_ns(body_ready, Clock::now());
        kernels::quantize_q8_1_rows(x_.as<float>(), run_.tokens, manifest_.embedding, xq_.data, stream_.value);
        auto* m = metadata_.as<HostMetadata>();
        kernels::native_expert_grouped(layouts_[run_.layer], m->pointers, m->starts, &m->groups,
            m->destinations, m->tokens, groups, run_.jobs, xq_.data, scratch_.data, result_.as<float>(), stream_.value, groups);
        cucheck(cudaGetLastError(), "grouped expert launch");
        if (profile) run_events_.record(RunEvents::D2HBegin, stream_.value);
        cucheck(cudaMemcpyAsync(output_.data(), result_.data, result_bytes, cudaMemcpyDeviceToHost, stream_.value), "download expert results");
        if (profile) run_events_.record(RunEvents::End, stream_.value);
        cucheck(cudaStreamSynchronize(stream_.value), "complete expert RUN");
        timing.execute_ns = elapsed_ns(body_ready, Clock::now());
        if (profile) {
            timing.h2d_stage_ns = run_events_.duration_ns(RunEvents::H2DBegin, RunEvents::KernelBegin);
            timing.kernel_ns = run_events_.duration_ns(RunEvents::KernelBegin, RunEvents::D2HBegin);
            timing.d2h_ns = run_events_.duration_ns(RunEvents::D2HBegin, RunEvents::End);
        }
        timing.cache_hits = stats_.cache_hits - hits;
        timing.cache_misses = stats_.cache_misses - misses;
        timing.upload_bytes = stats_.upload_bytes - uploaded;
        stats_.activation_bytes += activation_bytes;
        ++stats_.requests; stats_.jobs += run_.jobs;
    }

    net::Manifest manifest_;
    net::Stats stats_;
    AnonymousWeights weights_;
    std::array<kernels::NativeExpertLayout, 48> layouts_{};
    std::array<uint64_t, 48> offsets_{};
    size_t stride_ = 0;
    std::string device_name_;
    std::vector<uint8_t> reply_manifest_;
    std::vector<uint8_t> probe_input_, probe_zero_;
    std::vector<Slot> slots_;
    std::vector<int32_t> index_;
    uint64_t clock_ = 0;
    DeviceMemory cache_, x_, xq_, scratch_, result_, metadata_;
    Stream stream_;
    RunEvents run_events_;
    net::Run run_;
    std::array<net::Job, net::kMaxJobs> jobs_{};
    std::array<float, net::kMaxTokens * net::kEmbedding> input_;
    std::array<float, net::kMaxJobs * net::kEmbedding> output_;
    HostMetadata host_metadata_;
};

void configure_client(int fd) {
    const int one = 1;
    const timeval timeout{60, 0};
    syscheck(::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) == 0, "TCP_NODELAY");
    syscheck(::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) == 0, "SO_RCVTIMEO");
    syscheck(::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) == 0, "SO_SNDTIMEO");
#ifdef SO_NOSIGPIPE
    syscheck(::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) == 0, "SO_NOSIGPIPE");
#endif
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Options o = options(argc, argv);
        struct sigaction action{};
        action.sa_handler = stop_worker;
        ::sigemptyset(&action.sa_mask);
        syscheck(::sigaction(SIGTERM, &action, nullptr) == 0, "SIGTERM handler");
        syscheck(::sigaction(SIGINT, &action, nullptr) == 0, "SIGINT handler");
        ::signal(SIGPIPE, SIG_IGN);
        Worker worker;
        worker.load(o);
        Fd listener(::socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC | SOCK_NONBLOCK, 0));
        syscheck(listener.value >= 0, "worker socket");
        const int one = 1;
        syscheck(::setsockopt(listener.value, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) == 0, "SO_REUSEADDR");
        sockaddr_in addr{};
        addr.sin_family = AF_INET; addr.sin_port = htons(o.port); addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        syscheck(::bind(listener.value, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0, "bind 127.0.0.1");
        syscheck(::listen(listener.value, 1) == 0, "listen expert worker");
        const auto initial = worker.snapshot();
        std::printf("READY port=%u bind=127.0.0.1 device=%s host_bytes=%llu vram_cache_bytes=%llu slot_stride=%zu slots=%zu pid=%llu VmRSS=%llu\n",
                    o.port, worker.device_name(), static_cast<unsigned long long>(initial.host_bytes),
                    static_cast<unsigned long long>(initial.vram_bytes), worker.stride(), worker.slots(),
                    static_cast<unsigned long long>(initial.pid), static_cast<unsigned long long>(initial.resident_bytes));
        std::fflush(stdout);
        while (!stopping) {
            pollfd event{listener.value, POLLIN, 0};
            const int n = ::poll(&event, 1, 1000);
            if (n < 0 && errno == EINTR) continue;
            syscheck(n >= 0, "poll listener");
            if (!n || stopping) continue;
            if (!(event.revents & POLLIN)) fail("expert listener failed");
            Fd client(::accept(listener.value, nullptr, nullptr));
            if (client.value < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
            syscheck(client.value >= 0, "accept client");
            try {
                configure_client(client.value);
                active_socket = client.value;
                if (!stopping) worker.session(client.value);
                active_socket = -1;
            } catch (const std::exception& e) {
                active_socket = -1;
                std::fprintf(stderr, "client setup error: %s\n", e.what());
            }
            worker.report("DISCONNECT");
        }
        worker.report("SHUTDOWN");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-network-experts: %s\n", e.what());
        return stopping ? 0 : 1;
    }
}
