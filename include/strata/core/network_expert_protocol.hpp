#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

namespace strata::core::network {

static_assert(std::endian::native == std::endian::little,
              "The experimental expert protocol requires a little-endian host");
constexpr uint32_t kMagic = 0x3158454e; // NEX1
constexpr uint32_t kVersion = 2;
constexpr uint32_t kReply = 0x80000000u;
constexpr uint32_t kError = 0xffffffffu;
constexpr size_t kMaxFrame = 4u << 20;
constexpr uint32_t kMaxTokens = 8, kMaxJobs = 80;
constexpr uint32_t kProfileRun = 1;
constexpr size_t kMaxProbe = 1u << 20;
constexpr uint32_t kEmbedding = 2560, kFeedForward = 640;
constexpr char kFileMagic[8] = {'S','T','R','N','E','X','P','1'};

enum class Op : uint32_t { Hello = 1, Run = 2, Stats = 3, Probe = 4 };
struct Header { uint32_t magic = kMagic, op = 0; uint64_t sequence = 0, bytes = 0; };
static_assert(sizeof(Header) == 24);
struct Job { uint32_t expert = 0, token = 0, row = 0; };
static_assert(sizeof(Job) == 12);
struct Run { uint32_t layer = 0, tokens = 0, jobs = 0, flags = 0; };
static_assert(sizeof(Run) == 16);
// RUN replies begin with these durations, then unweighted FP32 expert rows.
// All clocks are host-local: durations never require synchronized device clocks.
// prepare_ns overlaps H2D; do not add it to the GPU stages or execute_ns.
// h2d_stage_ns includes GPU idle time while the host enqueues pageable uploads.
struct RunTiming {
    uint64_t receive_ns = 0, prepare_ns = 0, execute_ns = 0;
    uint64_t h2d_stage_ns = 0, kernel_ns = 0, d2h_ns = 0;
    uint64_t cache_hits = 0, cache_misses = 0, upload_bytes = 0;
};
static_assert(sizeof(RunTiming) == 72);
// A bounded no-compute application round trip, using the same SSH/TCP path.
// Body: Probe + arbitrary request padding; reply: ProbeReply + zero padding.
struct Probe { uint32_t response_bytes = 0, reserved = 0; };
struct ProbeReply { uint64_t receive_ns = 0; };
static_assert(sizeof(Probe) == 8 && sizeof(ProbeReply) == 8);
struct Layer {
    uint32_t gu_type = 0, down_type = 0;
    uint64_t gu_row = 0, down_row = 0, blob_bytes = 0;
};
static_assert(sizeof(Layer) == 32);
struct Manifest {
    std::string model_id;
    uint32_t layers = 48, experts = 512, owned = 0;
    uint32_t embedding = kEmbedding, feed_forward = kFeedForward;
    uint64_t weight_bytes = 0;
    std::vector<Layer> layout;
};
struct Stats {
    uint64_t host_bytes = 0, vram_bytes = 0, requests = 0, jobs = 0;
    uint64_t cache_hits = 0, cache_misses = 0, upload_bytes = 0;
    uint64_t activation_bytes = 0, result_bytes = 0, pid = 0, resident_bytes = 0;
};
static_assert(sizeof(Stats) == 88);
struct Part { const void* data; size_t bytes; };

inline std::string system_error(const char* operation) {
    return std::string(operation) + ": " + std::strerror(errno);
}
inline bool read_exact(int fd, void* dst, size_t bytes, std::string& err) {
    auto* p = static_cast<uint8_t*>(dst);
    while (bytes) {
        const ssize_t n = ::recv(fd, p, bytes, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { err = n == 0 ? "expert worker disconnected" : system_error("expert recv"); return false; }
        p += n; bytes -= static_cast<size_t>(n);
    }
    return true;
}
inline bool read_header(int fd, Header& h, std::string& err) {
    if (!read_exact(fd, &h, sizeof(h), err)) return false;
    if (h.magic != kMagic || h.bytes > kMaxFrame) { err = "invalid expert frame header"; return false; }
    return true;
}
inline bool send_parts(int fd, uint32_t op, uint64_t sequence, const Part* parts,
                       size_t count, std::string& err) {
    if (count > 4) { err = "too many expert frame parts"; return false; }
    Header h; h.op = op; h.sequence = sequence;
    std::array<iovec, 5> io{};
    io[0] = {&h, sizeof(h)};
    for (size_t i = 0; i < count; ++i) {
        if (parts[i].bytes > kMaxFrame - h.bytes) { err = "expert frame is too large"; return false; }
        h.bytes += parts[i].bytes;
        io[i + 1] = {const_cast<void*>(parts[i].data), parts[i].bytes};
    }
    size_t first = 0;
    while (first < count + 1) {
        msghdr msg{}; msg.msg_iov = io.data() + first; msg.msg_iovlen = count + 1 - first;
#ifdef MSG_NOSIGNAL
        const ssize_t sent = ::sendmsg(fd, &msg, MSG_NOSIGNAL);
#else
        const ssize_t sent = ::sendmsg(fd, &msg, 0);
#endif
        if (sent < 0 && errno == EINTR) continue;
        if (sent <= 0) { err = system_error("expert send"); return false; }
        size_t n = static_cast<size_t>(sent);
        while (first < count + 1 && n >= io[first].iov_len) { n -= io[first].iov_len; ++first; }
        if (first < count + 1 && n) {
            io[first].iov_base = static_cast<uint8_t*>(io[first].iov_base) + n;
            io[first].iov_len -= n;
        }
    }
    return true;
}
inline bool send_frame(int fd, uint32_t op, uint64_t sequence, const void* data,
                       size_t bytes, std::string& err) {
    const Part p{data, bytes}; return send_parts(fd, op, sequence, &p, 1, err);
}
inline bool send_error(int fd, uint64_t sequence, const std::string& message) {
    std::string ignored;
    return send_frame(fd, kError, sequence, message.data(), std::min<size_t>(message.size(), 4096), ignored);
}
inline bool response_header(int fd, Op op, uint64_t sequence, Header& h, std::string& err) {
    if (!read_header(fd, h, err)) return false;
    if (h.sequence != sequence) { err = "expert response sequence mismatch"; return false; }
    if (h.op == kError) {
        if (h.bytes > 4096) { err = "oversized expert error"; return false; }
        err.resize(static_cast<size_t>(h.bytes));
        std::string io_error;
        if (!read_exact(fd, err.data(), err.size(), io_error)) err = io_error;
        return false;
    }
    if (h.op != (static_cast<uint32_t>(op) | kReply)) { err = "unexpected expert response"; return false; }
    return true;
}

inline bool valid_manifest(const Manifest& m, std::string& err) {
    if (m.model_id.size() != 64 || m.layers != 48 || m.experts != 512 ||
        m.embedding != kEmbedding || m.feed_forward != kFeedForward ||
        m.owned == 0 || m.owned >= m.experts || m.layout.size() != m.layers) {
        err = "unsupported Qwen expert manifest geometry"; return false;
    }
    for (char c : m.model_id) if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
        err = "invalid expert model identity"; return false;
    }
    uint64_t total = 0;
    for (const auto& l : m.layout) {
        if (l.gu_row == 0 || l.down_row == 0 || l.gu_row > 65536 || l.down_row > 65536 ||
            l.blob_bytes != 2 * l.gu_row * m.feed_forward + l.down_row * m.embedding ||
            l.blob_bytes > (16u << 20)) { err = "invalid expert layer layout"; return false; }
        total += l.blob_bytes * m.owned;
    }
    if (total != m.weight_bytes) { err = "expert manifest byte count mismatch"; return false; }
    return true;
}
inline std::vector<uint8_t> encode_manifest(const Manifest& m) {
    const size_t fixed = 64 + 5 * sizeof(uint32_t) + sizeof(uint64_t);
    std::vector<uint8_t> out(fixed + m.layout.size() * sizeof(Layer));
    uint8_t* p = out.data();
    std::memcpy(p, m.model_id.data(), 64); p += 64;
    const uint32_t dims[5] = {m.layers, m.experts, m.owned, m.embedding, m.feed_forward};
    std::memcpy(p, dims, sizeof(dims)); p += sizeof(dims);
    std::memcpy(p, &m.weight_bytes, sizeof(m.weight_bytes)); p += sizeof(m.weight_bytes);
    std::memcpy(p, m.layout.data(), m.layout.size() * sizeof(Layer));
    return out;
}
inline bool decode_manifest(const uint8_t* data, size_t bytes, Manifest& m, std::string& err) {
    const size_t fixed = 64 + 5 * sizeof(uint32_t) + sizeof(uint64_t);
    if (bytes < fixed || bytes > 65536) { err = "invalid expert manifest length"; return false; }
    const uint8_t* p = data;
    m.model_id.assign(reinterpret_cast<const char*>(p), 64); p += 64;
    uint32_t dims[5]; std::memcpy(dims, p, sizeof(dims)); p += sizeof(dims);
    m.layers=dims[0]; m.experts=dims[1]; m.owned=dims[2]; m.embedding=dims[3]; m.feed_forward=dims[4];
    std::memcpy(&m.weight_bytes, p, sizeof(m.weight_bytes)); p += sizeof(m.weight_bytes);
    if (m.layers != 48 || bytes != fixed + m.layers * sizeof(Layer)) { err = "invalid expert layout table"; return false; }
    m.layout.resize(m.layers); std::memcpy(m.layout.data(), p, m.layout.size() * sizeof(Layer));
    return valid_manifest(m, err);
}

} // namespace strata::core::network
