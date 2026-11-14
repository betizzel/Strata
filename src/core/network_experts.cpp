#include "strata/core/network_experts.hpp"

#include <arpa/inet.h>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <netinet/tcp.h>

namespace strata::core {

namespace {
uint64_t monotonic_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
std::atomic<uint64_t> next_window_id{0};
}

NetworkExperts::~NetworkExperts() {
    if (window_active_) {
        std::string err;
        if (!end_window(false, err)) std::fprintf(stderr, "%s\n", err.c_str());
    }
    close();
    close_trace();
}

bool NetworkExperts::trace_status(std::string& err, bool flush) {
    if (!trace_) return true;
    if (flush && std::fflush(trace_) != 0) trace_failed_ = true;
    if (std::ferror(trace_)) trace_failed_ = true;
    if (!trace_failed_) return true;
    err = "network experts: requested JSONL trace write/flush failed";
    return false;
}

void NetworkExperts::close_trace() {
    if (!trace_) return;
    std::string err;
    if (!trace_status(err, true)) std::fprintf(stderr, "%s\n", err.c_str());
    if (std::fclose(trace_) != 0)
        std::fprintf(stderr, "network experts: requested JSONL trace close failed: %s\n", std::strerror(errno));
    trace_ = nullptr;
}

bool NetworkExperts::begin_window(const char* phase, int64_t pos0, uint32_t tokens, std::string& err) {
    if (!trace_) return true;
    if (!trace_status(err)) return false;
    if (window_active_ || pending_sequence_) {
        err = "network experts: overlapping trace window"; return false;
    }
    window_phase_ = phase && std::strcmp(phase, "prefill") == 0 ? "prefill" :
                    phase && std::strcmp(phase, "decode") == 0 ? "decode" : "unknown";
    window_id_ = next_window_id.fetch_add(1, std::memory_order_relaxed) + 1;
    window_pos0_ = pos0;
    window_tokens_ = tokens;
    window_start_ns_ = monotonic_ns();
    window_active_ = true;
    return true;
}

bool NetworkExperts::end_window(bool success, std::string& err) {
    if (!trace_ || !window_active_) return trace_status(err);
    const uint64_t wall_ns = monotonic_ns() - window_start_ns_;
    const bool complete = success && !trace_failed_ && !pending_sequence_;
    if (std::fprintf(trace_, "{\"type\":\"window\",\"window_id\":%" PRIu64
                     ",\"phase\":\"%s\",\"pos0\":%" PRId64 ",\"tokens\":%u,\"wall_ns\":%" PRIu64
                     ",\"success\":%s}\n", window_id_, window_phase_, window_pos0_, window_tokens_,
                     wall_ns, complete ? "true" : "false") < 0) trace_failed_ = true;
    window_active_ = false;
    window_id_ = 0;
    window_phase_ = "unknown";
    window_pos0_ = 0;
    window_tokens_ = 0;
    // Resident benchmark engines need each completed window visible immediately,
    // without turning each layer request into a separate file flush.
    return trace_status(err, true);
}

void NetworkExperts::close() {
    if (fd_ >= 0) { ::shutdown(fd_, SHUT_RDWR); ::close(fd_); }
    fd_ = -1;
    pending_sequence_ = 0;
    pending_values_ = 0;
}

bool NetworkExperts::open(const std::string& endpoint, const network::Manifest& expected, std::string& err) {
    close();
    if (window_active_ && !end_window(false, err)) return false;
    close_trace();
    trace_failed_ = false;
    if (expected.model_id.size() != 64 || expected.layout.size() != 48) {
        err = "network experts: missing local model identity or layout"; return false;
    }
    const size_t colon = endpoint.rfind(':');
    int port = 0;
    if (colon == std::string::npos || endpoint.substr(0, colon) != "127.0.0.1") {
        err = "network experts: use a private SSH tunnel at 127.0.0.1:PORT"; return false;
    }
    const char* start = endpoint.data() + colon + 1;
    const char* end = endpoint.data() + endpoint.size();
    const auto parsed = std::from_chars(start, end, port);
    if (parsed.ec != std::errc() || parsed.ptr != end || port < 1 || port > 65535) {
        err = "network experts: invalid worker port"; return false;
    }
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd_ < 0) { err = network::system_error("expert socket"); return false; }
    const int one = 1;
    const timeval timeout{60, 0};
    if (::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)) != 0 ||
        ::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) != 0 ||
        ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) != 0) {
        err = network::system_error("expert socket options"); close(); return false;
    }
#ifdef SO_NOSIGPIPE
    if (::setsockopt(fd_, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)) != 0) {
        err = network::system_error("expert SO_NOSIGPIPE"); close(); return false;
    }
#endif
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd_, reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
        err = network::system_error("expert connect"); close(); return false;
    }
    sequence_ = 0; completed_ = jobs_ = input_bytes_ = output_bytes_ = 0;
    std::array<uint8_t, 68> hello{};
    const uint32_t version = network::kVersion;
    std::memcpy(hello.data(), &version, sizeof(version));
    std::memcpy(hello.data() + sizeof(version), expected.model_id.data(), 64);
    network::Header response;
    if (!network::send_frame(fd_, static_cast<uint32_t>(network::Op::Hello), ++sequence_, hello.data(), hello.size(), err) ||
        !network::response_header(fd_, network::Op::Hello, sequence_, response, err)) {
        close(); return false;
    }
    if (response.bytes > 65536) { err = "network experts: oversized worker manifest"; close(); return false; }
    std::vector<uint8_t> bytes(static_cast<size_t>(response.bytes));
    if (!network::read_exact(fd_, bytes.data(), bytes.size(), err) ||
        !network::decode_manifest(bytes.data(), bytes.size(), manifest_, err)) { close(); return false; }
    if (manifest_.model_id != expected.model_id || manifest_.layers != expected.layers ||
        manifest_.experts != expected.experts || manifest_.embedding != expected.embedding ||
        manifest_.feed_forward != expected.feed_forward) {
        err = "network experts: worker has different model weights or geometry"; close(); return false;
    }
    for (size_t i = 0; i < expected.layout.size(); ++i) {
        const auto& a = expected.layout[i]; const auto& b = manifest_.layout[i];
        if (a.gu_type != b.gu_type || a.down_type != b.down_type || a.gu_row != b.gu_row ||
            a.down_row != b.down_row || a.blob_bytes != b.blob_bytes) {
            err = "network experts: worker quant layout differs at layer " + std::to_string(i); close(); return false;
        }
    }
    if (!stats(stats_, err)) { close(); return false; }
    if (const char* path = std::getenv("STRATA_NETWORK_TRACE"); path && *path) {
        if (path[0] != '/') {
            err = "network experts: STRATA_NETWORK_TRACE must be an absolute path"; close(); return false;
        }
        trace_ = std::fopen(path, "w");
        if (!trace_) { err = network::system_error("network experts trace open"); close(); return false; }
        if (std::setvbuf(trace_, trace_buffer_.data(), _IOFBF, trace_buffer_.size()) != 0) {
            err = "network experts: cannot buffer requested JSONL trace"; close(); close_trace(); return false;
        }
        if (std::fprintf(trace_, "{\"type\":\"session\",\"schema_version\":1,\"protocol_version\":%u,"
                         "\"model_id\":\"%s\",\"owned\":%u,\"embedding\":%u}\n",
                         network::kVersion, manifest_.model_id.c_str(), manifest_.owned, manifest_.embedding) < 0)
            trace_failed_ = true;
        if (!trace_status(err, true)) { close(); close_trace(); return false; }
    }
    return true;
}

bool NetworkExperts::begin(uint32_t layer, uint32_t tokens, const float* x,
                           const network::Job* jobs, size_t count, std::string& err) {
    if (!active() || pending_sequence_ || !x || !jobs || count == 0 || count > network::kMaxJobs ||
        layer >= manifest_.layers || tokens == 0 || tokens > network::kMaxTokens) {
        err = "network experts: invalid or overlapping expert request"; return false;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!owns(layer, jobs[i].expert) || jobs[i].token >= tokens || jobs[i].row >= network::kMaxJobs) {
            err = "network experts: invalid owned expert or token index"; return false;
        }
    }
    const network::Run run{layer, tokens, static_cast<uint32_t>(count), trace_ ? network::kProfileRun : 0u};
    const size_t activation_bytes = static_cast<size_t>(tokens) * manifest_.embedding * sizeof(float);
    const network::Part parts[] = {{&run, sizeof(run)}, {jobs, count * sizeof(network::Job)}, {x, activation_bytes}};
    pending_sequence_ = ++sequence_;
    pending_values_ = count * manifest_.embedding;
    if (trace_) {
        if (!trace_status(err)) { close(); return false; }
        pending_layer_ = layer;
        pending_jobs_ = static_cast<uint32_t>(count);
        pending_request_bytes_ = sizeof(network::Header) + sizeof(run) + count * sizeof(network::Job) + activation_bytes;
        pending_send_start_ns_ = monotonic_ns();
    }
    if (!network::send_parts(fd_, static_cast<uint32_t>(network::Op::Run), pending_sequence_, parts, 3, err)) {
        close(); return false;
    }
    if (trace_) pending_send_end_ns_ = monotonic_ns();
    jobs_ += count; input_bytes_ += activation_bytes;
    return true;
}

bool NetworkExperts::finish(float* results, size_t values, std::string& err) {
    const uint64_t finish_entry_ns = trace_ ? monotonic_ns() : 0;
    if (!active() || !pending_sequence_ || !results || values != pending_values_) {
        err = "network experts: result buffer does not match pending request"; return false;
    }
    network::Header response;
    if (!network::response_header(fd_, network::Op::Run, pending_sequence_, response, err)) { close(); return false; }
    const size_t bytes = values * sizeof(float);
    if (response.bytes != sizeof(network::RunTiming) + bytes) {
        err = "network experts: result length mismatch"; close(); return false;
    }
    network::RunTiming timing;
    if (!network::read_exact(fd_, &timing, sizeof(timing), err) ||
        !network::read_exact(fd_, results, bytes, err)) { close(); return false; }
    if (trace_) {
        const uint64_t reply_read_ns = monotonic_ns();
        const uint64_t send_ns = pending_send_end_ns_ - pending_send_start_ns_;
        const uint64_t overlap_ns = finish_entry_ns - pending_send_end_ns_;
        const uint64_t wait_ns = reply_read_ns - finish_entry_ns;
        // This includes CUDA, local Metal overlap and delayed reply consumption.
        // Worker fields are independent durations; no cross-host subtraction.
        if (std::fprintf(trace_, "{\"type\":\"request\",\"sequence\":%" PRIu64
                         ",\"window_id\":%" PRIu64 ",\"phase\":\"%s\",\"pos0\":%" PRId64
                         ",\"window_tokens\":%u,\"layer\":%u,\"jobs\":%u"
                         ",\"request_bytes\":%" PRIu64 ",\"response_bytes\":%" PRIu64
                         ",\"send_ns\":%" PRIu64 ",\"overlap_ns\":%" PRIu64 ",\"wait_ns\":%" PRIu64
                         ",\"elapsed_ns\":%" PRIu64 ",\"worker_receive_ns\":%" PRIu64
                         ",\"worker_prepare_ns\":%" PRIu64 ",\"worker_execute_ns\":%" PRIu64
                         ",\"gpu_h2d_stage_ns\":%" PRIu64 ",\"gpu_kernel_ns\":%" PRIu64
                         ",\"gpu_d2h_ns\":%" PRIu64 ",\"cache_hits\":%" PRIu64
                         ",\"cache_misses\":%" PRIu64 ",\"upload_bytes\":%" PRIu64 ",\"success\":true}\n",
                         pending_sequence_, window_id_, window_phase_, window_pos0_, window_tokens_,
                         pending_layer_, pending_jobs_, pending_request_bytes_,
                         static_cast<uint64_t>(sizeof(network::Header) + response.bytes),
                         send_ns, overlap_ns, wait_ns, send_ns + overlap_ns + wait_ns,
                         timing.receive_ns, timing.prepare_ns, timing.execute_ns,
                         timing.h2d_stage_ns, timing.kernel_ns, timing.d2h_ns,
                         timing.cache_hits, timing.cache_misses, timing.upload_bytes) < 0) trace_failed_ = true;
        if (!trace_status(err)) { close(); return false; }
    }
    output_bytes_ += bytes; ++completed_;
    pending_sequence_ = 0; pending_values_ = 0;
    return true;
}

bool NetworkExperts::stats(network::Stats& out, std::string& err) {
    if (!active() || pending_sequence_) { err = "network experts: stats requested during work or after disconnect"; return false; }
    network::Header response;
    if (!network::send_frame(fd_, static_cast<uint32_t>(network::Op::Stats), ++sequence_, nullptr, 0, err) ||
        !network::response_header(fd_, network::Op::Stats, sequence_, response, err)) { close(); return false; }
    if (response.bytes != sizeof(network::Stats)) { err = "network experts: invalid stats length"; close(); return false; }
    if (!network::read_exact(fd_, &out, sizeof(out), err)) { close(); return false; }
    stats_ = out;
    return true;
}

void NetworkExperts::report(FILE* out) {
    std::string err;
    if (!trace_status(err, true)) std::fprintf(out, "%s\n", err.c_str());
    network::Stats current;
    if (active() && !stats(current, err)) std::fprintf(out, "network experts: stats unavailable: %s\n", err.c_str());
    std::fprintf(out, "network experts: %llu completed requests, %llu expert rows; %.3f MiB activations, %.3f MiB results; no runtime weight transfers\n",
                 static_cast<unsigned long long>(completed_), static_cast<unsigned long long>(jobs_),
                 input_bytes_ / 1048576.0, output_bytes_ / 1048576.0);
    std::fprintf(out, "network worker: %.3f GiB host weights, %.3f MiB VRAM cache, %.3f GiB RSS; %llu hits, %llu misses, %.3f MiB PC-RAM uploads\n",
                 stats_.host_bytes / 1073741824.0, stats_.vram_bytes / 1048576.0, stats_.resident_bytes / 1073741824.0,
                 static_cast<unsigned long long>(stats_.cache_hits), static_cast<unsigned long long>(stats_.cache_misses),
                 stats_.upload_bytes / 1048576.0);
}

} // namespace strata::core
