#pragma once

#include "strata/core/network_expert_protocol.hpp"
#include <cstdio>

namespace strata::core {

class NetworkExperts {
public:
    NetworkExperts() = default;
    ~NetworkExperts();
    NetworkExperts(const NetworkExperts&) = delete;
    NetworkExperts& operator=(const NetworkExperts&) = delete;

    // expected contains the local source SHA-256 and each layer's raw GGUF layout.
    // owned/weight_bytes are learned from the worker; they are not guessed locally.
    bool open(const std::string& endpoint, const network::Manifest& expected, std::string& err);
    void close();
    bool active() const { return fd_ >= 0; }
    bool owns(uint32_t layer, uint32_t expert) const {
        return active() && layer < manifest_.layers && expert < manifest_.owned;
    }
    const network::Manifest& manifest() const { return manifest_; }
    const network::Stats& last_stats() const { return stats_; }

    // Optional trace context. Plain begin/finish users remain valid and use
    // window_id=0, phase=unknown until they supply a window context.
    bool begin_window(const char* phase, int64_t pos0, uint32_t tokens, std::string& err);
    bool end_window(bool success, std::string& err);

    // One in-flight request. Inputs are normalized FP32 rows; jobs carry original
    // plan row indices, but results are UNWEIGHTED and ordered exactly like jobs.
    // begin sends synchronously; CUDA work overlaps local Metal until finish.
    bool begin(uint32_t layer, uint32_t tokens, const float* x,
               const network::Job* jobs, size_t count, std::string& err);
    bool finish(float* results, size_t values, std::string& err);
    bool stats(network::Stats& out, std::string& err);
    void report(FILE* out);

private:
    int fd_ = -1;
    uint64_t sequence_ = 0, pending_sequence_ = 0;
    size_t pending_values_ = 0;
    uint64_t completed_ = 0, jobs_ = 0, input_bytes_ = 0, output_bytes_ = 0;
    FILE* trace_ = nullptr;
    std::array<char, 65536> trace_buffer_;
    bool trace_failed_ = false, window_active_ = false;
    const char* window_phase_ = "unknown";
    uint64_t window_id_ = 0, window_start_ns_ = 0;
    int64_t window_pos0_ = 0;
    uint32_t window_tokens_ = 0, pending_layer_ = 0, pending_jobs_ = 0;
    uint64_t pending_request_bytes_ = 0, pending_send_start_ns_ = 0, pending_send_end_ns_ = 0;
    bool trace_status(std::string& err, bool flush = false);
    void close_trace();
    network::Manifest manifest_;
    network::Stats stats_;
};

} // namespace strata::core
