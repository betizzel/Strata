// metal/src/engine/missing.cpp - what a kernel entry point that has no Metal implementation yet does when called.
//
// Default: say which one and stop (exit code 86), so a run never continues on outputs nobody wrote.
// STRATA_METAL_STUBS=log: say it once per entry point and return (outputs untouched) - for listing which entry points a
// code path reaches, never for answers.
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

namespace strata::metal_engine {

void missing(const char* name) {
    static const bool log_only = [] {
        const char* v = std::getenv("STRATA_METAL_STUBS");
        return v != nullptr && std::string(v) == "log";
    }();
    if (!log_only) {
        std::fprintf(stderr, "strata metal: %s has no Metal implementation yet\n", name);
        std::fflush(stderr);
        std::_Exit(86);
    }
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lk(mu);
    if (seen.insert(name).second) std::fprintf(stderr, "strata metal: STUB %s\n", name);
}

}  // namespace strata::metal_engine

#include "strata/kernels/cvec.hpp"
namespace strata::kernels {
namespace {
Cvec g_cvec;
std::vector<float> g_cvec_dir, g_cvec_s;
int g_cvec_on = 1;
int* g_cvec_metal_flag = nullptr;
}

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > 4096) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    g_cvec_dir = dir;
    g_cvec_s = s;
    g_cvec_on = 1;
    if (g_cvec_metal_flag) *g_cvec_metal_flag = g_cvec_on;
    g_cvec.dir = g_cvec_dir.data();
    g_cvec.s = g_cvec_s.data();
    g_cvec.on = &g_cvec_on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string&) { return true; }

void cvec_set_enabled(bool on) {
    if (g_cvec.loaded()) {
        g_cvec_on = on ? 1 : 0;
        if (g_cvec_metal_flag) *g_cvec_metal_flag = g_cvec_on;
    }
}

void cvec_metal_bind_flag(int* ptr) {
    g_cvec_metal_flag = ptr;
    if (ptr) *ptr = g_cvec_on;
}

bool cvec_enabled() { return g_cvec.loaded() && g_cvec_on != 0; }

bool cvec_tables(const float** dir, const float** s, const int** on) {
    if (!g_cvec.loaded()) return false;
    *dir = g_cvec.dir; *s = g_cvec.s; *on = g_cvec.on;
    return true;
}
}  // namespace strata::kernels

// an uncaught exception or abort on the Metal build prints where it came from (macOS gives no core dump by default and
// lldb needs a debugging permission a user's session may not have)
#include <csignal>
#include <signal.h>
#include <execinfo.h>
#include <exception>
#include <unistd.h>
namespace {
[[noreturn]] void metal_terminate() {
    void* fr[64];
    const int n = backtrace(fr, 64);
    const char msg[] = "strata metal: terminate - the call stack:\n";
    (void) !write(2, msg, sizeof msg - 1);
    backtrace_symbols_fd(fr, n, 2);
    std::abort();
}
const bool g_terminate_set = (std::set_terminate(metal_terminate), true);
// a crash (SIGSEGV / SIGBUS) prints its call stack too: macOS keeps no report for a command-line process by default
void metal_crash(int sig, siginfo_t* si, void*) {
    void* fr[64];
    const int n = backtrace(fr, 64);
    char msg[160];
    const int m = std::snprintf(msg, sizeof msg, "strata metal: crashed (signal %d at address %p) - the call stack:\n", sig,
                                si ? si->si_addr : nullptr);
    (void) !write(2, msg, (size_t) m);
    backtrace_symbols_fd(fr, n, 2);
    signal(sig, SIG_DFL);
    raise(sig);
}
bool set_crash() {
    struct sigaction sa {};
    sa.sa_sigaction = metal_crash;
    sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
    return true;
}
const bool g_crash_set = set_crash();
}  // namespace
