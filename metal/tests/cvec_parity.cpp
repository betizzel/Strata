// Metal control-vector projection: compare every residual stream against a CPU reference.
#include "mtl_rt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct CvecArgs { uint32_t n_embd, hc, layer, mode; };

int main() {
    constexpr int N = 2560, HC = 4, LAYERS = 48, LAYER = 20, MAXT = 11;
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "units.metal", "gr.metal"});
    mtl::Pipeline kernel = cx.pipeline(lib, "gr_cvec");
    mtl::Buffer r = cx.alloc(MAXT * HC * N * sizeof(float));
    mtl::Buffer bo = cx.alloc(MAXT * N * sizeof(float));
    mtl::Buffer inj = cx.alloc(MAXT * HC * sizeof(float));
    mtl::Buffer dir = cx.alloc(LAYERS * N * sizeof(float));
    mtl::Buffer scale = cx.alloc(LAYERS * sizeof(float));
    mtl::Buffer enabled = cx.alloc(sizeof(int));
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> rnd(-1.0f, 1.0f);
    std::vector<float> initial(MAXT * HC * N);
    for (float& x : initial) x = rnd(rng);
    for (int i = 0; i < MAXT * N; ++i) bo.as<float>()[i] = rnd(rng);
    for (int i = 0; i < MAXT * HC; ++i) inj.as<float>()[i] = rnd(rng) * 3;
    double norm = 0;
    for (int i = 0; i < N; ++i) {
        float x = rnd(rng);
        dir.as<float>()[LAYER * N + i] = x;
        norm += (double)x * x;
    }
    for (int i = 0; i < N; ++i) dir.as<float>()[LAYER * N + i] /= (float)std::sqrt(norm);
    scale.as<float>()[LAYER] = 0.7f;
    int failures = 0;
    for (int T : {1, 11}) for (int on : {0, 1}) for (int mode : {0, 1}) {
        std::memcpy(r.ptr, initial.data(), initial.size() * sizeof(float));
        *enabled.as<int>() = on;
        CvecArgs args{N, HC, LAYER, (uint32_t)mode};
        mtl::Dispatch d;
        d.p = &kernel;
        d.args = {mtl::Arg::B(r), mtl::Arg::B(bo), mtl::Arg::B(inj), mtl::Arg::B(dir),
                  mtl::Arg::B(scale), mtl::Arg::B(enabled), mtl::Arg::V(args)};
        d.grid[0] = HC; d.grid[1] = T; d.tg[0] = 256;
        cx.run({d});
        double max_error = 0;
        for (int t = 0; t < T; ++t) for (int c = 0; c < HC; ++c) {
            double gate = 2.0 / (1.0 + std::exp(-(double)inj.as<float>()[t * HC + c] / HC));
            std::vector<double> expected(N);
            double dot = 0;
            for (int j = 0; j < N; ++j) {
                expected[j] = initial[(t * HC + c) * N + j] + bo.as<float>()[t * N + j] * gate;
                dot += expected[j] * dir.as<float>()[LAYER * N + j];
            }
            for (int j = 0; j < N; ++j) {
                const double v = dir.as<float>()[LAYER * N + j];
                if (on) expected[j] += mode == 0 ? -scale.as<float>()[LAYER] * dot * v : v;
                max_error = std::max(max_error, std::abs(r.as<float>()[(t * HC + c) * N + j] - expected[j]));
            }
        }
        const bool ok = max_error < 2e-6;
        failures += !ok;
        std::printf("tokens=%d enabled=%d mode=%s max_error=%.3g %s\n", T, on, mode ? "add" : "project", max_error, ok ? "PASS" : "FAIL");
    }
    return failures ? 1 : 0;
}
