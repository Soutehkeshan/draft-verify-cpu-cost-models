/*
 * FP32 NEON FMA throughput check for the Jetson Nano (Cortex-A57).
 *
 * Verifies the FLOP/cycle assumption behind the analytic model's initial
 * peak_flops guess; it does not change clocks or power settings.
 *
 *   gcc -O2 -march=armv8-a -o /tmp/dvcm_fma_peak nano_fma_peak.c -lpthread
 *   /tmp/dvcm_fma_peak <threads> <iterations>
 *
 * Each iteration issues CHAINS independent 128-bit FMLA (4 lanes x 2 FLOP),
 * enough independent accumulators to cover the FMLA latency.
 */
#include <arm_neon.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CHAINS 16

static long iterations;
static volatile float sink[64];

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Sixteen named accumulators stay in V registers (an indexed array plus a memory
 * barrier made GCC 7 spill them, measuring store-to-load latency instead). */
#define FMA(x) x = vfmaq_f32(x, b, c)

static void *worker(void *arg) {
    float base = (float)(long)arg;
    const float32x4_t b = vdupq_n_f32(0.999999f);
    const float32x4_t c = vdupq_n_f32(1e-7f);
    float32x4_t a0 = vdupq_n_f32(base + 0), a1 = vdupq_n_f32(base + 1);
    float32x4_t a2 = vdupq_n_f32(base + 2), a3 = vdupq_n_f32(base + 3);
    float32x4_t a4 = vdupq_n_f32(base + 4), a5 = vdupq_n_f32(base + 5);
    float32x4_t a6 = vdupq_n_f32(base + 6), a7 = vdupq_n_f32(base + 7);
    float32x4_t a8 = vdupq_n_f32(base + 8), a9 = vdupq_n_f32(base + 9);
    float32x4_t a10 = vdupq_n_f32(base + 10), a11 = vdupq_n_f32(base + 11);
    float32x4_t a12 = vdupq_n_f32(base + 12), a13 = vdupq_n_f32(base + 13);
    float32x4_t a14 = vdupq_n_f32(base + 14), a15 = vdupq_n_f32(base + 15);
    for (long n = 0; n < iterations; n++) {
        FMA(a0); FMA(a1); FMA(a2); FMA(a3); FMA(a4); FMA(a5); FMA(a6); FMA(a7);
        FMA(a8); FMA(a9); FMA(a10); FMA(a11); FMA(a12); FMA(a13); FMA(a14); FMA(a15);
    }
    float32x4_t total = vaddq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)),
                                  vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7)));
    total = vaddq_f32(total, vaddq_f32(vaddq_f32(vaddq_f32(a8, a9), vaddq_f32(a10, a11)),
                                       vaddq_f32(vaddq_f32(a12, a13), vaddq_f32(a14, a15))));
    sink[(long)arg % 64] = vgetq_lane_f32(total, 0);
    return NULL;
}

int main(int argc, char **argv) {
    int threads = argc > 1 ? atoi(argv[1]) : 1;
    iterations = argc > 2 ? atol(argv[2]) : 100000000L;
    pthread_t pool[64];
    if (threads < 1 || threads > 64 || iterations < 1) return 2;
    double start = now_s();
    for (long t = 0; t < threads; t++) pthread_create(&pool[t], NULL, worker, (void *)t);
    for (int t = 0; t < threads; t++) pthread_join(pool[t], NULL);
    double seconds = now_s() - start;
    double flop = (double)threads * (double)iterations * CHAINS * 4.0 * 2.0;
    printf("{\"threads\": %d, \"iterations\": %ld, \"chains\": %d, \"seconds\": %.6f, "
           "\"flop\": %.0f, \"gflop_s\": %.3f}\n",
           threads, iterations, CHAINS, seconds, flop, flop / seconds / 1e9);
    return 0;
}
