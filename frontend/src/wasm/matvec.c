/**
 * WASM SIMD128 kernels for int8-quantized LSTM inference.
 *
 * Exports:
 *   matvec_fused_i8  — fused gates: out = W_ih@input*s1 + W_hh@h*s2 + bias
 *   matvec_i8        — simple:      out = A@x * scale
 *
 * All pointers are byte offsets into the imported linear memory.
 * Compile:
 *   clang --target=wasm32 -O3 -msimd128 -nostdlib \
 *     -Wl,--no-entry -Wl,--import-memory \
 *     -Wl,--stack-first -Wl,-z,stack-size=4096 \
 *     -o matvec.wasm matvec.c
 */
#include <wasm_simd128.h>
#include <stdint.h>

/* ---- helpers ---- */

static inline float hsum(v128_t v) {
    v128_t t1 = wasm_i32x4_shuffle(v, v, 2, 3, 0, 1);
    v128_t s1 = wasm_f32x4_add(v, t1);
    v128_t t2 = wasm_i32x4_shuffle(s1, s1, 1, 0, 2, 3);
    v128_t s2 = wasm_f32x4_add(s1, t2);
    return wasm_f32x4_extract_lane(s2, 0);
}

/**
 * Dot product of int8 weight row (len) with float32 vector (len).
 * Processes 16 int8 elements (one v128 load) per iteration,
 * extending i8→i16→i32→f32 for the multiply-accumulate.
 */
static float dot_i8_f32(const int8_t *w, const float *x, int len) {
    v128_t a0 = wasm_f32x4_splat(0.0f);
    v128_t a1 = wasm_f32x4_splat(0.0f);
    v128_t a2 = wasm_f32x4_splat(0.0f);
    v128_t a3 = wasm_f32x4_splat(0.0f);

    int j = 0;
    const int len16 = len & ~15;

    for (; j < len16; j += 16) {
        /* 16 × int8 weights */
        v128_t wi8 = wasm_v128_load(w + j);

        /* widen to four groups of 4 × float32 */
        v128_t lo16 = wasm_i16x8_extend_low_i8x16(wi8);
        v128_t hi16 = wasm_i16x8_extend_high_i8x16(wi8);

        v128_t wf0 = wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(lo16));
        v128_t wf1 = wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(lo16));
        v128_t wf2 = wasm_f32x4_convert_i32x4(wasm_i32x4_extend_low_i16x8(hi16));
        v128_t wf3 = wasm_f32x4_convert_i32x4(wasm_i32x4_extend_high_i16x8(hi16));

        /* 4 × float32 input loads */
        v128_t x0 = wasm_v128_load(x + j);
        v128_t x1 = wasm_v128_load(x + j + 4);
        v128_t x2 = wasm_v128_load(x + j + 8);
        v128_t x3 = wasm_v128_load(x + j + 12);

        a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(wf0, x0));
        a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(wf1, x1));
        a2 = wasm_f32x4_add(a2, wasm_f32x4_mul(wf2, x2));
        a3 = wasm_f32x4_add(a3, wasm_f32x4_mul(wf3, x3));
    }

    float result = hsum(wasm_f32x4_add(wasm_f32x4_add(a0, a1),
                                        wasm_f32x4_add(a2, a3)));

    /* scalar remainder (cols not multiple of 16) */
    for (; j < len; j++)
        result += (float)w[j] * x[j];

    return result;
}

/* ---- exported kernels ---- */

/**
 * Fused LSTM gate computation:
 *   out[i] = (W_ih[i,:] · input) * scale_ih
 *          + (W_hh[i,:] · h)     * scale_hh
 *          + bias[i]
 *
 * for i in [0, rows).  rows = 4 * hidden_dim.
 */
__attribute__((export_name("matvec_fused_i8")))
void matvec_fused_i8(
    float       *out,
    const int8_t *w_ih, const float *input, int cols_ih, float scale_ih,
    const int8_t *w_hh, const float *h,     int cols_hh, float scale_hh,
    const float  *bias,
    int rows)
{
    for (int i = 0; i < rows; i++) {
        float s_ih = dot_i8_f32(w_ih + i * cols_ih, input, cols_ih) * scale_ih;
        float s_hh = dot_i8_f32(w_hh + i * cols_hh, h,     cols_hh) * scale_hh;
        out[i] = s_ih + s_hh + bias[i];
    }
}

/**
 * Simple int8 matrix-vector multiply:
 *   out[i] = (A[i,:] · x) * scale
 */
__attribute__((export_name("matvec_i8")))
void matvec_i8(
    float       *out,
    const int8_t *A, const float *x,
    int rows, int cols,
    float scale)
{
    for (int i = 0; i < rows; i++)
        out[i] = dot_i8_f32(A + i * cols, x, cols) * scale;
}
