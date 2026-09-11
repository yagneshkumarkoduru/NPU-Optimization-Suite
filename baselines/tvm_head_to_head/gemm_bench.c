/*
 * Deterministic native INT8 x INT8 -> INT32 GEMM kernels for the TVM
 * head-to-head benchmark. The Python driver owns input generation and checks
 * this program's raw INT32 output against its shared NumPy reference.
 *
 * Build: x86_64-w64-mingw32-gcc -O3 -std=c11 -Wall -Wextra -Werror \
 *        -o gemm_bench_x64.exe gemm_bench.c
 * Usage: gemm_bench_x64.exe <naive|tiled> <M> <N> <K> <A.bin> <B.bin> \
 *        <out.bin> <warmups> <repetitions>
 */

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define TI 64
#define TJ 128
#define TK 128

#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

typedef void (*matmul_kernel)(const int8_t *, const int8_t *, int32_t *, int, int, int);

static volatile int64_t result_guard = 0;

static int min_int(int left, int right) {
    return left < right ? left : right;
}

static int parse_nonnegative_int(const char *text, const char *name, int *out) {
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        return 0;
    }
    *out = (int)parsed;
    return 1;
}

static int checked_product(size_t left, size_t right, size_t *out) {
    if (left != 0 && right > SIZE_MAX / left) {
        return 0;
    }
    *out = left * right;
    return 1;
}

static unsigned char *read_file(const char *path, size_t expected_size) {
    FILE *file = fopen(path, "rb");
    unsigned char *buffer;
    size_t actual_size;
    long file_end;
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (file_end = ftell(file)) < 0) {
        fprintf(stderr, "cannot size %s\n", path);
        fclose(file);
        return NULL;
    }
    actual_size = (size_t)file_end;
    if (actual_size != expected_size) {
        fprintf(stderr, "unexpected matrix file size for %s\n", path);
        fclose(file);
        return NULL;
    }
    if (fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "cannot seek %s\n", path);
        fclose(file);
        return NULL;
    }
    buffer = (unsigned char *)malloc(expected_size);
    if (buffer == NULL) {
        fprintf(stderr, "out of memory reading %s\n", path);
        fclose(file);
        return NULL;
    }
    if (fread(buffer, 1, expected_size, file) != expected_size) {
        fprintf(stderr, "read error for %s\n", path);
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    return buffer;
}

static int write_file(const char *path, const int32_t *data, size_t bytes) {
    FILE *file = fopen(path, "wb");
    size_t written;
    int close_status;
    if (file == NULL) {
        fprintf(stderr, "cannot open %s\n", path);
        return 0;
    }
    written = fwrite(data, 1, bytes, file);
    close_status = fclose(file);
    if (written != bytes || close_status != 0) {
        fprintf(stderr, "write error for %s\n", path);
        return 0;
    }
    return 1;
}

static NOINLINE void matmul_naive(
    const int8_t *a, const int8_t *b, int32_t *c, int m, int n, int k
) {
    int i;
    int j;
    int kk;
    for (i = 0; i < m; ++i) {
        for (j = 0; j < n; ++j) {
            int32_t accumulator = 0;
            for (kk = 0; kk < k; ++kk) {
                accumulator += (int32_t)a[(size_t)i * (size_t)k + (size_t)kk]
                    * (int32_t)b[(size_t)kk * (size_t)n + (size_t)j];
            }
            c[(size_t)i * (size_t)n + (size_t)j] = accumulator;
        }
    }
}

static NOINLINE void matmul_tiled(
    const int8_t *a, const int8_t *b, int32_t *c, int m, int n, int k
) {
    static int32_t c_tile[TI * TJ];
    int i0;
    int j0;
    int k0;
    for (i0 = 0; i0 < m; i0 += TI) {
        const int i_count = min_int(TI, m - i0);
        for (j0 = 0; j0 < n; j0 += TJ) {
            const int j_count = min_int(TJ, n - j0);
            int i;
            memset(c_tile, 0, sizeof(c_tile));
            for (k0 = 0; k0 < k; k0 += TK) {
                const int k_count = min_int(TK, k - k0);
                for (i = 0; i < i_count; ++i) {
                    const int8_t *a_row = a + (size_t)(i0 + i) * (size_t)k + (size_t)k0;
                    int32_t *c_row = c_tile + (size_t)i * (size_t)TJ;
                    int kk;
                    for (kk = 0; kk < k_count; ++kk) {
                        const int32_t a_value = (int32_t)a_row[kk];
                        const int8_t *b_row = b + (size_t)(k0 + kk) * (size_t)n + (size_t)j0;
                        int j;
                        for (j = 0; j < j_count; ++j) {
                            c_row[j] += a_value * (int32_t)b_row[j];
                        }
                    }
                }
            }
            for (i = 0; i < i_count; ++i) {
                memcpy(
                    c + (size_t)(i0 + i) * (size_t)n + (size_t)j0,
                    c_tile + (size_t)i * (size_t)TJ,
                    (size_t)j_count * sizeof(int32_t)
                );
            }
        }
    }
}

static double now_seconds(void) {
    static LARGE_INTEGER frequency;
    LARGE_INTEGER counter;
    if (frequency.QuadPart == 0) {
        if (!QueryPerformanceFrequency(&frequency)) {
            return -1.0;
        }
    }
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
}

static void consume_result(const int32_t *c, size_t elements) {
    result_guard ^= (int64_t)c[0];
    result_guard ^= (int64_t)c[elements / 2];
    result_guard ^= (int64_t)c[elements - 1];
}

static int compare_double(const void *left, const void *right) {
    const double left_value = *(const double *)left;
    const double right_value = *(const double *)right;
    return (left_value > right_value) - (left_value < right_value);
}

static double median_seconds(const double *samples, int count) {
    double *sorted = (double *)malloc((size_t)count * sizeof(double));
    double value;
    if (sorted == NULL) {
        return -1.0;
    }
    memcpy(sorted, samples, (size_t)count * sizeof(double));
    qsort(sorted, (size_t)count, sizeof(double), compare_double);
    if ((count % 2) == 1) {
        value = sorted[count / 2];
    } else {
        value = (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0;
    }
    free(sorted);
    return value;
}

int main(int argc, char **argv) {
    const char *mode;
    const char *a_path;
    const char *b_path;
    const char *out_path;
    matmul_kernel kernel;
    int m;
    int n;
    int k;
    int warmups;
    int repetitions;
    size_t a_elements;
    size_t b_elements;
    size_t c_elements;
    size_t c_bytes;
    unsigned char *a_raw;
    unsigned char *b_raw;
    int32_t *c;
    double *samples;
    int iteration;
    double measured_median;

    if (argc != 10) {
        fprintf(
            stderr,
            "usage: %s <naive|tiled> <M> <N> <K> <A.bin> <B.bin> <out.bin> <warmups> <repetitions>\n",
            argv[0]
        );
        return 1;
    }

    mode = argv[1];
    if (strcmp(mode, "naive") == 0) {
        kernel = matmul_naive;
    } else if (strcmp(mode, "tiled") == 0) {
        kernel = matmul_tiled;
    } else {
        fprintf(stderr, "unsupported mode: %s\n", mode);
        return 1;
    }
    if (!parse_nonnegative_int(argv[2], "M", &m) || m == 0
        || !parse_nonnegative_int(argv[3], "N", &n) || n == 0
        || !parse_nonnegative_int(argv[4], "K", &k) || k == 0
        || !parse_nonnegative_int(argv[8], "warmups", &warmups)
        || !parse_nonnegative_int(argv[9], "repetitions", &repetitions) || repetitions == 0) {
        return 1;
    }
    a_path = argv[5];
    b_path = argv[6];
    out_path = argv[7];

    if (!checked_product((size_t)m, (size_t)k, &a_elements)
        || !checked_product((size_t)k, (size_t)n, &b_elements)
        || !checked_product((size_t)m, (size_t)n, &c_elements)
        || !checked_product(c_elements, sizeof(int32_t), &c_bytes)) {
        fprintf(stderr, "matrix dimensions overflow size_t\n");
        return 1;
    }

    a_raw = read_file(a_path, a_elements);
    b_raw = read_file(b_path, b_elements);
    if (a_raw == NULL || b_raw == NULL) {
        free(a_raw);
        free(b_raw);
        return 1;
    }
    c = (int32_t *)malloc(c_bytes);
    samples = (double *)malloc((size_t)repetitions * sizeof(double));
    if (c == NULL || samples == NULL) {
        fprintf(stderr, "out of memory\n");
        free(a_raw);
        free(b_raw);
        free(c);
        free(samples);
        return 1;
    }

    for (iteration = 0; iteration < warmups; ++iteration) {
        kernel((const int8_t *)a_raw, (const int8_t *)b_raw, c, m, n, k);
        consume_result(c, c_elements);
    }
    for (iteration = 0; iteration < repetitions; ++iteration) {
        double started = now_seconds();
        if (started < 0.0) {
            fprintf(stderr, "QueryPerformanceCounter initialization failed\n");
            free(a_raw);
            free(b_raw);
            free(c);
            free(samples);
            return 1;
        }
        kernel((const int8_t *)a_raw, (const int8_t *)b_raw, c, m, n, k);
        samples[iteration] = now_seconds() - started;
        if (samples[iteration] <= 0.0) {
            fprintf(stderr, "QueryPerformanceCounter measurement failed\n");
            free(a_raw);
            free(b_raw);
            free(c);
            free(samples);
            return 1;
        }
        consume_result(c, c_elements);
    }

    if (!write_file(out_path, c, c_bytes)) {
        free(a_raw);
        free(b_raw);
        free(c);
        free(samples);
        return 1;
    }
    measured_median = median_seconds(samples, repetitions);
    if (measured_median <= 0.0) {
        fprintf(stderr, "out of memory computing timing median\n");
        free(a_raw);
        free(b_raw);
        free(c);
        free(samples);
        return 1;
    }
    printf(
        "{\"mode\":\"%s\",\"warmups\":%d,\"repetitions\":%d,\"samples_seconds\":[",
        mode,
        warmups,
        repetitions
    );
    for (iteration = 0; iteration < repetitions; ++iteration) {
        printf("%s%.9f", iteration == 0 ? "" : ",", samples[iteration]);
    }
    printf(
        "],\"median_seconds\":%.9f,\"result_guard\":%" PRId64 "}\n",
        measured_median,
        result_guard
    );

    free(a_raw);
    free(b_raw);
    free(c);
    free(samples);
    return 0;
}
