#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>

void binary_matmul_apple_m(
    const uint32_t *packed_weights,
    const float *scaling_factors,
    const float *input,
    float *output,
    int batch_size,
    int in_features,
    int out_features)
{
    const int bits_per_int = 32;
    const int ints_per_row = (in_features + bits_per_int - 1) / bits_per_int;

    uint32_t *input_packed = (uint32_t *)malloc(batch_size * ints_per_row * sizeof(uint32_t));

    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < ints_per_row; i++)
        {
            uint32_t packed = 0;
            for (int bit = 0; bit < bits_per_int; bit++)
            {
                int idx = i * bits_per_int + bit;
                if (idx >= in_features)
                    break;

                float val = input[b * in_features + idx];
                if (val >= 0)
                {
                    packed |= (1u << bit);
                }
            }
            input_packed[b * ints_per_row + i] = packed;
        }
    }

#pragma omp parallel for
    for (int o = 0; o < out_features; o++)
    {
        float scale = scaling_factors[o];

        for (int b = 0; b < batch_size; b++)
        {
            int32x4_t acc = vdupq_n_s32(0);

            int i = 0;
            for (; i + 4 <= ints_per_row; i += 4)
            {
                uint32x4_t w_bits = vld1q_u32(&packed_weights[o * ints_per_row + i]);
                uint32x4_t in_bits = vld1q_u32(&input_packed[b * ints_per_row + i]);

                uint32x4_t xor_result = veorq_u32(w_bits, in_bits);
                uint32x4_t xnor_result = vmvnq_u32(xor_result);

#ifdef __APPLE__
                int popcount0 = __builtin_popcount(vgetq_lane_u32(xnor_result, 0));
                int popcount1 = __builtin_popcount(vgetq_lane_u32(xnor_result, 1));
                int popcount2 = __builtin_popcount(vgetq_lane_u32(xnor_result, 2));
                int popcount3 = __builtin_popcount(vgetq_lane_u32(xnor_result, 3));

                acc = vaddq_s32(acc, vsetq_lane_s32(popcount0, acc, 0));
                acc = vaddq_s32(acc, vsetq_lane_s32(popcount1, acc, 1));
                acc = vaddq_s32(acc, vsetq_lane_s32(popcount2, acc, 2));
                acc = vaddq_s32(acc, vsetq_lane_s32(popcount3, acc, 3));
#else
                printf("not working")
#endif
            }

            int total_popcount = vaddvq_s32(acc);

            for (; i < ints_per_row; i++)
            {
                uint32_t w_bits = packed_weights[o * ints_per_row + i];
                uint32_t in_bits = input_packed[b * ints_per_row + i];
                uint32_t xnor = ~(w_bits ^ in_bits);
                total_popcount += __builtin_popcount(xnor);
            }

            float result = (2.0f * total_popcount - in_features) * scale;
            output[b * out_features + o] = result;
        }
    }

    free(input_packed);
}