#include <arm_neon.h>
#include <stdint.h>

void binary_matmul_arm_neon(
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

    for (int o = 0; o < out_features; o++)
    {
        float scale = scaling_factors[o];

        for (int b = 0; b < batch_size; b++)
        {
            float dot_product = 0.0f;

            int i = 0;
            for (; i + 4 <= ints_per_row; i += 4)
            {
                uint32x4_t w_packed = vld1q_u32(&packed_weights[o * ints_per_row + i]);

                uint32x4_t result = vdupq_n_u32(0);

                for (int bit = 0; bit < bits_per_int; bit++)
                {
                    int idx = i * bits_per_int + bit;
                    if (idx >= in_features)
                        break;

                    float in_val = input[b * in_features + idx];
                    uint32_t in_bit = (in_val >= 0) ? 1 : 0;

                    uint32x4_t mask = vdupq_n_u32(1u << bit);
                    uint32x4_t w_bit = vandq_u32(w_packed, mask);

                    uint32x4_t is_zero = vceqq_u32(w_bit, vdupq_n_u32(0));
                    uint32x4_t xnor_result;

                    if (in_bit == 0)
                    {
                        xnor_result = is_zero;
                    }
                    else
                    {
                        xnor_result = vmvnq_u32(is_zero);
                    }

                    result = vaddq_u32(result, xnor_result);
                }

                uint32x2_t sum = vadd_u32(vget_low_u32(result), vget_high_u32(result));
                sum = vpadd_u32(sum, sum);
                dot_product += vget_lane_u32(sum, 0);
            }

            for (; i < ints_per_row; i++)
            {
                uint32_t w_packed_val = packed_weights[o * ints_per_row + i];
                uint32_t xnor_sum = 0;

                for (int bit = 0; bit < bits_per_int; bit++)
                {
                    int idx = i * bits_per_int + bit;
                    if (idx >= in_features)
                        break;

                    float in_val = input[b * in_features + idx];
                    uint32_t in_bit = (in_val >= 0) ? 1 : 0;
                    uint32_t w_bit = (w_packed_val >> bit) & 1;

                    xnor_sum += (in_bit == w_bit) ? 1 : 0;
                }

                dot_product += xnor_sum;
            }

            float popcount_adjusted = 2.0f * dot_product - in_features;
            output[b * out_features + o] = popcount_adjusted * scale;
        }
    }
}