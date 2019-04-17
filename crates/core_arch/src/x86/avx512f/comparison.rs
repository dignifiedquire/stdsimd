//! Intrinsics for Comparison Operations

use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem,
};

// _mm512_cmp_pd_mask

// extern __mmask8 __cdecl _mm512_cmp_pd_mask(__m512d a, __m512d b, const int imm);

// Compares float64 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector.

// _mm512_cmp_round_pd_mask

// extern __mmask8 __cdecl _mm512_cmp_round_pd_mask(__m512d a, __m512d b, const int imm, const int round);

// Compares float64 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector.

// Note

// Pass __MM_FROUND_NO_EXC to round to suppress all exceptions.

// _mm512_mask_cmp_round_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmp_round_pd_mask(__mmask8 k, __m512d a, __m512d b, const int imm, const int round);

// Compares float64 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// Note

// Pass __MM_FROUND_NO_EXC to round to suppress all exceptions.

// _mm512_mask_cmp_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmp_pd_mask(__mmask8 k, __m512d a, __m512d b, const int imm);

// Compares float64 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpeq_pd_mask

// extern __mmask8 __cdecl _mm512_cmp_pd_mask(__m512d a, __m512d b);

// Compares float64 elements in a and b for equality.

// The result is stored in mask vector.

// _mm512_mask_cmpeq_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmpeq_pd_mask(__mmask8 k, __m512d a, __m512d b);

// Compares float64 elements in a and b for equality.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmple_pd_mask

// extern __mmask8 __cdecl _mm512_cmple_pd_mask(__m512d a, __m512d b);

// Compares float64 elements in a and b for less-than-or-equal.

// The result is stored in mask vector.

// _mm512_mask_cmple_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmple_pd_mask(__mmask8 k, __m512d a, __m512d b);

// Compares float64 elements in a and b for less-than-or-equal.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmplt_pd_mask

// extern __mmask8 __cdecl _mm512_cmplt_pd_mask(__m512d a, __m512d b);

// Compares float64 elements in a and b for less-than.

// The result is stored in mask vector.

// _mm512_mask_cmplt_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmplt_pd_mask(__mmask8 k, __m512d a, __m512d b);

// Compares float64 elements in a and b for less-than.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpneq_pd_mask

// extern __mmask8 __cdecl _mm512_cmpneq_pd_mask(__m512d a, __m512d b);

// Compares float64 elements in a and b for not-equal.

// The result is stored in mask vector.

// _mm512_mask_cmpneq_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmpneq_pd_mask(__mmask8 k, __m512d a, __m512d b);

// Compares float64 elements in a and b for not-equal.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpnle_pd_mask

// extern __mmask8 __cdecl _mm512_cmpnle_pd_mask(__m512d a, __m512d b);

// Compares float64 elements in a and b for not-less-than-or-equal.

// The result is stored in mask vector.

// _mm512_mask_cmpnle_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmpnle_pd_mask(__mmask8 k, __m512d a, __m512d b);

// Compares float64 elements in a and b for not-less-than-or-equal.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpnlt_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmpnlt_pd_mask(__m512d a, __m512d b);

// Compares float64 elements in a and b for not-less-than.

// The result is stored in mask vector.

// _mm512_mask_cmpnlt_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmpnlt_pd_mask(__mmask8 k, __m512d a, __m512d b);

// Compares float64 elements in a and b for not-less-than.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpord_pd_mask

// extern __mmask8 __cdecl _mm512_cmpord_pd_mask(__m512d a, __m512d b);

// Compares float64 elements in a and b to see if neither is NaN.

// The result is stored in mask vector.

// _mm512_mask_cmpord_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmpord_pd_mask(__mmask8 k, __m512d a, __m512d b);

// Compares float64 elements in a and b to see if neither is NaN.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpunord_pd_mask

// extern __mmask8 __cdecl _mm512_cmpunord_pd_mask(__m512d a, __m512d b);

// Compares float64 elements in a and b to see if either is NaN.

// The result is stored in mask vector.

// _mm512_mask_cmpord_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmpord_pd_mask(__mmask8 k, __m512d a, __m512d b);

// Compares float64 elements in a and b to see if neither is NaN.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_mask_cmpunord_pd_mask

// extern __mmask8 __cdecl _mm512_mask_cmpord_pd_mask(__mmask8 k, __m512d a, __m512d b);

// Compares float64 elements in a and b to see if either is NaN.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmp_ps_mask

// extern __mmask16 __cdecl _mm512_cmp_ps_mask(__m512 a, __m512 b, const int imm);

// Compares float32 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector.

// _mm512_mask_cmp_ps_mask

// extern __mmask16 __cdecl _mm512_mask_cmp_ps_mask(__mmask16 k, __m512 a, __m512 b, const int imm);

// Compares float32 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmp_round_ps_mask

// extern __mmask16 __cdecl _mm512_cmp_round_ps_mask(__m512 a, __m512 b, const int imm, const int round);

// Compares float32 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector.

// Note

// Pass __MM_FROUND_NO_EXC to round to suppress all exceptions.

// _mm512_mask_cmp_round_ps_mask

// extern __mmask16 __cdecl _mm512_mask_cmp_round_ps_mask(__mmask16 k, __m512 a, __m512 b, const int imm, const int round);

// Compares float32 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// Note

// Pass __MM_FROUND_NO_EXC to round to suppress all exceptions.

// _mm512_cmpeq_ps_mask

// extern __mmask16 __cdecl _mm512_cmpeq_ps_mask(__m512 a, __m512 b);

// Compares float32 elements in a and b for equality.

// The result is stored in mask vector.

// _mm512_mask_cmpeq_ps_mask

// extern __mmask16 __cdecl _mm512_mask_cmpeq_ps_mask(__mmask16 k, __m512 a, __m512 b);

// Compares float32 elements in a and b for equality.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmple_ps_mask

// extern __mmask16 __cdecl _mm512_cmple_ps_mask(__m512 a, __m512 b);

// Compares float32 elements in a and b for less-than-or-equal.

// The result is stored in mask vector.

// _mm512_mask_cmple_ps_mask

// extern __mmask16 __cdecl _mm512_mask_cmple_ps_mask(__mmask16 k, __m512 a, __m512 b);

// Compares float32 elements in a and b for less-than-or-equal.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpunord_ps_mask

// extern __mmask16 __cdecl _mm512_cmpunord_ps_mask(__m512 a, __m512 b);

// Compares float32 elements in a and b to see if either is NaN.

// The result is stored in mask vector.

// _mm512_mask_cmpunord_ps_mask

// extern __mmask16 __cdecl _mm512_mask_cmpunord_ps_mask(__mmask16 k, __m512 a, __m512 b);

// Compares float32 elements in a and b to see if neither is NaN.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmplt_ps_mask

// extern __mmask16 __cdecl _mm512_cmplt_ps_mask(__m512 a, __m512 b);

// Compares float32 elements in a and b for less-than.

// The result is stored in mask vector.

// _mm512_mask_cmplt_ps_mask

// extern __mmask16 __cdecl _mm512_mask_cmplt_ps_mask(__mmask16 k, __m512 a, __m512 b);

// Compares float32 elements in a and b for less-than.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpneq_ps_mask

// extern __mmask16 __cdecl _mm512_cmpneq_ps_mask(__m512 a, __m512 b);

// Compares float32 elements in a and b for not-equal.

// The result is stored in mask vector.

// _mm512_mask_cmpneq_ps_mask

// extern __mmask16 __cdecl _mm512_mask_cmpneq_ps_mask(__mmask16 k, __m512 a, __m512 b, const int round);

// Compares float32 elements in a and b for not-equal.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpnle_ps_mask

// extern __mmask16 __cdecl _mm512_cmpnle_ps_mask(__m512 a, __m512 b);

// Compares float32 elements in a and b for not-less-than-or-equal.

// The result is stored in mask vector.

// _mm512_mask_cmpnle_ps_mask

// extern __mmask16 __cdecl _mm512_mask_cmpnle_ps_mask(__mmask16 k, __m512 a, __m512 b);

// Compares float32 elements in a and b for not-less-than-or-equal.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpnlt_ps_mask

// extern __mmask16 __cdecl _mm512_cmpnlt_ps_mask(__m512 a, __m512 b);

// Compares float32 elements in a and b for not-less-than.

// The result is stored in mask vector.

// _mm512_mask_cmpnlt_ps_mask

// extern __mmask16 __cdecl _mm512_mask_cmpnlt_ps_mask(__mmask16 k, __m512 a, __m512 b);

// Compares float32 elements in a and b for not-less-than.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm512_cmpord_ps_mask

// extern __mmask16 __cdecl _mm512_cmpord_ps_mask(__m512 a, __m512 b);

// Compares float32 elements in a and b to see if either is NaN.

// The result is stored in mask vector.

// _mm512_mask_cmpord_ps_mask

// extern __mmask16 __cdecl _mm512_mask_cmpord_ps_mask(__mmask16 k, __m512 a, __m512 b);

// Compares float32 elements in a and b to see if either is NaN.

// The result is stored in mask vector using zeromask k (elements are zeroed out when the corresponding mask bit is not set).

// _mm_cmp_round_sd_mask

// extern __mmask8 __cdecl _mm_cmp_round_sd_mask(__m128d a, __m128d b, const int imm, const round);

// Compares lower float64 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector.

// Note

// Pass __MM_FROUND_NO_EXC to round to suppress all exceptions.

// _mm_mask_cmp_round_sd_mask

// extern __mmask8 __cdecl _mm_mask_cmp_round_sd_mask(__mmask8 k, __m128d a, __m128d b, const int imm, const int round);

// Compares lower float64 elements in a and b based on the comparison operand specified by imm, and store the result in mask vector k using zeromask k (the element is zeroed out when mask bit 0 is not set).

// Note

// Pass __MM_FROUND_NO_EXC to round to suppress all exceptions.

// _mm_cmp_sd_mask

// extern __mmask8 __cdecl _mm_cmp_sd_mask(__m128d a, __m128d b, const int imm);

// Compares lower float64 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector.

// _mm_mask_cmp_sd_mask

// extern __mmask8 __cdecl _mm_mask_cmp_sd_mask(__mmask8 k, __m128d a, __m128d b, const int imm);

// Compares lower float64 elements in a and b based on the comparison operand specified by imm

// The result is stored in mask vector using zeromask k (the element is zeroed out when mask bit 0 is not set).

// _mm_cmp_round_ss_mask

// extern __mmask8 __cdecl _mm_cmp_round_ss_mask(__m128 a, __m128 b, const int imm, const int round);

// Compares lower float32 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector.

// Note

// Pass __MM_FROUND_NO_EXC to round to suppress all exceptions.

// _mm_mask_cmp_round_ss_mask

// extern __mmask8 __cdecl _mm_mask_cmp_round_ss_mask(__mmask8 k, __m128 a, __m128 b, const int imm, const int round);

// Compares lower float32 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector using zeromask k (the element is zeroed out when mask bit 0 is not set).

// Note

// Pass __MM_FROUND_NO_EXC to round to suppress all exceptions.

// _mm_cmp_ss_mask

// extern __mmask8 __cdecl _mm_cmp_ss_mask(__m128 a, __m128 b, const int imm);

// Compares lower float32 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector.

// _mm_mask_cmp_ss_mask

// extern __mmask8 __cdecl _mm_mask_cmp_ss_mask(__mmask8 k, __m128 a, __m128 b, const int imm);

// Compares lower float32 elements in a and b based on the comparison operand specified by imm.

// The result is stored in mask vector using zeromask k (the element is zeroed out when mask bit 0 is not set).
