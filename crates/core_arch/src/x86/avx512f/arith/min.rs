//! Intrinsics for Determining Minimum and Maximum Values

use crate::{
    core_arch::{simd_llvm::*, x86::*},
    mem::transmute,
};

#[cfg(test)]
use stdsimd_test::assert_instr;

// -- Intrinsics for Determining Minimum and Maximum FP Values

/// Compares packed float64 elements in a and b, and stores packed maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd))]
pub unsafe fn _mm512_max_pd(a: __m512d, b: __m512d) -> __m512d {
    _mm512_max_round_pd(a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Compares packed float64 elements in a and b, and stores packed maximum values using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd))]
pub unsafe fn _mm512_mask_max_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    _mm512_mask_max_round_pd(src, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Compares packed float64 elements in a and b, and stores packed maximum values using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd))]
pub unsafe fn _mm512_maskz_max_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    _mm512_maskz_max_round_pd(k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Compares packed float64 elements in a and b, and stores packed maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_round_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd))]
pub unsafe fn _mm512_max_round_pd(a: __m512d, b: __m512d, round: i32) -> __m512d {
    let zero = _mm512_setzero_pd();
    _mm512_mask_max_round_pd(zero, 255u8 as i8, a, b, round)
}

/// Compares packed float64 elements in a and b, and stores packed maximum values using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_round_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd))]
pub unsafe fn _mm512_mask_max_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    round: i32,
) -> __m512d {
    macro_rules! call {
        ($imm8:expr) => {
            maxpd512(a, b, src, k, $imm8)
        };
    }
    constify_imm8!(round, call)
}

/// Compares packed float64 elements in a and b, and stores packed maximum values using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_round_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxpd))]
pub unsafe fn _mm512_maskz_max_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    round: i32,
) -> __m512d {
    let zero = _mm512_setzero_pd();
    _mm512_mask_max_round_pd(zero, k, a, b, round)
}

/// Compares packed float32 elements in a and b, and stores packed maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm512_max_ps(a: __m512, b: __m512) -> __m512 {
    _mm512_max_round_ps(a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Compares packed float32 elements in a and b, and stores packed maximum values using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm512_mask_max_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    _mm512_mask_max_round_ps(src, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Compares packed float32 elements in a and b, and stores packed maximum values using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm512_maskz_max_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    _mm512_maskz_max_round_ps(k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Compares packed float32 elements in a and b, and stores packed maximum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_round_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm512_max_round_ps(a: __m512, b: __m512, round: i32) -> __m512 {
    let zero = _mm512_setzero_ps();
    _mm512_mask_max_round_ps(zero, 0xFFFFu16 as __mmask16, a, b, round)
}

/// Compares packed float32 elements in a and b, and stores packed maximum values using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_round_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm512_mask_max_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
    round: i32,
) -> __m512 {
    macro_rules! call {
        ($imm8:expr) => {
            maxps512(a, b, src, k, $imm8)
        };
    }
    constify_imm8!(round, call)
}

/// Compares packed float32 elements in a and b, and stores packed maximum values using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_round_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm512_maskz_max_round_ps(k: __mmask16, a: __m512, b: __m512, round: i32) -> __m512 {
    let zero = _mm512_setzero_ps();
    _mm512_mask_max_round_ps(zero, k, a, b, round)
}

/// Compares the lower float64 elements in a and b, stores the maximum value in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_max_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm_mask_max_sd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    _mm_mask_max_round_sd(src, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Compares the lower float64 elements in a and b, stores the maximum value in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_max_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm_maskz_max_sd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let zero = _mm_setzero_pd();
    maxsdround(a, b, zero, k, _MM_FROUND_CUR_DIRECTION)
}

/// Compares the lower float64 elements in a and b, stores the maximum value in the lower destination element, and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_max_round_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm_max_round_sd(a: __m128d, b: __m128d, round: i32) -> __m128d {
    let zero = _mm_setzero_pd();
    _mm_mask_max_round_sd(zero, -1i8 as __mmask8, a, b, round)
}

/// Compares the lower float64 elements in a and b, stores the maximum value in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_max_round_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm_mask_max_round_sd(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
    round: i32,
) -> __m128d {
    macro_rules! call {
        ($imm8:expr) => {
            maxsdround(a, b, src, k, $imm8)
        };
    }
    constify_imm8!(round, call)
}

/// Compares the lower float64 elements in a and b, stores the maximum value in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_max_round_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxps))]
pub unsafe fn _mm_maskz_max_round_sd(k: __mmask8, a: __m128d, b: __m128d, round: i32) -> __m128d {
    let zero = _mm_setzero_pd();
    _mm_mask_max_round_sd(zero, k, a, b, round)
}

/// Compares the lower float32 elements in a and b, stores the maximum value in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_max_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxss))]
pub unsafe fn _mm_mask_max_ss(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    _mm_mask_max_round_ss(src, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Compares the lower float32 elements in a and b, stores the maximum value in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_max_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxss))]
pub unsafe fn _mm_maskz_max_ss(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let zero = _mm_setzero_ps();
    _mm_mask_max_round_ss(zero, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Compares the lower float32 elements in a and b, stores the maximum value in the lower destination element, and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_max_round_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxss))]
pub unsafe fn _mm_max_round_ss(a: __m128, b: __m128, round: i32) -> __m128 {
    let zero = _mm_setzero_ps();
    _mm_mask_max_round_ss(zero, -1i8 as __mmask8, a, b, round)
}

/// Compares the lower float32 elements in a and b, stores the maximum value in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_max_round_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxss))]
pub unsafe fn _mm_mask_max_round_ss(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
    round: i32,
) -> __m128 {
    macro_rules! call {
        ($imm8:expr) => {
            maxssround(a, b, src, k, $imm8)
        };
    }
    constify_imm8!(round, call)
}

/// Compares the lower float32 elements in a and b, stores the maximum value in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_max_round_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmaxss))]
pub unsafe fn _mm_maskz_max_round_ss(k: __mmask8, a: __m128, b: __m128, round: i32) -> __m128 {
    let zero = _mm_setzero_ps();
    _mm_mask_max_round_ss(zero, k, a, b, round)
}

/// Compares packed float64 elements in a and b, and stores packed minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_min_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd))]
pub unsafe fn _mm512_min_pd(a: __m512d, b: __m512d) -> __m512d {
    unimplemented!()
}

/// Compares packed float64 elements in a and b, and stores packed minimum values using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_min_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd))]
pub unsafe fn _mm512_mask_min_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    unimplemented!()
}

/// Compares packed float64 elements in a and b, and store packed minimum values using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_min_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd))]
pub unsafe fn _mm512_maskz_min_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    unimplemented!()
}

/// Compares packed float64 elements in a and b, and stores packed minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_min_round_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd))]
pub unsafe fn _mm512_min_round_pd(a: __m512d, b: __m512d, round: i32) -> __m512d {
    unimplemented!()
}

/// Compares packed float64 elements in a and b, and stores packed minimum values using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_min_round_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd))]
pub unsafe fn _mm512_mask_min_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    round: i32,
) -> __m512d {
    unimplemented!()
}

/// Compares packed float64 elements in a and b, and stores packed minimum values using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_min_round_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminpd))]
pub unsafe fn _mm512_maskz_min_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    round: i32,
) -> __m512d {
    unimplemented!()
}

/// Compares packed float32 elements in a and b, and stores packed minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_min_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps))]
pub unsafe fn _mm512_min_ps(a: __m512, b: __m512) -> __m512 {
    unimplemented!()
}

/// Compares packed float32 elements in a and b, and stores packed minimum values using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_min_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps))]
pub unsafe fn _mm512_mask_min_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unimplemented!()
}

/// Compares packed float32 elements in a and b, and store packed minimum values using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_min_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps))]
pub unsafe fn _mm512_maskz_min_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unimplemented!()
}

/// Compares packed float32 elements in a and b, and stores packed minimum values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_min_round_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps))]
pub unsafe fn _mm512_min_round_ps(a: __m512, b: __m512, round: i32) -> __m512 {
    unimplemented!()
}

/// Compares packed float32 elements in a and b, and stores packed minimum values using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_min_round_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps))]
pub unsafe fn _mm512_mask_min_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
    round: i32,
) -> __m512 {
    unimplemented!()
}

/// Compares packed float32 elements in a and b, and stores packed minimum values using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_min_round_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminps))]
pub unsafe fn _mm512_maskz_min_round_ps(k: __mmask16, a: __m512, b: __m512, round: i32) -> __m512 {
    unimplemented!()
}

/// Compares the lower float64 elements in a and b, stores the minimum value in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_min_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminsd))]
pub unsafe fn _mm_mask_min_sd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unimplemented!()
}

/// Compares the lower float64 elements in a and b, stores the minimum value in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_min_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminsd))]
pub unsafe fn _mm_maskz_min_sd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unimplemented!()
}

/// Compares the lower float64 elements in a and b, stores the minimum value in the lower destination element, and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_min_round_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminsd))]
pub unsafe fn _mm_min_round_sd(a: __m128d, b: __m128d, round: i32) -> __m128d {
    unimplemented!()
}

/// Compares the lower float64 elements in a and b, stores the minimum value in the lower destination element of using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_min_round_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminsd))]
pub unsafe fn _mm_mask_min_round_sd(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
    round: i32,
) -> __m128d {
    unimplemented!()
}

/// Compares the lower float64 elements in a and b, stores the minimum value in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_min_round_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminsd))]
pub unsafe fn _mm_maskz_min_round_sd(k: __mmask8, a: __m128d, b: __m128d, round: i32) -> __m128d {
    unimplemented!()
}

/// Compares the lower float32 elements in a and b, stores the minimum value in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_min_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminss))]
pub unsafe fn _mm_mask_min_ss(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unimplemented!()
}

/// Compares the lower float32 elements in a and b, stores the minimum value in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_min_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminss))]
pub unsafe fn _mm_maskz_min_ss(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unimplemented!()
}

/// Compares the lower float32 elements in a and b, stores the minimum value in the lower destination element, and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_min_round_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminss))]
pub unsafe fn _mm_min_round_ss(a: __m128, b: __m128, round: i32) -> __m128 {
    unimplemented!()
}

/// Compares the lower float32 elements in a and b, stores the minimum value in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_min_round_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminss))]
pub unsafe fn _mm_mask_min_round_ss(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
    round: i32,
) -> __m128 {
    unimplemented!()
}

/// Compares the lower float32 elements in a and b, stores the minimum value in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_min_round_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vminss))]
pub unsafe fn _mm_maskz_min_round_ss(k: __mmask8, a: __m128, b: __m128, round: i32) -> __m128 {
    unimplemented!()
}

// -- Intrinsics for Determining Minimum and Maximum Integer Values

/// LLVM intrinsics used in the above functions
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.mask.max.pd.512"]
    fn maxpd512(a: __m512d, b: __m512d, src: __m512d, k: __mmask8, round: i32) -> __m512d;
    #[link_name = "llvm.x86.avx512.mask.max.ps.512"]
    fn maxps512(a: __m512, b: __m512, src: __m512, k: __mmask16, round: i32) -> __m512;

    #[link_name = "llvm.x86.avx512.mask.max.sd.round"]
    fn maxsdround(a: __m128d, b: __m128d, src: __m128d, k: __mmask8, round: i32) -> __m128d;
    #[link_name = "llvm.x86.avx512.mask.max.ss.round"]
    fn maxssround(a: __m128, b: __m128, src: __m128, k: __mmask8, round: i32) -> __m128;

    #[link_name = "llvm.x86.avx512.mask.min.pd.512"]
    fn minpd512(a: __m512d, b: __m512d, src: __m512d, k: __mmask8, round: i32) -> __m512d;
    #[link_name = "llvm.x86.avx512.mask.min.ps.512"]
    fn minps512(a: __m512, b: __m512, src: __m512, k: __mmask16, round: i32) -> __m512;

    #[link_name = "llvm.x86.avx512.mask.min.sd.round"]
    fn minsdround(a: __m128d, b: __m128d, src: __m128d, k: __mmask8, round: i32) -> __m128d;
    #[link_name = "llvm.x86.avx512.mask.min.ss.round"]
    fn minssround(a: __m128, b: __m128, src: __m128, k: __mmask8, round: i32) -> __m128;

}

#[cfg(test)]
mod tests {
    use std;
    use stdsimd_test::simd_test;

    use crate::core_arch::x86::*;

    // #[simd_test(enable = "avx512f")]
    // unsafe fn test__mm512_add_pd() {
    // }
}
