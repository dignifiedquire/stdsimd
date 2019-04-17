//! Intrinsics for Addition Operations

use crate::{
    core_arch::{simd_llvm::*, x86::*},
    mem::transmute,
};

#[cfg(test)]
use stdsimd_test::assert_instr;

// --  Intrinsics for FP Addition Operations

/// Adds packed float64 elements in a and b, and stores the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_add_pd(a: __m512d, b: __m512d) -> __m512d {
    _mm512_add_round_pd(a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float64 elements in a and b, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_mask_add_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    _mm512_mask_add_round_pd(src, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float64 elements in a and , and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_maskz_add_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    _mm512_maskz_add_round_pd(k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float64 elements in a and b using rounding control round, and stores the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_round_pd)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_add_round_pd(a: __m512d, b: __m512d, round: i32) -> __m512d {
    let zero = _mm512_setzero_pd();
    _mm512_mask_add_round_pd(zero, 255u8 as i8, a, b, round)
}

/// Adds packed float64 elements in a and b using rounding control round, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_round_pd)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_mask_add_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    round: i32,
) -> __m512d {
    macro_rules! call {
        ($imm8:expr) => {
            addpd512(a, b, src, k, $imm8)
        };
    }
    constify_imm8!(round, call)
}

/// Adds packed float64 elements in a and b using rounding control round, and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_round_pd)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_maskz_add_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    round: i32,
) -> __m512d {
    let zero = _mm512_setzero_pd();
    _mm512_mask_add_round_pd(zero, k, a, b, round)
}

/// Adds packed float32 elements in a and b, and stores the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm512_add_ps(a: __m512, b: __m512) -> __m512 {
    _mm512_add_round_ps(a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float32 elements in a and b, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm512_mask_add_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    _mm512_mask_add_round_ps(src, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float32 elements in a and b, and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm512_maskz_add_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    _mm512_maskz_add_round_ps(k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float32 elements in a and b using rounding control round, and stores the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_round_ps)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm512_add_round_ps(a: __m512, b: __m512, round: i32) -> __m512 {
    let zero = _mm512_setzero_ps();
    _mm512_mask_add_round_ps(zero, 0xFFFFu16 as __mmask16, a, b, round)
}

/// Adds packed float32 elements in a and b using rounding control round, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_round_ps)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddps))]
pub unsafe fn _mm512_mask_add_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
    round: i32,
) -> __m512 {
    macro_rules! call {
        ($imm8:expr) => {
            addps512(a, b, src, k, $imm8)
        };
    }
    constify_imm8!(round, call)
}

/// Adds packed float32 elements in a and b using rounding control round, and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_round_ps)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_maskz_add_round_ps(k: __mmask16, a: __m512, b: __m512, round: i32) -> __m512 {
    let zero = _mm512_setzero_ps();
    _mm512_mask_add_round_ps(zero, k, a, b, round)
}

/// Adds the lower float64 element in a and b using rounding control round, stores the result in the lower destination element, and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_add_round_sd)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddsd))]
pub unsafe fn _mm_add_round_sd(a: __m128d, b: __m128d, round: i32) -> __m128d {
    let zero = _mm_setzero_pd();
    _mm_mask_add_round_sd(zero, -1i8 as __mmask8, a, b, round)
}

/// Adds the lower float64 element in a and b using rounding control round, stores the result in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_add_round_sd)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddsd))]
pub unsafe fn _mm_mask_add_round_sd(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
    round: i32,
) -> __m128d {
    macro_rules! call {
        ($imm8:expr) => {
            addsdround(a, b, src, k, $imm8)
        };
    }
    constify_imm8!(round, call)
}

/// Adds the lower float64 element in a and b using rounding control round, stores the result in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_add_round_sd)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddsd))]
pub unsafe fn _mm_maskz_add_round_sd(k: __mmask8, a: __m128d, b: __m128d, round: i32) -> __m128d {
    let zero = _mm_setzero_pd();
    _mm_mask_add_round_sd(zero, k, a, b, round)
}

/// Adds the lower float64 element in a and b, stores the result in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_add_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddsd))]
pub unsafe fn _mm_mask_add_sd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    _mm_mask_add_round_sd(src, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds the lower float64 element in a and b, stores the result in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_add_sd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddsd))]
pub unsafe fn _mm_maskz_add_sd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    let zero = _mm_setzero_pd();
    addsdround(a, b, zero, k, _MM_FROUND_CUR_DIRECTION)
}

/// Add the lower float32 element in a and b using rounding control round, stores the result in the lower destination element, and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_add_round_ss)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddss))]
pub unsafe fn _mm_add_round_ss(a: __m128, b: __m128, round: i32) -> __m128 {
    let zero = _mm_setzero_ps();
    _mm_mask_add_round_ss(zero, -1i8 as __mmask8, a, b, round)
}

/// Add the lower float32 element in a and b using rounding control round, stores the result in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_add_round_ss)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddss))]
pub unsafe fn _mm_mask_add_round_ss(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
    round: i32,
) -> __m128 {
    macro_rules! call {
        ($imm8:expr) => {
            addssround(a, b, src, k, $imm8)
        };
    }
    constify_imm8!(round, call)
}

/// Add the lower float32 element in a and b using rounding control round, stores the result in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_add_round_ss)
#[inline]
#[target_feature(enable = "avx512f")]
// FIXME: #[cfg_attr(test, assert_instr(vaddss))]
pub unsafe fn _mm_maskz_add_round_ss(k: __mmask8, a: __m128, b: __m128, round: i32) -> __m128 {
    let zero = _mm_setzero_ps();
    _mm_mask_add_round_ss(zero, k, a, b, round)
}

/// Add the lower float32 element in a and b, stores the result in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_add_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddss))]
pub unsafe fn _mm_mask_add_ss(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    _mm_mask_add_round_ss(src, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Add the lower float32 element in a and b, stores the result in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_add_ss)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddss))]
pub unsafe fn _mm_maskz_add_ss(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    let zero = _mm_setzero_ps();
    _mm_mask_add_round_ss(zero, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

// --  Intrinsics for Integer Addition Operations

/// Adds packed int32 elements in a and b, and stores the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#test=_mm512_add_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_add_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i32x16(), b.as_i32x16()))
}

/// Adds packed int32 elements in a and b, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#test=_mm512_mask_add_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_mask_add_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let res = simd_add(a.as_i32x16(), b.as_i32x16());
    transmute(simd_select_bitmask(k, res, src.as_i32x16()))
}

/// Adds packed int32 elements in a and b, and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#test=_mm512_maskz_add_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_maskz_add_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let res = simd_add(a.as_i32x16(), b.as_i32x16());
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, res, zero))
}

/// Adds packed int64 elements in a and b, and stores the result.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#test=_mm512_add_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_add_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i64x8(), b.as_i64x8()))
}

/// Adds packed int64 elements in a and b, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#test=_mm512_mask_add_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_mask_add_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let res = simd_add(a.as_i64x8(), b.as_i64x8());
    transmute(simd_select_bitmask(k, res, src.as_i64x8()))
}

/// Adds packed int64 elements in a and b, and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#test=_mm512_maskz_add_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_maskz_add_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let res = simd_add(a.as_i64x8(), b.as_i64x8());
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, res, zero))
}

/// LLVM intrinsics used in the above functions
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.mask.add.pd.512"]
    fn addpd512(a: __m512d, b: __m512d, src: __m512d, k: __mmask8, round: i32) -> __m512d;
    #[link_name = "llvm.x86.avx512.mask.add.ps.512"]
    fn addps512(a: __m512, b: __m512, src: __m512, k: __mmask16, round: i32) -> __m512;
    #[link_name = "llvm.x86.avx512.mask.add.sd.round"]
    fn addsdround(a: __m128d, b: __m128d, src: __m128d, k: __mmask8, round: i32) -> __m128d;
    #[link_name = "llvm.x86.avx512.mask.add.ss.round"]
    fn addssround(a: __m128, b: __m128, src: __m128, k: __mmask8, round: i32) -> __m128;
}

#[cfg(test)]
mod tests {
    use std;
    use stdsimd_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm512_add_pd() {
        let a = _mm512_set_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm512_set_pd(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let expected = _mm512_set_pd(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);

        let r: __m512d = _mm512_add_pd(a, b);
        assert_eq_m512d(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm512_maskz_add_pd() {
        let a = _mm512_set_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm512_set_pd(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let mask = 0b1010_1010u8 as i8;

        let expected = _mm512_set_pd(2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0);

        let r: __m512d = _mm512_maskz_add_pd(mask, a, b);
        assert_eq_m512d(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm512_add_ps() {
        let a = _mm512_set_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        );
        let b = _mm512_set_ps(
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        );
        let expected = _mm512_set_ps(
            2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
        );

        let r: __m512 = _mm512_add_ps(a, b);
        assert_eq_m512(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm512_maskz_add_ps() {
        let a = _mm512_set_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        );
        let b = _mm512_set_ps(
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        );
        let mask = 0b1010_1010_1010_1010u16 as i16;

        let expected = _mm512_set_ps(
            2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0, 2.0, 0.0, 4.0, 0.0, 6.0, 0.0, 8.0, 0.0,
        );

        let r: __m512 = _mm512_maskz_add_ps(mask, a, b);
        assert_eq_m512(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm_mask_add_sd() {
        let a = _mm_set_pd(1.0, 2.0);
        let b = _mm_set_pd(1.0, 1.0);
        let src = _mm_set_pd(9.0, 9.0);

        let mask = 0b1010_1010u8 as i8;

        let expected = _mm_set_pd(1.0, 9.0);

        let r = _mm_mask_add_sd(src, mask, a, b);
        assert_eq_m128d(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm_add_round_sd() {
        let a = _mm_set_pd(8.0, 2.0);
        let b = _mm_set_pd(1.0, 1.0);

        let expected = _mm_set_pd(8.0, 3.0);

        let r = _mm_add_round_sd(a, b, _MM_FROUND_CUR_DIRECTION);
        assert_eq_m128d(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm_maskz_add_sd() {
        let a = _mm_set_pd(1.0, 2.0);
        let b = _mm_set_pd(2.0, 3.0);
        let mask = 255u8 as i8;

        let expected = _mm_set_pd(1.0, 5.0);

        let r = _mm_maskz_add_sd(mask, a, b);
        assert_eq_m128d(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm_mask_add_ss() {
        let a = _mm_set_ps(1.0, 2.0, 1.0, 2.0);
        let b = _mm_set_ps(1.0, 1.0, 1.0, 1.0);
        let src = _mm_set_ps(9.0, 9.0, 9.0, 9.0);

        let mask = 1i8;

        let expected = _mm_set_ps(1.0, 2.0, 1.0, 3.0);

        let r = _mm_mask_add_ss(src, mask, a, b);
        assert_eq_m128(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm_add_round_ss() {
        let a = _mm_set_ps(8.0, 2.0, 8.0, 2.0);
        let b = _mm_set_ps(1.0, 1.0, 1.0, 1.0);

        let expected = _mm_set_ps(8.0, 2.0, 8.0, 3.0);

        let r = _mm_add_round_ss(a, b, _MM_FROUND_CUR_DIRECTION);
        assert_eq_m128(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm_maskz_add_ss() {
        let a = _mm_set_ps(1.0, 2.0, 1.0, 2.0);
        let b = _mm_set_ps(2.0, 3.0, 2.0, 3.0);
        let mask = 255u8 as i8;

        let expected = _mm_set_ps(1.0, 2.0, 1.0, 5.0);

        let r = _mm_maskz_add_ss(mask, a, b);
        assert_eq_m128(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm512_maskz_add_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let mask = 0b0000_1111_1111_1111 as i16;

        let expected = _mm512_set_epi32(0, 0, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);

        let r = _mm512_maskz_add_epi32(mask, a, b);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm512_add_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let expected = _mm512_set_epi32(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);

        let r = _mm512_add_epi32(a, b);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm512_maskz_add_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_epi64(1, 1, 1, 1, 1, 1, 1, 1);
        let mask = 0b0000__1111 as i8;

        let expected = _mm512_set_epi64(0, 0, 0, 0, 6, 7, 8, 9);

        let r = _mm512_maskz_add_epi64(mask, a, b);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm512_add_epi64() {
        let a = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm512_set_epi64(1, 1, 1, 1, 1, 1, 1, 1);

        let expected = _mm512_set_epi64(2, 3, 4, 5, 6, 7, 8, 9);

        let r = _mm512_add_epi64(a, b);
        assert_eq_m512i(r, expected);
    }
}
