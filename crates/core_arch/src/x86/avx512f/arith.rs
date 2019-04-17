//! # Intrinsics for Arithmetic Operations
//!
//! This module contains the following intrinsics.
//! - Intrinsics for Addition Operations
//! - Intrinsics for Determining Minimum and Maximum Values
//! - Intrinsics for FP Fused Multiply-Add (FMA) Operations
//! - Intrinsics for Multiplication Operations
//! - Intrinsics for Subtraction Operations
//! - Intrinsics for Short Vector Math Library (SVML) Operations
//! - Intrinsics for Other Mathematics Operations
use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::{self, transmute},
    ptr,
};

#[cfg(test)]
use stdsimd_test::assert_instr;

// -- Intrinsics for Addition Operations

// ---  Intrinsics for FP Addition Operations

/// Adds packed float64 elements in a and b, and stores the result.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpadpd))]
pub unsafe fn _mm512_add_pd(a: __m512d, b: __m512d) -> __m512d {
    _mm512_add_round_pd(a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float64 elements in a and b, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpadpd))]
pub unsafe fn _mm512_mask_add_pd(src: __m512d, k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    _mm512_mask_add_round_pd(src, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float64 elements in a and , and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddpd))]
pub unsafe fn _mm512_maskz_add_pd(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    _mm512_maskz_add_round_pd(k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float64 elements in a and b using rounding control round, and stores the result.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_add_round_pd(a: __m512d, b: __m512d, round: i32) -> __m512d {
    let zero = _mm512_setzero_pd();
    _mm512_mask_add_round_pd(zero, 255u8 as i8, a, b, round)
}

/// Adds packed float64 elements in a and b using rounding control round, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_mask_add_round_pd(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    round: i32,
) -> __m512d {
    macro_rules! call {
        ($imm8:expr) => {
            addpd512(src, a, b, k, $imm8)
        };
    }
    constify_imm8!(round, call)
}

/// Adds packed float64 elements in a and b using rounding control round, and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_maskz_add_round_pd(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
    round: i32,
) -> __m512d {
    let res: i64x8 = transmute(_mm512_add_round_pd(a, b, round));
    let zero: i64x8 = transmute(_mm512_setzero_pd());
    transmute(simd_select_bitmask(k, res, zero))
}

/// Adds packed float32 elements in a and b, and stores the result.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_add_ps(a: __m512, b: __m512) -> __m512 {
    unimplemented!()
}

/// Adds packed float32 elements in a and b, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_mask_add_ps(src: __m512, k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unimplemented!()
}

/// Adds packed float32 elements in a and b, and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_maskz_add_ps(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    unimplemented!()
}

/// Adds packed float32 elements in a and b using rounding control round, and stores the result.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_add_round_ps(a: __m512, b: __m512, round: i32) -> __m512 {
    unimplemented!()
}

/// Adds packed float32 elements in a and b using rounding control round, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_mask_add_round_ps(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
    round: i32,
) -> __m512 {
    unimplemented!()
}

/// Adds packed float32 elements in a and b using rounding control round, and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_maskz_add_round_ps(k: __mmask16, a: __m512, b: __m512, round: i32) -> __m512 {
    unimplemented!()
}

/// Adds the lower float64 element in a and b using rounding control round, stores the result in the lower destination element, and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_add_round_sd(a: __m128d, b: __m128d, round: i32) -> __m128d {
    unimplemented!()
}

/// Adds the lower float64 element in a and b using rounding control round, stores the result in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_mask_add_round_sd(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
    round: i32,
) -> __m128d {
    unimplemented!()
}

/// Adds the lower float64 element in a and b using rounding control round, stores the result in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_maskz_add_round_sd(k: __mmask8, a: __m128d, b: __m128d, round: i32) -> __m128d {
    unimplemented!()
}

/// Adds the lower float64 element in a and b, stores the result in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_mask_add_sd(src: __m128d, k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unimplemented!()
}

/// Adds the lower float64 element in a and b, stores the result in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_maskz_add_sd(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    unimplemented!()
}

/// Add the lower float32 element in a and b using rounding control round, stores the result in the lower destination element, and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_add_round_ss(a: __m128, b: __m128, round: i32) -> __m128 {
    unimplemented!()
}

/// Add the lower float32 element in a and b using rounding control round, stores the result in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_mask_add_round_ss(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
    round: i32,
) -> __m128 {
    unimplemented!()
}

/// Add the lower float32 element in a and b using rounding control round, stores the result in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_maskz_add_round_ss(k: __mmask8, a: __m128, b: __m128, round: i32) -> __m128 {
    unimplemented!()
}

/// Add the lower float32 element in a and b, stores the result in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_mask_add_ss(src: __m128, k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unimplemented!()
}

/// Add the lower float32 element in a and b, stores the result in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_maskz_add_ss(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    unimplemented!()
}

// -- Intrinsics for Determining Minimum and Maximum Values

// -- Intrinsics for FP Fused Multiply-Add (FMA) Operations

// -- Intrinsics for Multiplication Operations

// -- Intrinsics for Subtraction Operations

// -- Intrinsics for Short Vector Math Library (SVML) Operations

// -- Intrinsics for Other Mathematics Operations

/// LLVM intrinsics used in the above functions
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.mask.add.pd.512"]
    // <8 x double> @llvm.x86.avx512.mask.add.pd.512(
    //   <8 x double> %a,
    //   <8 x double> %b,
    //   <8 x double> %c,
    //   i8 %mask,
    //   i32 8
    // )
    fn addpd512(src: __m512d, a: __m512d, b: __m512d, k: __mmask8, round: i32) -> __m512d;
}

#[cfg(test)]
mod tests {
    use std;
    use stdsimd_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm512_add_pd() {
        let a = _mm512_set_pd(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let b = _mm512_set_pd(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let expected = _mm512_set_pd(2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0);

        let r: __m512d = _mm512_add_pd(a, b);
        assert_eq_m512d(r, expected);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test__mm512_maskz_add_pd() {
        let a = _mm512_set_pd(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let b = _mm512_set_pd(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        let mask = 0b1010_1010u8 as i8;

        let expected = _mm512_set_pd(2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0);

        let r: __m512d = _mm512_maskz_add_pd(mask, a, b);
        assert_eq_m512d(r, expected);
    }
}
