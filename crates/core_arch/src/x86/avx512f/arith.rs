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

const _MM_K0_REG: i32 = 0xFFFF;

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
pub unsafe fn _mm512_mask_add_pd(src: __m512d, k: i32, a: __m512d, b: __m512d) -> __m512d {
    _mm512_mask_add_round_pd(src, k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float64 elements in a and , and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddpd))]
pub unsafe fn _mm512_maskz_add_pd(k: i32, a: __m512d, b: __m512d) -> __m512d {
    _mm512_maskz_add_round_pd(k, a, b, _MM_FROUND_CUR_DIRECTION)
}

/// Adds packed float64 elements in a and b using rounding control round, and stores the result.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_add_round_pd(a: __m512d, b: __m512d, round: i32) -> __m512d {
    _mm512_maskz_add_round_pd(_MM_K0_REG, a, b, round)
}

/// Adds packed float64 elements in a and b using rounding control round, and stores the result using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_mask_add_round_pd(
    src: __m512d,
    k: i32,
    a: __m512d,
    b: __m512d,
    round: i32,
) -> __m512d {
    addpd512(src, k, a, b, round)
}

/// Adds packed float64 elements in a and b using rounding control round, and stores the result using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vaddpd))]
pub unsafe fn _mm512_maskz_add_round_pd(k: i32, a: __m512d, b: __m512d, round: i32) -> __m512d {
    let res = _mm512_add_round_pd(a, b, round);
    let zero = _mm512_setzero_pd();
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
    k: i32,
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
pub unsafe fn _mm_maskz_add_round_sd(k: i32, a: __m128d, b: __m128d, round: i32) -> __m128d {
    unimplemented!()
}

/// Adds the lower float64 element in a and b, stores the result in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_mask_add_sd(src: __m128d, k: i32, a: __m128d, b: __m128d) -> __m128d {
    unimplemented!()
}

/// Adds the lower float64 element in a and b, stores the result in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper element from a to the upper destination element.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_maskz_add_sd(k: i32, a: __m128d, b: __m128d) -> __m128d {
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
    k: i32,
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
pub unsafe fn _mm_maskz_add_round_ss(k: i32, a: __m128, b: __m128, round: i32) -> __m128 {
    unimplemented!()
}

/// Add the lower float32 element in a and b, stores the result in the lower destination element using writemask k (the element is copied from src when mask bit 0 is not set), and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_mask_add_ss(src: __m128, k: i32, a: __m128, b: __m128) -> __m128 {
    unimplemented!()
}

/// Add the lower float32 element in a and b, stores the result in the lower destination element using zeromask k (the element is zeroed out when mask bit 0 is not set), and copies the upper three packed elements from a to the upper destination elements.
///
/// [Intel's documentation]()
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm_maskz_add_ss(k: i32, a: __m128, b: __m128) -> __m128 {
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
    fn addpd512(src: __m512d, k: i32, a: __m512d, b: __m512d, round: i32) -> __m512d;
}
