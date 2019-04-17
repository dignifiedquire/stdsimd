//! Intrinsics for Set Operations
use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem,
};

/// Returns vector of type `__m512i` with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setzero_si512)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_setzero_si512() -> __m512i {
    // All-0 is a properly initialized __m512i
    mem::zeroed()
}

/// Returns vector of type `__m512d` with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setzero_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vxorpd))]
pub unsafe fn _mm512_setzero_pd() -> __m512d {
    // All-0 is a properly initialized __m512d
    mem::zeroed()
}

/// Returns vector of type `__m512` with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setzero_ps)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_setzero_ps() -> __m512 {
    // All-0 is a properly initialized __m512
    mem::zeroed()
}

/// Sets packed double-precision (64-bit) floating-point elements in returned
/// vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set_pd)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_set_pd(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
) -> __m512d {
    _mm512_setr_pd(h, g, f, e, d, c, b, a)
}

/// Sets packed double-precision (64-bit) floating-point elements in returned
/// vector with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_setr_pd)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_setr_pd(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
) -> __m512d {
    __m512d(a, b, c, d, e, f, g, h)
}

/// Sets packed single-precision (32-bit) floating-point elements in returned
/// vector with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set_pd)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_set_ps(
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
    i: f32,
    j: f32,
    k: f32,
    l: f32,
    m: f32,
    n: f32,
    o: f32,
    p: f32,
) -> __m512 {
    _mm512_setr_ps(p, o, n, m, l, k, j, i, h, g, f, e, d, c, b, a)
}

/// Sets packed single-precision (32-bit) floating-point elements in returned
/// vector with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_setr_ps)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding instruction.
pub unsafe fn _mm512_setr_ps(
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
    i: f32,
    j: f32,
    k: f32,
    l: f32,
    m: f32,
    n: f32,
    o: f32,
    p: f32,
) -> __m512 {
    __m512(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
}
