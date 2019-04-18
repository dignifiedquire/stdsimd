//! Intrinsics for Set Operations

use crate::{
    core_arch::{simd::*, x86::*},
    mem::{self, transmute},
};

/// Returns vector of type `__m512i` with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setzero_si512)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vxorps))] // FIXME: intrinsics guide says vpxorq
pub unsafe fn _mm512_setzero_si512() -> __m512i {
    // All-0 is a properly initialized __m512i
    mem::zeroed()
}

/// Returns vector of type `__m512d` with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setzero_pd)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vxorps))] // FIXME: intrinsics guide say vxorpd
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

/// Sets packed 32-bit integers in `dst` with the supplied values.
///
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_set_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding intstruction.
pub unsafe fn _mm512_set_epi32(
    e15: i32,
    e14: i32,
    e13: i32,
    e12: i32,
    e11: i32,
    e10: i32,
    e9: i32,
    e8: i32,
    e7: i32,
    e6: i32,
    e5: i32,
    e4: i32,
    e3: i32,
    e2: i32,
    e1: i32,
    e0: i32,
) -> __m512i {
    let r = i32x16(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    );
    transmute(r)
}

/// Sets packed 32-bit integers in `dst` with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setr_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding intstruction.
pub unsafe fn _mm512_setr_epi32(
    e15: i32,
    e14: i32,
    e13: i32,
    e12: i32,
    e11: i32,
    e10: i32,
    e9: i32,
    e8: i32,
    e7: i32,
    e6: i32,
    e5: i32,
    e4: i32,
    e3: i32,
    e2: i32,
    e1: i32,
    e0: i32,
) -> __m512i {
    let r = i32x16(
        e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0,
    );
    transmute(r)
}

/// Sets packed 64-bit integers in `dst` with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_set_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding intstruction.
pub unsafe fn _mm512_set_epi64(
    e7: i64,
    e6: i64,
    e5: i64,
    e4: i64,
    e3: i64,
    e2: i64,
    e1: i64,
    e0: i64,
) -> __m512i {
    let r = i64x8(e0, e1, e2, e3, e4, e5, e6, e7);
    transmute(r)
}

/// Sets packed 64-bit integers in `dst` with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setr_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding intstruction.
pub unsafe fn _mm512_setr_epi64(
    e7: i64,
    e6: i64,
    e5: i64,
    e4: i64,
    e3: i64,
    e2: i64,
    e1: i64,
    e0: i64,
) -> __m512i {
    let r = i64x8(e7, e6, e5, e4, e3, e2, e1, e0);
    transmute(r)
}

/// Broadcast 32-bit integer a to all elements of `dst`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_set1_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
// This intrinsic has no corresponding intstruction.
pub unsafe fn _mm512_set1_epi32(a: i32) -> __m512i {
    let r = i64x8::splat(a as i64);
    transmute(r)
}
